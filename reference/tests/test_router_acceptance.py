"""Lead-owned REF-006 tests: scalar decimal arithmetic and explicit FP32 boundaries."""

import importlib
import math
from decimal import Decimal, localcontext

import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")
ARRAYS = ("x", "weight", "correction_bias")
U = 2.0**-24


@pytest.fixture
def router():
    return importlib.import_module("tq_reference.router")


def budget(k):
    n = 2 * k + 16
    assert n * U < 1
    return n * U / (1 - n * U)


def _case(phase=0, k=8):
    x = ((np.arange(16).reshape(4, 4) * 7 + phase) % 17 - 8) / 8
    weight = ((np.arange(48).reshape(12, 4) * 5 + phase * 3) % 31 - 15) / 16
    bias = (((np.arange(12) * 11 + phase) % 17 - 8) / 64).astype(np.float32)
    return {"x": x, "weight": weight, "correction_bias": bias, "top_k": k}


def logit_case(values, bias=None, k=8):
    values = np.asarray(values, dtype=np.float32)
    return {
        "x": np.ones((1, 1), dtype=np.float32),
        "weight": values.reshape(-1, 1),
        "correction_bias": np.zeros(values.size, dtype=np.float32)
        if bias is None
        else np.asarray(bias, dtype=np.float32),
        "top_k": k,
    }


def _d(value):
    return Decimal.from_float(float(value))


def _f32(value):
    with np.errstate(over="ignore", under="ignore"):
        return np.float32(float(value))


def scalar_sigmoid(logit):
    """Decimal exp followed by the three distinct float32 operation boundaries."""
    with localcontext() as ctx:
        ctx.prec = 80
        if logit < -100:
            return np.float32(0)
        if logit > 150:
            return np.float32(1)
        exponential = _f32((-_d(logit)).exp())
        if np.isinf(exponential):
            return np.float32(0)
        denominator = _f32(Decimal(1) + _d(exponential))
        return _f32(Decimal(1) / _d(denominator))


def scalar_oracle(x, weight, correction_bias, *, top_k=8):
    """Projection fixtures have exact dyadic sums; no NumPy exp, GEMM or reductions."""
    tokens, width = x.shape
    experts = weight.shape[0]
    logits = np.empty((tokens, experts), dtype=np.float32)
    ids = np.empty((tokens, top_k), dtype=np.int64)
    weights = np.empty((tokens, top_k), dtype=np.float32)
    with localcontext() as ctx:
        ctx.prec = 80
        for t in range(tokens):
            scores = []
            for e in range(experts):
                value = sum(_d(_f32(x[t, h])) * _d(_f32(weight[e, h])) for h in range(width))
                logits[t, e] = _f32(value)
                scores.append(scalar_sigmoid(logits[t, e]))
            choice = [_f32(_d(s) + _d(b)) for s, b in zip(scores, correction_bias, strict=True)]
            selected = sorted(sorted(range(experts), key=lambda e: (-float(choice[e]), e))[:top_k])
            denominator = np.float32(0)
            for e in selected:
                denominator = _f32(_d(denominator) + _d(scores[e]))
            denominator = _f32(_d(denominator) + _d(np.float32(1e-20)))
            ids[t] = selected
            for slot, e in enumerate(selected):
                divided = _f32(_d(scores[e]) / _d(denominator))
                weights[t, slot] = _f32(_d(divided) * Decimal("2.5"))
    return logits, weights, ids


def assert_weights(actual, expected, k):
    assert actual.dtype == np.float32
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all()
    assert np.all(actual >= 0)
    np.testing.assert_array_equal(actual[expected == 0], 0)
    for got, want in zip(actual.flat, expected.flat, strict=True):
        if want != 0:
            assert abs(float(got) - float(want)) / abs(float(want)) <= budget(k)
    for got, want in [(actual.ravel(), expected.ravel()), *zip(actual, expected, strict=True)]:
        denominator = math.hypot(*want)
        if denominator == 0:
            np.testing.assert_array_equal(got, want)
        else:
            assert math.hypot(*(got.astype(float) - want.astype(float))) / denominator <= budget(k)


def assert_result(actual, expected):
    assert isinstance(actual, tuple) and len(actual) == 3
    for value in actual:
        assert isinstance(value, np.ndarray)
        assert value.flags.c_contiguous
    logits, weights, ids = actual
    assert logits.dtype == np.float32
    assert ids.dtype == np.int64
    assert np.isfinite(logits).all()
    np.testing.assert_array_equal(logits, expected[0])
    np.testing.assert_array_equal(ids, expected[2])
    assert np.all(np.diff(ids, axis=-1) > 0)
    assert_weights(weights, expected[1], ids.shape[-1])


def _cast_case():
    weight = np.zeros((12, 2), dtype=np.float64)
    weight[:4] = [1, -(2**24)]
    weight[4:, 1] = np.arange(1, 9) / 16
    return {
        "x": np.array([[2**24 + 1.0, 1.0]]),
        "weight": weight,
        "correction_bias": (np.arange(12) / 1024).astype(np.float32),
        "top_k": 8,
    }


def test_oracle_analytical_sigmoid_and_normalization():
    assert scalar_sigmoid(0) == np.float32(0.5)
    rounded_three_quarters = np.array(0x3F400001, dtype=np.uint32).view(np.float32)
    assert scalar_sigmoid(np.float32(math.log(3))) == rounded_three_quarters
    case = logit_case(np.zeros(8))
    logits, weights, ids = scalar_oracle(**case)
    np.testing.assert_array_equal(logits, np.zeros((1, 8)))
    np.testing.assert_array_equal(weights, np.full((1, 8), 0.3125))
    np.testing.assert_array_equal(ids, [np.arange(8)])
    np.testing.assert_array_equal(scalar_oracle(**logit_case([-1000] * 8))[1], 0)


@pytest.mark.parametrize(
    "mistake", ["ids", "pairing", "scale", "uniform", "small", "zeros", "fp64"]
)
def test_weight_gate_rejects_deliberately_broken_results(mistake):
    case = logit_case([-46, -10, -5, -3, -2, -1, 0, 1, 2], k=8)
    expected = scalar_oracle(**case)
    wrong = tuple(a.copy() for a in expected)
    if mistake == "ids":
        wrong[2][0, 0] = 0
    elif mistake == "pairing":
        wrong[1][0] = wrong[1][0, ::-1]
    elif mistake == "scale":
        wrong[1][:] *= np.float32(2.5)
    elif mistake == "uniform":
        wrong[1][:] = np.float32(2.5 / 8)
    elif mistake == "small":
        wrong[1][0, 0] *= np.float32(1.001)
    elif mistake == "zeros":
        wrong[1][:] = 0
    else:
        wrong = (wrong[0], wrong[1].astype(np.float64), wrong[2])
    with pytest.raises(AssertionError):
        assert_result(wrong, expected)


def test_comparison_does_not_hide_zero_or_tiny_outputs():
    zero = np.zeros((1, 8), dtype=np.float32)
    tiny = np.full((1, 8), np.nextafter(np.float32(0), np.float32(1)))
    assert_weights(zero, zero, 8)
    assert_weights(tiny, tiny, 8)
    with pytest.raises(AssertionError):
        assert_weights(zero, tiny, 8)
    with pytest.raises(AssertionError):
        assert_weights(tiny, zero, 8)


@pytest.mark.parametrize("phase", range(12))
@pytest.mark.parametrize("k", [1, 3, 8, 12])
def test_dense_projection_and_routes(router, phase, k):
    case = _case(phase, k)
    assert_result(router.route(**case), scalar_oracle(**case))


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("name", ["x", "weight"])
def test_supplied_float_dtypes(router, name, dtype):
    case = _case()
    case[name] = case[name].astype(dtype)
    assert_result(router.route(**case), scalar_oracle(**case))


def test_cast_to_fp32_occurs_before_projection(router):
    case = _cast_case()
    expected = scalar_oracle(**case)
    assert_result(router.route(**case), expected)
    np.testing.assert_array_equal(expected[0][0, :4], 0)
    np.testing.assert_array_equal(expected[2], [np.arange(4, 12)])
    wrong = dict(case)
    wrong["weight"] = np.zeros_like(case["weight"])
    wrong["weight"][:, 1] = [1] * 4 + list(np.arange(1, 9) / 16)
    with pytest.raises(AssertionError):
        assert_result(router.route(**wrong), expected)


def test_weight_cast_occurs_before_projection(router):
    case = logit_case(np.zeros(8))
    case["x"] = np.array([[1.0, 1.0]])
    case["weight"] = np.array([[2**24 + 1.0, -(2**24)]] * 8)
    actual = router.route(**case)
    np.testing.assert_array_equal(actual[0], 0)
    assert_result(actual, scalar_oracle(**case))


def test_correction_bias_changes_selection_only(router):
    case = logit_case(np.linspace(-2, 2, 12), bias=[2] * 8 + [0] * 4)
    actual = router.route(**case)
    np.testing.assert_array_equal(actual[2], [np.arange(8)])
    assert_result(actual, scalar_oracle(**case))
    other = dict(case, correction_bias=case["correction_bias"] + np.float32(0.125))
    for got, want in zip(router.route(**other), actual, strict=True):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("values", [[0] * 12, [100] * 12, [-100] * 12])
def test_ties_choose_lowest_ids_and_canonical_order(router, values):
    case = logit_case(values)
    actual = router.route(**case)
    np.testing.assert_array_equal(actual[2], [np.arange(8)])
    assert_result(actual, scalar_oracle(**case))


def test_bias_addition_really_is_fp32(router):
    values = np.linspace(-2, 2, 10, dtype=np.float32)
    bias = np.full(10, 2**24, dtype=np.float32)
    case = logit_case(values, bias=bias)
    actual = router.route(**case)
    np.testing.assert_array_equal(actual[2], [np.arange(8)])


def test_output_pairing_is_by_increasing_expert_id(router):
    case = logit_case([1, -1, 3, -3, 2, -2, 4, -4, 0, 5, -5, 6])
    actual = router.route(**case)
    assert_result(actual, scalar_oracle(**case))
    assert not np.all(np.diff(actual[1][0]) >= 0)
    assert not np.all(np.diff(actual[1][0]) <= 0)


@pytest.mark.parametrize("logit", [-46, -50, -70, -88, -89, -100, -1000, 1000])
def test_epsilon_and_fp32_exponential_boundaries(router, logit):
    case = logit_case([logit] * 8)
    actual = router.route(**case)
    assert_result(actual, scalar_oracle(**case))
    if -88 <= logit <= -46:
        assert 0 < math.fsum(actual[1][0]) < 2.5
    if logit <= -89:
        np.testing.assert_array_equal(actual[1], 0)


def test_no_float64_or_fused_normalization(router):
    actual = router.route(**logit_case([0, 0, 0], k=3))
    sequential = np.float32(np.float32(0.5) / np.float32(1.5)) * np.float32(2.5)
    fused = np.float32(2.5 / 3)
    assert sequential != fused
    np.testing.assert_array_equal(actual[1], np.full((1, 3), sequential))


def test_normalization_uses_the_fixed_fp32_addition_order(router):
    tiny_logit = np.float32(-16.63553237915039)
    assert scalar_sigmoid(tiny_logit) == np.float32(2.0**-24)
    case = logit_case([100] + [tiny_logit] * 7)
    actual = router.route(**case)
    # Each half-ULP increment rounds back to one; summing tiny terms first differs.
    assert actual[1][0, 0] == np.float32(2.5)
    np.testing.assert_array_equal(actual[1][0, 1:], np.full(7, np.float32(2.5 * 2.0**-24)))


def test_glm_hidden_width_with_dense_dyadic_projection(router):
    x = np.tile(np.array([1, 0.5, -1, -0.5], dtype=np.float32), 1024)[None, :]
    columns = np.arange(4096)[None, :]
    rows = np.arange(8)[:, None]
    weight = (((columns * 17 + rows * 7) % 31 - 15) / 1024).astype(np.float32)
    case = {"x": x, "weight": weight, "correction_bias": np.zeros(8, dtype=np.float32)}
    assert_result(router.route(**case), scalar_oracle(**case))


def test_allowed_underflow_ignores_external_warning_policy_locally(router):
    case = logit_case([-88] * 7 + [100])
    with np.errstate(all="warn"):
        policy = np.geterr()
        assert_result(router.route(**case), scalar_oracle(**case))
        tiny_input = _case()
        tiny_input["x"][:] = 1e-300
        assert_result(router.route(**tiny_input), scalar_oracle(**tiny_input))
        assert np.geterr() == policy


def test_extreme_finite_logits_and_bias(router):
    maximum = np.finfo(np.float32).max
    case = logit_case([-maximum, maximum] * 6, bias=[maximum, -maximum] * 6, k=6)
    actual = router.route(**case)
    assert_result(actual, scalar_oracle(**case))
    np.testing.assert_array_equal(actual[2], [[0, 2, 4, 6, 8, 10]])
    np.testing.assert_array_equal(actual[1], 0)


def test_all_288_experts_can_be_selected(router):
    case = logit_case(np.zeros(288), bias=np.arange(288), k=8)
    actual = router.route(**case)
    np.testing.assert_array_equal(actual[2], [np.arange(280, 288)])
    np.testing.assert_array_equal(actual[1], np.full((1, 8), 0.3125))
    case["top_k"] = 288
    assert_result(router.route(**case), scalar_oracle(**case))


def test_default_top8(router):
    case = _case()
    explicit = router.route(**case)
    del case["top_k"]
    for got, want in zip(router.route(**case), explicit, strict=True):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("k", [np.int8(1), np.int32(3), np.int64(8), np.uint64(12)])
def test_numpy_integer_top_k(router, k):
    case = _case(k=k)
    assert_result(router.route(**case), scalar_oracle(**case))


def test_empty_tokens(router):
    case = _case()
    case["x"] = np.empty((0, 4), dtype=np.float64)
    assert_result(router.route(**case), scalar_oracle(**case))


@pytest.mark.parametrize("layout", ["readonly", "strided", "negative"])
def test_layouts_are_supported_without_aliasing_or_mutation(router, layout):
    case = _case()
    for name in ARRAYS:
        original = case[name]
        if layout == "strided":
            padded = np.zeros(tuple(2 * n for n in original.shape), dtype=original.dtype)
            view = padded[(slice(None, None, 2),) * original.ndim]
            view[:] = original
            case[name] = view
        elif layout == "negative":
            case[name] = original[(slice(None, None, -1),) * original.ndim]
        case[name].flags.writeable = False
    snapshots = {name: case[name].tobytes() for name in ARRAYS}
    actual = router.route(**case)
    assert_result(actual, scalar_oracle(**case))
    for name in ARRAYS:
        assert case[name].tobytes() == snapshots[name]
        assert all(not np.shares_memory(output, case[name]) for output in actual)
    again = router.route(**case)
    for got, want in zip(actual, again, strict=True):
        np.testing.assert_array_equal(got, want)
        assert not np.shares_memory(got, want)


@pytest.mark.parametrize("name", ARRAYS)
@pytest.mark.parametrize("bad", [None, [1.0], 1.0])
def test_requires_arrays(router, name, bad):
    case = _case()
    case[name] = bad
    with pytest.raises(router.RouterError, match=r"numpy\.ndarray"):
        router.route(**case)


@pytest.mark.parametrize("name", ["x", "weight"])
@pytest.mark.parametrize("dtype", [np.int32, np.uint8, np.bool_, np.complex128, object])
def test_rejects_nonfloating_dtype(router, name, dtype):
    case = _case()
    case[name] = case[name].astype(dtype)
    with pytest.raises(router.RouterError, match="floating dtype"):
        router.route(**case)


@pytest.mark.parametrize("dtype", [np.float16, np.float64, np.int32, np.complex64, object])
def test_bias_must_stay_float32(router, dtype):
    case = _case()
    case["correction_bias"] = case["correction_bias"].astype(dtype)
    with pytest.raises(router.RouterError, match="float32"):
        router.route(**case)


@pytest.mark.parametrize("name", ARRAYS)
@pytest.mark.parametrize("rank", [0, 3, 4])
def test_rejects_wrong_dimensions(router, name, rank):
    case = _case()
    case[name] = np.zeros((1,) * rank, dtype=case[name].dtype)
    with pytest.raises(router.RouterError, match="dimensions"):
        router.route(**case)


@pytest.mark.parametrize("name", ARRAYS)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_rejects_nonfinite_inputs(router, name, bad):
    case = _case()
    case[name].flat[0] = bad
    with pytest.raises(router.RouterError, match="non-finite"):
        router.route(**case)


@pytest.mark.parametrize("name", ["x", "weight"])
@pytest.mark.parametrize("value", [np.finfo(np.float64).max, -np.finfo(np.float64).max])
def test_rejects_overflow_in_cast(router, name, value):
    case = _case()
    case[name].flat[0] = value
    with pytest.raises(router.RouterError, match="float32 range"):
        router.route(**case)


@pytest.mark.parametrize("which", ["x_width", "weight_width", "bias_size"])
def test_rejects_shape_mismatch(router, which):
    case = _case()
    name = {"x_width": "x", "weight_width": "weight", "bias_size": "correction_bias"}[which]
    case[name] = case[name][..., :-1]
    with pytest.raises(router.RouterError, match="shape mismatch"):
        router.route(**case)


@pytest.mark.parametrize("which", ["width", "no_experts", "one_expert"])
def test_rejects_empty_required_dimensions(router, which):
    case = _case(k=1)
    if which == "width":
        case["x"], case["weight"] = case["x"][:, :0], case["weight"][:, :0]
    else:
        experts = 0 if which == "no_experts" else 1
        case["weight"] = case["weight"][:experts]
        case["correction_bias"] = case["correction_bias"][:experts]
    with pytest.raises(router.RouterError, match="empty"):
        router.route(**case)


BAD_K = [0, -1, 13, True, np.bool_(False), 8.0, "8", [8], np.array(8), 8j, 10**5000]


@pytest.mark.parametrize("bad", BAD_K, ids=[f"case_{i}" for i in range(len(BAD_K))])
def test_rejects_invalid_top_k(router, bad):
    case = _case(k=bad)
    with pytest.raises(router.RouterError, match="top_k"):
        router.route(**case)


@pytest.mark.parametrize("name", ["weight", "correction_bias"])
def test_empty_tokens_still_validate_values(router, name):
    case = _case()
    case["x"] = np.empty((0, 4), dtype=np.float64)
    case[name].flat[0] = np.nan
    with pytest.raises(router.RouterError, match="non-finite"):
        router.route(**case)


@pytest.mark.parametrize("sign", [1, -1])
def test_rejects_nonfinite_projection_before_sigmoid(router, sign):
    maximum = np.finfo(np.float32).max
    case = logit_case([2 * sign, 1], k=1)
    case["x"][0, 0] = maximum
    with pytest.raises(router.RouterError, match="overflow in projection"):
        router.route(**case)


def test_error_type_and_numeric_policy_unchanged(router):
    assert issubclass(router.RouterError, ValueError)
    policy = np.geterr()
    case = logit_case([-1000] * 8)
    router.route(**case)
    assert np.geterr() == policy
    case["x"][0, 0] = np.nan
    with pytest.raises(router.RouterError):
        router.route(**case)
    assert np.geterr() == policy
