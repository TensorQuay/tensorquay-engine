"""Independent REF-002 acceptance: scalar Decimal math and explicit contract boundaries."""

import importlib
import math
from decimal import Decimal, localcontext

import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")
DATA_NAMES = ("x", "w13", "w2", "topk_weights")
ARRAY_NAMES = (*DATA_NAMES, "topk_ids")


@pytest.fixture
def moe():
    return importlib.import_module("tq_reference.moe_experts")


def _values(shape, phase):
    positions = np.arange(math.prod(shape), dtype=np.int64).reshape(shape)
    return ((positions * 7 + phase) % 29 - 14).astype(np.float64) / 8


def _case(phase=0):
    return {
        "x": _values((3, 3), phase + 1),
        "w13": _values((5, 8, 3), phase + 5),
        "w2": _values((5, 3, 4), phase + 11),
        "topk_ids": np.array([[4, 0, 2], [3, 4, 1], [0, 2, 3]], dtype=np.int64),
        "topk_weights": np.array([[1.5, 0.75, 0.25], [0, 0.125, 0.5], [0.5, 0.25, 2.0]]),
        "swiglu_limit": 10.0,
    }


def _scalar_case(gate=1.0, up=1.0, down=1.0, weight=1.0, limit=10.0):
    return {
        "x": np.ones((1, 1), dtype=np.float64),
        "w13": np.array([[[gate], [up]]], dtype=np.float64),
        "w2": np.array([[[down]]], dtype=np.float64),
        "topk_ids": np.zeros((1, 1), dtype=np.int64),
        "topk_weights": np.array([[weight]], dtype=np.float64),
        "swiglu_limit": limit,
    }


def _decimal(value):
    return Decimal.from_float(float(value))


def _oracle(x, w13, w2, topk_ids, topk_weights, *, swiglu_limit):
    """80-digit scalar expert path; no NumPy products, activation or reductions."""
    tokens, width = x.shape
    intermediate = w2.shape[-1]
    result = np.zeros((tokens, width), dtype=np.float64)
    with localcontext() as ctx:
        ctx.prec = 80
        limit = _decimal(swiglu_limit)
        for t in range(tokens):
            combined = [Decimal(0) for _ in range(width)]
            for slot in sorted(range(topk_ids.shape[1]), key=lambda j: int(topk_ids[t, j])):
                expert = int(topk_ids[t, slot])
                activated = []
                for i in range(intermediate):
                    gate = sum(
                        _decimal(x[t, h]) * _decimal(w13[expert, i, h]) for h in range(width)
                    )
                    up = sum(
                        _decimal(x[t, h]) * _decimal(w13[expert, intermediate + i, h])
                        for h in range(width)
                    )
                    gate = min(gate, limit)
                    up = min(max(up, -limit), limit)
                    activated.append(gate / (1 + (-gate).exp()) * up)
                for h in range(width):
                    projected = sum(
                        _decimal(w2[expert, h, i]) * activated[i] for i in range(intermediate)
                    )
                    combined[h] += _decimal(topk_weights[t, slot]) * projected
            result[t] = [float(value) for value in combined]
    return result


def _assert_close(actual, expected):
    assert isinstance(actual, np.ndarray)
    assert actual.dtype == np.float64
    assert actual.shape == expected.shape
    assert actual.flags.c_contiguous
    assert np.isfinite(actual).all()
    for got, want in [(actual.ravel(), expected.ravel()), *zip(actual, expected, strict=True)]:
        denominator = math.hypot(*want)
        if denominator == 0:
            np.testing.assert_array_equal(got, want)
        else:
            assert math.hypot(*(got - want)) / denominator <= 1e-12


def test_oracle_known_sigmoid_and_exact_zero():
    expected = np.array([[2.1931757358900147]])
    _assert_close(_oracle(**_scalar_case(up=2, down=3, weight=0.5)), expected)
    np.testing.assert_array_equal(_oracle(**_scalar_case(gate=0)), 0)
    np.testing.assert_array_equal(_oracle(**_scalar_case(weight=0)), 0)


def test_oracle_silu_difference_identity():
    case = _scalar_case(gate=2)
    case["w13"] = np.array([[[2.0], [1.0]], [[-2.0], [1.0]]])
    case["w2"] = np.array([[[1.0]], [[-1.0]]])
    case["topk_ids"] = np.array([[1, 0]], dtype=np.int64)
    case["topk_weights"] = np.ones((1, 2), dtype=np.float64)
    np.testing.assert_array_equal(_oracle(**case), [[2.0]])


@pytest.mark.parametrize("mistake", ["swap", "normalize", "scale_twice", "slot", "fp32", "row"])
def test_oracle_gate_catches_broken_results(mistake):
    case = _case()
    expected = _oracle(**case)
    if mistake == "swap":
        case["w13"] = np.concatenate((case["w13"][:, 4:], case["w13"][:, :4]), axis=1)
    elif mistake == "normalize":
        case["topk_weights"] /= case["topk_weights"].sum(axis=1, keepdims=True)
    elif mistake == "scale_twice":
        case["topk_weights"] *= 2.5
    elif mistake == "slot":
        case["topk_ids"] = case["topk_ids"][:, ::-1].copy()
    wrong = _oracle(**case)
    if mistake == "fp32":
        wrong = expected.astype(np.float32).astype(np.float64)
    elif mistake == "row":
        wrong = expected[::-1].copy()
    with pytest.raises(AssertionError):
        _assert_close(wrong, expected)


@pytest.mark.parametrize("phase", range(7))
def test_experts_match_independent_decimal(moe, phase):
    case = _case(phase)
    _assert_close(moe.moe_experts(**case), _oracle(**case))


def test_full_expert_count_top8_and_last_bank(moe):
    case = {
        "x": _values((2, 3), 2),
        "w13": _values((288, 8, 3), 5),
        "w2": _values((288, 3, 4), 9),
        "topk_ids": np.array(
            [[287, 255, 0, 1, 16, 64, 128, 256], [3, 17, 63, 127, 254, 256, 286, 287]],
            dtype=np.int32,
        ),
        "topk_weights": np.array([[1, 3, 2, 0, 4, 7, 5, 6], [8, 7, 6, 5, 4, 3, 2, 1]]) / 16,
        "swiglu_limit": 10.0,
    }
    _assert_close(moe.moe_experts(**case), _oracle(**case))


@pytest.mark.parametrize("gate", [-12.0, -10.0, -1.0, 0.0, 10.0, np.nextafter(10.0, np.inf), 12.0])
@pytest.mark.parametrize("up", [-12.0, -10.0, 0.0, 10.0, 12.0])
def test_asymmetric_clamp_edges(moe, gate, up):
    case = _scalar_case(gate=gate, up=up, down=1.25, weight=0.3)
    _assert_close(moe.moe_experts(**case), _oracle(**case))


@pytest.mark.parametrize("limit", [1, np.int64(2), np.float32(0.5), 7.5, np.float64(12)])
def test_supplied_limit_is_used(moe, limit):
    case = _scalar_case(gate=8, up=-9, limit=limit)
    _assert_close(moe.moe_experts(**case), _oracle(**case))


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_id_dtypes_and_joint_slot_permutation(moe, dtype):
    case = _case()
    case["topk_ids"] = case["topk_ids"].astype(dtype)
    original = moe.moe_experts(**case)
    case["topk_ids"] = case["topk_ids"][:, ::-1].copy()
    case["topk_weights"] = case["topk_weights"][:, ::-1].copy()
    np.testing.assert_array_equal(moe.moe_experts(**case), original)


def test_correct_intermediate_shards_add_to_full_result(moe):
    case = _case()
    full = moe.moe_experts(**case)
    shards = []
    for low, high in [(0, 2), (2, 4)]:
        w13 = np.concatenate((case["w13"][:, low:high], case["w13"][:, 4 + low : 4 + high]), axis=1)
        shard = case | {"w13": w13, "w2": case["w2"][:, :, low:high]}
        shards.append(moe.moe_experts(**shard))
    _assert_close(shards[0] + shards[1], full)
    wrong = sum(
        moe.moe_experts(
            **(
                case
                | {
                    "w13": case["w13"][:, j * 4 : (j + 1) * 4],
                    "w2": case["w2"][:, :, j * 2 : (j + 1) * 2],
                }
            )
        )
        for j in range(2)
    )
    with pytest.raises(AssertionError):
        _assert_close(wrong, full)


@pytest.mark.parametrize("mode", ["readonly", "strided", "negative"])
def test_layouts_fresh_output_and_no_mutation(moe, mode):
    case = _case()
    for name in ARRAY_NAMES:
        original = case[name]
        if mode == "strided":
            expanded = np.empty(
                (*original.shape[:-1], original.shape[-1] * 2), dtype=original.dtype
            )
            expanded[..., ::2] = original
            case[name] = expanded[..., ::2]
        elif mode == "negative":
            case[name] = original[..., ::-1].copy()[..., ::-1]
        case[name].flags.writeable = False
    before = {name: case[name].copy() for name in ARRAY_NAMES}
    result = moe.moe_experts(**case)
    _assert_close(result, _oracle(**case))
    for name in ARRAY_NAMES:
        np.testing.assert_array_equal(case[name], before[name])
        assert not np.shares_memory(result, case[name])
    again = moe.moe_experts(**case)
    assert not np.shares_memory(result, again)
    np.testing.assert_array_equal(result, again)


@pytest.mark.parametrize("name", ["w13", "w2"])
@pytest.mark.parametrize("poison", [np.nan, np.inf, -np.inf])
def test_unselected_poison_is_not_inspected(moe, name, poison):
    case = _case()
    case["topk_ids"][:] = [4, 0, 2]
    expected = _oracle(**case)
    case[name][[1, 3]] = poison
    _assert_close(moe.moe_experts(**case), expected)


@pytest.mark.parametrize("zero", ["x", "topk_weights"])
def test_exact_zero_outputs(moe, zero):
    case = _case()
    case[zero].fill(0)
    np.testing.assert_array_equal(moe.moe_experts(**case), 0)


def test_empty_token_batch_never_inspects_experts(moe):
    case = _case()
    for name in ("x", "topk_ids", "topk_weights"):
        case[name] = case[name][:0]
    case["w13"].fill(np.nan)
    case["w2"].fill(np.inf)
    result = moe.moe_experts(**case)
    _assert_close(result, np.empty((0, 3), dtype=np.float64))
    assert not np.shares_memory(result, case["x"])


@pytest.mark.parametrize("gate", [-1000.0, -np.finfo(np.float64).max])
def test_negative_silu_tail_does_not_overflow(moe, gate):
    np.testing.assert_array_equal(moe.moe_experts(**_scalar_case(gate=gate)), 0)


def test_representable_negative_silu_tail(moe):
    case = _scalar_case(gate=-720.0)
    expected = _oracle(**case).item()
    actual = moe.moe_experts(**case).item()
    assert expected < 0 and actual < 0
    # A subnormal exp(-720) can lose half an ulp before multiplication by 720.
    assert abs(actual - expected) <= 721 * np.nextafter(0.0, 1.0)


@pytest.mark.parametrize("name", ARRAY_NAMES)
def test_reject_non_array(moe, name):
    case = _case()
    case[name] = case[name].tolist()
    with pytest.raises(moe.MoeError, match=r"numpy\.ndarray"):
        moe.moe_experts(**case)


@pytest.mark.parametrize("name", DATA_NAMES)
@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.complex128])
def test_reject_data_dtype(moe, name, dtype):
    case = _case()
    case[name] = case[name].astype(dtype)
    with pytest.raises(moe.MoeError, match="float64"):
        moe.moe_experts(**case)


@pytest.mark.parametrize("dtype", [np.float64, np.uint64, np.int16, np.bool_])
def test_reject_id_dtype(moe, dtype):
    case = _case()
    case["topk_ids"] = case["topk_ids"].astype(dtype)
    with pytest.raises(moe.MoeError, match="int32 or int64"):
        moe.moe_experts(**case)


@pytest.mark.parametrize("name", ARRAY_NAMES)
@pytest.mark.parametrize("kind", ["scalar", "extra"])
def test_reject_rank(moe, name, kind):
    case = _case()
    case[name] = np.array(1, dtype=case[name].dtype) if kind == "scalar" else case[name][None]
    with pytest.raises(moe.MoeError, match="dimensions"):
        moe.moe_experts(**case)


@pytest.mark.parametrize(
    "name,axis",
    [
        ("x", 1),
        ("w13", 0),
        ("w13", 1),
        ("w13", 2),
        ("w2", 0),
        ("w2", 1),
        ("w2", 2),
        ("topk_ids", 1),
        ("topk_weights", 1),
    ],
)
def test_reject_empty_non_token_dimensions(moe, name, axis):
    case = _case()
    index = [slice(None)] * case[name].ndim
    index[axis] = slice(0, 0)
    case[name] = case[name][tuple(index)]
    with pytest.raises(moe.MoeError, match="empty"):
        moe.moe_experts(**case)


@pytest.mark.parametrize(
    "name,axis",
    [
        ("x", 0),
        ("x", 1),
        ("w13", 0),
        ("w13", 1),
        ("w13", 2),
        ("w2", 0),
        ("w2", 1),
        ("w2", 2),
        ("topk_ids", 0),
        ("topk_ids", 1),
        ("topk_weights", 0),
        ("topk_weights", 1),
    ],
)
def test_reject_inconsistent_shapes(moe, name, axis):
    case = _case()
    index = [slice(None)] * case[name].ndim
    index[axis] = slice(0, -1)
    case[name] = case[name][tuple(index)]
    with pytest.raises(moe.MoeError, match="shape mismatch"):
        moe.moe_experts(**case)


def test_reject_more_slots_than_experts(moe):
    case = _scalar_case()
    # Empty tokens isolate the shape error from inevitable duplicate/range errors at K > E.
    case["x"] = case["x"][:0]
    case["topk_ids"] = np.empty((0, 2), dtype=np.int64)
    case["topk_weights"] = np.empty((0, 2), dtype=np.float64)
    with pytest.raises(moe.MoeError, match="shape mismatch"):
        moe.moe_experts(**case)


@pytest.mark.parametrize("bad", [-1, 5, np.iinfo(np.int64).max, np.iinfo(np.int64).min])
def test_reject_out_of_range_ids_including_upstream_sentinel(moe, bad):
    case = _case()
    case["topk_ids"][0, 0] = bad
    with pytest.raises(moe.MoeError, match="range"):
        moe.moe_experts(**case)


def test_reject_duplicate_even_with_zero_weight(moe):
    case = _case()
    case["topk_ids"][1, 0] = case["topk_ids"][1, 1]
    with pytest.raises(moe.MoeError, match="duplicate"):
        moe.moe_experts(**case)


@pytest.mark.parametrize("name", DATA_NAMES)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_reject_required_nonfinite_values(moe, name, bad):
    case = _case()
    case[name].flat[0] = bad
    with pytest.raises(moe.MoeError, match="non-finite"):
        moe.moe_experts(**case)


@pytest.mark.parametrize("name", ["w13", "w2"])
def test_selected_zero_weight_does_not_allow_poison(moe, name):
    case = _case()
    case["topk_ids"][2, 2] = 1  # Expert 3 is now selected only by the zero-weight slot.
    case[name][3].fill(np.nan)
    with pytest.raises(moe.MoeError, match="non-finite"):
        moe.moe_experts(**case)


def test_reject_negative_routing_weight(moe):
    case = _case()
    case["topk_weights"][0, 0] = -np.nextafter(0.0, 1.0)
    with pytest.raises(moe.MoeError, match="nonnegative"):
        moe.moe_experts(**case)


@pytest.mark.parametrize(
    "bad",
    [
        True,
        np.bool_(False),
        "10",
        None,
        1 + 0j,
        np.array(10.0),
        [10.0],
        0,
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
        pytest.param(10**5000, id="huge_integer"),
    ],
)
def test_reject_invalid_limit_with_bounded_diagnostic(moe, bad):
    with pytest.raises(moe.MoeError, match="swiglu_limit"):
        moe.moe_experts(**_scalar_case(limit=bad))


def test_limit_required_even_for_empty_batch(moe):
    case = _case()
    for name in ("x", "topk_ids", "topk_weights"):
        case[name] = case[name][:0]
    del case["swiglu_limit"]
    with pytest.raises(TypeError):
        moe.moe_experts(**case)


@pytest.mark.parametrize(
    "stage", ["gate/up projection", "activation", "down projection", "weighted output", "combine"]
)
def test_reject_first_arithmetic_overflow(moe, stage):
    largest = np.finfo(np.float64).max
    case = _scalar_case()
    if stage == "gate/up projection":
        case["x"].fill(2)
        case["w13"].fill(largest)
    elif stage == "activation":
        case = _scalar_case(gate=2, up=largest, limit=largest)
    elif stage == "down projection":
        case = _scalar_case(gate=2, up=2, down=largest)
    elif stage == "weighted output":
        case = _scalar_case(gate=2, up=2, weight=largest)
    else:
        case["w13"] = np.ones((2, 2, 1), dtype=np.float64)
        case["w2"] = np.full((2, 1, 1), largest, dtype=np.float64)
        case["topk_ids"] = np.array([[1, 0]], dtype=np.int64)
        case["topk_weights"] = np.ones((1, 2), dtype=np.float64)
    with pytest.raises(moe.MoeError, match=f"overflow in {stage}"):
        moe.moe_experts(**case)
