"""Independent REF-001 tests; the scalar oracle expands K/V rather than absorbing Q."""

import importlib
import math
from dataclasses import FrozenInstanceError, is_dataclass
from decimal import Decimal, localcontext
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")
DATA_NAMES = ("q", "latent", "w_uk", "w_uv")
ARRAY_NAMES = (*DATA_NAMES, "mask")
DIM_NAMES = ("num_heads", "qk_nope_head_dim", "v_head_dim")


@pytest.fixture
def mla():
    """Import only for implementation checks, allowing the oracle to be checked first."""
    return importlib.import_module("tq_reference.mla_attention")


def _values(shape, phase):
    indices = np.arange(math.prod(shape), dtype=np.int64).reshape(shape)
    return ((indices * 7 + phase) % 31 - 15).astype(np.float64) / 16


def _case(seed=0):
    return {
        "q": _values((3, 2, 3), seed + 1),
        "latent": _values((5, 4), seed + 7),
        "w_uk": _values((2, 3, 4), seed + 11),
        "w_uv": _values((2, 2, 4), seed + 19),
        "mask": np.array(
            [[True, True, False, True, False], [False, True, True, False, True], [False] * 5]
        ),
        "scale": 1 / 16,
    }


def _scalar_case(q=1.0, latent=(1.0,), key=1.0, value=1.0, scale=1.0):
    return {
        "q": np.array([[[q]]], dtype=np.float64),
        "latent": np.array(latent, dtype=np.float64).reshape(-1, 1),
        "w_uk": np.array([[[key]]], dtype=np.float64),
        "w_uv": np.array([[[value]]], dtype=np.float64),
        "mask": np.ones((1, len(latent)), dtype=bool),
        "scale": scale,
    }


def _decimal(value):
    return Decimal.from_float(float(value))


def _expanded_oracle(q, latent, w_uk, w_uv, mask, *, scale):
    """Scalar expanded attention, with exact float input conversion and 80-digit arithmetic."""
    tokens, heads, key_dim = q.shape
    value_dim, rank = w_uv.shape[1:]
    out = np.zeros((tokens, heads, value_dim), dtype=np.float64)
    out_latent = np.zeros((tokens, heads, rank), dtype=np.float64)
    lse = np.full((tokens, heads), -np.inf, dtype=np.float64)
    with localcontext() as ctx:
        ctx.prec = 80
        c = [[_decimal(x) for x in row] for row in latent]
        for t in range(tokens):
            selected = np.flatnonzero(mask[t]).tolist()
            if not selected:
                continue
            for h in range(heads):
                keys = [
                    [
                        sum(c[n][r] * _decimal(w_uk[h, d, r]) for r in range(rank))
                        for d in range(key_dim)
                    ]
                    for n in selected
                ]
                values = [
                    [
                        sum(c[n][r] * _decimal(w_uv[h, d, r]) for r in range(rank))
                        for d in range(value_dim)
                    ]
                    for n in selected
                ]
                scores = [
                    sum(_decimal(q[t, h, d]) * key[d] for d in range(key_dim)) * _decimal(scale)
                    for key in keys
                ]
                maximum = max(scores)
                weights = [(score - maximum).exp() for score in scores]
                denominator = sum(weights)
                probabilities = [weight / denominator for weight in weights]
                lse[t, h] = float(maximum + denominator.ln())
                for d in range(value_dim):
                    out[t, h, d] = float(
                        sum(p * value[d] for p, value in zip(probabilities, values, strict=True))
                    )
                for r in range(rank):
                    out_latent[t, h, r] = float(
                        sum(p * c[n][r] for p, n in zip(probabilities, selected, strict=True))
                    )
    return SimpleNamespace(out=out, out_latent=out_latent, lse=lse)


def _assert_vector(actual, expected):
    reference_norm = math.hypot(*expected.ravel())
    if reference_norm == 0:
        np.testing.assert_array_equal(actual, expected)
    else:
        error_norm = math.hypot(*(actual - expected).ravel())
        assert error_norm / reference_norm <= 1e-12


def _assert_result(actual, expected):
    for name in ("out", "out_latent", "lse"):
        value, reference = getattr(actual, name), getattr(expected, name)
        assert isinstance(value, np.ndarray)
        assert value.dtype == np.float64
        assert value.shape == reference.shape
        assert value.flags.c_contiguous
        if name == "lse":
            np.testing.assert_array_equal(np.isneginf(value), np.isneginf(reference))
            finite = np.isfinite(reference)
            assert np.isfinite(value[finite]).all()
            assert np.all(
                np.abs(value[finite] - reference[finite])
                <= 1e-12 * np.maximum(1, np.abs(reference[finite]))
            )
        else:
            assert np.isfinite(value).all()
            _assert_vector(value, reference)
            for t in range(reference.shape[0]):
                for h in range(reference.shape[1]):
                    _assert_vector(value[t, h], reference[t, h])


def test_oracle_one_key_and_empty_selection():
    case = _scalar_case(q=9, latent=(2, 5), key=3, value=4)
    case["q"] = np.array([[[9.0]], [[-7.0]]])
    case["mask"] = np.array([[True, False], [False, False]])
    result = _expanded_oracle(**case)
    np.testing.assert_array_equal(result.out.ravel(), [8, 0])
    np.testing.assert_array_equal(result.out_latent.ravel(), [2, 0])
    np.testing.assert_array_equal(result.lse.ravel(), [54, -np.inf])


def test_oracle_uniform_attention():
    result = _expanded_oracle(**_scalar_case(q=0, latent=(2, 5), key=3, value=4))
    assert result.out.item() == 14
    assert result.out_latent.item() == 3.5
    assert result.lse.item() == math.log(2)


def test_oracle_large_common_offset():
    result = _expanded_oracle(**_scalar_case(latent=(1, 1), key=1e16, value=3))
    assert result.out.item() == 3
    assert result.out_latent.item() == 1
    assert result.lse.item() == 1e16


def test_oracle_exponential_underflow_limit():
    result = _expanded_oracle(**_scalar_case(latent=(-800, 800)))
    assert result.out.item() == 800
    assert result.out_latent.item() == 800
    assert result.lse.item() == 800


@pytest.mark.parametrize("mistake", ["scale", "mask", "head", "fp32", "latent", "lse"])
def test_oracle_and_gate_detect_deliberately_wrong_results(mistake):
    case = _case()
    expected = _expanded_oracle(**case)
    wrong = SimpleNamespace(**{name: getattr(expected, name).copy() for name in vars(expected)})
    if mistake == "scale":
        wrong = _expanded_oracle(**(case | {"scale": 1 / math.sqrt(512)}))
    elif mistake == "mask":
        wrong = _expanded_oracle(**(case | {"mask": np.ones((3, 5), dtype=bool)}))
    elif mistake == "head":
        wrong.out = expected.out[:, ::-1, :].copy()
    elif mistake == "fp32":
        wrong.out = expected.out.astype(np.float32).astype(np.float64)
    elif mistake == "latent":
        wrong.out_latent[0, 0, 0] += 0.1
    else:
        wrong.lse[0, 0] += 0.1
    with pytest.raises(AssertionError):
        _assert_result(wrong, expected)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("scale", [1 / 16, 0.37, 2.0])
def test_ref001_expanded_decimal_equivalence(mla, seed, scale):
    case = _case(seed) | {"scale": scale}
    _assert_result(mla.mla_attention_core(**case), _expanded_oracle(**case))


@pytest.mark.parametrize(
    "case",
    [
        _scalar_case(q=0, latent=(2, 5), key=3, value=4),
        _scalar_case(latent=(1, 1), key=1e16, value=3),
        _scalar_case(latent=(-800, 800)),
        _scalar_case(latent=(0,), key=-3, value=5),
        _scalar_case(q=0, latent=(-2, 2), value=7),
    ],
    ids=["uniform", "large-offset", "exp-underflow", "zero-latent", "exact-cancellation"],
)
def test_ref001_analytical_boundaries(mla, case):
    _assert_result(mla.mla_attention_core(**case), _expanded_oracle(**case))


def test_ref001_scale_uses_model_key_width_not_latent_width(mla):
    case = {
        "q": np.zeros((1, 1, 256)),
        "latent": np.zeros((2, 512)),
        "w_uk": np.zeros((1, 256, 512)),
        "w_uv": np.zeros((1, 1, 512)),
        "mask": np.ones((1, 2), dtype=bool),
        "scale": 1 / 16,
    }
    case["q"][0, 0, 0] = 16
    case["latent"][1, 0] = 1
    case["w_uk"][0, 0, 0] = 1
    case["w_uv"][0, 0, 0] = 1
    result = mla.mla_attention_core(**case)
    probability = 1 / (1 + math.exp(-1))
    assert abs(result.out.item() - probability) < 1e-15
    assert abs(result.lse.item() - math.log1p(math.e)) < 1e-15
    expected_latent = np.zeros((1, 1, 512))
    expected_latent[0, 0, 0] = probability
    _assert_vector(result.out_latent, expected_latent)


def test_ref001_one_selected_key_per_query_and_head(mla):
    case = _case()
    case["mask"][:] = False
    case["mask"][0, 0] = case["mask"][1, 3] = case["mask"][2, 4] = True
    result = mla.mla_attention_core(**case)
    _assert_result(result, _expanded_oracle(**case))
    for t, n in enumerate((0, 3, 4)):
        for h in range(2):
            np.testing.assert_array_equal(result.out_latent[t, h], case["latent"][n])


def test_ref001_all_masked_skips_unused_overflow(mla):
    case = _scalar_case(q=1e308, key=2)
    case["mask"][:] = False
    _assert_result(mla.mla_attention_core(**case), _expanded_oracle(**case))


def test_ref001_inactive_query_does_not_invalidate_active_query(mla):
    case = _scalar_case(q=1, key=2)
    case["q"] = np.array([[[1.0]], [[1e308]]])
    case["mask"] = np.array([[True], [False]])
    _assert_result(mla.mla_attention_core(**case), _expanded_oracle(**case))


def test_ref001_does_not_evaluate_unselected_scores(mla):
    case = _scalar_case(q=2, latent=(1, 1e308))
    case["mask"][0, 1] = False
    _assert_result(mla.mla_attention_core(**case), _expanded_oracle(**case))


def _view(array, layout):
    if layout == "fortran":
        return np.asfortranarray(array)
    if layout == "negative":
        return array[..., ::-1].copy()[..., ::-1]
    if layout == "strided":
        storage = np.empty((*array.shape[:-1], array.shape[-1] * 2), dtype=array.dtype)
        view = storage[..., ::2]
        view[...] = array
        return view
    view = array.copy()
    view.flags.writeable = False
    return view


@pytest.mark.parametrize("layout", ["fortran", "negative", "strided", "readonly"])
def test_ref001_views_no_mutation_or_aliases(mla, layout):
    case = _case()
    expected = _expanded_oracle(**case)
    inputs = {name: _view(case[name], layout) for name in ARRAY_NAMES}
    saved = {name: array.copy() for name, array in inputs.items()}
    result = mla.mla_attention_core(**inputs, scale=case["scale"])
    _assert_result(result, expected)
    outputs = (result.out, result.out_latent, result.lse)
    for name, array in inputs.items():
        np.testing.assert_array_equal(array, saved[name])
        assert all(not np.shares_memory(array, output) for output in outputs)
    for i, output in enumerate(outputs):
        assert all(not np.shares_memory(output, other) for other in outputs[i + 1 :])
    result.out[:] = 777
    _assert_result(mla.mla_attention_core(**inputs, scale=case["scale"]), expected)


def test_ref001_frozen_result_and_error_type(mla):
    result = mla.mla_attention_core(**_case())
    assert is_dataclass(result)
    assert isinstance(result, mla.MlaCoreResult)
    assert issubclass(mla.MlaError, ValueError)
    with pytest.raises(FrozenInstanceError):
        result.out = np.empty(1)


@pytest.mark.parametrize("name", ARRAY_NAMES)
@pytest.mark.parametrize("value", [None, [], 1.0])
def test_ref001_requires_ndarrays(mla, name, value):
    with pytest.raises(mla.MlaError, match=r"numpy\.ndarray"):
        mla.mla_attention_core(**(_case() | {name: value}))


@pytest.mark.parametrize("name", DATA_NAMES)
@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int64, np.complex128, object])
def test_ref001_requires_float64(mla, name, dtype):
    case = _case()
    case[name] = case[name].astype(dtype)
    with pytest.raises(mla.MlaError, match="float64"):
        mla.mla_attention_core(**case)


@pytest.mark.parametrize("dtype", [np.uint8, np.int64, np.float64, object])
def test_ref001_requires_boolean_mask(mla, dtype):
    case = _case()
    case["mask"] = case["mask"].astype(dtype)
    with pytest.raises(mla.MlaError, match="bool"):
        mla.mla_attention_core(**case)


@pytest.mark.parametrize("name", ARRAY_NAMES)
def test_ref001_rejects_wrong_rank(mla, name):
    case = _case()
    case[name] = case[name].ravel()
    with pytest.raises(mla.MlaError, match="dimensions"):
        mla.mla_attention_core(**case)


EMPTY_DIMENSIONS = [(name, axis) for name in ARRAY_NAMES for axis in range(_case()[name].ndim)]


@pytest.mark.parametrize(("name", "axis"), EMPTY_DIMENSIONS)
def test_ref001_rejects_empty_dimensions(mla, name, axis):
    case = _case()
    shape = list(case[name].shape)
    shape[axis] = 0
    case[name] = np.empty(shape, dtype=case[name].dtype)
    with pytest.raises(mla.MlaError, match="empty"):
        mla.mla_attention_core(**case)


@pytest.mark.parametrize(
    ("name", "axis"),
    [(name, axis) for name, axis in EMPTY_DIMENSIONS if (name, axis) != ("w_uv", 1)],
)
def test_ref001_rejects_inconsistent_shapes(mla, name, axis):
    case = _case()
    shape = list(case[name].shape)
    shape[axis] += 1
    case[name] = np.resize(case[name], shape)
    with pytest.raises(mla.MlaError, match="shape mismatch"):
        mla.mla_attention_core(**case)


@pytest.mark.parametrize("name", DATA_NAMES)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_ref001_rejects_nonfinite_even_when_all_masked(mla, name, bad):
    case = _case()
    case[name].flat[0] = bad
    case["mask"][:] = False
    with pytest.raises(mla.MlaError, match="non-finite"):
        mla.mla_attention_core(**case)


@pytest.mark.parametrize(
    "scale",
    [True, np.bool_(True), None, "1", 1j, np.array(1.0), np.array([1.0]), Decimal(1)],
)
def test_ref001_rejects_nonscalar_real_scale(mla, scale):
    with pytest.raises(mla.MlaError, match="scale"):
        mla.mla_attention_core(**(_case() | {"scale": scale}))


@pytest.mark.parametrize("scale", [0, -1, np.nan, np.inf, -np.inf, 10**400])
def test_ref001_rejects_unrepresentable_or_nonpositive_scale(mla, scale):
    with pytest.raises(mla.MlaError, match="scale"):
        mla.mla_attention_core(**(_case() | {"scale": scale}))


@pytest.mark.parametrize("scale", [1, np.int64(2), np.float32(0.3), np.float64(0.7), 5e-324])
def test_ref001_accepts_real_scalar_types_and_subnormal_scale(mla, scale):
    case = _case() | {"scale": scale}
    _assert_result(mla.mla_attention_core(**case), _expanded_oracle(**case))


def test_ref001_scale_has_no_implicit_default(mla):
    case = _case()
    del case["scale"]
    with pytest.raises(TypeError, match="scale"):
        mla.mla_attention_core(**case)


@pytest.mark.parametrize(
    ("case", "stage"),
    [
        (_scalar_case(q=2, key=1e308), "absorbed query"),
        (_scalar_case(q=1e200, latent=(1e200,)), "score"),
        (_scalar_case(q=1e150, latent=(1e150,), scale=1e100), "score"),
        (_scalar_case(latent=(1e308, -1e308)), "shifted score"),
        (_scalar_case(q=0, latent=(1e308,), value=2), "output"),
    ],
)
def test_ref001_reports_evaluated_overflow_as_domain_error(mla, case, stage):
    with pytest.raises(mla.MlaError, match=f"overflow in {stage}"):
        mla.mla_attention_core(**case)


def test_ref001_latent_accumulation_at_float64_limit(mla):
    # Exact averaging stays at max. Float64 product/sum rounding can overflow first.
    case = _scalar_case()
    case["latent"] = np.array([[np.finfo(np.float64).max, 0], [np.finfo(np.float64).max, -7 / 16]])
    case["w_uk"] = np.array([[[0.0, 1.0]]])
    case["w_uv"] = np.zeros((1, 1, 2))
    case["mask"] = np.ones((1, 2), dtype=bool)
    expected = _expanded_oracle(**case)
    assert expected.out_latent[0, 0, 0] == np.finfo(np.float64).max
    try:
        result = mla.mla_attention_core(**case)
    except mla.MlaError as error:
        assert "overflow in latent output" in str(error)
    else:
        # Another reduction order can remain finite; it must still be accurate.
        _assert_result(result, expected)


@pytest.mark.parametrize("sign", [-1, 1])
def test_ref001_lse_stays_finite_at_both_float64_limits(mla, sign):
    offset = sign * np.finfo(np.float64).max
    case = _scalar_case(q=offset, latent=(1, 1), value=3)
    result = mla.mla_attention_core(**case)
    assert result.lse.item() == offset
    assert result.out.item() == 3
    assert result.out_latent.item() == 1


@pytest.mark.parametrize("target", ["scale", "negative-dimension", "positive-dimension"])
def test_ref001_extreme_integer_errors_do_not_format_unbounded_values(mla, target):
    too_large = 10**5000
    if target == "scale":
        with pytest.raises(mla.MlaError, match="scale"):
            mla.mla_attention_core(**(_case() | {"scale": too_large}))
    else:
        size = -too_large if target == "negative-dimension" else too_large
        fragment = "> 0" if size < 0 else "rows must equal"
        with pytest.raises(mla.MlaError, match=fragment):
            _split(mla, _projection(), num_heads=size)


def _projection():
    return np.arange(40, dtype=np.float64).reshape(10, 4) / 8


def _split(mla, weight, **overrides):
    dims = {"num_heads": 2, "qk_nope_head_dim": 3, "v_head_dim": 2} | overrides
    return mla.split_kv_b(weight, **dims)


@pytest.mark.parametrize("layout", ["fortran", "negative", "strided", "readonly"])
def test_ref001_projection_split_is_per_head_interleaved_and_copied(mla, layout):
    raw = _view(_projection(), layout)
    before = raw.copy()
    keys, values = _split(mla, raw)
    assert keys.shape == (2, 3, 4)
    assert values.shape == (2, 2, 4)
    for h in range(2):
        np.testing.assert_array_equal(keys[h], raw[h * 5 : h * 5 + 3])
        np.testing.assert_array_equal(values[h], raw[h * 5 + 3 : h * 5 + 5])
    for array in (keys, values):
        assert array.dtype == np.float64
        assert array.flags.c_contiguous
        assert not np.shares_memory(raw, array)
    assert not np.shares_memory(keys, values)
    keys[:] = -999
    np.testing.assert_array_equal(raw, before)


def test_ref001_split_then_attention_agrees_with_expanded_oracle(mla):
    case = _case()
    expected = _expanded_oracle(**case)
    weight = np.concatenate([case["w_uk"], case["w_uv"]], axis=1).reshape(10, 4)
    case["w_uk"], case["w_uv"] = _split(mla, weight)
    _assert_result(mla.mla_attention_core(**case), expected)


@pytest.mark.parametrize("name", DIM_NAMES)
@pytest.mark.parametrize("bad", [True, np.bool_(False), 1.0, "2", np.array(2), None])
def test_ref001_split_dimensions_require_integers(mla, name, bad):
    with pytest.raises(mla.MlaError, match="integer"):
        _split(mla, _projection(), **{name: bad})


@pytest.mark.parametrize("name", DIM_NAMES)
@pytest.mark.parametrize("bad", [0, -1])
def test_ref001_split_dimensions_require_positive_values(mla, name, bad):
    with pytest.raises(mla.MlaError, match="> 0"):
        _split(mla, _projection(), **{name: bad})


def test_ref001_split_accepts_numpy_integers(mla):
    keys, values = _split(
        mla,
        _projection(),
        num_heads=np.int64(2),
        qk_nope_head_dim=np.int32(3),
        v_head_dim=np.uint8(2),
    )
    np.testing.assert_array_equal(keys[1], _projection()[5:8])
    np.testing.assert_array_equal(values[1], _projection()[8:10])


@pytest.mark.parametrize("heads", [3, 10**100, np.int64(np.iinfo(np.int64).max)])
def test_ref001_split_wrong_row_count_including_huge_dimension(mla, heads):
    with pytest.raises(mla.MlaError, match="rows must equal"):
        _split(mla, _projection(), num_heads=heads)


@pytest.mark.parametrize(
    ("weight", "message"),
    [
        (None, "numpy.ndarray"),
        (_projection().astype(np.float32), "float64"),
        (_projection().ravel(), "dimensions"),
        (np.empty((0, 4)), "empty"),
        (np.empty((10, 0)), "empty"),
        (np.full((10, 4), np.nan), "non-finite"),
        (np.full((10, 4), np.inf), "non-finite"),
        (np.full((10, 4), -np.inf), "non-finite"),
    ],
)
def test_ref001_split_rejects_malformed_weight(mla, weight, message):
    with pytest.raises(mla.MlaError, match=message):
        _split(mla, weight)
