# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "numpy==2.3.3",
#     "torch==2.8.0",
#     "transformers @ https://github.com/huggingface/transformers/archive/770e4c40d0436082a52dc380f07a9d3f389c99d4.zip",
# ]
# ///
"""Generate or check the REF-001 upstream fixture with Transformers' own GLM-5.3-Flash MLA code.

Independent of tq_reference: this script never imports it. Transformers is installed from the
pinned source archive, and the three upstream files whose code this fixture depends on are checked
against pinned SHA-256 digests on every run. Importing Transformers and constructing the attention
module naturally runs much more upstream code than that, so the claim made here is narrower.

Inputs are ours, not upstream's. The two projection weights, ``q_resid`` and ``latent`` are drawn
here from the case's recorded seed, and ``topk_indices`` is a literal list; upstream computes none
of them. They are fixture inputs, and both comparison arms consume them unchanged.

What upstream produces, and what an acceptance test should treat as the expectation, is the
projected query ``inputs.q`` together with everything under ``upstream``: the expanded
``key_states`` and ``value_states``, the boolean ``mask`` and the attention ``output``. Those come
from these four installed callables, whose three defining files are byte-identical to the commit:

- ``Glm5NextTextAttention.q_b_proj``: the query expansion projection, giving ``inputs.q``;
- ``Glm5NextTextAttention.expand_kv``: latent rows to per-head key and value states;
- ``Glm5NextTextAttention.build_attention_mask_from_topk``: top-k indices to a boolean mask;
- ``transformers.integrations.sdpa_attention.sdpa_attention_forward``: attention itself.

No model forward, indexer, ``o_proj`` or norm contributes to any stored value. Only the glue moving
tensors between those four calls is written here, independently; no upstream implementation is
reproduced. Attention runs in float64 on CPU inside ``sdpa_kernel(SDPBackend.MATH)``, and a CPU
profiler trace around the call records which SDPA kernel actually ran: the MATH event must be
observed and no fused event may appear.

``config.validate_architecture`` forces ``qk_rope_head_dim == 0`` (NoPE) and
``num_attention_heads == num_key_value_heads``, so ``num_key_value_groups == 1`` and
``sdpa_attention_forward`` never calls ``repeat_kv`` or the GQA path. ``scaling`` is
``qk_head_dim**-0.5``, which for GLM-5.3-Flash is ``256**-0.5`` while the latent is 512 wide; the
``glm_width`` case reproduces that distinction. Supplying a boolean mask makes upstream set
``is_causal=False``, so SDPA sees exactly the top-k set mask and adds no causal mask of its own.

Transformers is Apache-2.0; no upstream code is stored in this repository, only generated values.

    uv run tools/transformers_mla_fixture.py           # regenerate the two fixtures
    uv run tools/transformers_mla_fixture.py --check   # re-run upstream; exit 1 on a difference
"""

import argparse
import base64
import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from importlib.metadata import distributions
from pathlib import Path

import numpy as np
import torch
import transformers
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextAttention

UPSTREAM = "https://github.com/huggingface/transformers"
COMMIT = "770e4c40d0436082a52dc380f07a9d3f389c99d4"
ARCHIVE = f"{UPSTREAM}/archive/{COMMIT}.zip"

# SHA-256 of the three upstream files that define the callables this fixture executes, pinned to
# COMMIT and asserted on every run, including --check.
FILES_SHA256 = {
    "models/glm5_next/modeling_glm5_next.py": (
        "431eae74b857ec2a8e38a96f9f76cdc0b366bf7e35bd739b071c35761f113888"
    ),
    "models/glm5_next/configuration_glm5_next.py": (
        "f12b5876701d000f930cd8102124e07d05675a79a5591db29c18244f5f025dec"
    ),
    "integrations/sdpa_attention.py": (
        "53c7229daca9ade4c5df874194448938c1edc925abbc71809f9750dd66381e6f"
    ),
}

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
FIXTURES = {
    "core": FIXTURE_DIR / "transformers_mla_770e4c40_core.json",
    "glm_width": FIXTURE_DIR / "transformers_mla_770e4c40_glm_width.json",
}

# Positive backend evidence: the profiler must observe this event, and no fused SDPA event.
MATH_EVENT = "aten::_scaled_dot_product_attention_math"
FUSED_MARKERS = ("flash", "efficient", "cudnn")

# The precision case must mismatch a float32 rerun by at least this much. The bound is fixed, not
# fitted: it sits three orders above the contract's 1e-12 float64 acceptance tolerance and well
# below float32's 2**-24 unit roundoff.
FP32_PROBE_FLOOR = 1e-9

FP32_PROBE_DESCRIPTION = (
    "the whole case is rerun with the upstream module and every input cast to float32, then "
    "compared against the float64 expectation as a relative difference over finite nonzero "
    "entries. It detects a float32 cast of this specific path; it is not a guarantee that any "
    "conceivable single downcast elsewhere would be caught."
)

# Provenance fields --check compares alongside the cases. `platform` is informational and excluded.
CHECKED_PROVENANCE_FIELDS = (
    "upstream",
    "commit",
    "archive",
    "files_sha256",
    "executed",
    "claim",
    "attn_implementation",
    "sdpa_backend",
    "sdpa_dispatched_ops",
    "fused_sdpa_ops_seen",
    "backend_evidence",
    "dtype",
    "device",
    "encoding",
    "seed_recipe",
    "fp32_probe",
    "torch",
    "numpy",
    "transformers",
    "python",
    "dependencies",
)


def verify_sources() -> None:
    """Fail unless the three pinned upstream source files are byte-identical to the commit."""
    root = Path(transformers.__file__).resolve().parent
    for relative, expected in FILES_SHA256.items():
        digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if digest != expected:
            sys.exit(f"sha256 mismatch for transformers/{relative}: {digest} != {expected}")


def resolved_dependencies() -> dict[str, str]:
    """Installed distribution names and versions only; never paths."""
    found = {}
    for dist in distributions():
        name = dist.metadata["Name"]
        if name:
            found[name] = dist.version
    return dict(sorted(found.items()))


def pack(value) -> dict:
    """Encode an array bit-exactly: big-endian raw buffer in base64, with its shape and dtype."""
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    array = np.ascontiguousarray(array)
    if array.dtype.byteorder in ("<", "="):
        array = array.astype(array.dtype.newbyteorder(">"))
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "b64": base64.b64encode(array.tobytes()).decode("ascii"),
    }


@dataclass(frozen=True)
class Spec:
    """One fixture case. ``indices`` is [batch][q_len][topk] and is never random."""

    name: str
    note: str
    heads: int
    qk_nope: int
    v_dim: int
    rank: int
    q_lora: int
    q_len: int
    indices: list[list[list[int]]]
    kv_len: int
    seed: int
    fixture: str = "core"
    w_scale: float = 0.5
    x_scale: float = 1.0
    require_fp32_lossy: bool = False
    fp32_probe_min: float | None = None

    @property
    def batch(self) -> int:
        return len(self.indices)

    @property
    def topk(self) -> int:
        return len(self.indices[0][0])


SPECS = [
    Spec(
        name="asym_multihead",
        note=(
            "Asymmetric key/value/latent widths with two heads, two batch entries and a different "
            "key set per query."
        ),
        heads=2,
        qk_nope=3,
        v_dim=5,
        rank=7,
        q_lora=8,
        q_len=4,
        kv_len=6,
        indices=[
            [[0, 1, 2], [1, 3, 5], [2, 4, 5], [0, 3, 4]],
            [[5, 4, 3], [0, 2, 4], [3, 4, 5], [1, 2, 3]],
        ],
        seed=20260920,
    ),
    Spec(
        name="mask_edges",
        note=(
            "Row 0 selects exactly one key, row 1 is fully masked, rows 2 and 3 select disjoint "
            "blocks. Fully masked rows are kept and stored exactly as upstream returns them: "
            "upstream softmaxes them with aten::_safe_softmax, which yields exact zeros, not NaN."
        ),
        heads=2,
        qk_nope=3,
        v_dim=5,
        rank=7,
        q_lora=8,
        q_len=4,
        kv_len=6,
        indices=[[[2, -1, -1], [-1, -1, -1], [0, 1, 2], [3, 4, 5]]],
        seed=20260921,
    ),
    Spec(
        name="dup_invalid_indices",
        note=(
            "Duplicate, negative and out-of-range indices. Upstream keeps set-mask behaviour only: "
            "duplicates collapse to one key and invalid entries are dropped."
        ),
        heads=2,
        qk_nope=3,
        v_dim=5,
        rank=7,
        q_lora=8,
        q_len=3,
        kv_len=5,
        indices=[[[1, 1, 1, 1], [0, -1, 7, 3], [4, 4, 2, -5]]],
        seed=20260922,
    ),
    Spec(
        name="precision_full_mantissa",
        note=(
            "Designated precision-sensitive case: well-conditioned float64 values whose mantissas "
            "are all lossy under a float32 round trip, so a float32 cast of this path shows up in "
            "diagnostics.fp32_probe_max_rel_diff, which must clear fp32_probe_floor."
        ),
        heads=2,
        qk_nope=8,
        v_dim=6,
        rank=12,
        q_lora=10,
        q_len=3,
        kv_len=8,
        indices=[[[0, 1, 2, 3, 4], [1, 3, 5, 6, 7], [0, 2, 4, 6, 7]]],
        seed=20260923,
        require_fp32_lossy=True,
        fp32_probe_min=FP32_PROBE_FLOOR,
    ),
    Spec(
        name="glm_width",
        note=(
            "GLM-5.3-Flash widths for the scale distinction: qk_head_dim 256 drives the scale "
            "while the latent is 512 wide. Heads and v_head_dim are kept small only to hold the "
            "file under 2 MB; 256 and 512 are the point of the case."
        ),
        heads=1,
        qk_nope=256,
        v_dim=2,
        rank=512,
        q_lora=8,
        q_len=3,
        kv_len=4,
        indices=[[[0, 1, 2], [1, 2, 3], [0, 2, 3]]],
        seed=20260924,
        fixture="glm_width",
        w_scale=0.05,
    ),
]


def assert_fp32_lossy(name: str, values: torch.Tensor) -> None:
    """Fail unless every value loses bits in a float32 round trip.

    Values of the form ``(1 + m·2⁻⁵²)·2ᵉ`` look full-mantissa but sit beside a number float32
    represents exactly, so a cast loses only the ~1e-15 perturbation. Requiring every entry to move
    under the round trip is the property that actually matters, and normal draws satisfy it.
    """
    survived = values.to(torch.float32).to(torch.float64) == values
    if survived.any():
        sys.exit(f"{name}: {int(survived.sum())} value(s) survive a float32 round trip unchanged")


def build_module(spec: Spec, dtype: torch.dtype) -> Glm5NextTextAttention:
    """An upstream attention module at the case's widths, with a `shared` indexer so none is
    constructed."""
    config = Glm5NextTextConfig(
        num_hidden_layers=2,
        hidden_size=8,
        num_attention_heads=spec.heads,
        num_key_value_heads=spec.heads,
        qk_nope_head_dim=spec.qk_nope,
        qk_rope_head_dim=0,
        v_head_dim=spec.v_dim,
        kv_lora_rank=spec.rank,
        q_lora_rank=spec.q_lora,
        indexer_types=["full", "shared"],
        attention_bias=False,
    )
    config._attn_implementation = "sdpa"
    module = Glm5NextTextAttention(config, layer_idx=1).to(dtype)
    module.eval()
    if module.indexer is not None:
        sys.exit("expected a shared-indexer layer so no indexer is constructed")
    return module


def draw(spec: Spec, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Inputs in the fixed order recorded in the fixture: q_b, kv_b, q_resid, latent."""
    shapes = {
        "q_b_proj_weight": (spec.heads * spec.qk_nope, spec.q_lora),
        "kv_b_proj_weight": (spec.heads * (spec.qk_nope + spec.v_dim), spec.rank),
        "q_resid": (spec.batch, spec.q_len, spec.q_lora),
        "latent": (spec.batch, 1, spec.kv_len, spec.rank),
    }
    generator = torch.Generator().manual_seed(spec.seed)
    scales = {
        "q_b_proj_weight": spec.w_scale,
        "kv_b_proj_weight": spec.w_scale,
        "q_resid": spec.x_scale,
        "latent": spec.x_scale,
    }
    drawn = {
        name: torch.randn(shape, generator=generator, dtype=torch.float64) * scales[name]
        for name, shape in shapes.items()
    }
    if spec.require_fp32_lossy:
        for name, value in drawn.items():
            assert_fp32_lossy(f"{spec.name}.{name}", value)
    return {name: value.to(dtype) for name, value in drawn.items()}


def run_upstream(spec: Spec, dtype: torch.dtype, trace: bool = False) -> dict:
    """Execute the four upstream callables for one case under the MATH backend."""
    module = build_module(spec, dtype)
    inputs = draw(spec, dtype)
    with torch.no_grad():
        module.q_b_proj.weight.copy_(inputs["q_b_proj_weight"])
        module.kv_b_proj.weight.copy_(inputs["kv_b_proj_weight"])

    indices = torch.tensor(spec.indices, dtype=torch.int64)
    latent = inputs["latent"]
    # qk_rope_head_dim is 0, so the rotary half of the key is a zero-width tensor.
    k_rot = torch.zeros(spec.batch, 1, spec.kv_len, 0, dtype=dtype)

    with torch.no_grad():
        projected = module.q_b_proj(inputs["q_resid"])  # upstream query projection
        # Glue: [B, S, H*D] -> [B, H, S, D], the layout the upstream attention call expects.
        query = projected.unflatten(-1, (spec.heads, module.qk_head_dim)).transpose(1, 2)
        key, value = module.expand_kv(latent, k_rot)  # upstream
        mask = module.build_attention_mask_from_topk(  # upstream
            topk_indices=indices, query_states=query, kv_length=spec.kv_len
        )
        if mask is None or mask.dtype != torch.bool:
            sys.exit(f"{spec.name}: expected a boolean sdpa mask, got {mask}")

        def attend():
            return sdpa_attention_forward(  # upstream
                module, query, key, value, mask, dropout=0.0, scaling=module.scaling
            )[0]

        events: list[str] = []
        with sdpa_kernel(SDPBackend.MATH):
            if not trace:
                output = attend()
            else:
                with profile(activities=[ProfilerActivity.CPU]) as prof:
                    output = attend()
                events = sorted(
                    {event.name for event in prof.events() if "scaled_dot_product" in event.name}
                )

    return {
        "module": module,
        "inputs": inputs,
        "indices": indices,
        "query": query,
        "key": key,
        "value": value,
        "mask": mask,
        "output": output,
        "events": events,
    }


def require_float64_cpu(spec: Spec, tensors: dict[str, torch.Tensor]) -> None:
    for name, tensor in tensors.items():
        if tensor.dtype != torch.float64:
            sys.exit(f"{spec.name}: {name} is {tensor.dtype}, expected float64")
        if tensor.device.type != "cpu":
            sys.exit(f"{spec.name}: {name} is on {tensor.device}, expected cpu")


def fp32_probe(spec: Spec, expected: torch.Tensor, counts: torch.Tensor) -> float | None:
    """Relative difference when the whole case is rerun in float32; see FP32_PROBE_DESCRIPTION."""
    got = run_upstream(spec, torch.float32)["output"]
    if not torch.isfinite(got).all():
        sys.exit(f"{spec.name}: the float32 rerun produced a non-finite output")
    got = got.to(torch.float64)

    empty = (counts == 0)[:, :, None, None].expand_as(expected)
    for label, values in (("float64", expected), ("float32", got)):
        if not bool((values[empty] == 0.0).all()):
            sys.exit(f"{spec.name}: a fully masked row is not exactly zero in the {label} run")

    comparable = ~empty & (expected != 0.0)
    if not comparable.any():
        return None
    reference = expected[comparable]
    return float(((got[comparable] - reference).abs() / reference.abs()).max())


def build_case(spec: Spec) -> tuple[dict, list[str]]:
    traced = run_upstream(spec, torch.float64, trace=True)
    plain = run_upstream(spec, torch.float64)["output"]
    module, mask, output = traced["module"], traced["mask"], traced["output"]

    require_float64_cpu(
        spec,
        {
            "q_resid": traced["inputs"]["q_resid"],
            "latent": traced["inputs"]["latent"],
            "q": traced["query"],
            "key_states": traced["key"],
            "value_states": traced["value"],
            "output": output,
        },
    )
    for label, values in (("traced", output), ("untraced", plain)):
        if not torch.isfinite(values).all():
            sys.exit(f"{spec.name}: the {label} float64 output is not finite")
    if not torch.equal(output, plain):
        sys.exit(f"{spec.name}: profiling the call changed the result")

    events = traced["events"]
    fused = [name for name in events if any(marker in name for marker in FUSED_MARKERS)]
    if fused:
        sys.exit(f"{spec.name}: a fused SDPA kernel ran: {fused}")
    if MATH_EVENT not in events:
        sys.exit(f"{spec.name}: {MATH_EVENT} was not observed; saw {events}")

    counts = mask[:, 0].sum(dim=-1)  # [B, q_len]

    def rows_with(selected: int) -> list[list[int]]:
        """[batch, query] pairs whose mask row selects exactly ``selected`` keys, as plain ints."""
        return [[int(b), int(q)] for b, q in np.argwhere(counts.numpy() == selected)]

    probe = fp32_probe(spec, output, counts)
    if spec.fp32_probe_min is not None and (probe is None or probe < spec.fp32_probe_min):
        sys.exit(f"{spec.name}: fp32 probe {probe} below the required {spec.fp32_probe_min}")

    case = {
        "name": spec.name,
        "note": spec.note,
        "dims": {
            "batch": spec.batch,
            "q_len": spec.q_len,
            "kv_len": spec.kv_len,
            "num_heads": spec.heads,
            "qk_nope_head_dim": spec.qk_nope,
            "qk_rope_head_dim": 0,
            "qk_head_dim": module.qk_head_dim,
            "v_head_dim": spec.v_dim,
            "kv_lora_rank": spec.rank,
            "q_lora_rank": spec.q_lora,
            "topk": spec.topk,
        },
        "scale": {
            "value": pack(np.array(module.scaling, dtype=np.float64)),
            "repr": repr(module.scaling),
            "formula": "qk_head_dim ** -0.5",
            "qk_head_dim": module.qk_head_dim,
            "kv_lora_rank": spec.rank,
        },
        "seed": spec.seed,
        "draw": {
            "w_scale": spec.w_scale,
            "x_scale": spec.x_scale,
            "require_fp32_lossy": spec.require_fp32_lossy,
            "fp32_probe_floor": spec.fp32_probe_min,
        },
        "inputs": {
            "q_resid": pack(traced["inputs"]["q_resid"]),
            "q_b_proj_weight": pack(traced["inputs"]["q_b_proj_weight"]),
            "q": pack(traced["query"]),
            "latent": pack(traced["inputs"]["latent"]),
            "kv_b_proj_weight": pack(traced["inputs"]["kv_b_proj_weight"]),
            "topk_indices": pack(traced["indices"]),
        },
        "upstream": {
            "key_states": pack(traced["key"]),
            "value_states": pack(traced["value"]),
            "mask": pack(mask),
            "output": pack(output),
        },
        "rows": {
            "selected_counts": [[int(n) for n in row] for row in counts.numpy()],
            "all_masked": rows_with(0),
            "single_key": rows_with(1),
        },
        "diagnostics": {
            "sdpa_events": events,
            "fp32_probe_max_rel_diff": probe,
            "output_has_nan": bool(torch.isnan(output).any()),
        },
    }
    return case, events


def provenance(observed_events: list[str]) -> dict:
    return {
        "upstream": UPSTREAM,
        "commit": COMMIT,
        "archive": ARCHIVE,
        "files_sha256": FILES_SHA256,
        "executed": [
            "Glm5NextTextAttention.q_b_proj (query expansion projection)",
            "Glm5NextTextAttention.expand_kv",
            "Glm5NextTextAttention.build_attention_mask_from_topk",
            "transformers.integrations.sdpa_attention.sdpa_attention_forward",
        ],
        "claim": (
            "inputs.q_b_proj_weight, inputs.kv_b_proj_weight, inputs.q_resid and inputs.latent are "
            "drawn here from the case seed, and inputs.topk_indices is a literal list: these are "
            "fixture inputs, not upstream outputs. Upstream produces inputs.q (the query "
            "projection) and all of upstream.{key_states, value_states, mask, output}, via the "
            "four callables in `executed`, whose three defining files in `files_sha256` are "
            "byte-identical to `commit`. Importing transformers and constructing the module runs "
            "further upstream code, pinned only by `dependencies` and the archive, not by those "
            "three digests. No model forward, indexer, o_proj or norm contributes to any stored "
            "value."
        ),
        "attn_implementation": "sdpa",
        "sdpa_backend": "MATH, forced by torch.nn.attention.sdpa_kernel(SDPBackend.MATH)",
        "sdpa_dispatched_ops": observed_events,
        "fused_sdpa_ops_seen": [
            name for name in observed_events if any(m in name for m in FUSED_MARKERS)
        ],
        "backend_evidence": (
            f"a torch.profiler CPU trace is taken around the sdpa_attention_forward call, inside "
            f"the MATH context. Each case must observe {MATH_EVENT} and no event matching "
            f"{list(FUSED_MARKERS)}. sdpa_dispatched_ops holds the event names actually observed; "
            f"a fused kernel would appear there as "
            f"aten::_scaled_dot_product_flash_attention_for_cpu. "
            f"The traced and untraced outputs are required to be finite and bit-identical."
        ),
        "dtype": "float64 throughout, on CPU",
        "device": (
            "cpu. Per case, q_resid, latent, q, key_states, value_states and output are each "
            "asserted to be float64 and on cpu before anything is stored."
        ),
        "encoding": (
            "arrays are {shape, dtype, b64}; b64 is base64 of the C-contiguous big-endian buffer; "
            "decode with np.frombuffer(base64.b64decode(b64), dtype=np.dtype(dtype)).reshape(shape)"
        ),
        "seed_recipe": (
            "per case, one torch.Generator().manual_seed(seed), drawn in this order as float64: "
            "q_b_proj_weight[H*qk_head_dim, q_lora_rank]*w_scale, "
            "kv_b_proj_weight[H*(qk_nope+v_head_dim), kv_lora_rank]*w_scale, "
            "q_resid[batch, q_len, q_lora_rank]*x_scale, "
            "latent[batch, 1, kv_len, kv_lora_rank]*x_scale. "
            "topk_indices are literal, never random. Every case uses this one recipe. The stored "
            "arrays are authoritative; the seed only makes regeneration reproducible on the pinned "
            "environment."
        ),
        "fp32_probe": FP32_PROBE_DESCRIPTION,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "transformers": transformers.__version__,
        "python": sys.version.split()[0],
        "dependencies": resolved_dependencies(),
        "platform": platform.platform(),
        "checked_provenance_fields": list(CHECKED_PROVENANCE_FIELDS),
        "generator": "reference/tools/transformers_mla_fixture.py",
        "regenerate": "cd reference && uv run tools/transformers_mla_fixture.py",
        "check": "cd reference && uv run tools/transformers_mla_fixture.py --check",
        "licence": "Transformers is Apache-2.0; only generated values are stored here",
    }


def generate() -> dict[str, dict]:
    verify_sources()
    built: dict[str, dict] = {}
    for spec in SPECS:
        case, events = build_case(spec)
        file = built.setdefault(spec.fixture, {"cases": [], "events": []})
        file["cases"].append(case)
        file["events"] = sorted(set(file["events"]) | set(events))
    return {
        name: {"provenance": provenance(f["events"]), "cases": f["cases"]}
        for name, f in built.items()
    }


def differences(stored: dict, fresh: dict) -> list[str]:
    """Case values plus the stable provenance fields; `platform` stays informational."""
    found = ["cases"] if stored["cases"] != fresh["cases"] else []
    found += [
        f"provenance.{field}"
        for field in CHECKED_PROVENANCE_FIELDS
        if stored["provenance"].get(field) != fresh["provenance"][field]
    ]
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare with the stored fixtures")
    args = parser.parse_args()
    fresh = generate()

    if args.check:
        report = []
        for name, path in FIXTURES.items():
            if not path.exists():
                report.append(f"{path.name}: missing")
                continue
            changed = differences(json.loads(path.read_text()), fresh[name])
            if changed:
                report.append(f"{path.name}: DIFFERS in {', '.join(changed)}")
        total = sum(len(f["cases"]) for f in fresh.values())
        checked = len(CHECKED_PROVENANCE_FIELDS)
        print(
            f"{total} upstream cases, {checked} provenance fields checked; "
            + ("; ".join(report) if report else "fixtures match")
        )
        sys.exit(1 if report else 0)

    for name, path in FIXTURES.items():
        path.write_text(json.dumps(fresh[name], indent=1) + "\n")
        size = path.stat().st_size / 1e6
        print(f"wrote {len(fresh[name]['cases'])} cases to {path.name} ({size:.2f} MB)")


if __name__ == "__main__":
    main()
