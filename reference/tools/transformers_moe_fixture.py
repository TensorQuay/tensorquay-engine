# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "numpy==2.3.3",
#     "torch==2.8.0",
#     "transformers @ https://github.com/huggingface/transformers/archive/770e4c40d0436082a52dc380f07a9d3f389c99d4.zip",
# ]
# ///
"""Generate or check the REF-002 upstream fixture with Transformers' own GLM-5.3-Flash experts.

Independent of tq_reference: this script never imports it. Transformers is installed from the
pinned source archive, and the three upstream files whose code this fixture depends on are checked
against pinned SHA-256 digests on every run. Importing Transformers and constructing the modules
runs much more upstream code than that, so the claim made here is narrower.

Fixture inputs are ours, and they come in two flavours. ``gate_up_proj``, ``down_proj`` and
``hidden_states`` are seeded random draws, except in ``clamp_boundaries``, where the gate/up rows
are the literal boundary values and the hidden states are exact ones. ``top_k_index`` is literal
everywhere but ``router_authentic``; ``top_k_weights`` is literal in ``zero_weights`` and
``clamp_boundaries`` and a seeded ``torch.rand`` draw elsewhere. ``router_weight`` and
``router_bias`` are generated fixture inputs too, drawn here in float32.

Upstream produces exactly five things: ``upstream.output``, ``upstream.output_f32`` from the eager
``Glm5NextTextExperts.forward``, and ``router.router_logits``, ``router.top_k_weights_f32`` and
``router.top_k_index`` from a real ``Glm5NextTextTopkRouter``. Those router outputs then serve as
inputs to both expert arms, which records the router's float32 input boundary only; it does not
accept REF-006 router semantics.

The router is float32 end to end, including ``e_score_correction_bias``. Casting it to float64
would make selection partially float64, because ``scores + bias`` promotes while the returned
weights stay float32. The router is handed the original float64 hidden states, which its own source
hard-casts to float32. Its float32 weights are then widened exactly to float64, and those same IDs
and weights feed both expert arms.

Eager execution is proven positively: a ``sys.setprofile`` observer counts entries into the code
objects of ``inspect.unwrap(Glm5NextTextExperts.forward)`` and the class-owned ``_apply_gate``,
whose identity is checked by qualified name and defining source file so the decorator's unclamped
``_default_apply_gate`` cannot pass unnoticed. Nothing upstream is replaced or copied; the observer
saves ``sys.getprofile()`` and restores that exact hook on both the normal and the exceptional
path, which is verified at startup. Dispatch is separately asserted to return that same original
callable. A CPU profiler op list is kept as supporting data only; op-name resemblance alone would
be weaker evidence.

Transformers is Apache-2.0; no upstream code is stored in this repository, only generated values.

    uv run tools/transformers_moe_fixture.py           # regenerate the fixture
    uv run tools/transformers_moe_fixture.py --check   # re-run upstream; exit 1 on a difference
"""

import argparse
import base64
import hashlib
import inspect
import json
import math
import platform
import sys
from dataclasses import dataclass, field
from importlib.metadata import distributions
from pathlib import Path

import numpy as np
import torch
import transformers
from torch.profiler import ProfilerActivity, profile
from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextTextExperts,
    Glm5NextTextTopkRouter,
)

UPSTREAM = "https://github.com/huggingface/transformers"
COMMIT = "770e4c40d0436082a52dc380f07a9d3f389c99d4"
ARCHIVE = f"{UPSTREAM}/archive/{COMMIT}.zip"

FILES_SHA256 = {
    "models/glm5_next/modeling_glm5_next.py": (
        "431eae74b857ec2a8e38a96f9f76cdc0b366bf7e35bd739b071c35761f113888"
    ),
    "models/glm5_next/configuration_glm5_next.py": (
        "f12b5876701d000f930cd8102124e07d05675a79a5591db29c18244f5f025dec"
    ),
    "integrations/moe.py": ("706047c3850f41fd298c21df6fdd79b639c385351f1c12422c63b8ef3733f423"),
}

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "transformers_moe_770e4c40_experts.json"
)

# The eager body is the fallback of the dispatch, so "eager" must not be a registered key.
EXPERTS_IMPLEMENTATION = "eager"
REGISTERED = ("batched_mm", "deepgemm", "grouped_mm", "sonicmoe")
FUSED_MARKERS = ("grouped", "deepgemm", "sonic")

# The designated case must differ from a float32 expert path by at least this global relative L2.
# Fixed, not fitted: three orders above the 1e-12 comparison gate, well below float32 roundoff.
FP32_PROBE_FLOOR = 1e-9

# What the fixture does and does not claim; recorded verbatim in provenance.
CLAIM = (
    "Fixture inputs, none of them upstream outputs: inputs.gate_up_proj and "
    "inputs.down_proj are seeded random draws, except clamp_boundaries, whose "
    "gate_up_proj rows are the literal boundary values; inputs.hidden_states is a seeded "
    "random draw, except clamp_boundaries, which uses exact ones; inputs.top_k_index is "
    "literal for every case except router_authentic; inputs.top_k_weights is literal for "
    "zero_weights and clamp_boundaries and a seeded torch.rand draw for "
    "nonuniform_routes, poisoned_unselected and precision_sensitive. In "
    "poisoned_unselected the banks of the never-routed experts are overwritten with "
    "NaN and infinity after the seeded draws, so the selected banks keep exactly the "
    "drawn values. In router_authentic alone the IDs and weights come from the "
    "router. router_weight and router_bias are "
    "also generated fixture inputs, drawn here in float32. Upstream produces exactly: "
    "upstream.output, upstream.output_f32, and router.router_logits, "
    "router.top_k_weights_f32 and router.top_k_index. The three files in files_sha256 "
    "that define these callables are byte-identical to commit. Importing transformers and "
    "constructing the modules runs further upstream code, pinned only by dependencies and "
    "the archive. No MoE wrapper, shared expert or model forward contributes to a stored "
    "value. Stored arrays are authoritative."
)

CHECKED_PROVENANCE_FIELDS = (
    "upstream",
    "commit",
    "archive",
    "files_sha256",
    "executed",
    "claim",
    "api_mapping",
    "experts_implementation",
    "experts_registered_keys",
    "apply_gate_owner",
    "experts_dispatched_ops",
    "fused_experts_ops_seen",
    "eager_evidence",
    "router_boundary",
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
    found = {d.metadata["Name"]: d.version for d in distributions() if d.metadata["Name"]}
    return dict(sorted(found.items()))


def pack(value) -> dict:
    """Encode an array bit-exactly, NaN and infinity included: big-endian buffer in base64."""
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    array = np.ascontiguousarray(array)
    if array.dtype.byteorder in ("<", "="):
        array = array.astype(array.dtype.newbyteorder(">"))
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "b64": base64.b64encode(array.tobytes()).decode("ascii"),
    }


class CodeObserver:
    """Counts entries into specific code objects, without replacing anything upstream."""

    def __init__(self, targets: dict) -> None:
        self.targets = targets
        self.counts = dict.fromkeys(targets.values(), 0)

    def __enter__(self):
        self.previous = sys.getprofile()  # restored verbatim, so a caller's hook survives
        sys.setprofile(self._hook)
        return self

    def __exit__(self, *exc) -> None:
        sys.setprofile(self.previous)

    def _hook(self, frame, event, arg) -> None:
        if event == "call":
            name = self.targets.get(frame.f_code)
            if name is not None:
                self.counts[name] += 1


def verify_observer_restores_the_hook() -> None:
    """The observer must hand back the exact hook it found, on the normal and the failing path."""

    def harmless(frame, event, arg):
        return None

    previous = sys.getprofile()
    try:
        sys.setprofile(harmless)
        with CodeObserver({}):
            pass
        if sys.getprofile() is not harmless:
            sys.exit("CodeObserver did not restore the previous profile hook")
        try:
            with CodeObserver({}):
                raise RuntimeError("deliberate")
        except RuntimeError:
            pass
        if sys.getprofile() is not harmless:
            sys.exit("CodeObserver did not restore the previous hook after an exception")
    finally:
        sys.setprofile(previous)


def verified_apply_gate():
    """The class-owned ``_apply_gate``, confirmed by qualified name and defining source file.

    The decorator installs ``moe._default_apply_gate``, which does not clamp at all, on any experts
    class that defines none. GLM's asymmetric clamps exist only while this identity holds, so the
    owner recorded in provenance comes from the verified function rather than a literal.
    """
    function = Glm5NextTextExperts._apply_gate
    owner, _, name = function.__qualname__.rpartition(".")
    source = Path(inspect.getsourcefile(function)).resolve()
    expected = Path(inspect.getsourcefile(Glm5NextTextExperts)).resolve()
    if name != "_apply_gate" or owner != Glm5NextTextExperts.__name__ or source != expected:
        sys.exit(f"_apply_gate is {function.__qualname__} from {source}, not the class's own")
    if "_apply_gate" not in Glm5NextTextExperts.__dict__:
        sys.exit("_apply_gate is inherited, so the decorator's unclamped default may be in use")
    return function, owner


EAGER_FORWARD = inspect.unwrap(Glm5NextTextExperts.forward)
APPLY_GATE, APPLY_GATE_OWNER = verified_apply_gate()
OBSERVED_CODES = {
    EAGER_FORWARD.__code__: "eager_forward",
    APPLY_GATE.__code__: "apply_gate",
}


@dataclass(frozen=True)
class Spec:
    """One fixture case. Literal routes are used verbatim; nothing here is renormalised."""

    name: str
    note: str
    tokens: int
    width: int
    intermediate: int
    experts: int
    slots: int
    seed: int
    swiglu_limit: float = 10.0
    ids: tuple = ()
    weights: tuple = ()
    gate_up_rows: tuple = ()
    ones_input: bool = False
    from_router: bool = False
    poison_unselected: bool = False
    fp32_floor: float | None = None
    scales: dict = field(default_factory=lambda: {"weight": 0.5, "input": 1.0})


SPECS = [
    Spec(
        name="router_authentic",
        note=(
            "IDs and weights come from a real float32 Glm5NextTextTopkRouter at GLM's top-8 shape. "
            "Its float32 weights are widened exactly to float64 and fed to both expert arms. "
            "Router normalisation and the 2.5 factor already happened upstream; the reference must "
            "not reapply either."
        ),
        tokens=4,
        width=6,
        intermediate=3,
        experts=12,
        slots=8,
        seed=20260930,
        from_router=True,
    ),
    Spec(
        name="nonuniform_routes",
        note="Asymmetric widths, a different expert set per token and no repeated route.",
        tokens=4,
        width=6,
        intermediate=2,
        experts=5,
        slots=3,
        seed=20260931,
        ids=((0, 1, 2), (2, 3, 4), (0, 2, 4), (1, 3, 4)),
    ),
    Spec(
        name="zero_weights",
        note=(
            "Individual zero routing weights plus one token whose every weight is zero, giving an "
            "exactly zero output row. A zero weight never deselects its expert, which is still "
            "evaluated and must stay finite."
        ),
        tokens=3,
        width=4,
        intermediate=3,
        experts=6,
        slots=2,
        seed=20260932,
        ids=((0, 1), (2, 3), (4, 5)),
        weights=((0.0, 1.5), (0.0, 0.0), (0.25, 0.0)),
    ),
    Spec(
        name="clamp_boundaries",
        note=(
            "Unit inputs make gate_up_proj the pre-clamp values directly. Gate rows are "
            "[-12, -10, 0, 10, 12] and up rows [12, 10, 0, -10, -12], so both the gate's "
            "upper-only clamp and the up's symmetric clamp are visible below, at and above "
            "the +/-10 boundary."
        ),
        tokens=4,
        width=1,
        intermediate=5,
        experts=4,
        slots=1,
        seed=20260933,
        ids=((0,), (1,), (2,), (3,)),
        weights=((1.0,), (0.5,), (2.0,), (0.25,)),
        gate_up_rows=(-12.0, -10.0, 0.0, 10.0, 12.0, 12.0, 10.0, 0.0, -10.0, -12.0),
        ones_input=True,
    ),
    Spec(
        name="poisoned_unselected",
        note=(
            "Experts 1, 3, 5 and 7 are never routed to and their banks hold NaN and infinity, "
            "encoded in the fixture. Upstream never reads them: the output must equal the clean "
            "run bitwise."
        ),
        tokens=3,
        width=4,
        intermediate=3,
        experts=8,
        slots=2,
        seed=20260934,
        ids=((0, 2), (4, 6), (0, 6)),
        poison_unselected=True,
    ),
    Spec(
        name="precision_sensitive",
        note=(
            "Designated case: a complete float32 expert path must differ from float64 by at least "
            "fp32_probe_floor in global relative L2, with both runs finite."
        ),
        tokens=4,
        width=6,
        intermediate=5,
        experts=6,
        slots=2,
        seed=20260935,
        ids=((0, 1), (2, 3), (4, 5), (0, 3)),
        fp32_floor=FP32_PROBE_FLOOR,
    ),
]


def make_config(spec: Spec) -> Glm5NextTextConfig:
    config = Glm5NextTextConfig(
        num_hidden_layers=2,
        hidden_size=spec.width,
        num_attention_heads=2,
        num_key_value_heads=2,
        qk_nope_head_dim=4,
        qk_rope_head_dim=0,
        v_head_dim=4,
        kv_lora_rank=8,
        q_lora_rank=8,
        n_routed_experts=spec.experts,
        moe_intermediate_size=spec.intermediate,
        num_experts_per_tok=spec.slots,
        n_group=1,
        topk_group=1,
        swiglu_limit=spec.swiglu_limit,
        indexer_types=["full", "shared"],
    )
    config._experts_implementation = EXPERTS_IMPLEMENTATION
    return config


def draw(spec: Spec) -> dict:
    """Case inputs in a fixed order: gate_up_proj, down_proj, hidden_states, then any weights."""
    generator = torch.Generator().manual_seed(spec.seed)
    scale = spec.scales["weight"]
    if spec.gate_up_rows:
        rows = torch.tensor(spec.gate_up_rows, dtype=torch.float64)
        gate_up = rows[None, :, None].expand(spec.experts, 2 * spec.intermediate, spec.width)
        gate_up = gate_up.clone()
    else:
        shape = (spec.experts, 2 * spec.intermediate, spec.width)
        gate_up = torch.randn(shape, generator=generator, dtype=torch.float64) * scale
    down = (
        torch.randn(
            (spec.experts, spec.width, spec.intermediate), generator=generator, dtype=torch.float64
        )
        * scale
    )
    if spec.ones_input:
        hidden = torch.ones((spec.tokens, spec.width), dtype=torch.float64)
    else:
        hidden = (
            torch.randn((spec.tokens, spec.width), generator=generator, dtype=torch.float64)
            * spec.scales["input"]
        )
    drawn = {"gate_up_proj": gate_up, "down_proj": down, "hidden_states": hidden}
    if spec.weights:
        drawn["top_k_weights"] = torch.tensor(spec.weights, dtype=torch.float64)
    elif not spec.from_router:
        drawn["top_k_weights"] = torch.rand(
            (spec.tokens, spec.slots), generator=generator, dtype=torch.float64
        )
    if spec.ids:
        drawn["top_k_index"] = torch.tensor(spec.ids, dtype=torch.int64)
    return drawn


def run_router(spec: Spec, config, hidden: torch.Tensor) -> dict:
    """A float32 router, bias included, handed the original float64 hidden states."""
    generator = torch.Generator().manual_seed(spec.seed + 1)
    router = Glm5NextTextTopkRouter(config).to(torch.float32)
    with torch.no_grad():
        router.weight.copy_(
            torch.randn((spec.experts, spec.width), generator=generator, dtype=torch.float32) * 0.5
        )
        router.e_score_correction_bias.copy_(
            torch.randn((spec.experts,), generator=generator, dtype=torch.float32) * 0.1
        )
        logits, weights, index = router(hidden)
    for name, tensor in (("weight", router.weight), ("bias", router.e_score_correction_bias)):
        if tensor.dtype != torch.float32:
            sys.exit(f"{spec.name}: router {name} is {tensor.dtype}, expected float32")
    if logits.dtype != torch.float32 or weights.dtype != torch.float32:
        sys.exit(f"{spec.name}: router outputs are not float32")
    if not torch.isfinite(router.e_score_correction_bias).all():
        sys.exit(f"{spec.name}: router correction bias must be finite")
    return {
        "router_weight": router.weight.detach().clone(),
        "router_bias": router.e_score_correction_bias.detach().clone(),
        "router_logits": logits,
        "top_k_weights_f32": weights,
        "top_k_index": index,
    }


def build_experts(spec: Spec, config, gate_up: torch.Tensor, down: torch.Tensor, dtype):
    experts = Glm5NextTextExperts(config).to(dtype)
    with torch.no_grad():
        experts.gate_up_proj.copy_(gate_up.to(dtype))
        experts.down_proj.copy_(down.to(dtype))
    experts.eval()
    if experts.swiglu_limit != spec.swiglu_limit:
        sys.exit(f"{spec.name}: upstream swiglu_limit is {experts.swiglu_limit}")
    return experts


def call(experts, hidden, index, weights):
    with torch.no_grad():
        return experts(hidden, index, weights)


def run_case(spec: Spec) -> dict:
    """Execute upstream for one case and return every checked tensor and piece of evidence."""
    config = make_config(spec)
    if ALL_EXPERTS_FUNCTIONS.get_interface(EXPERTS_IMPLEMENTATION, EAGER_FORWARD) is not (
        EAGER_FORWARD
    ):
        sys.exit(f"{spec.name}: dispatch did not return the original eager forward")
    if tuple(sorted(ALL_EXPERTS_FUNCTIONS.keys())) != REGISTERED:
        sys.exit(f"{spec.name}: registered implementations changed: {list(ALL_EXPERTS_FUNCTIONS)}")

    drawn = draw(spec)
    hidden, gate_up, down = drawn["hidden_states"], drawn["gate_up_proj"], drawn["down_proj"]
    router = run_router(spec, config, hidden) if spec.from_router else None
    if router is None:
        index, weights = drawn["top_k_index"], drawn["top_k_weights"]
    else:
        index = router["top_k_index"]
        weights = router["top_k_weights_f32"].to(torch.float64)  # exact widening

    selected = sorted({int(e) for e in index.flatten()})
    unselected = [e for e in range(spec.experts) if e not in selected]
    clean_output = None
    if spec.poison_unselected:
        if not unselected:
            sys.exit(f"{spec.name}: poisoning requires at least one unselected expert")
        clean_output = call(
            build_experts(spec, config, gate_up, down, torch.float64), hidden, index, weights
        )
        gate_up = gate_up.clone()
        down = down.clone()
        for position, expert in enumerate(unselected):
            gate_up[expert] = torch.nan if position % 2 == 0 else torch.inf
            down[expert] = -torch.inf if position % 2 == 0 else torch.nan

    experts = build_experts(spec, config, gate_up, down, torch.float64)
    output = call(experts, hidden, index, weights)
    with CodeObserver(OBSERVED_CODES) as observer:
        observed_output = call(experts, hidden, index, weights)
    with torch.no_grad(), profile(activities=[ProfilerActivity.CPU]) as prof:
        profiled_output = call(experts, hidden, index, weights)

    for label, tensor in (("observed", observed_output), ("profiled", profiled_output)):
        if not torch.equal(output, tensor):
            sys.exit(f"{spec.name}: the {label} run changed the result")
    for name, tensor in (
        ("hidden_states", hidden),
        ("gate_up_proj", gate_up),
        ("down_proj", down),
        ("top_k_weights", weights),
        ("output", output),
    ):
        if tensor.dtype != torch.float64 or tensor.device.type != "cpu":
            sys.exit(f"{spec.name}: {name} is {tensor.dtype} on {tensor.device}, want float64 cpu")
    if index.dtype != torch.int64:
        sys.exit(f"{spec.name}: top_k_index is {index.dtype}, expected int64")
    if not torch.isfinite(output).all():
        sys.exit(f"{spec.name}: the float64 output is not finite")
    if clean_output is not None and not torch.equal(output, clean_output):
        sys.exit(f"{spec.name}: poisoning an unselected expert changed the output")

    counts = observer.counts
    if counts["eager_forward"] != 1 or counts["apply_gate"] != len(selected):
        sys.exit(f"{spec.name}: eager call counts {counts} do not match {len(selected)} experts")
    ops = sorted({event.name for event in prof.events() if event.name.startswith("aten::")})
    fused = [name for name in ops if any(marker in name for marker in FUSED_MARKERS)]
    if fused:
        sys.exit(f"{spec.name}: a fused experts kernel ran: {fused}")

    # Same routes, same supplied weights, cast to float32: a complete float32 expert path.
    experts32 = build_experts(spec, config, gate_up, down, torch.float32)
    output32 = call(experts32, hidden.to(torch.float32), index, weights.to(torch.float32))
    if not torch.isfinite(output32).all():
        sys.exit(f"{spec.name}: the float32 output is not finite")
    return {
        "config": config,
        "hidden": hidden,
        "gate_up": gate_up,
        "down": down,
        "index": index,
        "weights": weights,
        "router": router,
        "selected": selected,
        "unselected": unselected,
        "output": output,
        "output32": output32,
        "counts": counts,
        "ops": ops,
        "rel_l2": fp32_probe(spec, output, output32),
    }


def norm(values: torch.Tensor) -> float:
    """Euclidean norm of every element, via math.hypot.

    torch.linalg.vector_norm squares before summing, so a reference of 1e-300 underflows to a norm
    of exactly zero for two or more elements and a tiny but genuinely nonzero output would be
    misread as an exact-zero case. math.hypot scales internally and does not.
    """
    return math.hypot(*values.to(torch.float64).flatten().tolist())


def fp32_probe(spec: Spec, output: torch.Tensor, output32: torch.Tensor) -> float | None:
    """Global relative L2 over every output element, with no epsilon and no excluded rows.

    A truly zero reference norm leaves the ratio undefined rather than zero, so a float32 result
    that differs from an all-zero float64 result can never be reported as a perfect match.
    """
    reference = norm(output)
    difference = norm(output32.to(torch.float64) - output)
    if reference == 0:
        if not bool((output32 == 0).all()):
            sys.exit(f"{spec.name}: float64 output is all zero but the float32 rerun is not")
        if spec.fp32_floor is not None:
            sys.exit(f"{spec.name}: the designated case needs a nonzero reference norm")
        return None
    rel_l2 = float(difference / reference)
    if not np.isfinite(rel_l2):
        sys.exit(f"{spec.name}: the fp32 probe is not finite: {rel_l2}")
    if spec.fp32_floor is not None and rel_l2 < spec.fp32_floor:
        sys.exit(f"{spec.name}: fp32 probe {rel_l2} below the required {spec.fp32_floor}")
    return rel_l2


def verify_fp32_probe_boundary() -> None:
    """A tiny but nonzero reference must not be mistaken for an exact zero.

    Element counts matter: a single 1e-300 value survives torch.linalg.vector_norm, while two or
    more underflow, so both a small and a larger count are checked here.
    """
    spec = next(candidate for candidate in SPECS if candidate.fp32_floor is None)
    for count in (2, 24):
        tiny = torch.full((1, count), 1e-300, dtype=torch.float64)
        ratio = fp32_probe(spec, tiny, torch.zeros((1, count), dtype=torch.float32))
        if ratio != 1.0:
            sys.exit(f"fp32 probe scored a {count}-element 1e-300 reference {ratio}, wanted 1.0")
        if fp32_probe(spec, tiny, tiny.to(torch.float32)) is None:
            sys.exit(f"fp32 probe called a {count}-element 1e-300 reference an exact zero")
    zeros = torch.zeros((1, 4), dtype=torch.float64)
    if fp32_probe(spec, zeros, torch.zeros((1, 4), dtype=torch.float32)) is not None:
        sys.exit("fp32 probe must report an undefined ratio for a true zero reference")


def build_case(spec: Spec) -> tuple[dict, list[str]]:
    """Serialize one executed case."""
    ran = run_case(spec)
    hidden, gate_up, down = ran["hidden"], ran["gate_up"], ran["down"]
    index, weights, router = ran["index"], ran["weights"], ran["router"]
    output, selected, ops = ran["output"], ran["selected"], ran["ops"]

    case = {
        "name": spec.name,
        "note": spec.note,
        "dims": {
            "num_tokens": spec.tokens,
            "hidden_size": spec.width,
            "intermediate_size": spec.intermediate,
            "num_experts": spec.experts,
            "top_k": spec.slots,
            "swiglu_limit": spec.swiglu_limit,
        },
        "seed": spec.seed,
        "draw": {**spec.scales, "fp32_probe_floor": spec.fp32_floor},
        "inputs": {
            "hidden_states": pack(hidden),
            "gate_up_proj": pack(gate_up),
            "down_proj": pack(down),
            "top_k_index": pack(index),
            "top_k_weights": pack(weights),
        },
        "router": None
        if router is None
        else {name: pack(tensor) for name, tensor in router.items()}
        | {
            "note": (
                "router_weight and router_bias are generated fixture inputs, drawn here in "
                "float32; only router_logits, top_k_weights_f32 and top_k_index are produced by "
                "the router. It is float32 throughout, correction bias included, and received the "
                "float64 hidden states and hard-cast them itself. inputs.top_k_weights is the "
                "exact float64 widening of top_k_weights_f32."
            )
        },
        "upstream": {"output": pack(output), "output_f32": pack(ran["output32"])},
        "rows": {
            "selected_experts": selected,
            "unselected_experts": ran["unselected"],
            "zero_weight_slots": [
                [int(t), int(k)] for t, k in torch.nonzero(weights == 0).tolist()
            ],
            # Derived by sorting the selected IDs, which is the order the contract requires;
            # it is the expected order, not a trace of the upstream loop.
            "expected_expert_order": selected,
        },
        "diagnostics": {
            "call_counts": ran["counts"],
            "experts_dispatched_ops": ops,
            "fp32_probe_rel_l2": ran["rel_l2"],
            "output_has_nan": bool(torch.isnan(output).any()),
        },
    }
    return case, ops


def provenance(ops: list[str]) -> dict:
    return {
        "upstream": UPSTREAM,
        "commit": COMMIT,
        "archive": ARCHIVE,
        "files_sha256": FILES_SHA256,
        "executed": [
            "Glm5NextTextExperts.forward (eager, via the experts dispatch fallback)",
            "Glm5NextTextExperts._apply_gate (class-owned, with GLM's asymmetric clamps)",
            "Glm5NextTextTopkRouter.forward (router_authentic only)",
        ],
        "claim": CLAIM,
        "api_mapping": {
            "hidden_states": "x",
            "gate_up_proj": "w13",
            "down_proj": "w2",
            "top_k_index": "topk_ids",
            "top_k_weights": "topk_weights",
            "dims.swiglu_limit": "swiglu_limit",
        },
        "experts_implementation": EXPERTS_IMPLEMENTATION,
        "experts_registered_keys": list(REGISTERED),
        "apply_gate_owner": APPLY_GATE_OWNER,
        "experts_dispatched_ops": ops,
        "fused_experts_ops_seen": [],
        "eager_evidence": (
            "a sys.setprofile observer counts entries into the code objects of "
            "inspect.unwrap(Glm5NextTextExperts.forward) and the class-owned _apply_gate, whose "
            "identity is verified by qualified name and defining source file against the pinned "
            "modeling module, so the decorator's unclamped _default_apply_gate cannot pass; per "
            "case diagnostics.call_counts must show eager_forward == 1 and apply_gate equal to the "
            "number of distinct selected experts in that single traced run. Dispatch is separately "
            "asserted to return that original callable, and the registered key set is pinned. "
            "Nothing upstream is replaced; the observer saves sys.getprofile() and restores that "
            "exact hook, on the normal and the exceptional path, which is checked at startup. "
            "experts_dispatched_ops is supporting data only."
        ),
        "router_boundary": (
            "the router is float32 end to end, including e_score_correction_bias: casting it to "
            "float64 would promote scores + bias and make selection partially float64 while the "
            "returned weights stayed float32. Its float32 weights are widened exactly to float64 "
            "and the same IDs and weights feed both expert arms. This records the input boundary "
            "only and does not accept REF-006 router semantics."
        ),
        "dtype": "float64 experts throughout, on CPU; the router is float32 by construction",
        "device": (
            "cpu. Per case hidden_states, gate_up_proj, down_proj, top_k_weights and output are "
            "asserted float64 on cpu, and top_k_index int64, before anything is stored."
        ),
        "encoding": (
            "arrays are {shape, dtype, b64}; b64 is base64 of the C-contiguous big-endian buffer, "
            "so NaN and infinity survive exactly; decode with "
            "np.frombuffer(base64.b64decode(b64), dtype=np.dtype(dtype)).reshape(shape)"
        ),
        "seed_recipe": (
            "per case, one torch.Generator().manual_seed(seed) drawn in this order as float64: "
            "gate_up_proj[E, 2I, H]*weight_scale, down_proj[E, H, I]*weight_scale, "
            "hidden_states[T, H]*input_scale, then top_k_weights[T, K] from torch.rand when the "
            "case supplies no literal weights. Literal routes and weights are used verbatim. The "
            "router uses manual_seed(seed + 1) for its float32 weight then its bias. "
            "poisoned_unselected then overwrites the banks of its never-routed experts with NaN "
            "and infinity, after those draws, so every selected bank keeps exactly the drawn "
            "value. Stored arrays are authoritative; the seed only makes regeneration reproducible."
        ),
        "fp32_probe": (
            "the expert path is rerun with the module, hidden states and the same supplied weights "
            "cast to float32, keeping the identical routes; no rerouting. fp32_probe_rel_l2 is the "
            "global relative L2 over all output elements, ||out32 - out64|| / ||out64||, both "
            "norms taken with math.hypot so a tiny reference cannot underflow to a zero norm, "
            "with no "
            "epsilon and no excluded rows; a zero reference norm requires an exactly zero float32 "
            "output and records null rather than zero. Both runs are asserted finite, and the "
            "ratio itself is asserted finite. It detects a complete "
            "float32 expert path, not an arbitrary single downcast."
        ),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "transformers": transformers.__version__,
        "python": sys.version.split()[0],
        "dependencies": resolved_dependencies(),
        "platform": platform.platform(),
        "checked_provenance_fields": list(CHECKED_PROVENANCE_FIELDS),
        "generator": "reference/tools/transformers_moe_fixture.py",
        "regenerate": "cd reference && uv run tools/transformers_moe_fixture.py",
        "check": "cd reference && uv run tools/transformers_moe_fixture.py --check",
        "licence": "Transformers is Apache-2.0; only generated values are stored here",
    }


def generate() -> dict:
    verify_sources()
    verify_observer_restores_the_hook()
    verify_fp32_probe_boundary()
    cases, every_op = [], set()
    for spec in SPECS:
        case, ops = build_case(spec)
        cases.append(case)
        every_op |= set(ops)
    return {"provenance": provenance(sorted(every_op)), "cases": cases}


def differences(stored: dict, fresh: dict) -> list[str]:
    found = ["cases"] if stored["cases"] != fresh["cases"] else []
    return found + [
        f"provenance.{field_name}"
        for field_name in CHECKED_PROVENANCE_FIELDS
        if stored["provenance"].get(field_name) != fresh["provenance"][field_name]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare with the stored fixture")
    args = parser.parse_args()
    fresh = generate()

    if args.check:
        if not FIXTURE.exists():
            sys.exit(f"{FIXTURE.name}: missing")
        changed = differences(json.loads(FIXTURE.read_text()), fresh)
        verdict = f"DIFFERS in {', '.join(changed)}" if changed else "fixture matches"
        print(
            f"{len(fresh['cases'])} upstream cases, "
            f"{len(CHECKED_PROVENANCE_FIELDS)} provenance fields checked; {verdict}"
        )
        sys.exit(1 if changed else 0)

    FIXTURE.write_text(json.dumps(fresh, indent=1) + "\n")
    megabytes = FIXTURE.stat().st_size / 1e6
    print(f"wrote {len(fresh['cases'])} cases to {FIXTURE.name} ({megabytes:.2f} MB)")


if __name__ == "__main__":
    main()
