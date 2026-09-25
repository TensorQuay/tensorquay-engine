# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "numpy==2.3.3",
#     "torch==2.8.0",
#     "transformers @ https://github.com/huggingface/transformers/archive/770e4c40d0436082a52dc380f07a9d3f389c99d4.zip",
# ]
# ///
"""Generate or check the REF-006 upstream fixture with Transformers' own GLM-5.3-Flash router.

Independent of tq_reference: this script never imports it, nor any lead test. Transformers is
installed from the pinned source archive, and the two upstream files this fixture depends on are
checked against pinned SHA-256 digests on every run.

Everything under ``inputs`` is ours. Every case keeps the float32 projection itself exact, so no
vendor GEMM blocking can perturb a logit. ``cast_sensitive`` is not an exception to that: what is
deliberately inexact there is the *input cast*, where a float64 value does not survive the mandated
narrowing to float32. Once cast, its projection is exact like every other case. Upstream produces
only the three arrays it returns, recorded under ``upstream`` in the raw slot order
``topk(sorted=False)`` gave them; no reordering, no recomputation, no substituted maths.
``diagnostics`` holds values this script recomputed from those outputs for case design, clearly
separate from the captured arrays.

Positive evidence: the original ``Glm5NextTextTopkRouter.forward`` is verified by qualified name,
defining source file and code object, and a temporary ``sys.setprofile`` observer counts entries
into that exact code object. It must be entered exactly once per recorded case. The observer saves
any previous profile hook and restores it on both the normal and the exceptional path.

Transformers is Apache-2.0; no upstream source is stored in this repository, only generated values.

    uv run tools/transformers_router_fixture.py           # regenerate the fixture
    uv run tools/transformers_router_fixture.py --check   # re-run upstream; exit 1 on a difference
"""

import argparse
import hashlib
import inspect
import json
import platform
import sys
from dataclasses import dataclass
from importlib.metadata import distributions
from pathlib import Path

import numpy as np
import torch
from transformers import __version__ as transformers_version
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextTopkRouter

COMMIT = "770e4c40d0436082a52dc380f07a9d3f389c99d4"
SCHEMA_VERSION = 1
CALLABLE = "Glm5NextTextTopkRouter.forward"
GENERATOR = "reference/tools/transformers_router_fixture.py"
FIXTURE = (
    Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "transformers_router_770e4c40.json"
)

FILES_SHA256 = {
    "models/glm5_next/modeling_glm5_next.py": (
        "431eae74b857ec2a8e38a96f9f76cdc0b366bf7e35bd739b071c35761f113888"
    ),
    "models/glm5_next/configuration_glm5_next.py": (
        "f12b5876701d000f930cd8102124e07d05675a79a5591db29c18244f5f025dec"
    ),
}

U = 2.0**-24
GAP_FACTOR = 64.0  # the contract's cutoff-gap requirement for cross-library selection cases

INPUT_RECIPE = (
    "All inputs are built in this script, never captured from upstream. Three cases supply "
    "float64 x and weight: dense_dyadic, glm_288 and cast_sensitive. The first two hold dyadic "
    "values, x entries multiples of 1/8 and weight entries multiples of 1/16, small enough that "
    "the mandated cast to float32 is exact and the float32 projection is exact too, so GEMM "
    "blocking cannot change a logit. cast_sensitive differs only in its input cast: its float64 "
    "2**24+1 against -(2**24) does not survive the narrowing to float32, so casting before the "
    "projection gives a different logit than projecting in float64 would. Its projection after "
    "that cast is exact like every other case. The remaining cases supply float32 x and weight "
    "directly. Six of them use H=1 with x=[[1.0]], so weight holds the intended logits exactly "
    "and the projection is the identity; empty_tokens is the exception, with T=0 and H=4 and no "
    "token to project. correction_bias is always float32 and is chosen per case to separate the "
    "top-k cutoff."
)

CLAIM = (
    "inputs.x, inputs.weight and inputs.correction_bias are generated here and are fixture "
    "inputs, not upstream outputs. upstream.logits, upstream.weights and upstream.ids are the "
    "three arrays returned by one real call to Glm5NextTextTopkRouter.forward at the pinned "
    "commit, stored in the raw slot order upstream produced. diagnostics values are recomputed by "
    "this script from those captured outputs for case design and are not captured intermediates. "
    "The two files in provenance.source_sha256 that define the router are byte-identical to the "
    "commit; importing transformers and constructing the module runs further upstream code, "
    "pinned only by the archive and the recorded dependency versions. Shapes are deliberately "
    "small: this qualifies the router's arithmetic, not full-model tensor shapes."
)


def verify_sources() -> None:
    root = Path(inspect.getsourcefile(Glm5NextTextTopkRouter)).resolve().parents[2]
    for relative, expected in FILES_SHA256.items():
        digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if digest != expected:
            sys.exit(f"sha256 mismatch for transformers/{relative}: {digest} != {expected}")


def verify_callable():
    """Confirm the recorded callable really is the class's own forward, then return its code."""
    function = Glm5NextTextTopkRouter.forward
    source = Path(inspect.getsourcefile(function)).resolve()
    expected = Path(inspect.getsourcefile(Glm5NextTextTopkRouter)).resolve()
    if function.__qualname__ != CALLABLE or source != expected:
        sys.exit(f"router forward is {function.__qualname__} from {source}, not the class's own")
    if "forward" not in Glm5NextTextTopkRouter.__dict__:
        sys.exit("router forward is inherited, so it is not the pinned implementation")
    return function.__code__


class ForwardObserver:
    """Counts entries into one code object; restores any previous profile hook either way."""

    def __init__(self, code) -> None:
        self.code = code
        self.calls = 0

    def __enter__(self):
        self.previous = sys.getprofile()
        sys.setprofile(self._hook)
        return self

    def __exit__(self, *exc) -> None:
        sys.setprofile(self.previous)

    def _hook(self, frame, event, arg) -> None:
        if event == "call" and frame.f_code is self.code:
            self.calls += 1


def verify_observer_restores_the_hook(code) -> None:
    def harmless(frame, event, arg):
        return None

    previous = sys.getprofile()
    try:
        sys.setprofile(harmless)
        with ForwardObserver(code):
            pass
        if sys.getprofile() is not harmless:
            sys.exit("observer did not restore the previous profile hook")
        try:
            with ForwardObserver(code):
                raise RuntimeError("deliberate")
        except RuntimeError:
            pass
        if sys.getprofile() is not harmless:
            sys.exit("observer did not restore the previous hook after an exception")
    finally:
        sys.setprofile(previous)


def pack(value) -> dict:
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    return {"dtype": str(array.dtype), "shape": list(array.shape), "data": array.ravel().tolist()}


def dyadic(rows: int, columns: int, step: int, modulus: int, divisor: float) -> np.ndarray:
    values = (np.arange(rows * columns) * step) % modulus - modulus // 2
    return (values.reshape(rows, columns) / divisor).astype(np.float64)


def logit_inputs(logits, bias) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """H=1 with x=[[1.0]], so the projection is the identity and the logits are exact."""
    weight = np.asarray(logits, dtype=np.float32).reshape(-1, 1)
    return np.ones((1, 1), dtype=np.float32), weight, np.asarray(bias, dtype=np.float32)


@dataclass(frozen=True)
class Case:
    name: str
    x: np.ndarray
    weight: np.ndarray
    correction_bias: np.ndarray
    top_k: int


def build_cases() -> list[Case]:
    cases = []

    # A small dense projection whose every product and partial sum is exact in float32.
    cases.append(
        Case(
            "dense_dyadic",
            dyadic(4, 4, 7, 17, 8.0),
            dyadic(12, 4, 5, 31, 16.0),
            ((np.arange(12) * 11 % 17 - 8) / 64).astype(np.float32),
            8,
        )
    )

    # All 288 experts, narrow H. The bias ladder gives a wide, unambiguous cutoff.
    experts = 288
    cases.append(
        Case(
            "glm_288",
            dyadic(2, 4, 3, 13, 8.0),
            dyadic(experts, 4, 5, 31, 16.0),
            (np.arange(experts) / 256.0).astype(np.float32),
            8,
        )
    )

    # Rising logits, falling bias: the selected experts hold the lowest uncorrected scores.
    size = 16
    cases.append(
        Case(
            "bias_selection",
            *logit_inputs(np.arange(size) / 2.0 - 4.0, (size - np.arange(size)) / 8.0),
            8,
        )
    )

    # float64 inputs where casting to float32 before the projection changes the logit.
    weight64 = np.zeros((12, 2), dtype=np.float64)
    weight64[:4] = [1.0, -(2.0**24)]
    weight64[4:, 1] = np.arange(1, 9) / 16.0
    cases.append(
        Case(
            "cast_sensitive",
            np.array([[2.0**24 + 1.0, 1.0]], dtype=np.float64),
            weight64,
            (np.arange(12) / 1024.0).astype(np.float32),
            8,
        )
    )

    # Scores small enough that the 1e-20 epsilon materially changes the normalization.
    depth = np.array([-46.0, -47.5, -49.0, -50.5, -52.0, -53.5, -55.0, -57.0] * 2)
    cases.append(Case("epsilon_dominated", *logit_inputs(depth, np.arange(16) / 32.0), 8))

    # exp(-logit) overflows, so every score is exactly zero and only the bias orders them.
    tail = [-100.0 - i for i in range(16)]
    cases.append(Case("zero_tail", *logit_inputs(tail, np.arange(16) / 16.0), 8))

    # exp(-logit) underflows, so every score is exactly one and only the bias orders them.
    high = [100.0 + i for i in range(16)]
    cases.append(Case("saturated", *logit_inputs(high, np.arange(16) / 16.0), 8))

    # The exact logits recorded in the contract's numerical-gate revision.
    cases.append(
        Case(
            "rounding_discrepancy",
            *logit_inputs(
                [
                    -10.91289234161377,
                    -9.825783729553223,
                    -5.059233665466309,
                    -4.0,
                    -3.0,
                    -2.0,
                    -1.0,
                    0.0,
                ],
                np.zeros(8),
            ),
            8,
        )
    )

    # Subnormal but nonzero scores, just inside the float32 exponential range.
    tail = np.array([-87.5, -87.6, -87.7, -87.8, -87.9, -88.0, -88.1, -88.2] * 2)
    cases.append(Case("negative_subnormal", *logit_inputs(tail, np.arange(16) / 32.0), 8))

    # No tokens, valid metadata.
    cases.append(
        Case(
            "empty_tokens",
            np.zeros((0, 4), dtype=np.float32),
            dyadic(12, 4, 5, 31, 16.0).astype(np.float32),
            ((np.arange(12) * 11 % 17 - 8) / 64).astype(np.float32),
            8,
        )
    )
    return cases


def make_router(case: Case) -> tuple[Glm5NextTextTopkRouter, Glm5NextTextConfig]:
    experts, width = case.weight.shape
    config = Glm5NextTextConfig(
        hidden_size=width,
        n_routed_experts=experts,
        num_experts_per_tok=case.top_k,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        num_hidden_layers=2,
        indexer_types=["full", "shared"],
    )
    router = Glm5NextTextTopkRouter(config).to(torch.float32)
    weight = torch.from_numpy(np.ascontiguousarray(case.weight))
    with torch.no_grad():
        if case.weight.dtype == np.float64:
            router.weight = torch.nn.Parameter(weight.clone())  # upstream casts it itself
        else:
            router.weight.copy_(weight.to(torch.float32))
        router.e_score_correction_bias.copy_(torch.from_numpy(case.correction_bias))
    if router.e_score_correction_bias.dtype != torch.float32:
        sys.exit(f"{case.name}: correction bias must stay float32")
    router.eval()
    return router, config


def check_cutoff_gap(case: Case, logits: torch.Tensor, bias: torch.Tensor) -> dict:
    """Recomputed diagnostics: the corrected scores and the realised top-k cutoff gap."""
    scores = torch.sigmoid(logits)
    corrected = scores + bias
    gap, limit = None, None
    experts = corrected.shape[-1] if corrected.numel() else case.weight.shape[0]
    if corrected.shape[0] and case.top_k < experts:
        ordered = torch.sort(corrected, dim=-1, descending=True).values
        gaps = (ordered[:, case.top_k - 1] - ordered[:, case.top_k]).tolist()
        scale = float(corrected.abs().max())
        limit = GAP_FACTOR * U * max(1.0, scale)
        gap = min(gaps)
        if gap <= limit:
            sys.exit(
                f"{case.name}: cutoff gap {gap:.6e} does not exceed {limit:.6e}; "
                "investigate this case rather than removing it"
            )
    return {
        "note": "recomputed by the generator from upstream.logits; not a captured intermediate",
        "scores": pack(scores),
        "corrected_scores": pack(corrected),
        "cutoff_gap": gap,
        "cutoff_gap_required_above": limit,
    }


def run_case(case: Case, code) -> dict:
    router, config = make_router(case)
    x = torch.from_numpy(np.ascontiguousarray(case.x))
    with ForwardObserver(code) as observer, torch.no_grad():
        logits, weights, ids = router(x)
    if observer.calls != 1:
        sys.exit(f"{case.name}: upstream forward ran {observer.calls} times, expected exactly 1")
    for name, tensor, dtype in (
        ("logits", logits, torch.float32),
        ("weights", weights, torch.float32),
        ("ids", ids, torch.int64),
    ):
        if tensor.dtype != dtype or tensor.device.type != "cpu":
            sys.exit(f"{case.name}: {name} is {tensor.dtype} on {tensor.device}")
    if not torch.isfinite(logits).all() or not torch.isfinite(weights).all():
        sys.exit(f"{case.name}: upstream returned a non-finite value")

    bias = router.e_score_correction_bias.detach()
    return {
        "name": case.name,
        "top_k": case.top_k,
        "inputs": {
            "x": pack(case.x),
            "weight": pack(case.weight),
            "correction_bias": pack(case.correction_bias),
        },
        "upstream": {"logits": pack(logits), "weights": pack(weights), "ids": pack(ids)},
        "evidence": {"forward_calls": observer.calls},
        "config": {
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": True,
            "routed_scaling_factor": 2.5,
            "hidden_size": config.hidden_size,
            "n_routed_experts": config.n_routed_experts,
            "num_experts_per_tok": config.num_experts_per_tok,
        },
        "diagnostics": check_cutoff_gap(case, logits, bias),
    }


def resolved_dependencies() -> dict[str, str]:
    """Distribution names and versions only; no paths and no identities."""
    found = {d.metadata["Name"]: d.version for d in distributions() if d.metadata["Name"]}
    return dict(sorted(found.items()))


def provenance() -> dict:
    return {
        "commit": COMMIT,
        "source_sha256": FILES_SHA256,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "transformers": transformers_version,
        "python": sys.version.split()[0],
        "device": "cpu",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dependencies": resolved_dependencies(),
        "callable": CALLABLE,
        "generator": GENERATOR,
        "input_recipe": INPUT_RECIPE,
        "claim": CLAIM,
    }


def generate() -> dict:
    verify_sources()
    code = verify_callable()
    verify_observer_restores_the_hook(code)
    return {
        "schema_version": SCHEMA_VERSION,
        "provenance": provenance(),
        "cases": [run_case(case, code) for case in build_cases()],
    }


def differences(stored: dict, fresh: dict) -> list[str]:
    """Every differing root field, including one present in only one of the two documents."""
    missing = object()
    return sorted(
        key
        for key in set(stored) | set(fresh)
        if stored.get(key, missing) != fresh.get(key, missing)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare with the stored fixture")
    args = parser.parse_args()  # parse before running upstream, so --help stays cheap
    fresh = generate()

    if args.check:
        if not FIXTURE.exists():
            sys.exit(f"{FIXTURE.name}: missing")
        differing = differences(json.loads(FIXTURE.read_text()), fresh)
        verdict = f"DIFFERS in {', '.join(differing)}" if differing else "fixture matches"
        print(f"{len(fresh['cases'])} cases, {len(fresh)} root fields; {verdict}")
        sys.exit(1 if differing else 0)

    FIXTURE.write_text(json.dumps(fresh, indent=1) + "\n")
    megabytes = FIXTURE.stat().st_size / 1e6
    print(f"wrote {len(fresh['cases'])} cases to {FIXTURE.name} ({megabytes:.2f} MB)")


if __name__ == "__main__":
    main()
