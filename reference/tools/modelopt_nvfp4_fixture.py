# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.8.0"]
# ///
"""Generate or check the REF-004 upstream fixture by running NVIDIA ModelOpt's own NVFP4 code.

Independent of tq_reference: this script never imports it. It downloads the pinned upstream files
from github.com/NVIDIA/Model-Optimizer at a fixed commit, verifies their SHA-256, and executes:

- ``qtensor/nvfp4_tensor.py``: whole file, verbatim, with its relative imports resolved to the
  modules below;
- ``utils/numeric_utils.py``: whole file, verbatim;
- ``utils/core_utils.py``: ``_FP8_DTYPES``, ``reduce_amax``, ``reduce_block_amax`` and
  ``reduce_block_padding``, extracted verbatim with ``ast``;
- ``qtensor/base_qtensor.py``: the ``BaseQuantizedTensor`` class, extracted verbatim.

Only ``backends.utils.fp4_compatible`` is a stub (it returns False). The upstream code calls it
only when ``try_tensorrt=True``, which this script never passes. ModelOpt is Apache-2.0; no
upstream code is stored in this repository, only the generated values.

    uv run tools/modelopt_nvfp4_fixture.py            # regenerate tests/fixtures/...json
    uv run tools/modelopt_nvfp4_fixture.py --check    # re-run upstream; exit 1 on any difference
"""

import argparse
import ast
import hashlib
import json
import struct
import sys
import types
import urllib.request
from pathlib import Path

import torch

UPSTREAM = "https://github.com/NVIDIA/Model-Optimizer"
COMMIT = "b311c054de4052df9c7f3de9409b7598f44a0dba"
RAW = f"https://raw.githubusercontent.com/NVIDIA/Model-Optimizer/{COMMIT}/"
FILES_SHA256 = {
    "modelopt/torch/quantization/qtensor/nvfp4_tensor.py": (
        "e128fd62f0d0c4aa179e057b53a615bccf42f30f8ed9ef7c8f7e3dc84c5eb61e"
    ),
    "modelopt/torch/quantization/qtensor/base_qtensor.py": (
        "1d9c06291bbbd41300efa25af7513cd3402a8e80d031a6e537c329889c666cf7"
    ),
    "modelopt/torch/quantization/utils/core_utils.py": (
        "608e8caac495d9af64cf222b31d22cbbf820d3d1f359bf9f020298c72f5f2172"
    ),
    "modelopt/torch/quantization/utils/numeric_utils.py": (
        "f901f22030f43ba2cc095f601f5f19daf81df87e1fe37f47ab7770e8167a6084"
    ),
}
FIXTURE = (
    Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "modelopt_nvfp4_b311c054.json"
)
PKG = "mo_pinned.quantization"


def fetch(path: str) -> str:
    data = urllib.request.urlopen(RAW + path, timeout=60).read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != FILES_SHA256[path]:
        sys.exit(f"sha256 mismatch for {path}: {digest}")
    return data.decode("utf-8")


def extract(source: str, names: set[str]) -> str:
    """Verbatim source of the named top-level functions, classes and assignments."""
    tree = ast.parse(source)
    parts = []
    for node in tree.body:
        targets = [t.id for t in getattr(node, "targets", []) if isinstance(t, ast.Name)]
        if getattr(node, "name", None) in names or set(targets) & names:
            parts.append(ast.get_source_segment(source, node))
    return "\n\n".join(parts)


def module(name: str, package: bool = False) -> types.ModuleType:
    mod = types.ModuleType(name)
    if package:
        mod.__path__ = []
    sys.modules[name] = mod
    return mod


def load_upstream():
    for name in ("mo_pinned", PKG, f"{PKG}.qtensor", f"{PKG}.backends"):
        module(name, package=True)
    backends = module(f"{PKG}.backends.utils")
    backends.fp4_compatible = lambda: False  # stub: only reached with try_tensorrt=True

    numeric = module(f"{PKG}.utils.numeric_utils")
    exec(fetch("modelopt/torch/quantization/utils/numeric_utils.py"), numeric.__dict__)

    utils = module(f"{PKG}.utils", package=True)
    utils.__dict__.update(torch=torch, F=torch.nn.functional)
    core = fetch("modelopt/torch/quantization/utils/core_utils.py")
    wanted = {"_FP8_DTYPES", "reduce_amax", "reduce_block_amax", "reduce_block_padding"}
    exec(extract(core, wanted), utils.__dict__)

    base = module(f"{PKG}.qtensor.base_qtensor")
    base.__dict__.update(torch=torch, Any=object)
    base_src = fetch("modelopt/torch/quantization/qtensor/base_qtensor.py")
    exec(extract(base_src, {"BaseQuantizedTensor"}), base.__dict__)

    nvfp4 = module(f"{PKG}.qtensor.nvfp4_tensor")
    nvfp4.__package__ = f"{PKG}.qtensor"
    exec(fetch("modelopt/torch/quantization/qtensor/nvfp4_tensor.py"), nvfp4.__dict__)
    return nvfp4.NVFP4QTensor


def hexes(t: torch.Tensor) -> list[str]:
    return [struct.pack(">f", float(v)).hex() for v in t.to(torch.float32).flatten()]


def run(qt, x: torch.Tensor, group: int, g) -> dict:
    g_t = None if g is None else torch.tensor(g, dtype=torch.float32)
    q, s, g2 = qt.quantize(x.clone(), group, weights_scaling_factor_2=g_t)
    deq = q.dequantize(dtype=torch.float32, scale=s, double_scale=g2, block_sizes={-1: group})
    return {
        "packed_hex": bytes(q._quantized_data.flatten().tolist()).hex(),
        "scales_hex": bytes(s.view(torch.uint8).flatten().tolist()).hex(),
        "global_scale_f32_hex": hexes(g2.reshape(1))[0],
        "dequant_f32_hex": hexes(deq),
    }


def build_inputs() -> list[dict]:
    gen = torch.Generator().manual_seed(20260919)
    g = 2.0**-4
    ties = [6.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.75, -5.0, -0.1, 0.0, 1.0, -2.5, 3.0, 4.0]

    def randn(rows, cols, scale=0.02):
        return torch.randn(rows, cols, generator=gen) * scale

    cases = []
    a = randn(4, 64)
    a[3, 7] = 0.4
    cases.append(("g16_random_natural_scale", a, 16, None))
    cases.append(("g32_random_natural_scale", randn(4, 64), 32, None))
    rows = [torch.tensor(ties) * g * s for s in (1.0, 2.0, 0.5, 1.125)]
    cases.append(("g16_ties_supplied_pow2_scale", torch.stack(rows), 16, g))
    sat = torch.zeros(2, 32)
    sat[0, :3] = torch.tensor([6.36, -6.36, 1.0]) * g  # block scale rounds down: saturation
    sat[0, 16] = 6.0 * g * 1000.0  # block scale clamps to 448
    sat[1, :2] = torch.tensor([6.0, 0.4]) * g * 2.0**-12  # block scale clamps to 2^-9
    cases.append(("g16_saturation_and_clamps_supplied_scale", sat, 16, g))
    z = randn(2, 64)
    z[1, 16:32] = 0.0  # a zero block inside a non-zero tensor
    z[0, 5] = -1e-7  # a tiny negative value that rounds to code 8
    cases.append(("g16_zero_block_natural_scale", z, 16, None))
    cases.append(("g32_non_power_of_two_supplied_scale", randn(2, 64, 0.5), 32, 0.3))
    under = torch.zeros(1, 32)
    under[0, 0] = 3.0e38  # derived g follows this outlier
    under[0, 16] = -1.0e-45  # this block's ratio underflows to 0 in float32 -> scale 1.0
    cases.append(("g16_underflowing_block_ratio", under, 16, None))
    smallest = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
    subn = torch.cat([torch.full((16,), float(smallest)), torch.full((16,), 0.0006)])
    cases.append(("g16_subnormal_block_and_clamped_block_supplied_unit_scale", subn[None], 16, 1.0))
    sub_g = 2.0**-140  # a subnormal supplied global scale; every block step stays > 0
    sub = randn(2, 32, 0.3) * 2688.0 * sub_g
    cases.append(("g16_subnormal_supplied_scale", sub, 16, sub_g))
    nat = randn(2, 64, 0.3) * 2688.0 * 2.0**-135  # natural g = amax / 2688 is subnormal
    cases.append(("g32_subnormal_natural_scale", nat, 32, None))
    big = torch.full((1, 16), torch.finfo(torch.float32).max)
    big[0, 1::2] *= -1
    cases.append(("g16_supplied_tiny_scale_overflow_saturates", big, 16, 1e-30))
    return cases


def w13_cases(qt) -> list[dict]:
    gen = torch.Generator().manual_seed(1313)
    gate = torch.randn(2, 32, generator=gen) * 0.05
    up = torch.randn(2, 32, generator=gen) * 0.3
    amax = [qt.get_weights_scaling_factor_2(t) for t in (gate, up)]
    shared = float(torch.maximum(*amax))
    out = []
    for role, x, partner in (("gate", gate, up), ("up", up, gate)):
        out.append(
            {
                "name": f"w13_{role}_shared_scale_g16",
                "kind": "w13",
                "role": role,
                "group": 16,
                "shape": list(x.shape),
                "input_f32_hex": hexes(x),
                "partner_f32_hex": hexes(partner),
                "supplied_global_scale_f32_hex": None,
                "expected": run(qt, x, 16, shared),
            }
        )
    return out


def generate() -> dict:
    qt = load_upstream()
    cases = []
    for name, x, group, g in build_inputs():
        x = x.to(torch.float32)
        supplied = None if g is None else hexes(torch.tensor([g]))[0]
        cases.append(
            {
                "name": name,
                "kind": "single",
                "group": group,
                "shape": list(x.shape),
                "input_f32_hex": hexes(x),
                "supplied_global_scale_f32_hex": supplied,
                "expected": run(qt, x, group, g),
            }
        )
    cases += w13_cases(qt)
    return {
        "provenance": {
            "upstream": UPSTREAM,
            "commit": COMMIT,
            "files_sha256": FILES_SHA256,
            "executed_verbatim": [
                "qtensor/nvfp4_tensor.py (whole file)",
                "utils/numeric_utils.py (whole file)",
                "utils/core_utils.py: _FP8_DTYPES, reduce_amax, reduce_block_amax, "
                "reduce_block_padding",
                "qtensor/base_qtensor.py: BaseQuantizedTensor",
            ],
            "stubbed": ["backends.utils.fp4_compatible -> False (try_tensorrt path only)"],
            "torch": torch.__version__,
            "python": sys.version.split()[0],
            "generator": "reference/tools/modelopt_nvfp4_fixture.py",
            "regenerate": "cd reference && uv run tools/modelopt_nvfp4_fixture.py",
            "check": "cd reference && uv run tools/modelopt_nvfp4_fixture.py --check",
            "licence": "ModelOpt is Apache-2.0; only generated values are stored here",
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare with the stored fixture")
    args = parser.parse_args()
    fresh = generate()
    if args.check:
        stored = json.loads(FIXTURE.read_text())
        same = stored["cases"] == fresh["cases"]
        print(f"{len(fresh['cases'])} upstream cases, fixture {'matches' if same else 'DIFFERS'}")
        sys.exit(0 if same else 1)
    FIXTURE.write_text(json.dumps(fresh, indent=1) + "\n")
    print(f"wrote {len(fresh['cases'])} upstream cases to {FIXTURE.name}")


if __name__ == "__main__":
    main()
