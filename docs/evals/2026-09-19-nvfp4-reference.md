# NVFP4 CPU reference acceptance

Gate date: 19 Sep 2026. **Decision: accepted for REF-003/004.** This completes the NVFP4 portion of DEV-PLAN step 0.2.
The next increment may prepare the next CPU reference contract and tests. GPU gates and full-model quality remain open.

## Scope and reference

- Implementation: `reference/src/tq_reference/nvfp4.py`; G16 and G32 with E4M3 block scales.
- Independent acceptance: `reference/tests/test_nvfp4_acceptance.py`, authored by the technical lead before the first
  implementation review, then extended with review regressions. The developer did not edit this suite or its scalar
  fixture.
- Oracle: [NVIDIA ModelOpt b311c054](https://github.com/NVIDIA/Model-Optimizer/blob/b311c054de4052df9c7f3de9409b7598f44a0dba/modelopt/torch/quantization/qtensor/nvfp4_tensor.py).
  The lead inspected the upstream fixture generator and reran its check: it downloads SHA-256-verified pinned files,
  executes their numerical code, and never imports the TensorQuay codec. It does not require GPU execution.
- Environment: macOS ARM64, Python 3.12.4, NumPy 2.3.3, pytest 8.4.2, pytest-cov 7.0.0, Ruff 0.13.1.
  The separate upstream fixture generator uses PyTorch 2.8.0 on CPU.
- Repository state: uncommitted working tree; no new revision is claimed. Dependencies are locked in `reference/uv.lock`.

## Independently reproduced results

| Check | Result |
|---|---|
| Lead acceptance suite alone | **102 passed** |
| Coverage from lead acceptance alone | **166/166 statements; 38/38 branches; 100 %** |
| All tests, including developer checks | **214 passed**, 100 % line and branch coverage |
| Fresh pinned upstream fixture check | **13 cases reproduced** |
| Numerical comparison | Packed codes, scale bytes, global-scale bits and dequantized float32 bits match the fixture |
| Formatting and lints | Clean |
| Working-tree content scan and git guard audit | Clean |

The suite exhausts all FP4/FP8 decode codes and positive FP8 rounding boundaries. It also checks FP4 midpoint neighbors,
packing order, both group sizes, W13 shared scaling, scalar and strided arrays, canonical zero, saturation, subnormal
scales, invalid inputs and non-power-of-two scales. Independent scalar-generated vectors supplement the upstream data.

Review required three corrections after the initial coverage pass: accept usable subnormal global scales, return
arrays consistently for scalar-shaped inputs, and reject noninteger codes with the documented error. The lead's early
0.0006 clamp-case expectation was also corrected using direct arithmetic and upstream output; that upstream case
remains covered. Coverage alone did not establish correctness.

## Reproduce

```sh
cd reference
uv sync --locked
uv run --locked pytest tests/test_nvfp4_acceptance.py --cov --cov-branch
uv run --locked pytest --cov --cov-branch
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run tools/modelopt_nvfp4_fixture.py --check
```

Ordinary tests run offline after setup. The final command uses the network and a separate pinned PyTorch environment.

The following SHA-256 snapshot identifies the accepted implementation, lead tests, numerical fixtures and setup
before a commit exists. Paths are relative to `reference/`.

```text
67dd2483a4257331b2bb8b7597706393b3659e04c86ad125d00b76b4ae9be1d1  src/tq_reference/nvfp4.py
d608a8b125e60601203fd950296807238c9a242ba1b0ed279f0ab567e2ed0ff3  tests/test_nvfp4_acceptance.py
84c3c93e0d06d2ae1ceb215ce279fb6bf58d38f431e5d88ac60dc9fdd2989405  tests/fixtures/lead_nvfp4_vectors.json
50478a85d6d48dcfe6f58e605ef4df3792c70dbaa701abd0ee7ab8ad0149499a  tests/fixtures/modelopt_nvfp4_b311c054.json
fa1973163502ab0d698fd39663e8834239443b378aba60c4eaf7c13360858a44  tools/modelopt_nvfp4_fixture.py
de26565efb5c8eced41e82af98a51ea60e06cee033f4440500347990c9c8dfb9  pyproject.toml
28e01c7c3925bff1e7ffb5bbca58bd7aa5ddf43ed40ee5a244f8b7d70e1a3f7a  uv.lock
```

## Limits

Canonical all-zero encoding and rejection of non-finite source weights or an unusable float32 scale/step are explicit
deviations from the upstream helper. Primitive decoding preserves signed zero; compatibility dequantization follows
upstream's positive-zero convention. See TEST-PLAN §2 for the complete contract.

This report covers the CPU reference package. It makes no claim about GPU kernel correctness, model quality,
serving throughput, memory fit or an advantage over vLLM. G32 reference correctness does not establish support in a
full-model baseline or a native GPU kernel. Nothing was committed or pushed; no GPU session was used.
