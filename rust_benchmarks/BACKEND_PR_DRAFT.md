# Add an optional Rust canonicalization backend

## Summary

This PR adds an optional `RUST` canonicalization backend implemented as a PyO3
extension. It follows the existing CVXPY backend interface and can be selected with
`canon_backend="RUST"` or through the default-backend configuration.

The branch is rebased onto current CVXPY master and includes support for affine,
cone, complex, ND, broadcast, einsum, convolve, and parameterized expressions.
Parameterized ND matrix multiplication is lowered to the backend's existing sparse
2D parameter path, including batched and broadcast left/right parameter operands.

## Changes

- Add the `cvxpy_rust` PyO3 crate and Python packaging integration.
- Add Rust backend selection, settings, and solving-chain integration.
- Implement the CVXPY linear-operator protocol in Rust.
- Add dense-constant sparsification and specialized structural operations.
- Add parameterized ND matrix multiplication without a SciPy fallback.
- Add backend-selection, expression, and 142 Rust-backend tests.
- Add CI and pre-commit coverage for the Rust crate.

## Correctness

- Rust backend test suite: 142 passed.
- Python backend, selection, and kron tests: 199 passed.
- Broad `CVXPY_DEFAULT_CANON_BACKEND=RUST` smoke suite: 363 passed, 407 skipped.
- Rust unit tests: 31 passed.
- Cross-backend stuffed `A`, `b`, and `c` data matched for all 18 benchmark cases.
- All 124 atom cases compiled with no unexpected backend failures.
- Ruff, clippy with warnings denied, and `git diff --check` passed.

## Performance

Ratios are `OTHER/RUST`; values above 1 mean Rust is faster.

| Campaign | SCIPY/RUST | CPP/RUST | COO/RUST |
| --- | ---: | ---: | ---: |
| 124 atom cases | 1.50x | 1.07x | 1.28x |
| 40 synthetic build-matrix cases | 4.87x | 2.09x | 3.54x |
| 18 ASV full-compilation cases | 1.93x | 1.45x | 1.41x |
| 21 external benchmark classes | 1.07x | 1.27x | 1.20x |

The previous Murray regression is no longer significant in full compilation: Rust
measured 1107 ms versus 1066 ms for SciPy, 1135 ms for CPP, and 1067 ms for COO in
the process-isolated external sweep.

The external suite still exposes follow-up optimization opportunities in
`SDPSegfault1132Benchmark`, `UnconstrainedQP`, `QuantumHilbertMatrix`, and
`TvInpainting`. These are localized structural workloads and do not affect backend
correctness; they are documented in the benchmark report rather than hidden from
this review.

## Review notes

- The backend is optional and does not change the default installation path.
- CPP has 12 expected unsupported cells in the ND/broadcast atom matrix.
- The benchmark coverage is maintained in `cvxpy/benchmarks` PR #32 and should be
  merged before using those jobs as the long-term performance gate.
- This branch intentionally excludes local benchmark scripts, logs, and result data.
