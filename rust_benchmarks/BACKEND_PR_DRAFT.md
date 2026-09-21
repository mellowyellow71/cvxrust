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
- Add backend-selection, expression, and 150+ Rust-backend tests, plus a cross-backend
  affine-map oracle test (`A @ vec_F(x0) + b == expr.value`) parametrized over SCIPY, COO and RUST.
- Add CI and pre-commit coverage for the Rust crate.

## Correctness

- Backend, selection and Python-backend suites: 378 passed, 7 expected failures (upstream N-D
  `hstack`/`concatenate(axis=None)` and COO N-D sparse-constant discrepancies, filed separately).
- Broad `CVXPY_DEFAULT_CANON_BACKEND=RUST` suite (atoms, problem, dpp, conic/qp solvers, complex,
  constant atoms, canon methods, expressions, nd matmul, kron): 1002 passed, 449 skipped for absent solvers.
- Rust unit tests: 41 passed (randomized kernel tests against a per-product reference,
  restricted-vs-full lowering tests, shared-subtree test).
- RUST never had the parametric-lhs matmul bug fixed in #3398 for SCIPY/COO; the regression test passes.
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

The four structural losses of the July campaign were addressed in September 2026 (grouped
accumulating mul/rmul kernel, dense row-selection map, coalescing assembly, DAG memo, row push-down;
see `rust_benchmarks/RUST_BACKEND_CHANGES_2026-09.tex`). Cold compile on one Linux machine, RUST vs CPP:
UnconstrainedQP 0.42 s vs 5.24 s, QuantumHilbertMatrix 0.38 vs 2.08, TvInpainting 0.62 vs 1.03,
Cajas 0.35 vs 1.85, SDPSegfault1132 12.1 vs 29.6 (SCIPY 1.75, DIFFENGINE 4.97). SDPSegfault1132
remains slower than SCIPY: its 49.5M-nonzero Jacobian arrives as 148M un-coalesced entries in the COO
intermediate representation; that is stated up front rather than hidden. The campaign tables above are
from July and are to be refreshed on the merge commit.

## Review notes

- Rollout contract: the extension is built as `RustExtension(optional=True)`, so a source install
  without a Rust toolchain still succeeds; wheels from the release workflow include it. RUST is the
  default canonicalization backend only when the compiled entry point is present
  (`settings.rust_backend_available()` checks the attribute, not the import), otherwise the default is
  unchanged. MSRV is 1.80 (`rust-version` in Cargo.toml, `Cargo.lock` tracked); the module declares
  `gil_used = false` for free-threaded interpreters.
- DIFFENGINE (#3448) is parameter-free and opt-in; RUST is the fastest full-coverage tensor backend on
  the DPP/parametric path and coexists with it.
- CPP has 12 expected unsupported cells in the ND/broadcast atom matrix.
- The benchmark coverage is maintained in `cvxpy/benchmarks` PR #32 and should be
  merged before using those jobs as the long-term performance gate.
- This branch intentionally excludes local benchmark scripts, logs, and result data.
