# Why the Rust canon backend loses on some cvxpy/benchmarks

**Date:** 2026-06-17 · branch `alan/arena-allocator` · cvxpy 1.8.0.dev0

Two questions were investigated:
1. Are the benchmarks actually exercising *our* `cvxpy_rust` (a current release build)?
2. Why does the Rust backend lose on the external suite when our own suite shows 40/40 wins?

---

## 1. Build correctness (the trust question) — benchmarks DO use the release build

`cvxpy_rust` is installed **editable**; the canonical rebuild is **`maturin develop --release`**, which
installs the optimized `.so` (`opt-level=3, lto`, ~782 KB) into
`.venv/.../site-packages/cvxpy_rust/`. Because `run_one.py` runs as a **script** from `rust_benchmarks/`,
`import cvxpy_rust` resolves via normal `sys.path` to that **site-packages release build** — verified by
printing `sys.modules['cvxpy_rust.cvxpy_rust'].__file__` from the script context (md5 `7664af3b`, the
release binary). The site-packages `.so` (built Jun 17) is newer than the newest source (Jun 9), so it is
current. **The benchmark numbers were always measured on a current release build.**

Footgun (does NOT affect benchmarks): there is a stale, gitignored top-level
`cvxpy_rust.cpython-313-darwin.so` at the repo root — a DEBUG build (md5 `f732e35a`). It is loaded **only**
by `python -c`/`python -` run from the repo root (where `sys.path[0]=''`), never by the benchmark scripts.
An earlier draft of this doc wrongly concluded the suite ran on debug after testing with `python -c` from
the repo root; that was a misdiagnosis. After editing `cvxpy_rust/src/`, rebuild with
`maturin develop --release` (venv activated). Consider deleting the vestigial repo-root `.so`.

### Is debug-vs-release the reason for the losses? No — and it can't be, since the suite runs on release.
The losses below were all measured on the release build. (An attempted "debug vs release" comparison was
invalid: both sweeps loaded the same site-packages release binary, so the differences were run-to-run
noise.)

| Benchmark | R/S (release) |
| --- | --- |
| SDPSegfault1132 | **0.04×** |
| UnconstrainedQP | **0.26×** |
| Murray | **0.42×** |
| QuantumHilbertMatrix | **0.59×** |
| TvInpainting | **0.82×** |

---

## 2. Why we lose — three distinct mechanisms (source-grounded)

### A. `diag` of a dense-affine matrix — SDPSegfault1132 (0.04×)
`cp.diag(V @ G @ V.T)`, `G` a PSD variable. `V@G@V.T` is **dense-affine**: each of its m² entries is a
combination of *all* of G's entries. The COO tensor for that intermediate has ~m² entries, and
`process_diag_mat` (`operations/specialized.rs`) then iterates **all m² entries** to keep the m on the
diagonal, after which the whole COO block is sorted (`tensor.rs::from_tensor`, O(nnz log nnz)). SciPy's
specialized diag/mul paths never materialize the full dense block. The dominant cost is allocation + sort
of a huge COO array, so it is bounded by the m²-entry materialization regardless of optimization level.

### B. `kron` index blow-up — UnconstrainedQP (0.26×)
`cp.kron(np.diag(ones(14)), cp.diag(var))` sandwiched between dense DFT matrices.
`compute_kron_indices` (`operations/specialized.rs:521+`) eagerly allocates a **full dense
`lhs_size * rhs_size` row-index map** regardless of operand sparsity; `process_kron_r/l` then expands a
handful of variables into a dense block. The Cartesian index map dominates.

### C. dense Constant that should be sparse — Murray (0.42×)
No kron, no diag. The model builds `np.zeros((244650, 700))` (~171M cells, **99.7% zero**) and reaches
the backend as a **dense `Constant`**. `arithmetic.rs::mul_const_by_variable` (DenseColMajor/RowMajor
arms, ~lines 737 & 757) walks all 171M cells **twice** — once for the nnz pre-scan
(`data.iter().filter(...).count()`), once to emit — to produce only ~489K nonzeros. SciPy/CPP keep it
sparse and touch only the nonzeros. **Surgical fix:** detect a mostly-zero dense `Constant` and convert
it to CSC before the multiply.

`QuantumHilbertMatrix` / `TvInpainting` are sub-second, FFI+setup-bound — backend choice barely matters.

---

## 3. Why our own suite shows 40/40 but the external one doesn't

`rust_benchmarks/quick_benchmark.py` and `benchmark_suite.py` cover only **least-squares, LASSO, dense QP
(np.eye), and many-constraint LPs** — grep confirms **zero** `kron`, `diag`-of-dense-affine, or large
dense-zero constants. Those are precisely the three loss mechanisms above. The 40/40 is a real result on
a **biased sample**: it tests where the vectorized Rust paths are strong and never exercises the
structural operators where we're algorithmically weak. The external suite adds exactly those cases.

**Takeaway:** the wins are real and the geomean is favorable, but to close the gap we should (1) sparsify
mostly-zero dense constants (Murray — easiest, highest-confidence win), (2) avoid materializing
dense-affine intermediates before `diag` (SDP1132), and (3) compute kron indices lazily instead of a full
Cartesian allocation (UnconstrainedQP). Adding kron/diag/dense-constant cases to our own suite would stop
it from over-reporting.
