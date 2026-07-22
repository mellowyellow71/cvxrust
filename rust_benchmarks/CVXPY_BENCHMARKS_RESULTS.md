# cvxpy/benchmarks results — Rust canon backend vs SciPy, C++ & COO

> **Historical report (2026-07-03).** The current rebased results, expanded 124-atom
> coverage, and corrected ASV tables are in `CVXPY_BENCHMARKS_RESULTS_REBASED.md`.
> Do not use the values below as the current benchmark baseline.

Run of the **official [cvxpy/benchmarks](https://github.com/cvxpy/benchmarks)** suite plus the
in-repo synthetic suite, ASV backend suite, and the exhaustive per-atom sweep, comparing the Rust
canonicalization backend head-to-head with the `SCIPY`, `CPP` (cvxcore), and `COO` backends.

- **Date:** 2026-07-03
- **Branch:** `ray/latestfixes` — the Rust backend ported onto upstream cvxpy master
  (`40947203c`, ~1.9.2-dev) plus the dense-constant sparsification fix (`7bf07625c`)
- **Machine:** macOS (darwin 25.5.0), 18 GB, `.venv` Python 3.13, release build
- **What is measured:** wall-clock of `Problem.get_problem_data(...)` (canonicalization / matrix
  stuffing), or the backend-isolated `build_matrix` call where noted — never the numerical solve
- **Previous report:** 2026-06-17, `alan/arena-allocator` (pre-rebase, RUST/SCIPY/CPP only) —
  preserved as `results_prerebase.jsonl`; the Δ columns below compare against it

## What changed since the 2026-06-17 report

1. **Rebase onto upstream master (1.9.x).** The much-cited "murray optimization" on master
   (PR #3366) turned out to live in the NLP diff_engine, *not* the canonicalization path — it does
   not speed up SCIPY/CPP canon. What the rebase actually brought: the **COO backend** (#3031,
   now in the comparison), the **einsum atom** (#2970 — works on the Rust backend with zero Rust
   changes), the ND-parametric-matmul fix (#3401 — Rust path verified unaffected), and the
   `backends/` refactor.
2. **Dense-constant sparsification in the Rust serializer** (mirrors #3366's heuristic: 2D dense
   constants ≥4096 elements below `SPARSE_DENSITY_THRESHOLD`=5% nonzeros are serialized sparse).
   This **eliminated the Murray loss**: 2199 ms → 1024 ms (Δ 2.5×), now at parity with every
   backend. UnconstrainedQP also dropped 4938 → 3606 ms as a side effect.
3. **Correctness gate:** `verify_backends.py` asserts all four backends produce numerically
   identical stuffed tensors (max abs diff 0.0, parameter slices included) on every ASV case,
   including einsum, ND ops, both murray density regimes, and 256-deep/wide trees.

## Methodology

Same harness as before (`sweep.sh` → `run_one.py`, one subprocess per class for crash isolation,
fresh `Problem` per rep to defeat the solving-chain cache, backend injected via a transient
`get_problem_data` wrapper), with: **1 warm-up + 3 timed reps** (median), a 240 s SIGALRM watchdog,
and COO added. Sub-10% gaps are noise.

Four cells could not be measured, all on the two fully-parametrized classes (huge parameters —
the COO backend's native workload). `ParametrizedQPBenchmark`: **SCIPY timed out (>240 s)** and
**CPP was killed at >9 GB RSS** by a memory watchdog (unguarded, this cell froze/crashed the host
twice); RUST completed it in 1.6 s, COO in 1.2 s. `SimpleFullyParametrizedLPBenchmark`: **the OS
killed both SCIPY and CPP for memory exhaustion**, while RUST (325 ms) beat COO (451 ms).

---

## 1. External suite (official cvxpy/benchmarks, 21 classes)

All times are **median milliseconds** for canonicalization. Ratios are **>1 ⇒ Rust faster**.
Δ vs pre-rebase compares today's ratio with the 2026-06-17 ratio for the same pair
(>1 ⇒ Rust's relative position improved).

### RUST vs SCIPY

| Benchmark | Rust (ms) | SCIPY (ms) | SCIPY/Rust | Δ vs pre-rebase |
| --- | --- | --- | --- | --- |
| OptimalAdvertising | 261 | 542 | **2.08×** | 1.02× |
| SimpleQPBenchmark | 798 | 1614 | **2.02×** | 1.00× |
| Cajas | 360 | 677 | **1.88×** | 0.99× |
| LeastSquares | 798 | 1465 | **1.83×** | 1.03× |
| SimpleLPBenchmark | 2861 | 5134 | **1.79×** | 1.01× |
| SemidefiniteProgramming | 371 | 650 | **1.75×** | 0.93× |
| FactorCovarianceModel | 573 | 992 | **1.73×** | 1.03× |
| Yitzhaki | 525 | 865 | **1.65×** | 1.03× |
| SimpleScalarParametrizedLPBenchmark | 683 | 994 | **1.46×** | 1.01× |
| SVMWithL1Regularization | 1601 | 2293 | **1.43×** | 1.01× |
| HuberRegression | 1765 | 2495 | **1.41×** | 1.00× |
| SlowPruningBenchmark | 1353 | 1882 | **1.39×** | 1.02× |
| CVaRBenchmark | 4736 | 5838 | **1.23×** | 1.17× |
| ConvexPlasticity | 53 | 57 | **1.07×** | 0.16× |
| Murray | 1024 | 1069 | 1.04× | **2.51×** |
| TvInpainting | 889 | 769 | 0.87× | 1.05× |
| QuantumHilbertMatrix | 1370 | 800 | 0.58× | 0.99× |
| UnconstrainedQP | 3606 | 1003 | 0.28× | 1.06× |
| SDPSegfault1132Benchmark | 29113 | 1336 | 0.05× | 1.04× |
| ParametrizedQPBenchmark | 1561 | — | SCIPY timeout >240 s | |
| SimpleFullyParametrizedLPBenchmark | 325 | — | SCIPY killed by OS (memory) | |

**geomean 1.09× | 15/19 wins**

### RUST vs CPP

| Benchmark | Rust (ms) | CPP (ms) | CPP/Rust | Δ vs pre-rebase |
| --- | --- | --- | --- | --- |
| OptimalAdvertising | 261 | 1927 | **7.38×** | 1.07× |
| Cajas | 360 | 1743 | **4.84×** | 0.94× |
| SemidefiniteProgramming | 371 | 765 | **2.06×** | 0.90× |
| SimpleLPBenchmark | 2861 | 5502 | **1.92×** | 1.04× |
| SimpleQPBenchmark | 798 | 1487 | **1.86×** | 0.97× |
| FactorCovarianceModel | 573 | 964 | **1.68×** | 1.05× |
| SimpleScalarParametrizedLPBenchmark | 683 | 1074 | **1.57×** | 1.03× |
| Yitzhaki | 525 | 781 | **1.49×** | 1.02× |
| SVMWithL1Regularization | 1601 | 2321 | **1.45×** | 1.01× |
| CVaRBenchmark | 4736 | 6317 | **1.33×** | 0.99× |
| HuberRegression | 1765 | 2335 | **1.32×** | 0.98× |
| LeastSquares | 798 | 972 | **1.22×** | 0.99× |
| SlowPruningBenchmark | 1353 | 1488 | **1.10×** | 1.00× |
| ConvexPlasticity | 53 | 56 | **1.06×** | 0.92× |
| Murray | 1024 | 991 | 0.97× | **2.53×** |
| TvInpainting | 889 | 746 | 0.84× | 1.03× |
| QuantumHilbertMatrix | 1370 | 868 | 0.63× | 0.92× |
| UnconstrainedQP | 3606 | 973 | 0.27× | 0.85× |
| SDPSegfault1132Benchmark | 29113 | 6987 | 0.24× | 1.07× |
| ParametrizedQPBenchmark | 1561 | — | CPP killed: RSS >9 GB | |
| SimpleFullyParametrizedLPBenchmark | 325 | — | CPP killed by OS (memory) | |

**geomean 1.29× | 14/19 wins**

### RUST vs COO

| Benchmark | Rust (ms) | COO (ms) | COO/Rust | Δ vs pre-rebase |
| --- | --- | --- | --- | --- |
| SemidefiniteProgramming | 371 | 945 | **2.55×** | — |
| SimpleQPBenchmark | 798 | 1591 | **1.99×** | — |
| OptimalAdvertising | 261 | 487 | **1.86×** | — |
| LeastSquares | 798 | 1476 | **1.85×** | — |
| Cajas | 360 | 644 | **1.79×** | — |
| SimpleLPBenchmark | 2861 | 5069 | **1.77×** | — |
| FactorCovarianceModel | 573 | 1002 | **1.75×** | — |
| Yitzhaki | 525 | 869 | **1.66×** | — |
| SVMWithL1Regularization | 1601 | 2433 | **1.52×** | — |
| SimpleScalarParametrizedLPBenchmark | 683 | 987 | **1.44×** | — |
| HuberRegression | 1765 | 2510 | **1.42×** | — |
| SlowPruningBenchmark | 1353 | 1903 | **1.41×** | — |
| SimpleFullyParametrizedLPBenchmark | 325 | 451 | **1.38×** | — |
| CVaRBenchmark | 4736 | 5781 | **1.22×** | — |
| ConvexPlasticity | 53 | 56 | **1.05×** | — |
| Murray | 1024 | 1070 | 1.04× | — |
| ParametrizedQPBenchmark | 1561 | 1209 | 0.77× | — |
| TvInpainting | 889 | 685 | 0.77× | — |
| QuantumHilbertMatrix | 1370 | 841 | 0.61× | — |
| UnconstrainedQP | 3606 | 2078 | 0.58× | — |
| SDPSegfault1132Benchmark | 29113 | 8392 | 0.29× | — |

**geomean 1.23× | 16/21 wins** (no pre-rebase COO baseline exists)

---

## 2. In-repo synthetic suite (`benchmark_suite.py`, 40 cases, build_matrix layer)

- **SCIPY/RUST: geomean 4.81×**, range [1.96×, 72.2×], **RUST wins 40/40**
- **CPP/RUST: geomean 2.05×**, range [1.20×, 16.0×], **RUST wins 40/40**
- **COO/RUST: geomean 3.48×**, range [1.43×, 49.0×], **RUST wins 40/40**

(Raw data: `suite_4backend.json` / `suite_4backend.log`.)

## 3. Exhaustive per-atom sweep (`benchmark_suite.py --atoms`, 75 atoms, end-to-end)

- **SCIPY/RUST: geomean 1.49×, worst 1.07× — RUST wins 75/75**
- SCIPY/CPP: geomean 1.39× (70/70); SCIPY/COO: geomean 1.17× (69/75)

Every atom family — affine/structural (incl. einsum, ND ops, broadcast_to, convolve, both kron
orientations, partial_trace/transpose), elementwise, and matrix/reduction cone atoms — compiles
fastest through RUST. (Raw data: `atoms_4backend.json` / `atoms_4backend.log`.)

## 4. ASV backend suite (`asv run --python=same`, cvxpy-benchmarks)

Per-class geomeans of OTHER/RUST across all cases (full tables: `asv_report.md`):

| ASV class | SCIPY/RUST | CPP/RUST | COO/RUST |
| --- | --- | --- | --- |
| BackendCompileCanonicalization (16 cases) | 1.95× | 1.46× | 1.40× |
| BackendBuildMatrixCanonicalization (16 cases) | 3.96× | 2.13× | 1.83× |
| DeepExpressionTreeScaling (depth 4→256) | 1.31× | 1.14× | 1.10× |
| WideExpressionTreeScaling (width 8→256) | 1.99× | 1.03× | 2.82× |

Highlights: `murray_dense_above_threshold` (genuinely dense constant) — RUST 33.6 ms vs ~91 ms on
all three others (2.7×), so the sparsification heuristic does not tax the dense path;
`parameterized_lp` build_matrix — COO 0.30 ms vs RUST 0.51 ms vs CPP 75 ms vs SCIPY 249 ms
(COO's O(nnz) parameter handling is the one structural advantage RUST hasn't matched).

---

## Remaining losses and follow-ups

| Case | Ratio (vs best rival) | Mechanism | Status |
| --- | --- | --- | --- |
| ~~Murray (gini)~~ | ~~0.22×~~ → **1.04×** | dense mostly-zero constant walked in full | **fixed** (sparsification, `7bf07625c`) |
| SDPSegfault1132 | 0.05× vs SCIPY | `diag` of dense-affine: m² iteration + full COO sort in `specialized.rs::process_diag_mat` | follow-up |
| UnconstrainedQP | 0.27× vs CPP | `kron` eagerly allocates a dense lhs×rhs row-index map (`specialized.rs::process_kron_r/l`) | follow-up (sparsification already cut it 4938→3606 ms) |
| QuantumHilbertMatrix | 0.58× vs SCIPY | kron + partial_transpose, same mechanism family | follow-up |
| TvInpainting | 0.77× vs COO | small structured loss, unprofiled | follow-up |
| ParametrizedQP / parameterized_lp | 0.77× vs COO | COO's O(nnz) parameter tensors | COO's design point — yet RUST wins the sibling SimpleFullyParametrizedLP (1.38×) and is the only other backend to survive either |

Notes for reading the Δ column: `ConvexPlasticity`'s Δ (0.16×) reflects an upstream change to the
benchmark itself (it now compiles in ~55 ms vs ~18–57 s pre-rebase) — not a Rust regression.
`#3366` on master does not affect canonicalization timings for any backend (it optimizes the
NLP diff_engine path).
