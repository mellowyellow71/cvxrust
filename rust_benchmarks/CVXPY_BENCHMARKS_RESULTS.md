# cvxpy/benchmarks results — Rust canon backend vs SciPy & C++

Run of the **official [cvxpy/benchmarks](https://github.com/cvxpy/benchmarks)** suite against the
Rust canonicalization backend on this branch (`alan/arena-allocator`), compared head-to-head with
the `SCIPY` and `CPP` (cvxcore) backends.

- **Date:** 2026-06-17
- **Machine:** macOS (darwin 25.5.0), conda env `cvxpy-py313`, Python 3.13
- **What is measured:** wall-clock of `Problem.get_problem_data(...)` only — i.e. *canonicalization /
matrix stuffing*, not the numerical solve. This is exactly what each benchmark's
`time_compile_problem()` exercises.

---

## Methodology (and why it deviates from `asv run`)

The benchmarks repo is built for [Airspeed Velocity](https://asv.readthedocs.io/) (`asv run`). I did
**not** drive it through asv, for one decisive reason and a few practical ones:

1. **asv builds cvxpy from a git commit into an isolated virtualenv.** It would compile a *clean*
  cvxpy and never pick up the locally-built Rust extension (`cvxpy_rust`, built via maturin into the
   working env). The whole point here is to measure *our* backend, so asv's hermetic build defeats the
   experiment. Instead I imported each benchmark module directly and called its class.
2. `**get_problem_data` caches the solving chain on first call.** Calling it twice on the same
  `Problem` with different `canon_backend` values returns **stale** data from the first backend
   (verified empirically). So every single measurement builds a **fresh** problem (`Cls(); inst.setup()`)
   and times exactly one `get_problem_data`. The backend is injected by transiently wrapping
   `Problem.get_problem_data` to set `canon_backend=<backend>`, preserving each benchmark's native
   solver and other args.
3. **Crash isolation.** Each benchmark runs in its own subprocess (`run_one.py`), so an OOM-kill or
  segfault in one cannot poison the others' timings.
4. **No `timeout(1)` on macOS.** A per-measurement `SIGALRM` watchdog (150 s cap) stands in for
  `timeout`/`gtimeout`.

Per benchmark: **1 warm-up + 2 timed reps** per backend, median reported. (Light reps — the suite is
slow and the signal is large; treat sub-10% gaps as noise.)

Harness: `run_one.py` (single class, all backends) driven by `sweep.sh` over all 25 classes; raw
output in `results.jsonl`.

---

## Results

All times are **median milliseconds** for canonicalization. Ratios are **>1 ⇒ Rust faster**:
`R/S = SciPy / Rust`, `R/C = CPP / Rust`.


| Benchmark                  | Rust (ms) | SciPy (ms) | CPP (ms) | R/S       | R/C   |
| -------------------------- | --------- | ---------- | -------- | --------- | ----- |
| LeastSquares               | 910       | 2534       | 1545     | **2.79×** | 1.70× |
| ConvexPlasticity           | 18528     | 56943      | 16413    | **3.07×** | 0.89× |
| Cajas                      | 432       | 886        | 2078     | **2.05×** | 4.81× |
| FactorCovarianceModel      | 651       | 1224       | 1261     | **1.88×** | 1.94× |
| OptimalAdvertising         | 304       | 548        | 1910     | **1.80×** | 6.29× |
| SimpleQPBenchmark          | 1490      | 2695       | 2776     | **1.81×** | 1.86× |
| SimpleLPBenchmark          | 7026      | 12059      | 14137    | **1.72×** | 2.01× |
| SlowPruningBenchmark       | 1501      | 2498       | 1815     | **1.66×** | 1.21× |
| HuberRegression            | 1743      | 2858       | 3287     | **1.64×** | 1.89× |
| SemidefiniteProgramming    | 442       | 715        | 1087     | **1.62×** | 2.46× |
| Yitzhaki                   | 847       | 1337       | 1253     | **1.58×** | 1.48× |
| SVMWithL1Regularization    | 2036      | 2991       | 3419     | **1.47×** | 1.68× |
| SimpleScalarParametrizedLP | 1153      | 1502       | 1565     | **1.30×** | 1.36× |
| CVaRBenchmark              | 9211      | 10291      | 14665    | **1.12×** | 1.59× |
| QuantumHilbertMatrix       | 1114      | 1199       | 1238     | **1.08×** | 1.11× |
| TvInpainting               | 997       | 803        | 914      | 0.81×     | 0.92× |
| UnconstrainedQP            | 7808      | 3008       | 2406     | 0.39×     | 0.31× |
| Murray (gini)              | 7739      | 1723       | 2347     | 0.22×     | 0.30× |
| SDPSegfault1132            | 55255     | 3230       | 27075    | 0.06×     | 0.49× |


### Aggregates (19 measurable benchmarks)


| Comparison        | Geomean speedup | Rust wins |
| ----------------- | --------------- | --------- |
| **Rust vs SciPy** | **1.14×**       | 15 / 19   |
| **Rust vs CPP**   | **1.38×**       | 14 / 19   |


The Rust backend is the fastest of the three on the **majority** of real problems, and the wins are
often large (1.5–3×). The geomean is dragged down almost entirely by a small cluster of pathological
regressions described next.

---

## The regressions: two distinct mechanisms (not one cluster)

Four benchmarks where Rust *loses*. On reading the sources they split into **two unrelated causes** —
an earlier draft lumped them as one "kron/diag" cluster, which was wrong (Murray contains neither
`kron` nor `cp.diag`):


| Benchmark           | R/S                     | Mechanism                                                        |
| ------------------- | ----------------------- | --------------------------------------------------------------- |
| **SDPSegfault1132** | **0.06×** (≈17× slower) | kron + `diag` of a *dense affine* Gram matrix `V@G@V.T`; PSD var |
| **UnconstrainedQP** | 0.39×                   | `kron(I, diag(var))` sandwiched between dense DFT matrices       |
| **Murray (gini)**   | 0.22×                   | 244650×700 **dense** constant matrix (99.7% zeros) @ variable    |
| TvInpainting        | 0.81×                   | minor; sub-second problem, FFI/setup-bound                      |

### Mechanism 1 — kron + diag-of-dense-affine (SDP1132, UnconstrainedQP)

Both build a large, **fully dense** canonical block from a small variable:

- **SDP1132**: `cp.diag(V @ G @ V.T)` extracts the diagonal of a Gram matrix that is dense-affine in
  the PSD variable `G` (every one of the n² entries is a combination of *all* of G's entries), then
  `cp.kron(e, …)` broadcasts that diagonal n times. diag-of-dense-affine **× kron** is what explodes
  the COO tensor — not `diag` alone.
- **UnconstrainedQP**: `cp.kron(np.diag(ones(14)), cp.diag(var))` is a 252×252 block-diagonal operator
  in 18 variables, sandwiched as `H_H @ Err_est @ H` between dense (complex) DFT matrices — expanding
  18 variables into a dense ~252×252×2 coefficient.

The Rust backend appears to materialize/sort far more COO entries than SciPy's specialized paths for
these operators. SDP1132 alone dominates the geomean; drop it and the vs-SciPy geomean rises markedly.

### Mechanism 2 — dense constant that should be sparse (Murray)

Murray has **no kron and no `cp.diag`**. It builds `mat = np.zeros((244650, 700))` and writes a single
`+1` and `−1` per row — a ~171M-entry constant that is **99.7% zero** — then does `mat @ ret_w`. Because
`mat` reaches the backend as a *dense* `Constant`, the Rust dense-mul arm
(`arithmetic.rs::mul_const_by_variable`) pays a full **O(rows·cols)** walk over all 171M cells — once in
the `data.iter().filter(...).count()` nnz pre-scan (`arithmetic.rs:737`) and again in the
`for c { for r { … } }` emission loop (`arithmetic.rs:757`) — even though it correctly *skips emitting*
the zeros. SciPy/CPP exploit the sparsity and touch only the ~489K nonzeros. **Fix:** detect a
mostly-zero dense `Constant` and convert it to CSC before the multiply.

`TvInpainting` (0.81×) is a different, benign story: a sub-second problem where fixed per-call overhead
(FFI + setup) is a meaningful fraction of the total, so the backend choice barely matters.

---

## Not measurable (excluded from aggregates)

- **4× `matrix_stuffing.py` classes** — `ConeMatrixStuffingBench`, `ParamConeMatrixStuffing`,
`ParamSmallMatrixStuffing`, `SmallMatrixStuffing`. These call
`ConeMatrixStuffing().apply(self.problem)` directly inside `time_compile_problem()` and **never call
`get_problem_data`**, so the canon-backend injection has nothing to wrap. Reported as N/A — not a
failure, just outside this harness's measurement point.
- **2× fully-parametrized problems OOM-killed (rc=137), ignored per request** —
`SimpleFullyParametrizedLPBenchmark` (n=10⁶ `Parameter`) and `ParametrizedQPBenchmark`
(m=6000, n=2400 fully-parametrized A, b). Both emit cvxpy's *"too many parameters for efficient DPP
compilation"* warning and exhaust memory during DPP expansion before any backend timing is recorded.
Because the process dies before the first measurement, the cost cannot be attributed to a specific
backend; the DPP parameter blow-up is the proximate cause, not the Rust path. Excluded.

---

## Bottom line

- On real, solvable problems the **Rust backend is the fastest of the three backends** — geomean
**1.14× vs SciPy** and **1.38× vs CPP**, winning 15/19 and 14/19 respectively, with several 1.5–3×
wins.
- The losses come from **two separate weaknesses**, each a clear next step:
  1. **kron + diag-of-dense-affine SDP/QP construction** (SDPSegfault1132, UnconstrainedQP) — Rust
     materializes far more COO entries than SciPy's specialized operator paths. SDP1132 (0.06×)
     dominates the geomean and is the single biggest lever.
  2. **dense-constant-not-sparsified** (Murray, 0.22×) — a 99.7%-zero dense `Constant` walked
     densely by `mul_const_by_variable`; convert mostly-zero dense constants to CSC before the multiply.

### Reproduce

```sh
# from rust_benchmarks/ tmp harness:
zsh sweep.sh           # runs all 25 classes, one subprocess each -> results.jsonl
# single benchmark, all backends:
python run_one.py <benchmarks>/benchmark/simple_QP_benchmarks.py LeastSquares RUST,SCIPY,CPP 1 2 150
```

