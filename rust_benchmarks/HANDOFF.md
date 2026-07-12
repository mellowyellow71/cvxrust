# CVXRust project handoff

Last updated 2026-07-08 (session of 2026-07-02 → 07-08). Written for the next agent/session
picking this project up. Read this top to bottom before touching anything; the Pitfalls
section will save you real time. Ray (GitHub `mellowyellow71`) is the human owner;
Alan is a coworker (the pre-rebase branch bears his name).

## 1. What this project is

Get cvxpy's experimental **Rust canonicalization backend** (`cvxpy_rust`, PyO3 + rayon)
merged upstream. Upstream's stated blocker (cvxpy/cvxpy#3018) is *"more complete
benchmarking"*. Strategy (set by Ray's advisor): first land a **backend benchmark suite**
in the community repo `cvxpy/benchmarks` (the yardstick), then open the **main backend PR**
against `cvxpy/cvxpy` with reproducible evidence.

## 2. Current status — one line each

- **Benchmarks PR** [cvxpy/benchmarks#32](https://github.com/cvxpy/benchmarks/pull/32):
  review round 1 (PTNobel) fully addressed in commit `006e7cd7f` + PR description lists
  fix-per-comment. One good signal received; **waiting on a second reviewer**. Do nothing
  on it until feedback arrives.
- **Rust backend**: lives on fork branch `ray/latestfixes` (= upstream cvxpy master
  `40947203c` ≈ 1.9.2-dev + our port). All tests green, correctness gate ALL MATCH,
  two real performance fixes landed (dense-constant sparsification, ND-matmul correctness).
- **Benchmark evidence**: complete 4-suite comparison vs SCIPY/CPP/COO in
  `CVXPY_BENCHMARKS_RESULTS.md` (+ `benchmark_report.tex` for Overleaf, + an HTML
  artifact at https://claude.ai/code/artifact/c441ac9a-054d-44b0-ae92-cf657d87f82e).
- **Next milestones**: (a) benchmark PR merged → (b) finish testing (list in §7) →
  (c) main backend PR to cvxpy/cvxpy (plan in §8).

## 3. Repo & branch map

| What | Where |
|---|---|
| cvxpy fork (working repo) | `/Users/revantkasichainula/cvxpy` — `origin` = mellowyellow71/cvxpy, `upstream` = cvxpy/cvxpy |
| Live branch | `ray/latestfixes` (tracks origin; based on upstream/master `40947203c`) |
| Frozen pre-rebase archive | `alan/arena-allocator` (do not touch; source of the 2026-06-17 baseline) |
| Rust crate | `cvxpy_rust/` at fork root (PyO3 0.27, rayon; cdylib+rlib) |
| Python glue | `cvxpy/lin_ops/backends/rust_backend.py` (serializer + `RustCanonBackend`) |
| Benchmarks clone | `rust_benchmarks/cvxpy-benchmarks/` (untracked in the fork; `origin` = cvxpy/benchmarks, `fork` = mellowyellow71/benchmarks) |
| PR branch (benchmarks) | `backend-canonicalization-benchmarks` in that clone, pushed to `fork` |

Key commits on `ray/latestfixes` (oldest→newest): `2b09ac1be` port onto 1.9 master ·
`ec2a0c34c` benchmark_suite COO + `--atoms` · `7bf07625c` dense-constant sparsification ·
`b8a18bbf7` 4-backend report · `49e64b0f2` SimpleFullyParametrizedLP isolated cells ·
`d4c4f5a4a` LaTeX report · `894c2b501` **ND-matmul correctness fix**.

## 4. Build & test (the gotchas that bite)

- **Two cvxpy_rust builds exist**: maturin editable in `.venv/.../site-packages/cvxpy_rust/`
  (what scripts run from `rust_benchmarks/` import) and a gitignored repo-root
  `cvxpy_rust.cpython-313-darwin.so` (imported only by repo-root `python -c`). Both are
  release builds now (`setup.py` sets `debug=False`). After editing Rust sources refresh
  both: `pip install -e .` (rebuilds cvxcore + root .so) and `maturin develop --release`
  from `cvxpy_rust/`. **Always verify freshness before benchmarking**: print
  `cvxpy_rust.__file__` from the benchmark's cwd, compare mtime to newest `cvxpy_rust/src/*`.
- **Cargo tests**: `cargo test --release --no-default-features` (extension-module feature
  must be off to link libpython); clippy clean.
- **Python gauntlet** (run after any backend change, from fork root):
  `pytest cvxpy/tests/test_rust_backend.py` (139 tests) ·
  `pytest cvxpy/tests/test_python_backends.py cvxpy/tests/test_backend_selection.py cvxpy/tests/test_kron_canon.py` ·
  broad smoke `CVXPY_DEFAULT_CANON_BACKEND=RUST pytest cvxpy/tests/test_problem.py cvxpy/tests/test_expressions.py cvxpy/tests/test_conic_solvers.py -q`.
- **Correctness gate** (before ANY timing campaign): `python verify_backends.py` from
  `rust_benchmarks/` — must print ALL MATCH (every ASV case × RUST/CPP/COO vs SCIPY
  reference, on the **constraints** call).
- **CAUTION — env-var test sweeps lie for ND**: upstream `test_nd_matmul.py` passes
  `canon_backend=` explicitly, so `CVXPY_DEFAULT_CANON_BACKEND=RUST pytest ...` does NOT
  route those through RUST. That's how the ND-matmul bug hid. Use the explicit
  cross-backend tests in `test_rust_backend.py::test_nd_matmul_matches_scipy`.

## 5. Benchmark infrastructure inventory (all in `rust_benchmarks/`)

| File | Purpose |
|---|---|
| `cvxpy-benchmarks/benchmark/canonicalization_backends.py` | THE ASV suite (= PR #32 payload): 18 cases × 4 backends × {compile, build_matrix} + Deep/Wide tree + WideConstraint scaling classes; `LINOP_COVERAGE` checklist |
| `verify_backends.py` | cross-backend A-tensor equality gate |
| `gate_bench_cases.py` | runs every ASV cell once; expected n/a = CPP × {concatenate, nd_*, einsum} |
| `sweep.sh` + `run_one.py` | external whole-problem suite driver (21 classes, subprocess-isolated, RESULT jsonl lines) |
| `benchmark_suite.py` | in-repo synthetic suite (40 cases) + `--atoms` exhaustive 75-atom sweep, 4 backends |
| `asv_matrix_report.py` | pivots results.jsonl / asv json into pairwise markdown tables |
| `gen_tex.py` | regenerates `benchmark_report.tex` (Overleaf-ready) from the jsons |
| `results.jsonl` / `results_prerebase.jsonl` | current external-suite data / 2026-06-17 baseline |
| `suite_4backend.json` · `atoms_4backend.json` | synthetic + atom sweep raw data |
| `CVXPY_BENCHMARKS_RESULTS.md` | the canonical report (tables copied into tex/HTML) |
| `asv_local.conf.json` (in clone, untracked) | asv conf with `repo: "../.."`; run `asv run --python=same --set-commit-hash HEAD --config asv_local.conf.json` |

Headline numbers (2026-07-03, post-sparsification): external suite geomeans
**1.09× vs SCIPY (15/19), 1.29× vs CPP (14/19), 1.23× vs COO (16/21)**; synthetic 40/40
vs all; atoms 75/75 vs SCIPY (geomean 1.49×). Murray FIXED (0.22×→1.04×). Only RUST and
COO survive the two fully-parametrized classes (SCIPY times out / OOM; CPP OOMs —
**ParametrizedQPBenchmark × CPP crashed the host twice; never run it unguarded**).

Remaining known losses (mechanisms in `cvxpy_rust/src/operations/specialized.rs`):
SDPSegfault1132 0.05× (diag-of-dense-affine, `process_diag_mat` m² scan + full sort) ·
UnconstrainedQP 0.27× vs CPP (kron eager dense index map, `process_kron_r/l`) ·
QuantumHilbertMatrix 0.58× (same family) · TvInpainting 0.77× vs COO (unprofiled) ·
huge-param DPP 0.77× vs COO (COO's O(nnz) design point; RUST wins the sibling
SimpleFullyParametrizedLP 1.38×).

## 6. PR #32 — state and what to do on feedback

- Round-1 reviewer PTNobel (maintainer, COO author): all 15 comments addressed in code
  (`006e7cd7f`); the PR **description** carries the comment-by-comment fix list (Ray's
  chosen mechanism — do NOT post thread replies unless Ray says so).
- His review surfaced two real bugs, both fixed: the build_matrix **call-selection tie**
  (was timing the trivial objective call on single-constraint cases) and, downstream of
  that, the **ND-matmul correctness bug** in our Rust backend (`894c2b501`).
- A maintainer also asked how this differs from pre-existing benchmarks / why merge it —
  Ray has a drafted answer (in session history; gist: existing suite = whole problems on
  the default backend only; ours parameterizes backend × case at two measurement levels
  with LinOp-type coverage, and it caught a real backend bug; #3018's blocker is
  benchmarking). Check the PR thread for whether it was posted before re-answering.
- When approved/merged: nothing else needed on the benchmarks side except §7.1.
- If more review comments: research each before fixing (the pattern that worked);
  validate with ruff (`--config pyproject.toml` in the clone), `asv check --config
  asv_local.conf.json`, `gate_bench_cases.py`, `verify_backends.py`.

## 7. Remaining testing work (before the main PR)

1. **Refresh stale numbers**: local reports still carry pre-fix build_matrix cells for the
   previously-tied cases (flat 0.02–0.07 ms values are the artifact). Re-run the ASV suite
   (`asv run --python=same ...`) + regenerate `CVXPY_BENCHMARKS_RESULTS.md` §4,
   `benchmark_report.tex` (`gen_tex.py`), and note the correction. Corrected quick numbers
   already measured (see PR #32 description; e.g. RUST leads nearly every corrected cell;
   `shared_subexpressions` compile: RUST 8.2 / CPP 10.2 / COO 13.6 / SCIPY 14.3 ms).
2. **Atom-sweep completeness**: introspection found ~30 exported atoms not in the 75-atom
   sweep — real gaps include `sum_squares`, `mean`/`var`, `cummax`, `lambda_sum_largest/
   smallest`, `outer`/`vdot`, ND-manipulation (`squeeze`/`swapaxes`/`moveaxis`/`stack`),
   complex (`real`/`imag`/`conj`), quantum (`von_neumann_entr`, `quantum_rel_entr`,
   `tr_inv`), DGP/DQCP atoms (need `gp=True`/`qcp=True` compiles). Extend
   `make_atom_problems()` in `benchmark_suite.py`; many "missing" names are aliases —
   diff programmatically against `cvxpy.atoms` exports first.
3. **ND parametric matmul**: currently raises `ValueError` in the RUST serializer
   (loud, correct, but unsupported). Decide: implement (mirror `_expand_parametric_slices_
   mul/rmul` + `_build_interleaved_param_matrix_*` from scipy_backend.py) or document as
   limitation in the main PR. Note ND problems route to RUST **by default** on our branch
   (`solving_chain_utils.get_canon_backend`), so this raise is user-visible.
4. Optional hardening: run the full cvxpy test suite once with
   `CVXPY_DEFAULT_CANON_BACKEND=RUST` (not just the smoke subset); Linux/Windows CI has
   never run the Rust workflows (fork CI — check Actions on `ray/latestfixes`).

## 8. The main PR (cvxpy/cvxpy) — plan sketch

Payload = the fork's delta vs upstream/master, minus benchmark artifacts:
`cvxpy_rust/` crate · `cvxpy/lin_ops/backends/rust_backend.py` (replaces upstream's
experimental stub from #3018, keeping class name/registry) · `setup.py`/`pyproject.toml`
(setuptools-rust, `debug=False`) · CI workflow additions · `cvxpy/tests/test_rust_backend.py`
· small `solving_chain_utils.py` + `settings.py` deltas. Framing: cite the (hopefully
merged) benchmarks suite + `CVXPY_BENCHMARKS_RESULTS.md` numbers; be upfront about the
losses table and unsupported edges. **Open decisions to settle with maintainers** (flag in
the PR, don't presume): default-backend policy (our branch auto-selects RUST when
importable + routes ND to RUST; upstream default is CPP), `order='C'` unsupported
(raises), ND parametric matmul unsupported (raises), wheels/rust-toolchain implications
(cibuildwheel bootstrap is already in our build.yml delta), MSRV. Use upstream's PR
template (`.github/`), fill the checklist honestly.

Rust-side perf follow-ups (optional pre-PR, kron is the biggest lever): lazy kron indices;
diag-of-dense-affine filtered extraction; TvInpainting profiling; COO-style O(nnz)
parameter handling.

## 9. Pitfalls (each cost us real time — believe them)

- **Long-running benchmarks**: Claude session restarts kill harness-backgrounded AND
  nohup'd processes. Launch via
  `subprocess.Popen([...], start_new_session=True)` under `caffeinate -i`, append results
  incrementally (jsonl), and design every campaign to be resumable. Never run two timing
  campaigns concurrently.
- **ParametrizedQPBenchmark × CPP** exceeds 9 GB RSS and has hard-crashed this 18 GB Mac
  twice. Always wrap risky cells with an RSS-watchdog kill (pattern in session history /
  `resume_sweep.sh` in git history) and run them LAST.
- **The solving-chain cache**: `get_problem_data` results are cached per Problem instance —
  always build a fresh `cp.Problem` per timed/verified call.
- **`get_problem_matrix` is called twice per compile** (objective, then constraints);
  select the constraints call by `constr_length`, never by `len(lin_ops)` (tie-bug).
- **Released cvxpy (PyPI 1.9.2) lacks `SPARSE_DENSITY_THRESHOLD`** (1.9.x release branch
  predates #3366) — anything meant to run on PyPI cvxpy needs
  `getattr(cp.settings, "SPARSE_DENSITY_THRESHOLD", 0.05)`.
- **cvxcore OpenMP is compiled out of shipped wheels** and lock-serialized even when on
  (`cvxcore.cpp:78,174,177`) — don't attribute CPP wide-problem performance to threading.
- Ray's **PR policy**: every outward-facing artifact (PR bodies, replies, new PRs) is
  drafted → Ray reviews/edits → only then sent. Pushing to Ray's own fork branches is fine.

## 10. Memory & docs pointers

Agent memory (auto-loaded next session): `cvxpy-rust-build-load-gotcha.md`,
`cvxpy-rust-benchmark-losses.md` in the project memory dir. Long-form analyses from the
pre-rebase era: `LOSSES_ANALYSIS.md`, `ROUND2_CHANGES.md`, `CVXPY_BENCHMARKS_RESULTS.md`
(current), `PR_DRAFT.md` (original PR #32 text). Plan file from the last approved plan:
`~/.claude/plans/new-things-for-cvxrust-elegant-river.md` (review-response plan, executed).
