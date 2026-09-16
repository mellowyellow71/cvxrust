# CVXRust project handoff

## Resume on another machine (written 2026-09-15)

Everything lives on the fork `mellowyellow71/cvxrust` (remote `origin`). Three branches:

| Fork branch | What it is |
| --- | --- |
| `rust-rebase-20260720` | this doc, benchmark tooling, reports, the second-agent audit (`rust_benchmarks/ASTRA_AUDIT_2026-09-15.md`); backend code at the July base |
| `rust-backend-pr-20260915` | **the candidate**: clean backend payload rebased onto upstream `773163ebd` + `cargo fmt` + default-routing fix (3 commits). Start coding here. |
| `ray/latestfixes` | July archive checkout, snapshotted with its uncommitted artifacts |

```bash
git clone https://github.com/mellowyellow71/cvxrust.git cvxpy && cd cvxpy
git remote add upstream https://github.com/cvxpy/cvxpy.git && git fetch upstream
git checkout rust-rebase-20260720                       # docs + tooling
git worktree add rust_benchmarks/backend-pr-20260915 rust-backend-pr-20260915
cd rust_benchmarks/backend-pr-20260915
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e ".[testing]" maturin                     # needs Rust >= 1.80 (rustup)
(cd cvxpy_rust && maturin develop --release)
python -m pytest cvxpy/tests/test_rust_backend.py -q    # expect 142 passed
python -c "import cvxpy_rust; print(cvxpy_rust.build_matrix_serialized)"   # must not be a namespace package
```

For the external benchmark classes: `git clone https://github.com/cvxpy/benchmarks.git
rust_benchmarks/cvxpy-benchmarks` (add `fork` = mellowyellow71/benchmarks; PR #32 branch is
`backend-canonicalization-benchmarks`). `rust_benchmarks/benchmarks-integration` was a
local-only test merge and was intentionally not pushed.

Next task = step 0 and step 1 of "Revised work plan" below (safe availability check, then
the three N-D correctness bugs with an affine-map oracle test).

## 2026-09-15 status (first session after a two-month pause)

Read this section first. Everything under "Last updated 2026-07-22" below is still
accurate for tooling, safety and benchmark workflow; the upstream picture changed.

### Upstream since our base `359e9bb52` (44 commits, now `773163ebd`)

- **DIFFENGINE canon backend merged** (#3448, Transurgeon, 2026-09-09): builds the stuffed
  matrices by evaluating the canonicalized tree with sparsediffpy instead of tensor
  algebra. Opt-in, parameter-free only, `sparsediffpy>=0.6.1,<0.7` is now a hard
  dependency (the venv had 0.5.1; upgraded to 0.6.1 this session).
- **#3449 is open** (updated 2026-09-13): makes DIFFENGINE the default for every
  `ignore_dpp`/non-DPP solve and keeps parameters symbolic with a cached program. If it
  merges, CPP/COO/RUST stay the default only on the DPP-parametric path. The backend PR
  must be pitched with this in mind (see Next actions).
- **#3398** (PTNobel, 2026-08-23) fixed silent wrong coefficients in SCIPY/COO for
  parametric-lhs matmul (`(A @ p) @ x`, `broadcast_to(p) @ x`) and added
  `test_python_backends.py::test_matmul_with_parametric_expression_lhs`, parametrized
  over SCIPY/COO/**RUST**. RUST never had the bug (checked against a numpy reference on
  four shapes; the new test passes). Strong correctness talking point.
- New atom `cp.block` (#3482); `geo_mean` is an AxisAtom (#3493); stats atoms accept
  broader axes (#3496); `norm` raises on axis for fro/nuc (#3501); cone restructuring is a
  chain reduction applied as a permutation (#3525, no backend-interface change); cvxcore
  builds as `cvxpy.cvxcore.python._cvxcore` (#3522); `canon_backend` is in the solve
  cache key; CI matrix adds Python 3.14 and 3.14t.
- 3.14t note: `cvxpy_rust` uses a plain `#[pymodule]`, so a free-threaded interpreter
  re-enables the GIL on import with a RuntimeWarning (tests still pass). Decide whether
  to add `#[pymodule(gil_used = false)]` after a thread-safety pass, or document it.

### Validation done 2026-09-15 (payload rebased onto `773163ebd`)

- Trial rebase of `codex/rust-backend-pr-20260721` onto upstream/master: one conflict
  (`.github/workflows/build.yml`, cibuildwheel 4.1.0 vs 4.2.0), all else auto-merged.
- New local branch **`codex/rust-backend-pr-20260915`** = rebased payload + `cargo fmt`
  commit + routing-fix commit (below). Not pushed. Its worktree lived in the session
  scratchpad; materialize with
  `git worktree add rust_benchmarks/backend-pr-20260915 codex/rust-backend-pr-20260915`
  then `pip install -e .` and `maturin develop --release` from `cvxpy_rust/`.
- `test_rust_backend.py` 142 passed. `test_python_backends.py` + `test_backend_selection.py`
  + `test_kron_canon.py` + `test_nd_matmul.py` + `test_expressions.py`: 592 passed.
  `test_dpp.py -k cumsum_of_parameter`: passed. Broad smoke with
  `CVXPY_DEFAULT_CANON_BACKEND=RUST` over test_problem/test_conic_solvers/test_dpp/
  test_atoms/test_complex/test_constant_atoms/test_qp_solvers: 745 passed, 449 skipped
  (solvers absent), 0 failed. `cargo clippy -D warnings` clean, `cargo test` 31 passed.
- **Fork CI is red on every branch only because of `cargo fmt`** (the pre-commit hook we
  added). Fixed by the fmt commit on the new branch. build.yml / test_backends.yml have
  still never run on the fork: they trigger on pull_request or pushes to master.
- Atom gate on the rebased build: 125 cases (block added), 130 exports accounted;
  RUST/SCIPY/COO compile everything; CPP has the 12 expected ND/broadcast n/a cells plus
  `block` (it lowers to concatenate).
- Uncommitted edits in this worktree's `benchmark_suite.py`: `block` atom case
  (`requires="block"`, `supports_cpp=False`), deprecated `cp.conv` replaced by
  `cp.convolve` in the synthetic convolution case and the atom case.

### Routing fix on the new branch (commit `86bfab9dc`; drop it if maintainers object)

`get_canon_backend` warned and fell back to SCIPY for any CPP-unsupported expression even
with RUST importable; only the `_max_ndim() > 2` branch was special-cased for RUST. It now
resolves the default first (env var, then settings) and returns it when it is
SCIPY/COO/RUST; the warning/fallback and the explicit-CPP error survive for a CPP default.
Tests assert on the resolved default and pass under UNSET/RUST/SCIPY/COO/CPP, so the
`test_backends.yml` matrix stays green.

### Same-machine cold `get_problem_data` (one isolated process per cell, BLAS 1 thread)

| class | RUST | CPP | DIFFENGINE (ignore_dpp) | RUST (ignore_dpp) |
| --- | ---: | ---: | ---: | ---: |
| Cajas | 0.302 s | 2.206 s | 0.677 s | 0.313 s |
| Murray | 1.164 s | 1.45 s | 1.059 s | 1.129 s |
| QuantumHilbertMatrix | 1.336 s | 0.931 s | 0.653 s | 1.296 s |
| TvInpainting | 0.862 s | 0.741 s | 0.419 s | 0.823 s |
| HuberRegression | 0.861 s | 1.554 s | 0.884 s | 0.854 s |
| ConvexPlasticity | 0.056 s | 0.059 s | 0.483 s | 0.055 s |
| LeastSquares | 0.886 s | 1.131 s | 0.787 s | 1.009 s |
| SDPSegfault1132Benchmark | 30.253 s | 6.671 s | 2.764 s | 29.897 s |

One cold sample per cell, 2026-09-15, payload rebased onto `773163ebd`, `sparsediffpy` 0.6.1. Indicative only (single sample). RUST wins Cajas, HuberRegression, ConvexPlasticity and ties Murray/LeastSquares; it loses QuantumHilbertMatrix and TvInpainting to both, and SDPSegfault1132 badly (diag-of-dense-affine, the known `process_diag_mat` m^2 scan). DIFFENGINE is the reference the maintainers will compare against now.


### CPP-parity diagnosis (2026-09-15, native sampling with macOS `sample`)

Is RUST completely better than CPP? Not yet. It wins 122/124 atoms (the other two are
statistical ties), 40/40 synthetic, 17/18 ASV compile cells and 14/19 external classes,
and it supports everything CPP rejects (ND, broadcast, einsum, concatenate, `block`).
The remaining CPP wins all come from two backend mechanisms, not from FFI (deser is
<0.3% everywhere; Python serialization is 0.2-10 ms):

1. **mul/rmul kernels push one COO entry per (input nnz x constant nnz-per-column)
   product and rely on the final global sort to coalesce.** CPP's cvxcore does an Eigen
   sparse product (per-column accumulation, no explosion, no global sort).
   - UnconstrainedQP: `@ H` rmul, 252x252 dense H on a 2.3M-nnz tensor = ~576M pushes
     (build 4.2 s; the eight `process_mul` calls take 0.2 ms each).
   - SDPSegfault1132: `@ V.T` rmul (dense 100x99) explodes to ~1e8 entries, then
     `diag` keeps 1%; 21 s build, most of it in `rayon::slice::sort` (the mul calls are
     3 ms each). Pre-allocation is nnz x a_rows, i.e. ~3 GB.
   - QuantumHilbertMatrix: 83% of build samples in `multiply_block_diagonal_right`
     (`@ AxI`, 512x512 sparse const, density 0.29).
   - murray_dense_constant (ASV 0.59x): same pattern in `multiply_sparse_block_diagonal`.
2. **Final global sort in `BuildMatrixResult::from_tensor`** (`par_sort_unstable_by_key`
   over every entry). TvInpainting's 570 ms build is dominated by it; it is also why
   WideExpressionTreeScaling(256) is a tie with CPP. A counting sort by column, or a
   k-way merge of per-constraint sorted runs, is O(nnz).

DAG (PTNobel's point): the serializer has no memo, so shared LinOp objects are
re-emitted and re-lowered. Measured tree-nodes/unique-LinOps: QuantumHilbert 3.6x (a
`sparse_const(4096x2080) @ variable` subtree lowered 512 times), UnconstrainedQP 2.1x (the
two `H_H @ Err_est @ H` sums lowered twice), Cajas 1.5x, TvInpainting 1.4x. Synthetic
`e = e + e` chain: 65k emitted nodes for 16 unique at depth 16 (RUST 745 ms, CPP 1545 ms,
SCIPY 3992 ms: no backend exploits the DAG today). CPP memoizes only C++ node
construction (`linPy_to_linC`), not evaluation.

Work plan to reach "better than CPP everywhere measured" (about one week):

1. Accumulating mul/rmul kernels (dense + sparse): per-(block, column) dense workspace,
   emit coalesced entries. Fixes 1 above. ~2 days incl. tests against SCIPY.
2. Replace the global sort with a counting sort by column or a merge of sorted runs.
   ~1 day. Fixes TvInpainting, helps every large problem.
3. DAG memo: serializer emits each LinOp once (id -> index) plus `ref` nodes; Rust
   lowers shared nodes once into `Arc<SparseTensor>` before the parallel constraint
   pass. ~1 day. Fixes the rest of UnconstrainedQP and QuantumHilbert.
4. Optional: row-filter push-down for `diag`/`index` over mul (SDPSegfault beyond CPP
   parity), lazy kron index map. ~1-2 days.
5. Re-run the four-backend campaigns, refresh `BACKEND_PR_DRAFT.md`. ~0.5 day.

### Audit reconciliation (2026-09-15, second agent's `rust_benchmarks/ASTRA_AUDIT_2026-09-15.md`)

Verified and accepted:

- **Three RUST-only correctness bugs**, all N-D: `_emit_sparse` converts every sparse
  payload to CSC (fails on 3-D sparse constants: "CSC format must be 2D");
  `compute_vstack_indices` falls into the 1-D branch for 3-D/4-D output (wrong row
  order); `process_trace` treats a batched (3,2,2) input as one 2-D trace. Reproduced
  with an affine-map oracle (`A @ vecF(x0)` vs `expr.value`): vstack-3D and batched trace
  MISMATCH on RUST only; identity, stack, concatenate(axis=0), 2-D cases all match.
  Failing upstream test: `test_atoms.py::TestAtoms::test_lambda_sum_largest_nd_solve`
  (full suite with RUST default: 1 failed / 2831 passed). The same test also fails on
  COO (`inconsistent shapes (8, 2) and (4, 1)`), an upstream COO bug.
- **Upstream bug on all backends**: N-D `hstack` and `concatenate(axis=None)` canonicalize
  differently from their own `.value`; a least-squares recovery of `x0` fails on SCIPY and
  COO too (residual 8.5 / 21 instead of 0). File upstream with the 6-line repro.
- **Availability check is unsafe**: from a source checkout without the built extension,
  `import cvxpy_rust` succeeds as a namespace package (the crate directory), so settings
  select RUST and every solve then dies on a missing `build_matrix_serialized`. Check the
  attribute, not the import.
- Sparse right-multiply scans every CSC column for each input entry (O(lhs.nnz x nnz(A)));
  QuantumHilbert's `AxI` is 512x512 with 9,520 nnz (density 0.036, not 0.29).
- TvInpainting hotspot is `select_rows` (general HashMap path for strided index) plus the
  sort, not the sort alone. The sort key is the flattened (var column, row), so "counting
  sort by column" is not a drop-in; radix sort or merged sorted runs are the candidates.
- Accounting: 124 atoms = 112 timed CPP comparisons (110 RUST faster, 2 CPP faster) + 12
  CPP-unsupported; ASV compile = 14 timed (13 wins) + 4 unsupported. Report capability
  and speed separately. The 576M-push figure for UnconstrainedQP is a computed estimate.
- Packaging: README says Rust 1.70+ but rayon 1.12 needs 1.80; `Cargo.lock` is gitignored;
  `RustExtension(optional=True)` + auto-default is an inconsistent rollout contract; the
  wheel build step runs only on push events, so a PR never exercises cibuildwheel (push the
  candidate to the fork's master to trigger it); `multiply_parametric_left/right` silently
  pick one param when both operands are parametric (unreachable under DPP, but make it an
  explicit error).

Not accepted as stated: the audit's "no evidence for a one-week estimate" is fair as a
caveat; the estimates below remain estimates.

### Revised work plan (supersedes the list above)

0. Safe availability check (attribute, not import). ~30 min.
1. Fix the three N-D bugs with end-to-end RUST tests plus an affine-map oracle test that
   runs every backend against `expr.value`; file the upstream hstack/concatenate issue and
   the COO failure. ~1 day.
2. Instrument raw-vs-unique entry counts per op, then redesign dense/sparse mul and rmul:
   accumulate per (block, column), CSR traversal for sparse rmul. Re-profile. ~2-3 days.
3. Output assembly and `select_rows` from the new profile (radix sort / merged runs /
   strided-slice fast path). ~1-2 days.
4. DAG memo (node table + refs, shared nodes lowered once before the parallel pass, with
   copy-on-use). ~1 day.
5. Packaging contract: MSRV 1.80 in Cargo.toml and README, track Cargo.lock, decide
   `gil_used`, decide opt-in vs default rollout and make setup.py/settings/PR text agree,
   push to fork master for a real wheel build. ~1 day.
6. Re-run the four-backend campaigns and memory measurements on the merge commit; PR.

### Next actions, in order

1. **PR #32**: reply (draft below) and ask for merge; PTNobel approved 2026-07-08 and
   already answered Transurgeon's "how is this different" on 07-11; nothing since. Rebase
   on main is optional (3 merges, asv conf/deps only; still MERGEABLE).
2. Adopt `codex/rust-backend-pr-20260915` (or redo the rebase), push it to the fork, sync
   fork master to upstream/master, and open a **fork-internal PR** so build.yml and
   test_backends.yml finally run on Linux/Windows/3.14t before anything goes upstream.
3. Settle the pitch given DIFFENGINE: RUST is the fastest full-coverage *tensor* backend
   (DPP/parametric, ND, broadcast, einsum), it is correct where SCIPY/COO were not, and it
   coexists with DIFFENGINE, which is parameter-free/opt-in today. State the structural
   losses (diag-of-dense-affine, kron) up front.
4. Refresh the benchmark claims on the rebased build (ASV + external sweep) right before
   opening: cone restructuring (#3525) shrank the rest-of-chain cost, which changes ratios.
5. Open the backend PR from the upstream template; Ray reviews `BACKEND_PR_DRAFT.md` first.

Draft reply for PR #32 (Ray to edit/post):

> Thanks both. To add to Parth's summary: the existing benchmarks time whole problems on
> the default backend, while this suite parameterizes backend x case at two levels (full
> compile and the isolated `build_matrix` call) with a LinOp-coverage checklist, so a
> regression in one backend shows up immediately. Writing the `build_matrix` cases also
> surfaced an ND-matmul bug in the Rust backend, since fixed. Happy to make further changes.

Last updated 2026-07-22. Ray (`mellowyellow71`) owns the fork and requires review
before any outward-facing PR or comment is posted.

## Objective

Land the optional Rust canonicalization backend in `cvxpy/cvxpy`. The agreed workflow
is to land canonicalization coverage in `cvxpy/benchmarks`, establish correctness and
four-backend performance evidence, and only then open the backend PR.

## Current state

- `cvxpy/benchmarks` PR #32 is open, clean, approved by PTNobel, and green. Its head is
  `006e7cd7f`; no benchmark-side push is currently needed.
- The Rust backend is rebased onto CVXPY upstream master `359e9bb52`.
- Parameterized ND matrix multiplication is implemented for batched/broadcast left and
  right parameter operands. It does not fall back to SciPy.
- All 129 callable `cvxpy.atoms` exports are inventoried: 105 direct exports, 23 aliases,
  and one helper. The timing matrix contains 124 cases.
- Four-backend campaigns and correctness gates are complete.
- A backend-only branch and PR body are ready locally. No backend PR has been opened.

## Worktrees and branches

| Purpose | Path | Local branch | Commit |
| --- | --- | --- | --- |
| Original checkout; preserve its user changes | `/Users/revantkasichainula/cvxpy` | `ray/latestfixes` | `ac5a03906` |
| Rebased implementation, tools, and reports | `rust_benchmarks/cvxpy-rebased` | `codex/rust-rebase-20260720` | `aebe49d38` before this handoff update |
| Clean backend-only PR payload | `rust_benchmarks/backend-pr` | `codex/rust-backend-pr-20260721` | `ecdad7eee` |
| Local merge used to run ASV | `rust_benchmarks/benchmarks-integration` | `codex/benchmarks-integration-20260720` | `7131aebbc` |
| Live benchmark PR checkout | `rust_benchmarks/cvxpy-benchmarks` | `backend-canonicalization-benchmarks` | `006e7cd7f` |

The local `codex/*` names are worktree implementation details. GitHub branches are on
the `mellowyellow71/cvxrust` fork and intentionally omit that prefix:

- `mellowyellow71/cvxrust:rust-rebase-20260720`
- `mellowyellow71/cvxrust:rust-backend-pr-20260721`

Do not push `codex/benchmarks-integration-20260720`; it is only a local test merge of
benchmark main plus PR #32.

## Important commits

- `1cf5de2ef`: parameterized ND matmul, exhaustive atom coverage, Rust warning cleanup.
- `aebe49d38`: four-backend artifacts, corrected reporting, safe external sweep.
- `ecdad7eee`: one-commit backend-only diff from upstream master, with no benchmark data.

## Validation completed

- `pytest cvxpy/tests/test_rust_backend.py`: 142 passed.
- Python backend, backend-selection, and kron tests: 199 passed.
- Broad `CVXPY_DEFAULT_CANON_BACKEND=RUST` smoke suite: 363 passed, 407 skipped.
- `cargo test --release --no-default-features`: 31 passed.
- `cargo clippy --release --no-default-features -- -D warnings`: passed.
- Ruff on touched Python and benchmark files: passed.
- Cross-backend stuffed-data verification: all 18 ASV cases matched.
- Exhaustive atom compile gate: 124 cases, no unexpected failures.
- ASV discovery check and complete ASV run: passed.
- Clean backend-only worktree: 142 Python tests, 31 Rust tests, Ruff, clippy, and
  `git diff --check` passed again before commit.

## Current benchmark evidence

Ratios are `OTHER/RUST`; values above 1 mean Rust is faster.

| Campaign | SCIPY/RUST | CPP/RUST | COO/RUST |
| --- | ---: | ---: | ---: |
| 124 atom cases | 1.50x (124/124) | 1.07x (110/112) | 1.28x (120/124) |
| 40 synthetic build-matrix cases | 4.87x (40/40) | 2.09x (40/40) | 3.54x (40/40) |
| 18 ASV full-compilation cases | 1.93x | 1.45x | 1.41x |
| 18 ASV build-matrix cases | 8.64x | 2.94x | 4.75x |
| 21 external benchmark classes | 1.07x (13/19) | 1.27x (14/19) | 1.20x (14/21) |

The external sweep is a one-sample process-isolated smoke run. Use ASV and the atom/
synthetic repeated timings for stable microbenchmark conclusions.

Murray is no longer a significant full-compilation loss: Rust 1107 ms, SciPy 1066 ms,
CPP 1135 ms, COO 1067 ms. Focused ASV also shows Rust at 32.29 ms for the dense-above-
threshold case versus roughly 87-91 ms for the other backends.

Current artifacts in `rust_benchmarks/`:

- `CVXPY_BENCHMARKS_RESULTS_REBASED.md`: canonical concise report.
- `asv_report_rebased.md`: complete backend/scaling ASV tables.
- `external_report_rebased.md`: complete external pairwise tables.
- `atoms_4backend_rebased.json`: raw 124-case atom results.
- `synthetic_4backend_rebased.json`: raw synthetic/scaling results.
- `external_4backend_rebased.jsonl`: raw external results.

`CVXPY_BENCHMARKS_RESULTS.md` and `benchmark_report.tex` are the July 3 historical
report. Do not quote them as current results.

## Remaining performance work

The broad suite still has four substantial structural losses:

| Workload | Rust | Best comparison | Suspected mechanism |
| --- | ---: | ---: | --- |
| `SDPSegfault1132Benchmark` | 27.03 s | SciPy 1.69 s | diag of dense affine expression; quadratic scan/sort |
| `UnconstrainedQP` | 3.34 s | CPP 1.11 s | eager dense kron index map |
| `QuantumHilbertMatrix` | 1.37 s | SciPy 0.80 s | kron/partial-transpose family |
| `TvInpainting` | 0.94 s | COO 0.69 s | not yet profiled |

`ParametrizedQPBenchmark` is also faster on COO (1.22 s) than Rust (1.71 s), consistent
with COO's parameter-tensor design.

These losses are not correctness failures. The next owner should decide with Ray whether
to optimize the first two before opening the backend PR or document them as follow-ups.

## Safe build and test workflow

The main environment is `/Users/revantkasichainula/cvxpy/.venv`. ASV is 0.6.6.

Two extension builds may exist: a repository-root `.so` and a maturin editable package
in the venv. Before timing, print `cvxpy_rust.__file__` from the campaign working
directory and confirm the extension is newer than `cvxpy_rust/src/*`. Refresh both after
Rust edits with the established `pip install -e .` and `maturin develop --release`
workflow.

Run correctness before timing:

```bash
python rust_benchmarks/verify_backends.py
python rust_benchmarks/gate_bench_cases.py
python -m pytest cvxpy/tests/test_rust_backend.py -q
```

Run Rust checks without the extension-module feature so libpython links correctly:

```bash
cd cvxpy_rust
cargo test --release --no-default-features
cargo clippy --release --no-default-features -- -D warnings
```

## Benchmark safety

- Never run concurrent timing campaigns.
- Always construct a fresh `Problem` for each backend; `get_problem_data` caches the
  solving chain on a problem instance.
- `ParametrizedQPBenchmark` with CPP previously exceeded 9 GB RSS and crashed the host.
- `SimpleFullyParametrizedLPBenchmark` also exhausted memory on SCIPY/CPP.
- The updated `sweep.sh` restricts both fully parameterized classes to RUST/COO and uses
  one subprocess per benchmark class plus a SIGALRM watchdog.
- The benchmark integration worktree has an untracked machine-local
  `asv_local.conf.json`; do not commit it.

## Next actions

1. Wait for PR #32 to merge or respond to new review feedback on its existing branch.
2. Have Ray review `rust_benchmarks/BACKEND_PR_DRAFT.md` before any backend PR is opened.
3. Decide whether the structural losses block the backend PR or become explicit follow-up
   issues. Profile before changing implementation.
4. Optionally run the full CVXPY suite with Rust as default and exercise Linux/Windows CI;
   the completed smoke and focused suites are already green.
5. Rebase the clean backend branch onto the then-current upstream master immediately
   before opening the PR, rerun focused tests, and refresh benchmark-sensitive claims if
   upstream changed materially.

## PR boundaries

The clean branch contains only the backend payload: `cvxpy_rust/`, Python backend glue,
selection/settings integration, tests, packaging, and CI. Keep all benchmark scripts,
raw data, reports, and handoff documents off that branch.

Ray's policy is draft first, review locally, then publish. Pushing named branches to the
`mellowyellow71` forks is allowed; do not open or comment on a PR without explicit review.
