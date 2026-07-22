# CVXRust project handoff

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
the `mellowyellow71` fork and intentionally omit that prefix:

- `mellowyellow71/rust-rebase-20260720`
- `mellowyellow71/rust-backend-pr-20260721`

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
