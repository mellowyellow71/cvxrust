# Rust backend audit — Astra, 2026-09-15

## Assessment

Rust is a credible CPP replacement with strong existing performance evidence, but the current candidate is not ready for unconditional replacement. Three reproducible Rust correctness gaps, unvalidated release packaging, and substantial performance regressions remain. The received diagnosis identifies useful optimization targets but overcounts comparable wins and overstates what the proposed kernels and sorting changes will fix.

This audit was run with GPT-6 Astra. It independently recounted the stored results, rebuilt the native extension from the current candidate, ran the full local Python suite and Rust unit tests, tested selected affine maps against their numerical values, rechecked DAG sharing, inspected the native samples and kernels, and repeated two safe external benchmarks in separate processes. No implementation changes, Git history changes, pushes, PRs, or benchmark-suite edits were made.

## Branch and upstream state

- Preserved original checkout: `ray/latestfixes`, `ac5a03906`, at `/Users/revantkasichainula/cvxpy`. It contains existing dirty/untracked user files.
- Current candidate: `codex/rust-backend-pr-20260915`, `86bfab9dc`, at `/private/tmp/claude-501/-Users-revantkasichainula-cvxpy-rust-benchmarks/d81b59c9-09d3-4510-85bb-a699b04ae17b/scratchpad/rebase-test`. Clean before and after this audit.
- Its base is `773163ebd`, also the live `cvxpy/cvxpy` master hash confirmed with the remote during this audit.
- The candidate contains three commits: backend payload, Rust formatting, and default-backend routing. It has not been pushed.
- The original July checkout is 51 upstream commits behind; the July 21 candidate base `359e9bb52` is 44 commits behind. These are different bases, not conflicting counts.
- A current rebase already exists. The next branch-management task is making the candidate worktree durable and backing up the branch, not repeating the rebase.

[The original POC PR #3018](https://github.com/cvxpy/cvxpy/pull/3018) was closed unmerged in April. Its discussion does not establish a current commitment to merge this implementation immediately. There is no open Rust backend PR. [Benchmark PR #32](https://github.com/cvxpy/benchmarks/pull/32) is still open, has Parth's July approval, and has passing checks; it can remain deferred as requested.

## Corrected benchmark accounting

These broad campaigns are July data, principally the July 21 rebased report, not a new full September campaign.

| Campaign | Timed CPP comparisons | Rust faster | Separate coverage advantage |
| --- | ---: | ---: | --- |
| Atom sweep, 124 cases | 112 | 110 | CPP unsupported in 12 |
| Synthetic build-matrix suite | 40 | 40 | None in these 40 comparisons |
| ASV full compilation, 18 cases | 14 | 13 | CPP unsupported in 4 |
| External classes, 21 total | 19 | 14 | Two large parameterized classes omitted for CPP after earlier memory failures |

Thus “122/124 atom wins” combines 110 speed wins with 12 unsupported CPP cells. “17/18 ASV compile wins” similarly combines 13 speed wins with four unsupported cells. Capability advantages are valuable, but should be reported separately from timing wins.

The atom sweep and rebased synthetic campaign used quick mode with two timed repetitions. The external sweep used one sample per cell. Small differences cannot be called statistically established wins or ties. The two CPP-faster atom measurements are `unary_neg` and `sum`. The fifth raw external CPP loss was ConvexPlasticity at 57.94 vs 57.02 ms, a tiny gap; the four material external losses were UnconstrainedQP, QuantumHilbertMatrix, SDPSegfault1132, and TvInpainting.

The Murray figures 18.72 ms vs 10.96 ms are the ASV **build-matrix** layer. Its full-compilation figures are 27.32 ms vs 19.72 ms. Keep measurement layers explicit.

Sources: [July rebased summary](/Users/revantkasichainula/cvxpy/rust_benchmarks/cvxpy-rebased/rust_benchmarks/CVXPY_BENCHMARKS_RESULTS_REBASED.md), [atom JSON](/Users/revantkasichainula/cvxpy/rust_benchmarks/cvxpy-rebased/rust_benchmarks/atoms_4backend_rebased.json), [ASV table](/Users/revantkasichainula/cvxpy/rust_benchmarks/cvxpy-rebased/rust_benchmarks/asv_report_rebased.md), [recount output](/private/tmp/cvxrust-astra-audit.XLmzq0/counts.jsonl).

### Fresh bounded timing check

Fresh release extension from `86bfab9dc`, Python 3.13.5, SciPy 1.17.1. Three separate-process first-compilation samples per backend, one BLAS thread, setup excluded, Rayon left at its default. These are confirmation measurements, not a replacement full campaign.

| Class | RUST median | CPP median | DIFFENGINE median |
| --- | ---: | ---: | ---: |
| QuantumHilbertMatrix | 1,285.7 ms | 911.6 ms | 640.4 ms |
| TvInpainting | 887.7 ms | 676.1 ms | 429.8 ms |

Both CPP losses persist. DIFFENGINE also matters on these parameter-free problems. [Samples and procedure](/private/tmp/cvxrust-astra-audit.XLmzq0/timings.jsonl), [runner](/private/tmp/cvxrust-astra-audit.XLmzq0/check_timings.py).

## Three confirmed correctness gaps

The full suite with `CVXPY_DEFAULT_CANON_BACKEND=RUST` reports:

**1 failed, 2,831 passed, 878 skipped, 145 subtests passed.**

The failing test is `test_atoms.py::TestAtoms::test_lambda_sum_largest_nd_solve`. This test was introduced in March, so this is a newly discovered gap rather than evidence that September's upstream changes introduced it. Full-suite counts also include tests with explicit backend choices; they do not imply every test exercised Rust.

The test has two sequential solves. The first exception prevents the second solve from being reached. Isolating both reveals three distinct defects:

1. **N-D sparse constants fail serialization.** `_emit_sparse` converts every sparse payload to CSC, which only accepts 2-D inputs. A batched spectral mask has shape `(2,2,2)`. The native sparse payload also has a two-dimensional shape. [Serializer](/private/tmp/claude-501/-Users-revantkasichainula-cvxpy-rust-benchmarks/d81b59c9-09d3-4510-85bb-a699b04ae17b/scratchpad/rebase-test/cvxpy/lin_ops/backends/rust_backend.py:224).
2. **N-D vstack uses the wrong row order.** `compute_vstack_indices` interleaves rows only for two-dimensional output. For 3-D and 4-D it falls into a branch labelled “1D output” and simply concatenates flattened buffers. Direct affine evaluation disagrees with NumPy and both Python backends. [Stack mapping](/private/tmp/claude-501/-Users-revantkasichainula-cvxpy-rust-benchmarks/d81b59c9-09d3-4510-85bb-a699b04ae17b/scratchpad/rebase-test/cvxpy_rust/src/operations/structural.rs:312).
3. **Batched trace is implemented as a scalar 2-D trace.** `process_trace` uses `arg_shape[0]` as matrix size and always writes to output row zero. For input `(3,2,2)`, a simple example should produce `[11,13,15]`, but Rust produces `[15,0,0]`. [Trace kernel](/private/tmp/claude-501/-Users-revantkasichainula-cvxpy-rust-benchmarks/d81b59c9-09d3-4510-85bb-a699b04ae17b/scratchpad/rebase-test/cvxpy_rust/src/operations/specialized.rs:149).

Temporary diagnostic normalization of LinOps—not a repository fix—isolated causality:

| Rust diagnostic variant | Batched lambda_max, expected 3 | Batched lambda_sum_largest(k=2), expected 6 |
| --- | --- | --- |
| Original code | 3-D CSC error | 3-D CSC error |
| Normalize N-D sparse constant only | Wrong optimum 7.0953 | Incorrectly unbounded |
| Also normalize N-D vstack | Correct 3.0 | Incorrectly unbounded |
| Also normalize batched trace | Correct 3.0 | Correct 6.0 |

This establishes concrete repair targets. Production fixes still need to preserve sparsity, parameters, shapes and performance; replacing all sparse constants by dense arrays is not an acceptable general fix.

Additional probes found that N-D `hstack` and `concatenate(axis=None)` also disagree with their NumPy numeric semantics in **SCIPY, COO, and RUST**. These are shared upstream/backend issues rather than Rust-only regressions. They show why differential tests need a direct numeric oracle as well as another backend.

Evidence: [probe script](/private/tmp/cvxrust-astra-audit.XLmzq0/check_correctness.py), [all probe results](/private/tmp/cvxrust-astra-audit.XLmzq0/correctness.jsonl), [full suite log](/private/tmp/cvxrust-astra-audit.XLmzq0/full-suite.log), [JUnit results](/private/tmp/cvxrust-astra-audit.XLmzq0/full-suite.xml).

## Performance diagnosis: supported and unsupported claims

### Multiplication and duplicate accumulation

Confirmed: the multiplication kernels append raw contributions to four COO vectors. Duplicate `(expression row, variable column, parameter slice)` entries survive intermediate operations. The final Rust sort only orders entries; **SciPy's CSC conversion performs the duplicate summation**. CPP's Eigen products accumulate coefficients earlier.

Accumulating within products is a strong optimization target, especially for repeated-variable QP patterns. A bounded reproduction of the UnconstrainedQP algebra emitted 1,728 contributions for only 36 final unique coefficients. Early accumulation reduces intermediate work as well as final sorting work.

However, not every large tensor consists mostly of duplicates. A bounded symmetric SDP reproduction of `V @ G @ V.T`, with `n=12`, produced 17,424 raw entries and 9,504 unique ones: only a 1.83x reduction from coalescing. The dense map itself contains many distinct coefficients. Extracting its diagonal kept only 1/12 of its output rows. At the benchmark's n=100 scale, this dense product is intrinsically large even after symmetric duplicates are combined.

Therefore accumulator kernels are well justified, but “fixes four of five losses in two days” is a hypothesis, not a validated consequence. SDP also motivates selective evaluation of diagonal/indexed products, reuse, efficient final accumulation, and careful memory layout.

The quoted 576M UnconstrainedQP pushes should not be repeated as a measured fact. This audit did not run the large risky case with per-operation counters. A raw-versus-unique counter should be part of the optimization work.

[Dense right multiply](/private/tmp/claude-501/-Users-revantkasichainula-cvxpy-rust-benchmarks/d81b59c9-09d3-4510-85bb-a699b04ae17b/scratchpad/rebase-test/cvxpy_rust/src/operations/arithmetic.rs:1104), [bounded counts](/private/tmp/cvxrust-astra-audit.XLmzq0/counts.jsonl), [count probe](/private/tmp/cvxrust-astra-audit.XLmzq0/check_counts.py).

### Sparse right multiplication has a separate traversal problem

For each left-tensor entry, `multiply_sparse_block_diagonal_right` scans every CSC column and every stored value to find one row of the constant. This is approximately `lhs.nnz × constant.nnz` search work, rather than visiting only the matching sparse row.

QuantumHilbert's actual `AxI` is 512x512 with **9,520** nonzeros, density **0.0363**. The report's 0.29 density belongs to A before taking the Kronecker product with an 8x8 identity. A row-indexed representation or an equivalent properly organized sparse product directly addresses this wasted search. Accumulation and efficient traversal both belong in the kernel redesign.

The existing native sample strongly confirms this kernel as QuantumHilbert's hotspot. DAG reuse also helps this case.

[Sparse right multiply](/private/tmp/claude-501/-Users-revantkasichainula-cvxpy-rust-benchmarks/d81b59c9-09d3-4510-85bb-a699b04ae17b/scratchpad/rebase-test/cvxpy_rust/src/operations/arithmetic.rs:1228), [current graph measurements](/private/tmp/cvxrust-astra-audit.XLmzq0/dag.jsonl).

### Global sorting and row selection

Confirmed: `BuildMatrixResult::from_tensor` globally sorts by **flattened output row**. It does not sort by the returned parameter column. SciPy subsequently groups by that column. Consequently “counting sort by column” is not a drop-in replacement for the existing invariant.

Counting by flattened row may allocate a histogram proportional to `constraint_rows × (variables+1)`, vastly larger than nnz. K-way merging requires already-sorted input runs and is generally O(nnz log k), not unconditionally O(nnz). Radix sorting, direct CSC construction, or merging proven-sorted runs are candidates to measure after kernel changes reduce the input size.

TvInpainting's native sample has 1,932 top-of-stack samples in `select_rows`, versus 465 in the main Rayon sort-recursion symbol, plus other sorting helpers. These multithreaded sample counts are not wall-time percentages. They do establish that row selection is a major hotspot and that “the final sort dominates TV” is too narrow.

The current diagonal-extraction kernel already uses an O(nnz) arithmetic filter. The handoff's older “quadratic diag scan” diagnosis is stale.

[Final assembly](/private/tmp/claude-501/-Users-revantkasichainula-cvxpy-rust-benchmarks/d81b59c9-09d3-4510-85bb-a699b04ae17b/scratchpad/rebase-test/cvxpy_rust/src/tensor.rs:367), [row selection](/private/tmp/claude-501/-Users-revantkasichainula-cvxpy-rust-benchmarks/d81b59c9-09d3-4510-85bb-a699b04ae17b/scratchpad/rebase-test/cvxpy_rust/src/tensor.rs:135), [diagonal extraction](/private/tmp/claude-501/-Users-revantkasichainula-cvxpy-rust-benchmarks/d81b59c9-09d3-4510-85bb-a699b04ae17b/scratchpad/rebase-test/cvxpy_rust/src/operations/specialized.rs:238).

### FFI

The available profiles support “FFI is not the cause of the large remaining losses.” They do not support “deserialization is below 0.3% everywhere.” Small build calls can have a higher share; the saved SDP objective call itself reports 11.6% of a tiny call. Avoid extending percentages from the large constraint builds to every call or treating deserialization alone as all FFI/serialization overhead.

## DAG assessment

Parth's suggestion is well founded. Fresh traversal of the current canonical graphs reproduced:

| Class, constraint build | Recursive visits | Unique LinOps | Inflation |
| --- | ---: | ---: | ---: |
| QuantumHilbertMatrix | 4,181 | 1,161 | 3.60x |
| UnconstrainedQP | 91 | 43 | 2.12x |
| TvInpainting | 60 | 42 | 1.43x |

A synthetic binary sum graph with 15 additions has 16 unique nodes and 65,535 visits; 16 additions has 17 unique nodes and 131,071 visits. The report's depth convention was ambiguous, but exponential expansion is real.

Rust's serializer has no identity memo and deserialization builds owned nested nodes. CPP memoizes Python-to-C++ node construction but recursively evaluates shared nodes again in `lin_to_tensor`. The inspected DIFFENGINE Python converter also recursively converts shared nonleaf expressions; do not assume native engine internals from that fact alone.

A node table plus references and shared lowered tensors is promising. Caching every intermediate tensor eagerly can increase memory use, so node lifetimes, repeated-use counts and mutation/copy behavior need design. Node inflation does not translate directly into the same runtime speedup.

[Serializer recursion](/private/tmp/claude-501/-Users-revantkasichainula-cvxpy-rust-benchmarks/d81b59c9-09d3-4510-85bb-a699b04ae17b/scratchpad/rebase-test/cvxpy/lin_ops/backends/rust_backend.py:285), [CPP evaluation](/private/tmp/claude-501/-Users-revantkasichainula-cvxpy-rust-benchmarks/d81b59c9-09d3-4510-85bb-a699b04ae17b/scratchpad/rebase-test/cvxpy/cvxcore/src/LinOpOperations.cpp:154), [measurement script](/private/tmp/cvxrust-astra-audit.XLmzq0/check_dag.py).

## Upstream compatibility and merge preparation

Meaningful changes already included by the current base:

- DIFFENGINE merged as an opt-in backend for parameter-free problems in [#3448](https://github.com/cvxpy/cvxpy/pull/3448).
- [#3449](https://github.com/cvxpy/cvxpy/pull/3449) remains open and proposes default routing plus cached symbolic non-DPP handling. Its routing has capability checks and fallbacks; “all non-DPP uses DIFFENGINE” is too absolute.
- `cp.block`; broader `geo_mean` and statistics axes; bmat scalar/vector promotion; trace fixes; cone restructuring as a permutation; cvxcore namespace changes; Python 3.14/3.14t CI.
- The new parameterized-left-matmul regression passes for Rust.
- Inventory now finds 125 cases covering 130 callable exports: 106 direct names, 23 aliases, one helper. No missing export names were found. Existing tests do not cover all shapes or compositions.
- `cp.conv` is deprecated in favor of `cp.convolve`. The internal LinOp remains `conv`, so its Rust kernel must stay.

Packaging and policy need an explicit decision. The draft says “optional”; the code automatically selects Rust when `import cvxpy_rust` succeeds, while `RustExtension(optional=True)` allows compilation failure without failing installation. These behaviors do not describe the same rollout.

Availability detection also needs a stronger check: in a source checkout with no native extension, `import cvxpy_rust` can succeed as a namespace package for the crate directory. In a clean no-site-packages probe it imported with `__file__ = None` and no `build_matrix_serialized` attribute. Import success alone does not prove the backend exists.

Cross-platform feature CI has not run on this candidate. Fork master has older successful workflows, but those do not validate this branch. **A PR alone will not exercise the wheel build step:** the current workflow gates that step on a push event. Arrange actual wheel builds and installed-wheel import/backend smoke tests on supported platforms.

Other preparation items: resolve Cargo.lock policy (currently ignored), establish a truthful MSRV, decide free-threaded Python behavior, and update stale docs. The README says Rust 1.70+, while the resolved dependencies require at least Rust 1.80 through Rayon; the audit used Rust 1.93.1. Plain `#[pymodule]` is not a free-threading declaration. Silent parametric×parametric branches should become explicit invariant checks rather than returning a knowingly incomplete result.

## Proposed sequence toward the user's replacement goal

1. Repair N-D sparse constants, vstack and batched trace, with explicit Rust end-to-end tests and direct numeric/matrix checks. Track the shared hstack/axis=None discrepancies separately.
2. Establish the supported behavior and rollout contract: extension availability, packaged wheels, backend routing, DPP handling, and interaction with DIFFENGINE. Keep the PR description consistent with that contract.
3. Back up the clean current-master branch and get genuine platform and wheel validation. Preserve the old dirty checkout.
4. Instrument emitted/unique entries and operation phases. Redesign dense/sparse multiplication to combine contributions early and avoid repeated sparse searches. Re-profile afterward.
5. Address row selection and output assembly based on the new profile; do not commit to counting sort as the solution in advance.
6. Add DAG sharing with memory-lifetime discipline if measurements justify its position relative to the other work.
7. Refresh repeated performance and memory measurements on the intended merge commit, then prepare the upstream backend PR. The benchmark-suite PR can stay separate.
8. Treat “backend merged” and “CPP retired” as distinct acceptance decisions. Opt-in staging is a reasonable review strategy, not a technical requirement or an already-agreed maintainer plan. If the first PR is intended to change the default, correctness, release packaging, and the severe regressions must be resolved first.

No supported evidence establishes a four-day or one-week completion estimate. There are specific, tractable tasks, but kernel changes can expose new tradeoffs and cross-platform builds have not yet been exercised.

## Validation provenance

- Source: clean `86bfab9dc` on upstream `773163ebd`.
- Fresh extension: built using `maturin build --release --locked` into an isolated temporary target and extracted wheel; the shared virtual environment was not reinstalled or altered.
- Imported native file: `/private/tmp/cvxrust-astra-audit.XLmzq0/extension/cvxpy_rust/cvxpy_rust.cpython-313-darwin.so`.
- Full Python suite: 2,831 passed / 878 skipped / 1 failed / 145 subtests passed.
- Rust unit tests: 31 passed.
- Rust formatting and Git whitespace checks: clean.
- Release clippy with warnings denied: passed; see [clippy log](/private/tmp/cvxrust-astra-audit.XLmzq0/clippy.log).
- All diagnostic normalization occurred in temporary Python processes, not backend source.
- Large memory-dangerous CPP parameterized cases were not rerun.
- Diagnostic scripts, samples and build products live under `/private/tmp/cvxrust-astra-audit.XLmzq0`; this report preserves the essential findings even if that temporary directory is later removed.
