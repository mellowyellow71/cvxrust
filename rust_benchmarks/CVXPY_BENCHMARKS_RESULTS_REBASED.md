# Rebased CVXRust benchmark results

This report covers the Rust canonicalization backend rebased onto CVXPY master and
the benchmark additions from the approved, not-yet-merged `cvxpy/benchmarks` PR.

- Date: 2026-07-21
- CVXPY commit: `1cf5de2ef8671de76052e15615055d68d2cde98b`
- CVXPY base: upstream master `359e9bb52`
- Benchmark integration commit: `7131aebbc`
- Python: 3.13.5; ASV: 0.6.6; release Rust extension
- Backends: `RUST`, `SCIPY`, `CPP`, and `COO`
- Ratios below are `OTHER/RUST`; values above 1 mean Rust is faster.

## Summary

| Campaign | Cases | SCIPY/RUST | CPP/RUST | COO/RUST |
| --- | ---: | ---: | ---: | ---: |
| Exhaustive atom sweep | 124 | 1.50x (124/124) | 1.07x (110/112) | 1.28x (120/124) |
| Synthetic build-matrix suite | 40 | 4.87x (40/40) | 2.09x (40/40) | 3.54x (40/40) |
| ASV full compilation | 18 | 1.93x | 1.45x | 1.41x |
| ASV build-matrix only | 18 | 8.64x | 2.94x | 4.75x |
| External benchmark classes | 21 | 1.07x (13/19) | 1.27x (14/19) | 1.20x (14/21) |

The atom and synthetic campaigns used quick mode with one warmup and two timed
repetitions. ASV used its calibrated timing configuration. The external sweep was
a process-isolated smoke run with one timed measurement per backend, so small gaps
there should be treated as noise rather than release-quality estimates.

## Coverage

The atom inventory accounts for all 129 callable `cvxpy.atoms` exports: 105 direct
exports, 23 aliases, and one helper. The 124 benchmark cases cover DCP, DGP, DQCP,
complex and quantum atoms, ND manipulation, broadcasting, `einsum`, `convolve`, and
both numeric and parameterized ND matrix multiplication. CPP has 12 expected `n/a`
cells for unsupported ND and broadcast operations.

Expression-shape coverage includes:

- Deep negation trees at depths 4, 32, and 256.
- Wide expression trees and wide constraint sets at widths 8, 64, and 256.
- Shared subexpressions and wide sums.
- ND arrays, broadcasting, einsum, convolve, kron, diag, and trace composites.

At depth 256, full compilation was 5.56 ms for Rust, versus 8.35 ms for SciPy,
6.82 ms for CPP, and 6.42 ms for COO. At 256 wide constraints, Rust was 38.07 ms,
versus 85.00 ms, 51.40 ms, and 76.48 ms respectively.

## Murray

The prior Murray regression is no longer significant in full compilation: Rust
measured 1107 ms, versus 1066 ms for SciPy, 1135 ms for CPP, and 1067 ms for COO.
Those differences are within 4% in the single-sample external sweep.

The focused ASV matrix build shows the density split explicitly. Rust takes 18.72 ms
for the below-threshold Murray case and 32.29 ms for the dense-above-threshold case.
The latter is 2.69x faster than CPP and about 2.8x faster than SciPy and COO.

## Remaining losses

The broad external suite still identifies four structural workloads for follow-up:

| Workload | Rust | SciPy | CPP | COO |
| --- | ---: | ---: | ---: | ---: |
| `SDPSegfault1132Benchmark` | 27.03 s | 1.69 s | 7.22 s | 8.21 s |
| `UnconstrainedQP` | 3.34 s | 1.15 s | 1.11 s | 2.51 s |
| `QuantumHilbertMatrix` | 1.37 s | 0.80 s | 0.86 s | 0.84 s |
| `TvInpainting` | 0.94 s | 0.77 s | 0.75 s | 0.69 s |

`ParametrizedQPBenchmark` remains faster on COO (1.22 s) than Rust (1.71 s).
SCIPY and CPP were intentionally not run for that case or the fully parameterized
LP because prior runs exhausted host memory before a watchdog could intervene.

## Correctness and quality gates

- Rust backend tests: 142 passed.
- Python backend, backend-selection, and kron tests: 199 passed.
- Broad Rust-default canonicalization smoke suite: 363 passed, 407 skipped.
- Rust unit tests: 31 passed.
- Rust clippy with warnings denied: passed.
- Ruff on touched Python and benchmark files: passed.
- Cross-backend stuffed-data verification: all 18 ASV cases matched.
- Exhaustive 124-case compile gate: no unexpected backend failures.
- ASV benchmark discovery check: passed.

Parameterized ND matmul correctness is tested for left and right parameter operands,
including batched and broadcast forms, across two parameter updates. Stuffed `A`,
`b`, and `c` data match the established Python backends.

## Artifacts

- `atoms_4backend_rebased.json`: raw exhaustive atom timings.
- `synthetic_4backend_rebased.json`: raw synthetic and scaling timings.
- `external_4backend_rebased.jsonl`: raw external class timings.
- `external_report_rebased.md`: full pairwise external tables.
- `asv_report_rebased.md`: full ASV backend and scaling tables.

