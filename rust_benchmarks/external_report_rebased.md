# Rebased external benchmark sweep

- Date: 2026-07-21
- CVXPY commit: `1cf5de2ef8671de76052e15615055d68d2cde98b`
- Benchmark integration commit: `7131aebbc`
- Method: fresh problem per backend, one timed process-isolated measurement, 150-second watchdog.
- `ParametrizedQPBenchmark` and `SimpleFullyParametrizedLPBenchmark` compare only RUST/COO because SCIPY/CPP are known to exhaust host memory.
- Ratios are `OTHER/RUST`; values above 1 mean Rust is faster.


## RUST vs COO

| Benchmark | Rust (ms) | COO (ms) | COO/Rust |
| --- | --- | --- | --- |
| SemidefiniteProgramming | 396 | 950 | **2.40×** |
| LeastSquares | 797 | 1504 | **1.89×** |
| OptimalAdvertising | 274 | 487 | **1.78×** |
| SimpleQPBenchmark | 900 | 1593 | **1.77×** |
| SimpleLPBenchmark | 3076 | 4957 | **1.61×** |
| Yitzhaki | 539 | 857 | **1.59×** |
| FactorCovarianceModel | 620 | 978 | **1.58×** |
| Cajas | 408 | 637 | **1.56×** |
| HuberRegression | 1722 | 2542 | **1.48×** |
| SVMWithL1Regularization | 1679 | 2432 | **1.45×** |
| SlowPruningBenchmark | 1362 | 1901 | **1.40×** |
| SimpleFullyParametrizedLPBenchmark | 349 | 475 | **1.36×** |
| SimpleScalarParametrizedLPBenchmark | 727 | 981 | **1.35×** |
| CVaRBenchmark | 4173 | 5552 | **1.33×** |
| ConvexPlasticity | 58 | 56 | 0.97× |
| Murray | 1107 | 1067 | 0.96× |
| UnconstrainedQP | 3335 | 2511 | 0.75× |
| TvInpainting | 941 | 690 | 0.73× |
| ParametrizedQPBenchmark | 1707 | 1216 | 0.71× |
| QuantumHilbertMatrix | 1371 | 837 | 0.61× |
| SDPSegfault1132Benchmark | 27034 | 8206 | 0.30× |

**geomean 1.20× | 14/21 wins (ratio >1 ⇒ Rust faster)**

## RUST vs CPP

| Benchmark | Rust (ms) | CPP (ms) | CPP/Rust |
| --- | --- | --- | --- |
| OptimalAdvertising | 274 | 1915 | **6.99×** |
| Cajas | 408 | 1761 | **4.32×** |
| SemidefiniteProgramming | 396 | 767 | **1.94×** |
| SimpleLPBenchmark | 3076 | 5601 | **1.82×** |
| SimpleQPBenchmark | 900 | 1464 | **1.63×** |
| FactorCovarianceModel | 620 | 968 | **1.56×** |
| CVaRBenchmark | 4173 | 6214 | **1.49×** |
| SimpleScalarParametrizedLPBenchmark | 727 | 1081 | **1.49×** |
| Yitzhaki | 539 | 749 | **1.39×** |
| SVMWithL1Regularization | 1679 | 2324 | **1.38×** |
| HuberRegression | 1722 | 2320 | **1.35×** |
| LeastSquares | 797 | 963 | **1.21×** |
| SlowPruningBenchmark | 1362 | 1487 | **1.09×** |
| Murray | 1107 | 1135 | 1.03× |
| ConvexPlasticity | 58 | 57 | 0.98× |
| TvInpainting | 941 | 750 | 0.80× |
| QuantumHilbertMatrix | 1371 | 863 | 0.63× |
| UnconstrainedQP | 3335 | 1105 | 0.33× |
| SDPSegfault1132Benchmark | 27034 | 7224 | 0.27× |

**geomean 1.27× | 14/19 wins (ratio >1 ⇒ Rust faster)**

## RUST vs SCIPY

| Benchmark | Rust (ms) | SCIPY (ms) | SCIPY/Rust |
| --- | --- | --- | --- |
| OptimalAdvertising | 274 | 557 | **2.03×** |
| LeastSquares | 797 | 1460 | **1.83×** |
| SimpleQPBenchmark | 900 | 1625 | **1.80×** |
| SimpleLPBenchmark | 3076 | 5140 | **1.67×** |
| FactorCovarianceModel | 620 | 1004 | **1.62×** |
| SemidefiniteProgramming | 396 | 641 | **1.62×** |
| Cajas | 408 | 658 | **1.61×** |
| Yitzhaki | 539 | 856 | **1.59×** |
| HuberRegression | 1722 | 2457 | **1.43×** |
| SVMWithL1Regularization | 1679 | 2327 | **1.39×** |
| SimpleScalarParametrizedLPBenchmark | 727 | 1001 | **1.38×** |
| SlowPruningBenchmark | 1362 | 1864 | **1.37×** |
| CVaRBenchmark | 4173 | 5641 | **1.35×** |
| ConvexPlasticity | 58 | 57 | 0.98× |
| Murray | 1107 | 1066 | 0.96× |
| TvInpainting | 941 | 772 | 0.82× |
| QuantumHilbertMatrix | 1371 | 802 | 0.59× |
| UnconstrainedQP | 3335 | 1155 | 0.35× |
| SDPSegfault1132Benchmark | 27034 | 1688 | 0.06× |

**geomean 1.07× | 13/19 wins (ratio >1 ⇒ Rust faster)**

