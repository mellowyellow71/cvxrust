
## RUST vs COO

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
| CVaRBenchmark | 4736 | 5781 | **1.22×** | — |
| ConvexPlasticity | 53 | 56 | **1.05×** | — |
| Murray | 1024 | 1070 | 1.04× | — |
| ParametrizedQPBenchmark | 1561 | 1209 | 0.77× | — |
| TvInpainting | 889 | 685 | 0.77× | — |
| QuantumHilbertMatrix | 1370 | 841 | 0.61× | — |
| UnconstrainedQP | 3606 | 2078 | 0.58× | — |
| SDPSegfault1132Benchmark | 29113 | 8392 | 0.29× | — |

**geomean 1.22× | 15/20 wins (ratio >1 ⇒ Rust faster)**

## RUST vs CPP

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
| Murray | 1024 | 991 | 0.97× | 2.53× |
| TvInpainting | 889 | 746 | 0.84× | 1.03× |
| QuantumHilbertMatrix | 1370 | 868 | 0.63× | 0.92× |
| UnconstrainedQP | 3606 | 973 | 0.27× | 0.85× |
| SDPSegfault1132Benchmark | 29113 | 6987 | 0.24× | 1.07× |
| ParametrizedQPBenchmark | — | — | error | |

**geomean 1.29× | 14/19 wins (ratio >1 ⇒ Rust faster)**

## RUST vs SCIPY

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
| Murray | 1024 | 1069 | 1.04× | 2.51× |
| TvInpainting | 889 | 769 | 0.87× | 1.05× |
| QuantumHilbertMatrix | 1370 | 800 | 0.58× | 0.99× |
| UnconstrainedQP | 3606 | 1003 | 0.28× | 1.06× |
| SDPSegfault1132Benchmark | 29113 | 1336 | 0.05× | 1.04× |
| ParametrizedQPBenchmark | — | — | timeout | |

**geomean 1.09× | 15/19 wins (ratio >1 ⇒ Rust faster)**
