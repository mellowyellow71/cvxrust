# Rebased four-backend ASV results

- Date: 2026-07-21
- CVXPY commit: `1cf5de2ef8671de76052e15615055d68d2cde98b`
- Benchmark integration commit: `7131aebbc`
- Command: `asv run --python=same --set-commit-hash 1cf5de2e --config asv_local.conf.json`
- Ratios are `OTHER/RUST`; values above 1 mean Rust is faster.


## canonicalization_backends.BackendBuildMatrixCanonicalization.time_build_matrix

| case | COO (ms) | CPP (ms) | RUST (ms) | SCIPY (ms) | COO/RUST | CPP/RUST | SCIPY/RUST |
| --- | --- | --- | --- | --- | --- | --- | --- |
| concatenate | 0.70 | n/a | 0.08 | 0.98 | 9.19× | — | 12.76× |
| cone_atoms_composite | 0.64 | 0.46 | 0.23 | 1.10 | 2.78× | 1.99× | 4.75× |
| convolve | 0.34 | 0.11 | 0.06 | 0.39 | 5.84× | 1.93× | 6.66× |
| core_affine_atoms | 0.32 | 0.09 | 0.04 | 0.40 | 7.33× | 2.13× | 9.22× |
| deep_neg_tree | 0.21 | 0.30 | 0.05 | 0.74 | 3.79× | 5.47× | 13.57× |
| diag_trace_kron | 0.96 | 0.14 | 0.05 | 1.62 | 20.01× | 2.94× | 33.84× |
| einsum | 0.59 | n/a | 0.05 | 0.61 | 11.66× | — | 12.03× |
| hstack_vstack | 0.74 | 0.18 | 0.09 | 1.07 | 8.66× | 2.13× | 12.63× |
| kron_diag_dense_affine | 1.06 | 0.24 | 0.09 | 0.90 | 11.38× | 2.53× | 9.70× |
| matmul_multiply_divide | 1.13 | 0.51 | 0.21 | 0.71 | 5.32× | 2.43× | 3.34× |
| murray_dense_above_threshold | 90.51 | 86.98 | 32.29 | 89.63 | 2.80× | 2.69× | 2.78× |
| murray_dense_constant | 20.42 | 10.96 | 18.72 | 20.53 | 1.09× | 0.59× | 1.10× |
| nd_array_ops | 0.33 | n/a | 0.04 | 0.83 | 8.09× | — | 20.56× |
| nd_matmul | 0.22 | n/a | 0.18 | 0.33 | 1.21× | — | 1.85× |
| parameterized_lp | 0.30 | 74.69 | 0.52 | 249.66 | 0.57× | 143.96× | 481.22× |
| rmul_promote | 0.29 | 0.13 | 0.07 | 0.47 | 4.36× | 2.04× | 7.14× |
| shared_subexpressions | 5.93 | 2.97 | 0.83 | 6.32 | 7.17× | 3.59× | 7.64× |
| wide_sum_tree | 17.74 | 3.89 | 3.14 | 8.79 | 5.66× | 1.24× | 2.80× |

**COO: geomean 4.75× | CPP: geomean 2.94× | SCIPY: geomean 8.64×**

## canonicalization_backends.BackendCompileCanonicalization.time_get_problem_data

| case | COO (ms) | CPP (ms) | RUST (ms) | SCIPY (ms) | COO/RUST | CPP/RUST | SCIPY/RUST |
| --- | --- | --- | --- | --- | --- | --- | --- |
| concatenate | 2.04 | n/a | 1.35 | 2.30 | 1.51× | — | 1.70× |
| cone_atoms_composite | 5.14 | 4.34 | 4.01 | 5.76 | 1.28× | 1.08× | 1.44× |
| convolve | 1.53 | 1.26 | 1.19 | 1.59 | 1.29× | 1.06× | 1.34× |
| core_affine_atoms | 1.50 | 1.26 | 1.20 | 1.62 | 1.25× | 1.05× | 1.34× |
| deep_neg_tree | 2.23 | 2.32 | 2.01 | 2.82 | 1.11× | 1.15× | 1.40× |
| diag_trace_kron | 2.40 | 1.51 | 1.43 | 3.04 | 1.69× | 1.06× | 2.13× |
| einsum | 1.92 | n/a | 1.29 | 1.97 | 1.49× | — | 1.53× |
| hstack_vstack | 2.09 | 1.46 | 1.37 | 2.37 | 1.53× | 1.06× | 1.73× |
| kron_diag_dense_affine | 2.83 | 1.97 | 1.77 | 2.69 | 1.59× | 1.11× | 1.52× |
| matmul_multiply_divide | 2.56 | 1.93 | 1.62 | 2.24 | 1.58× | 1.20× | 1.39× |
| murray_dense_above_threshold | 168.52 | 167.43 | 109.40 | 169.98 | 1.54× | 1.53× | 1.55× |
| murray_dense_constant | 29.11 | 19.72 | 27.32 | 29.29 | 1.07× | 0.72× | 1.07× |
| nd_array_ops | 1.63 | n/a | 1.33 | 2.20 | 1.23× | — | 1.66× |
| nd_matmul | 1.27 | n/a | 1.21 | 1.41 | 1.05× | — | 1.17× |
| parameterized_lp | 2.08 | 108.23 | 2.16 | 252.19 | 0.96× | 50.03× | 116.58× |
| rmul_promote | 1.48 | 1.31 | 1.25 | 1.74 | 1.18× | 1.05× | 1.39× |
| shared_subexpressions | 14.23 | 10.80 | 8.52 | 15.14 | 1.67× | 1.27× | 1.78× |
| wide_sum_tree | 21.63 | 7.03 | 6.09 | 12.51 | 3.55× | 1.15× | 2.05× |

**COO: geomean 1.41× | CPP: geomean 1.45× | SCIPY: geomean 1.93×**

## canonicalization_backends.DeepExpressionTreeScaling.time_get_problem_data

| case | COO (ms) | CPP (ms) | RUST (ms) | SCIPY (ms) | COO/RUST | CPP/RUST | SCIPY/RUST |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | 6.42 | 6.82 | 5.56 | 8.35 | 1.15× | 1.23× | 1.50× |
| 32 | 1.65 | 1.71 | 1.53 | 1.96 | 1.08× | 1.12× | 1.29× |
| 4 | 1.14 | 1.15 | 1.09 | 1.25 | 1.05× | 1.05× | 1.14× |

**COO: geomean 1.09× | CPP: geomean 1.13× | SCIPY: geomean 1.30×**

## canonicalization_backends.WideConstraintScaling.time_get_problem_data

| case | COO (ms) | CPP (ms) | RUST (ms) | SCIPY (ms) | COO/RUST | CPP/RUST | SCIPY/RUST |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | 76.48 | 51.40 | 38.07 | 85.00 | 2.01× | 1.35× | 2.23× |
| 64 | 19.12 | 12.05 | 9.11 | 21.81 | 2.10× | 1.32× | 2.39× |
| 8 | 2.74 | 1.83 | 1.64 | 3.11 | 1.66× | 1.11× | 1.89× |

**COO: geomean 1.91× | CPP: geomean 1.26× | SCIPY: geomean 2.16×**

## canonicalization_backends.WideExpressionTreeScaling.time_get_problem_data

| case | COO (ms) | CPP (ms) | RUST (ms) | SCIPY (ms) | COO/RUST | CPP/RUST | SCIPY/RUST |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | 72.31 | 18.08 | 18.77 | 39.59 | 3.85× | 0.96× | 2.11× |
| 64 | 15.41 | 4.90 | 4.93 | 10.57 | 3.13× | 0.99× | 2.14× |
| 8 | 2.75 | 1.59 | 1.49 | 2.38 | 1.85× | 1.07× | 1.60× |

**COO: geomean 2.81× | CPP: geomean 1.01× | SCIPY: geomean 1.93×**

