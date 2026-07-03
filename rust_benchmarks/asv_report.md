
## canonicalization_backends.BackendBuildMatrixCanonicalization.time_build_matrix

| case | COO (ms) | CPP (ms) | RUST (ms) | SCIPY (ms) | COO/RUST | CPP/RUST | SCIPY/RUST |
| --- | --- | --- | --- | --- | --- | --- | --- |
| concatenate | 0.04 | n/a | 0.02 | 0.07 | 1.67× | — | 2.79× |
| cone_atoms_composite | 0.65 | 0.46 | 0.23 | 1.13 | 2.79× | 1.94× | 4.79× |
| convolve | 0.04 | 0.04 | 0.02 | 0.07 | 1.68× | 1.44× | 2.77× |
| core_affine_atoms | 0.04 | 0.04 | 0.02 | 0.07 | 1.67× | 1.42× | 2.77× |
| deep_neg_tree | 0.04 | 0.04 | 0.02 | 0.07 | 1.66× | 1.41× | 2.75× |
| diag_trace_kron | 0.04 | 0.04 | 0.02 | 0.07 | 1.68× | 1.42× | 2.78× |
| einsum | 0.04 | n/a | 0.02 | 0.07 | 1.68× | — | 2.79× |
| hstack_vstack | 0.04 | 0.04 | 0.02 | 0.07 | 1.67× | 1.42× | 2.76× |
| kron_diag_dense_affine | 1.08 | 0.23 | 0.09 | 0.92 | 11.41× | 2.48× | 9.75× |
| matmul_multiply_divide | 0.04 | 0.04 | 0.02 | 0.07 | 1.70× | 1.42× | 2.77× |
| murray_dense_above_threshold | 91.77 | 90.37 | 33.61 | 91.41 | 2.73× | 2.69× | 2.72× |
| murray_dense_constant | 20.69 | 11.02 | 18.86 | 20.94 | 1.10× | 0.58× | 1.11× |
| nd_array_ops | 0.04 | n/a | 0.02 | 0.07 | 1.69× | — | 2.79× |
| nd_matmul | 0.04 | n/a | 0.02 | 0.07 | 1.67× | — | 2.76× |
| parameterized_lp | 0.30 | 75.07 | 0.51 | 248.96 | 0.59× | 148.08× | 491.10× |
| rmul_promote | 0.04 | 0.04 | 0.02 | 0.07 | 1.67× | 1.41× | 2.76× |
| wide_sum_tree | 0.04 | 0.04 | 0.02 | 0.07 | 1.68× | 1.43× | 2.78× |

**COO: geomean 1.83× | CPP: geomean 2.13× | SCIPY: geomean 3.96×**

## canonicalization_backends.BackendCompileCanonicalization.time_get_problem_data

| case | COO (ms) | CPP (ms) | RUST (ms) | SCIPY (ms) | COO/RUST | CPP/RUST | SCIPY/RUST |
| --- | --- | --- | --- | --- | --- | --- | --- |
| concatenate | 2.02 | n/a | 1.34 | 2.30 | 1.50× | — | 1.72× |
| cone_atoms_composite | 5.13 | 4.34 | 4.05 | 5.73 | 1.27× | 1.07× | 1.42× |
| convolve | 1.51 | 1.26 | 1.19 | 1.59 | 1.27× | 1.06× | 1.34× |
| core_affine_atoms | 1.54 | 1.30 | 1.22 | 1.65 | 1.26× | 1.06× | 1.35× |
| deep_neg_tree | 2.24 | 2.33 | 2.04 | 2.83 | 1.09× | 1.14× | 1.39× |
| diag_trace_kron | 2.41 | 1.54 | 1.42 | 3.14 | 1.69× | 1.08× | 2.20× |
| einsum | 1.92 | n/a | 1.33 | 1.97 | 1.45× | — | 1.48× |
| hstack_vstack | 2.08 | 1.49 | 1.37 | 2.38 | 1.51× | 1.08× | 1.73× |
| kron_diag_dense_affine | 2.85 | 1.97 | 1.80 | 2.72 | 1.58× | 1.09× | 1.51× |
| matmul_multiply_divide | 2.63 | 1.97 | 1.65 | 2.31 | 1.59× | 1.19× | 1.40× |
| murray_dense_above_threshold | 172.90 | 168.63 | 111.86 | 168.95 | 1.55× | 1.51× | 1.51× |
| murray_dense_constant | 29.57 | 20.21 | 27.39 | 29.81 | 1.08× | 0.74× | 1.09× |
| nd_array_ops | 1.65 | n/a | 1.34 | 2.21 | 1.23× | — | 1.64× |
| nd_matmul | 1.30 | n/a | 1.08 | 1.45 | 1.20× | — | 1.34× |
| parameterized_lp | 2.13 | 108.96 | 2.15 | 250.92 | 0.99× | 50.73× | 116.82× |
| rmul_promote | 1.52 | 1.34 | 1.26 | 1.75 | 1.21× | 1.07× | 1.39× |
| wide_sum_tree | 21.75 | 7.01 | 6.22 | 12.51 | 3.49× | 1.13× | 2.01× |

**COO: geomean 1.40× | CPP: geomean 1.46× | SCIPY: geomean 1.95×**

## canonicalization_backends.DeepExpressionTreeScaling.time_get_problem_data

| case | COO (ms) | CPP (ms) | RUST (ms) | SCIPY (ms) | COO/RUST | CPP/RUST | SCIPY/RUST |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | 6.45 | 6.86 | 5.55 | 8.34 | 1.16× | 1.24× | 1.50× |
| 32 | 1.64 | 1.70 | 1.52 | 1.95 | 1.08× | 1.11× | 1.28× |
| 4 | 1.16 | 1.15 | 1.08 | 1.27 | 1.07× | 1.06× | 1.18× |

**COO: geomean 1.10× | CPP: geomean 1.14× | SCIPY: geomean 1.31×**

## canonicalization_backends.WideExpressionTreeScaling.time_get_problem_data

| case | COO (ms) | CPP (ms) | RUST (ms) | SCIPY (ms) | COO/RUST | CPP/RUST | SCIPY/RUST |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | 70.61 | 18.00 | 18.34 | 40.17 | 3.85× | 0.98× | 2.19× |
| 64 | 15.56 | 4.99 | 4.91 | 10.99 | 3.17× | 1.02× | 2.24× |
| 8 | 2.75 | 1.63 | 1.50 | 2.40 | 1.84× | 1.09× | 1.60× |

**COO: geomean 2.82× | CPP: geomean 1.03× | SCIPY: geomean 1.99×**
