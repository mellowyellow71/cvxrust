#!/usr/bin/env zsh
# Sweep all cvxpy/benchmarks classes across RUST, SCIPY, CPP backends.
# Each class runs in its own subprocess (run_one.py) for crash isolation.
# Append one JSON line per (class, backend) to results.jsonl.
#
# Usage: zsh sweep.sh [BENCH_REPO] [BACKENDS] [WARMUPS] [REPS] [TIMEOUT_S]
set -u
HERE=${0:A:h}
PY="${PY:-$HERE/../.venv/bin/python}"
B="${1:-/tmp/cvxpy-benchmarks}"
BACKENDS="${2:-RUST,SCIPY,CPP}"
WARMUPS="${3:-1}"
REPS="${4:-2}"
TIMEOUT_S="${5:-150}"
OUT="$HERE/results.jsonl"
: > "$OUT"

# module-file : ClassName  (one entry per timeable benchmark class)
benches=(
  "simple_QP_benchmarks.py:LeastSquares"
  "simple_QP_benchmarks.py:SimpleQPBenchmark"
  "simple_QP_benchmarks.py:UnconstrainedQP"
  "simple_QP_benchmarks.py:ParametrizedQPBenchmark"
  "simple_LP_benchmarks.py:SimpleLPBenchmark"
  "simple_LP_benchmarks.py:SimpleScalarParametrizedLPBenchmark"
  "simple_LP_benchmarks.py:SimpleFullyParametrizedLPBenchmark"
  "finance.py:CVaRBenchmark"
  "finance.py:FactorCovarianceModel"
  "gini_portfolio.py:Yitzhaki"
  "gini_portfolio.py:Murray"
  "gini_portfolio.py:Cajas"
  "high_dim_convex_plasticity.py:ConvexPlasticity"
  "huber_regression.py:HuberRegression"
  "quantum_hilbert_matrix.py:QuantumHilbertMatrix"
  "sdp_segfault_1132_benchmark.py:SDPSegfault1132Benchmark"
  "optimal_advertising.py:OptimalAdvertising"
  "semidefinite_programming.py:SemidefiniteProgramming"
  "slow_pruning_1668_benchmark.py:SlowPruningBenchmark"
  "svm_l1_regularization.py:SVMWithL1Regularization"
  "tv_inpainting.py:TvInpainting"
)

for entry in $benches; do
  mod="${entry%%:*}"
  cls="${entry##*:}"
  echo ">>> $cls ($mod)" >&2
  "$PY" "$HERE/run_one.py" "$B/benchmark/$mod" "$cls" "$BACKENDS" \
       "$WARMUPS" "$REPS" "$TIMEOUT_S" 2>>"$HERE/sweep.stderr.log" | tee -a "$OUT"
done

echo "=== done -> $OUT ===" >&2
