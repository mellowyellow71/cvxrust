import gc
import numpy as np
import cvxpy as cp
import time

np.random.seed(42)

n, m = 1000, 5000
A = np.random.randn(m, n)
b = np.random.randn(m)

WARMUP = 1
ITERATIONS = 5

def make_problem():
    x = cp.Variable(n)
    return cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))

backends = ["RUST", "CPP", "SCIPY"]

for backend in backends:
    # Warmup (first run pays import/init costs)
    for _ in range(WARMUP):
        problem = make_problem()
        problem.get_problem_data(cp.CLARABEL, canon_backend=backend)

    # Timed runs with GC disabled
    times = []
    for _ in range(ITERATIONS):
        problem = make_problem()
        gc.collect()
        gc.disable()
        start = time.perf_counter()
        problem.get_problem_data(cp.CLARABEL, canon_backend=backend)
        end = time.perf_counter()
        gc.enable()
        times.append(end - start)

    median = sorted(times)[len(times) // 2]
    mean = sum(times) / len(times)
    print(f"{backend}: median={median:.4f}s  mean={mean:.4f}s  "
          f"range=[{min(times):.4f}, {max(times):.4f}]  ({ITERATIONS} runs)")