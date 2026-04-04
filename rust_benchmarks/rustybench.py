import numpy as np
import cvxpy as cp
import time

n, m = 1000, 5000
A = np.random.randn(m, n)
b = np.random.randn(m)

def make_problem():
    x = cp.Variable(n)
    return cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))

backends = ["RUST", "CPP", "SCIPY"]

for backend in backends:
    problem = make_problem()
    start = time.perf_counter()
    problem.get_problem_data(cp.CLARABEL, canon_backend=backend)
    end = time.perf_counter()
    print(f"{backend}: {end - start:.4f}s")