import numpy as np
import cvxpy as cp
import time

def make_lasso(n: int, m: int) -> cp.Problem:
    A = cp.Constant(np.random.randn(m, n), name="A")
    b = cp.Constant(np.random.randn(m), name="b")
    x = cp.Variable(n, name="x")
    return cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))

def main():
    problem = make_lasso(2, 4)
    #time the operation
    start = time.perf_counter()
    problem.get_problem_data(cp.CLARABEL, canon_backend="RUST")
    end = time.perf_counter()

    print(f"Time taken: {end - start} seconds")

if __name__ == "__main__":
    print("Starting main")
    main()
    print("Done")
