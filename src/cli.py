# src/cli.py

import argparse

import matplotlib.pyplot as plt

from src.problems.matrix_factorisation import MatrixFactorisationProblem
from src.solvers.fista_palm import FISTAPALM
from src.utils.data import generate_low_rank_matrix_problem


def main():
    parser = argparse.ArgumentParser(
        description="Run FISTA-PALM on a synthetic matrix factorisation problem."
    )

    parser.add_argument("--m", type=int, default=100, help="Number of rows")
    parser.add_argument("--n", type=int, default=80, help="Number of columns")
    parser.add_argument("--rank", type=int, default=10, help="Factor rank")
    parser.add_argument("--sparsity", type=float, default=0.25, help="Sparsity level in Y")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--plot", action="store_true", help="Show convergence plot")

    args = parser.parse_args()

    A, X0, Y0, s_x, s_y = generate_low_rank_matrix_problem(
        args.m,
        args.n,
        args.rank,
        sparsity=args.sparsity,
        seed=args.seed,
    )

    problem = MatrixFactorisationProblem(
        A,
        s_x=s_x,
        s_y=s_y,
        gamma=1.2,
    )

    solver = FISTAPALM(max_iter=args.max_iter)
    _, _, history = solver.solve(problem, X0, Y0)

    print("\n FISTA-PALM Run Complete")
    print(f"Matrix size: {args.m} x {args.n}")
    print(f"Rank: {args.rank}")
    print(f"Sparsity in Y: {args.sparsity}")
    print(f"Iterations: {len(history) - 1}")
    print(f"Initial objective: {history[0]:.6f}")
    print(f"Final objective: {history[-1]:.6f}")

    print("\nFirst 5 objective values:")
    print([round(v, 4) for v in history[:5]])

    print("\nLast 5 objective values:")
    print([round(v, 4) for v in history[-5:]])

    if args.plot:
        plt.figure(figsize=(8, 5))
        plt.plot(history, color="red", linewidth=2, label="FISTA-PALM")
        plt.yscale("log")
        plt.xlabel("Iteration")
        plt.ylabel("Objective")
        plt.title("FISTA-PALM Convergence")
        plt.grid(True, linestyle=":")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()