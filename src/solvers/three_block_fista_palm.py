# src/solvers/three_block_fista_palm.py

import numpy as np


class ThreeBlockFISTAPALM:
    def __init__(self, max_iter=200):
        self.max_iter = max_iter

    def solve(self, problem, X0, B0, Y0):
        X, B, Y = X0.copy(), B0.copy(), Y0.copy()

        X_prev, B_prev, Y_prev = X.copy(), B.copy(), Y.copy()

        t_prev = 1.0

        history = [problem.objective(X, B, Y)]

        for _ in range(self.max_iter):
            # Nesterov step
            t_k = 0.5 * (1 + np.sqrt(1 + 4 * t_prev**2))
            beta = (t_prev - 1) / t_k
            t_prev = t_k

            X_ex = X + beta * (X - X_prev)
            B_ex = B + beta * (B - B_prev)
            Y_ex = Y + beta * (Y - Y_prev)

            # X update
            LX = problem.lipschitz_X(B_ex, Y_ex)
            X_new = problem.prox_X(
                X_ex - problem.grad_X(X_ex, B_ex, Y_ex) / LX
            )

            # B update
            LB = problem.lipschitz_B(X_new, B_ex, Y_ex)
            B_new = problem.prox_B(
                B_ex - problem.grad_B(X_new, B_ex, Y_ex) / LB
            )

            # Y update
            LY = problem.lipschitz_Y(X_new, B_new)
            Y_new = problem.prox_Y(
                Y_ex - problem.grad_Y(X_new, B_new, Y_ex) / LY
            )

            X_prev, B_prev, Y_prev = X.copy(), B.copy(), Y.copy()
            X, B, Y = X_new, B_new, Y_new

            history.append(problem.objective(X, B, Y))

        return X, B, Y, history