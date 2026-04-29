# src/solvers/fista_palm.py

import numpy as np


class FISTAPALM:
    def __init__(self, max_iter=100, tol=1e-4):
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, problem, X0, Y0):
        X, Y = X0.copy(), Y0.copy()
        X_extrap, Y_extrap = X.copy(), Y.copy()

        t_prev = 1.0
        history = [problem.objective(X, Y)]

        for _ in range(self.max_iter):
            Lx = problem.lipschitz_X(Y_extrap)
            X_new = problem.prox_X(
                X_extrap - problem.grad_X(X_extrap, Y) / Lx,
                step_size=1 / Lx,
            )

            Ly = problem.lipschitz_Y(X_new)
            Y_new = problem.prox_Y(
                Y_extrap - problem.grad_Y(X_new, Y_extrap) / Ly,
                step_size=1 / Ly,
            )

            loss = problem.objective(X_new, Y_new)
            history.append(loss)

            if abs(history[-2] - loss) < self.tol:
                break

            t_new = 0.5 * (1 + np.sqrt(1 + 4 * t_prev**2))
            beta = (t_prev - 1) / t_new
            t_prev = t_new

            X_extrap = X_new + beta * (X_new - X)
            Y_extrap = Y_new + beta * (Y_new - Y)

            X, Y = X_new, Y_new

        return X, Y, history