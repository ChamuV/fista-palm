# src/solvers/ipalm.py

import numpy as np


class IPALM:
    def __init__(
        self,
        max_iter=100,
        tol=1e-4,
        alpha1=0.2,
        beta1=0.3,
        alpha2=0.1,
        beta2=0.1,
        epsilon=1e-3,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.alpha1 = alpha1
        self.beta1 = beta1
        self.alpha2 = alpha2
        self.beta2 = beta2
        self.epsilon = epsilon

    def solve(self, problem, X0, Y0):
        X, Y = X0.copy(), Y0.copy()
        X_prev, Y_prev = X.copy(), Y.copy()

        history = [problem.objective(X, Y)]

        for _ in range(self.max_iter):
            X_in = X + self.alpha1 * (X - X_prev)
            Y_in = Y + self.beta1 * (Y - Y_prev)

            Lx = problem.lipschitz_X(Y_in)
            delta_x = (
                (self.alpha1 + self.beta1)
                / (1 - 2 * self.alpha1)
                * Lx
            )
            tau_x = (
                ((1 + self.epsilon) * delta_x + (1 + self.beta1) * Lx)
                / (1 - self.alpha1)
            )

            X_new = problem.prox_X(
                X_in - problem.grad_X(X_in, Y_in) / tau_x,
                step_size=1 / tau_x,
            )

            Ly = problem.lipschitz_Y(X_new)
            delta_y = (
                (self.alpha2**2 + 2 * self.beta2 * self.alpha2)
                / (2 * (1 - 2 * self.alpha2))
                * Ly
            )
            tau_y = (
                ((1 + self.epsilon) * delta_y + (1 + self.beta2) * Ly)
                / (2 - self.alpha2)
            )

            Y_new = problem.prox_Y(
                Y_in - problem.grad_Y(X_new, Y_in) / tau_y,
                step_size=1 / tau_y,
            )

            loss = problem.objective(X_new, Y_new)
            history.append(loss)

            if abs(history[-2] - loss) < self.tol:
                break

            X_prev, X = X, X_new
            Y_prev, Y = Y, Y_new

        return X, Y, history