# src/problems/l1_matrix_factorisation.py

import numpy as np


class L1MatrixFactorisationProblem:
    """
    Nonnegative L1-regularised matrix factorisation:

        min_{X,Y} 0.5 ||A - X Y^T||_F^2
                 + lam (||X||_1 + ||Y||_1)

    with:

        X >= 0, Y >= 0
    """

    def __init__(self, A, lam=0.3, gamma=1.2):
        self.A = A
        self.lam = lam
        self.gamma = gamma

    def objective(self, X, Y):
        recon_loss = 0.5 * np.linalg.norm(self.A - X @ Y.T, "fro") ** 2
        l1_penalty = self.lam * (np.sum(np.abs(X)) + np.sum(np.abs(Y)))
        return recon_loss + l1_penalty

    def grad_X(self, X, Y):
        return (X @ Y.T - self.A) @ Y

    def grad_Y(self, X, Y):
        return (X @ Y.T - self.A).T @ X

    @staticmethod
    def prox_l1_nonnegative(Z, threshold):
        return np.maximum(Z - threshold, 0)

    def prox_X(self, X, step_size=None):
        if step_size is None:
            raise ValueError("step_size must be supplied for L1 proximal step.")
        return self.prox_l1_nonnegative(X, self.lam * step_size)

    def prox_Y(self, Y, step_size=None):
        if step_size is None:
            raise ValueError("step_size must be supplied for L1 proximal step.")
        return self.prox_l1_nonnegative(Y, self.lam * step_size)

    def lipschitz_X(self, Y):
        return self.gamma * np.linalg.norm(Y.T @ Y, 2)

    def lipschitz_Y(self, X):
        return self.gamma * np.linalg.norm(X.T @ X, 2)