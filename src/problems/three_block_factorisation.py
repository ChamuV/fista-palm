# src/problems/three_block_factorisation.py

import numpy as np


class ThreeBlockFactorisationProblem:
    """
    Sparse three-block matrix factorisation problem:

        min_{X,B,Y} 0.5 ||A - X B Y^T||_F^2

    with:

        X >= 0, Y >= 0
        ||X||_0 <= s_x
        ||B||_0 <= s_b
        ||Y||_0 <= s_y
    """

    def __init__(self, A, s_x, s_b, s_y, gamma=1.5):
        self.A = A
        self.s_x = s_x
        self.s_b = s_b
        self.s_y = s_y
        self.gamma = gamma

    def objective(self, X, B, Y):
        return 0.5 * np.linalg.norm(self.A - X @ B @ Y.T, "fro") ** 2

    def grad_X(self, X, B, Y):
        return (X @ B @ Y.T - self.A) @ Y @ B.T

    def grad_B(self, X, B, Y):
        return X.T @ (X @ B @ Y.T - self.A) @ Y

    def grad_Y(self, X, B, Y):
        return (X @ B @ Y.T - self.A).T @ (X @ B)

    @staticmethod
    def prox_l0_nonnegative(Z, s):
        Z = np.maximum(Z, 0)
        flat = Z.flatten()

        if s >= flat.size:
            return Z

        threshold = np.partition(flat, -s)[-s]
        return np.where(Z >= threshold, Z, 0)

    @staticmethod
    def prox_l0(Z, s):
        flat = Z.flatten()

        if s >= flat.size:
            return Z

        threshold = np.partition(np.abs(flat), -s)[-s]
        return np.where(np.abs(Z) >= threshold, Z, 0)

    def prox_X(self, X):
        return self.prox_l0_nonnegative(X, self.s_x)

    def prox_B(self, B):
        return self.prox_l0(B, self.s_b)

    def prox_Y(self, Y):
        return self.prox_l0_nonnegative(Y, self.s_y)

    def lipschitz_X(self, B, Y):
        return self.gamma * np.linalg.norm(B @ Y.T @ Y @ B.T, 2)

    def lipschitz_B(self, X, B, Y):
        return self.gamma * np.linalg.norm(X.T @ X @ B @ Y.T @ Y, 2)
    
    def lipschitz_Y(self, X, B):
        return self.gamma * np.linalg.norm(B.T @ X.T @ X @ B, 2)