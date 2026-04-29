# src/problems/matrix_factorisation.py

import numpy as np

 
class MatrixFactorisationProblem:
    """
    Sparse nonnegative matrix factorisation problem:

        min_{X,Y} 0.5 ||A - X Y^T||_F^2

    with:
    
        X >= 0, Y >= 0
        ||X||_0 <= s_x, ||Y||_0 <= s_y
    """
    def __init__(self, A, s_x, s_y, gamma=1.2):
        self.A = A
        self.s_x = s_x
        self.s_y = s_y
        self.gamma = gamma

    def objective(self, X, Y):
        return 0.5 * np.linalg.norm(self.A - X @ Y.T, 'fro') ** 2
    
    def grad_X(self, X, Y):
        return (X @ Y.T - self.A) @ Y

    def grad_Y(self, X, Y):
        return (X @ Y.T - self.A).T @ X
    
    @staticmethod
    def prox_l0_nonnegative(Z, s):
        Z = np.maximum(Z, 0)
        flat = Z.flatten()
        if s >= flat.size:
            return Z
        
        threshold = np.partition(flat, -s)[-s]
        return np.where(Z >= threshold, Z, 0)

    def prox_X(self, X, step_size=None):
        return self.prox_l0_nonnegative(X, self.s_x)

    def prox_Y(self, Y, step_size=None):
        return self.prox_l0_nonnegative(Y, self.s_y)

    def lipschitz_X(self, Y):
        return self.gamma * np.linalg.norm(Y.T @ Y, 2)

    def lipschitz_Y(self, X):
        return self.gamma * np.linalg.norm(X.T @ X, 2)