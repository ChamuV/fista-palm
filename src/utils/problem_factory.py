# src/utils/problem_factory.py

from src.problems.matrix_factorisation import MatrixFactorisationProblem


def make_matrix_factorisation_problem(A, r, sparsity, gamma=1.2):
    m, n = A.shape
    s_x = m * r
    s_y = int(sparsity * n * r)
    return MatrixFactorisationProblem(A, s_x, s_y, gamma)