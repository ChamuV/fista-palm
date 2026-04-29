# src/solvers/three_block_palm.py

class ThreeBlockPALM:
    def __init__(self, max_iter=200):
        self.max_iter = max_iter

    def solve(self, problem, X0, B0, Y0):
        X, B, Y = X0.copy(), B0.copy(), Y0.copy()

        history = [problem.objective(X, B, Y)]

        for _ in range(self.max_iter):
            # X update 
            LX = problem.lipschitz_X(B, Y)
            X = problem.prox_X(
                X - problem.grad_X(X, B, Y) / LX
            )

            # B update 
            LB = problem.lipschitz_B(X, B, Y)
            B = problem.prox_B(
                B - problem.grad_B(X, B, Y) / LB
            )

            # Y update 
            LY = problem.lipschitz_Y(X, B)
            Y = problem.prox_Y(
                Y - problem.grad_Y(X, B, Y) / LY
            )

            history.append(problem.objective(X, B, Y))

        return X, B, Y, history