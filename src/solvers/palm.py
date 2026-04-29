# src/solvers/palm.py

class PALM:
    def __init__(self, max_iter=100, tol=1e-4):
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, problem, X0, Y0):
        X, Y = X0.copy(), Y0.copy()
        history = [problem.objective(X, Y)]

        for _ in range(self.max_iter):
            Lx = problem.lipschitz_X(Y)
            X = problem.prox_X(
                X - problem.grad_X(X, Y) / Lx,
                step_size=1 / Lx,
            )

            Ly = problem.lipschitz_Y(X)
            Y = problem.prox_Y(
                Y - problem.grad_Y(X, Y) / Ly,
                step_size=1 / Ly,
            )

            loss = problem.objective(X, Y)
            history.append(loss)

            if abs(history[-2] - loss) < self.tol:
                break

        return X, Y, history