import numpy as np

def newton_raphson(func, dfunc, x0, tol=1e-6, max_iter=100):
    """
    Perform the Newton-Raphson method for finding roots of a function.

    Parameters:
    func : callable
        The function for which we are trying to find a root.
    dfunc : callable
        The derivative of the function.
    x0 : float
        Initial guess for the root.
    tol : float
        Tolerance for convergence.
    max_iter : int
        Maximum number of iterations.

    Returns:
    float
        Estimated root of the function.
    """
    x = x0
    for i in range(max_iter):
        fx = func(x)
        dfx = dfunc(x)
        if dfx == 0:
            raise ValueError("Derivative is zero. No solution found.")
        x_new = x - fx / dfx
        if abs(x_new - x) < tol:
            return x_new,i
        x = x_new
    raise ValueError("Maximum iterations reached. No solution found.")

func = lambda x: x**3  + 2*x**2  - np.sin(x) - 3
dfunc = lambda x: 3*x**2 + 4*x - np.cos(x)
root,nstep= newton_raphson(func, dfunc, x0=-150.0)
print(f"Estimated root: {root} found in {nstep} steps")