import numpy as np
from scipy.optimize import fsolve


def equations(vars):
    x, y = vars

    # First equation: sum_{n=0}^{29} x y^n = 0.9
    eq1 = x * np.sum([y**n for n in range(30)]) - 0.9

    # Second equation: x y^29 = 0.01
    eq2 = x * (y**29) - 0.01

    return [eq1, eq2]


# Initial guesses
x0 = [1.0, 0.9]

sol = fsolve(equations, x0)
x_sol, y_sol = sol

print("x =", x_sol)
print("y =", y_sol)
