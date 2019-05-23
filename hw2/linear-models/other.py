import numpy as np
from scipy.optimize import minimize

w = np.array([0, 0, 0])
print(w)

x = np.array([1, 2, 1])
print(x)

cons = {
    'type': 'ineq',
    'fun': lambda t: np.linalg.norm(t, ord=1) - 1,
    'jac': lambda t: np.array([1, 1, 1])
}

w = minimize(
    fun=lambda t: np.linalg.norm(w * x, ord=2)**2 / 2,
    x0=w,
    jac=lambda t: t,
    constraints=cons
)

print(w)

w