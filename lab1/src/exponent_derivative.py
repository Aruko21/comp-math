import numpy as np
from scipy.special import comb


def coefficient_n_i(n, i):
    first_coef = 2 ** ((n - i) // 2)

    multiplication = 1
    # (n - 1) + 1 notation makes clear to understand the original formula and the inclusion of the last element in loop
    for j in np.arange(1, (i - 1) + 1, 2):
        multiplication *= (i - j)

    return first_coef * multiplication


def exponent_n_derivative(n, x):
    first_coef = 2 ** ((n + 1) // 2)

    polynomial = 0
    for i in np.arange(0, n + 1, 2):
        sign = (-1) ** ((n + (i/2) % 2) % 2)
        coef = comb(n, i, exact=True) * coefficient_n_i(n, i)
        x_exp = x ** (n - i)

        polynomial += sign * coef * x_exp

    result = first_coef * polynomial * np.exp(-(x ** 2))
    return result
