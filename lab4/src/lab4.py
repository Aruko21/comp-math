import numpy as np
import pathlib
import timeit
from scipy.optimize import root
from numpy.random import rand
import labs.lab4.src.graphics as graph

SAVE_FILE = False
ROOT_DIR = str(pathlib.Path().absolute())
PLOTS_DPI = 300


def gauss_substitution(A, b, type="backward"):
    n = len(A)
    x = np.zeros((n, 1))

    if type == "backward":
        for k in range(n - 1, -1, -1):
            x[k] = (b[k] - np.dot(A[k, k + 1:], x[k + 1:])) / A[k, k]
    elif type == "forward":
        for k in range(n):
            x[k] = (b[k] - np.dot(A[k, :k], x[:k])) / A[k, k]
    else:
        raise ValueError("Unknown type of substitution")

    return x


def gauss(A, b, pivoting=True):
    n = len(A)
    P_matrix = np.eye(n)
    A_gauss = np.copy(A)
    b_gauss = np.copy(b)

    # Elimination
    for k in range(n - 1):
        if pivoting:
            # Pivot
            # Сверху вырезали k строк, поэтому надо их вернуть.

            max_ind_flat = abs(A_gauss[k:, k:]).argmax()
            max_ind_col = max_ind_flat // (n - k) + k
            max_ind_row = max_ind_flat % (n - k) + k

            # Swap
            if max_ind_col != k:
                A_gauss[:, [k, max_ind_col]] = A_gauss[:, [max_ind_col, k]]
                P_matrix[:, [k, max_ind_col]] = P_matrix[:, [max_ind_col, k]]

            if max_ind_row != k:
                A_gauss[[k, max_ind_row]] = A_gauss[[max_ind_row, k]]
                b_gauss[[k, max_ind_row]] = b_gauss[[max_ind_row, k]]
        else:
            if A_gauss[k, k] == 0:
                raise ValueError("Pivot element is zero. Try setting pivoting to True.")

        # Eliminate
        for row in range(k + 1, n):
            multiplier = A_gauss[row, k] / A_gauss[k, k]
            A_gauss[row, k:] = A_gauss[row, k:] - multiplier * A_gauss[k, k:]
            b_gauss[row] = b_gauss[row] - multiplier * b_gauss[k]

    # Back Substitution
    # x = np.zeros((n, 1))
    # for k in range(n - 1, -1, -1):
    #     x[k] = (b_gauss[k] - np.dot(A_gauss[k, k + 1:], x[k + 1:])) / A_gauss[k, k]

    x = gauss_substitution(A_gauss, b_gauss)

    x_normalize = np.dot(P_matrix, x)
    return x_normalize


def silvester_criterion(A):
    n = len(A)

    for k in range(2, n, 1):
        if not np.linalg.det(A[:k, :k]) > 0:
            return False
    return True


def cholesky(A, b):
    n = len(A)

    # L - down triangle matrix with non-null diagonal elements
    L = np.zeros((n, n))

    for i in range(n):
        # i + 1 because there are i + 1 elements in each row in L matrix
        for j in range(i + 1):
            tmp_sum = sum([L[i, p] * L[j, p] for p in range(j)])  # max p is p=i-1, because max j is j=i

            if i == j:  # Diagonal element
                under_sqrt_expression = A[i, i] - tmp_sum
                if under_sqrt_expression < 0:
                    raise ValueError("Negative value under sqrt. A matrix isn't positive definitive")
                elif under_sqrt_expression == 0:
                    raise ValueError("L matrix is singular. A matrix isn't positive definitive")

                L[i, j] = np.sqrt(under_sqrt_expression)

            else:
                L[i, j] = (1.0 / L[j, j] * (A[i, j] - tmp_sum))

    y = gauss_substitution(L, b, type="forward")
    x = gauss_substitution(L.T, y, type="backward")

    return x


def thomas(A, b):
    n = len(A)

    gamma = np.zeros((n, 1))
    beta = np.zeros((n, 1))
    for i in range(n - 1):
        if (i != 0):
            A_i_im1 = A[i, i-1]
        else:
            A_i_im1 = 0

        gamma[i+1] = -A[i, i+1] / (A_i_im1 * gamma[i] + A[i, i])
        beta[i+1] = (b[i] - A_i_im1 * beta[i]) / (A_i_im1 * gamma[i] + A[i, i])

    x = np.zeros((n, 1))
    x[n - 1] = (b[n-1] - A[n-1, n - 2] * beta[n-1]) / (A[n-1, n-1] + A[n-1, n - 2] * gamma[n-1])

    for i in range(n - 1, 0, -1):
        x[i - 1] = gamma[i] * x[i] + beta[i]

    return x


def get_random_point():
    return [rand() * 100 - 50, rand() * 100 - 50, rand() * 70]


def main():
    print("Laboratory work #4 on the 'Computational Mathematics' course.\n Done by Kosenkov Aleksandr - RC6-64B\n")

    # A = np.array([[1., 5., 3.],
    #               [3., 4., 5.],
    #               [6., 7., 8.]])
    # b = np.array([[3.], [2.], [3.]])
    # print("Gauss without pivoting:\n", gauss(A, b, False))
    # print("Gauss with pivoting:\n", gauss(A, b))
    # print("Real solution:\n", np.linalg.solve(A, b))

    # A = np.array([[6., 3., 4.],
    #               [3., 6., 5.],
    #               [4., 5., 10.]])
    # b = np.array([[3.], [2.], [3.]])
    # print("Gauss with pivoting:\n", gauss(A, b))
    # print("Cholesky decomposition:\n", cholesky(A, b))
    # print("Real solution:\n", np.linalg.solve(A, b))

    A = np.array([[1., 5., 0.],
                  [3., 4., 5., ],
                  [0., 7., 8.]])
    b = np.array([[3.], [2.], [3.]])
    print("Gauss with pivoting:\n", gauss(A, b))
    print("Thomas method:\n", thomas(A, b))
    print("Real solution:\n", np.linalg.solve(A, b))


main()
