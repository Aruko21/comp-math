import numpy as np
import pathlib

import labs.lab4.src.graphics as graph

SAVE_FILE = False
ROOT_DIR = str(pathlib.Path().absolute())
PLOTS_DPI = 300

SINGULAR_EPSILON = 1e-1
SILVESTER_EPSILON = 5e-1


def silvester_criterion(A):
    n = len(A)

    for k in range(2, n, 1):
        if np.linalg.det(A[:k, :k]) < SILVESTER_EPSILON:
            return False
    return True


def is_matrix_singular_old(A):
    return np.linalg.det(A) == 0


def is_matrix_singular(A):
    return np.abs(np.linalg.det(A)) < SINGULAR_EPSILON


def is_strict_diag_prevalence(A):
    n = len(A)

    for i in range(n):
        tmp_sum = np.sum([np.abs(A[i, j]) for j in range(n) if i != j])

        if np.abs(A[i, i]) <= tmp_sum:
            return False

    return True


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
    # If the system is incompatible and has an infinite number of solutions
    if is_matrix_singular(A):
        raise ValueError("Matrix is singular.")

    n = len(A)

    A_gauss = np.copy(A)
    b_gauss = np.copy(b)

    # Permutation matrix
    P_matrix = np.eye(n)

    # Gaussian elimination
    for k in range(n - 1):
        if pivoting:
            # Pivoting
            max_ind_flat = abs(A_gauss[k:, k:]).argmax()
            # Since we cut 'k' rows, we need to return them back
            max_ind_row = max_ind_flat // (n - k) + k
            max_ind_col = max_ind_flat % (n - k) + k

            # Swapping
            if max_ind_col != k:
                A_gauss[:, [k, max_ind_col]] = A_gauss[:, [max_ind_col, k]]
                P_matrix[:, [k, max_ind_col]] = P_matrix[:, [max_ind_col, k]]

            if max_ind_row != k:
                A_gauss[[k, max_ind_row]] = A_gauss[[max_ind_row, k]]
                b_gauss[[k, max_ind_row]] = b_gauss[[max_ind_row, k]]
        else:
            if A_gauss[k, k] == 0:
                raise ValueError("Element on the main diagonal is zero. Try setting pivot flag to True.")

        # Eliminate
        for row in range(k + 1, n):
            multiplier = A_gauss[row, k] / A_gauss[k, k]
            A_gauss[row, k:] = A_gauss[row, k:] - multiplier * A_gauss[k, k:]
            b_gauss[row] = b_gauss[row] - multiplier * b_gauss[k]

    x = gauss_substitution(A_gauss, b_gauss)

    x_normalize = np.dot(P_matrix, x)
    return x_normalize


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
                    print("Check negative: ", under_sqrt_expression)
                    raise ValueError("Negative value under sqrt. A matrix isn't positive definitive")
                elif under_sqrt_expression == 0:
                    raise ValueError("L matrix is singular. A matrix isn't positive definitive")

                L[i, j] = np.sqrt(under_sqrt_expression)

            else:
                L[i, j] = ((A[i, j] - tmp_sum) / L[j, j])

    y = gauss_substitution(L, b, type="forward")
    x = gauss_substitution(L.T, y, type="backward")

    return x


def thomas(A, b):
    n = len(A)

    gamma = np.zeros((n, 1))
    beta = np.zeros((n, 1))

    for i in range(n - 1):
        if i != 0:
            A_i_prev = A[i, i - 1]
        else:
            A_i_prev = 0

        gamma[i + 1] = - A[i, i + 1] / (A_i_prev * gamma[i] + A[i, i])
        beta[i + 1] = (b[i] - A_i_prev * beta[i]) / (A_i_prev * gamma[i] + A[i, i])

    x = np.zeros((n, 1))
    # x_n initial condition
    x[n - 1] = (b[n - 1] - A[n - 1, n - 2] * beta[n - 1]) / (A[n - 1, n - 1] + A[n - 1, n - 2] * gamma[n - 1])

    for i in range(n - 1, 0, -1):
        x[i - 1] = gamma[i] * x[i] + beta[i]

    return x


def gen_rand_on_interval(n, low, high):
    if n == 1:
        return np.random.rand() * (high - low) + low

    return np.random.rand(n, n) * (high - low) + low


def modify_to_not_singular(matrix):
    n = len(matrix)
    while is_matrix_singular(matrix):
        coef = gen_rand_on_interval(1, -1, 1)
        matrix += coef * np.eye(n)

    return matrix


def modify_to_DP(matrix):
    n = len(matrix)

    for i in range(n):
        remaining_sum = np.sum([np.abs(matrix[i, j]) for j in range(n) if i != j])
        if matrix[i, i] == 0:
            matrix[i, i] = remaining_sum + 0.01
        elif np.abs(matrix[i, i]) <= remaining_sum:
            if matrix[i, i] > 0:
                matrix[i, i] += remaining_sum
            else:
                matrix[i, i] -= remaining_sum

    while is_matrix_singular(matrix):
        matrix = modify_to_not_singular(matrix)
        max_elem = np.max(np.abs(matrix))
        if max_elem >= 1:
            matrix /= (max_elem + 0.001)

    return matrix


def modify_to_PD(matrix):
    symm_matrix = np.dot(matrix, matrix.T)

    max_elem = np.max(np.abs(symm_matrix))
    if max_elem >= 1:
        symm_matrix /= (max_elem + 0.001)

    while is_matrix_singular(symm_matrix):
        symm_matrix = modify_to_not_singular(symm_matrix)
        max_elem = np.max(np.abs(symm_matrix))
        if max_elem >= 1:
            symm_matrix /= (max_elem + 0.001)

    return symm_matrix


def modify_to_TD(matrix):
    n = len(matrix)
    p = 2
    q = 2

    for i in range(n):
        for j in range(n):
            if (j - i) >= p or (i - j) >= q:
                matrix[i, j] = 0

    while is_matrix_singular(matrix):
        matrix = modify_to_not_singular(matrix)
        max_elem = np.max(np.abs(matrix))
        if max_elem >= 1:
            matrix /= (max_elem + 0.001)

    return matrix


def generate_rand_matrix(n, type="default"):
    gen_matrix = gen_rand_on_interval(n, -1, 1)

    while is_matrix_singular(gen_matrix):
        gen_matrix = gen_rand_on_interval(n, -1, 1)

    if type == "DP":
        gen_matrix = modify_to_DP(gen_matrix)
    elif type == "PD":
        gen_matrix = modify_to_PD(gen_matrix)
    elif type == "TD":
        gen_matrix = modify_to_TD(gen_matrix)

    return gen_matrix


def spectral_radius(matrix):
    return np.max(np.abs(np.linalg.eigvals(matrix)))


def cond_number(matrix):
    return np.linalg.cond(matrix)


def rel_err_rms(exact, approximate):
    # Relative error using the RMS norm
    n = len(exact)

    absolute_err = np.sum([(exact[i] - approximate[i]) ** 2 for i in range(n)])
    norm = np.sum([(exact[i] ** 2) for i in range(n)])

    return np.sqrt(absolute_err) / np.sqrt(norm)


def rel_err_supremum(exact, approximate):
    # Relative error using the supremum norm
    return np.max(np.abs(exact - approximate)) / np.max(np.abs(exact))


def eigenvals_rel(matrix):
    abs_matr_eigvals = np.abs(np.linalg.eigvals(matrix))
    return np.max(abs_matr_eigvals) / np.min(abs_matr_eigvals)


def main():
    print("Laboratory work #4 on the 'Computational Mathematics' course.\n Done by Kosenkov Aleksandr - RC6-64B\n")

    A = np.array([[1., 5., 3.],
                  [3., 4., 5.],
                  [6., 7., 8.]])
    b = np.array([[3.], [2.], [3.]])
    print("Gauss without pivoting:\n", gauss(A, b, False))
    print("Gauss with pivoting:\n", gauss(A, b))
    print("Real solution:\n", np.linalg.solve(A, b))

    A = np.array([[6., 3., 4.],
                  [3., 6., 5.],
                  [4., 5., 10.]])
    b = np.array([[3.], [2.], [3.]])
    print("Gauss with pivoting:\n", gauss(A, b))
    print("Cholesky decomposition:\n", cholesky(A, b))
    print("Real solution:\n", np.linalg.solve(A, b))

    A = np.array([[1., 5., 0.],
                  [3., 4., 5., ],
                  [0., 7., 8.]])
    b = np.array([[3.], [2.], [3.]])
    print("Gauss with pivoting:\n", gauss(A, b))
    print("Thomas method:\n", thomas(A, b))
    print("Real solution:\n", np.linalg.solve(A, b))

    types = ["default", "DP", "TD", "PD"]
    type_checkers = {
        "default": None,
        "DP": is_strict_diag_prevalence,
        "TD": None,
        "PD": silvester_criterion
    }

    default_methods = {
        "default": lambda A, b: gauss(A, b, pivoting=False),
        "DP": lambda A, b: gauss(A, b, pivoting=False),
        "TD": thomas,
        "PD": cholesky
    }
    # types = ["PD"]
    # type_checkers = {
    #     "PD": silvester_criterion
    # }
    #
    # default_methods = {
    #     "PD": cholesky
    # }

    b_sol = np.array([1., 1., 1., 1.]).T

    for type in types:
        matrices = []
        for i in range(1000):
            gen_matrix = generate_rand_matrix(n=4, type=type)
            if type_checkers[type]:
                while not type_checkers[type](gen_matrix):
                    gen_matrix = generate_rand_matrix(n=4, type=type)

            matrices.append(gen_matrix)

        matrix_plots = graph.MatricesPlots(save_file=SAVE_FILE)
        spectral_rad = [spectral_radius(matrix) for matrix in matrices]
        cond_numbers = [cond_number(matrix) for matrix in matrices]

        max_cond_index = np.array(cond_numbers).argmax()
        print("Matrix for type {} with max cond={} is:\n {}\n Inverse matrix is:\n {}\n determinant is = {}".format(
            type, cond_numbers[max_cond_index], matrices[max_cond_index], np.linalg.inv(matrices[max_cond_index]),
            np.linalg.det((matrices[max_cond_index]))
        ))

        matrix_plots.show_hist(data=spectral_rad, columns_number=50,
                               label="Spectral radius for {} matrices".format(type),
                               x_label="Spectral radius", name="spec_rad_{}_1e-1".format(type))
        matrix_plots.show_hist(data=cond_numbers, columns_number=50,
                               label="Condition numbers for {} matrices".format(type),
                               x_label="Condition number", name="cond_num_{}_1e-1".format(type))

        solutions_exact = [gauss(matrix, b_sol, pivoting=True) for matrix in matrices]
        solutions_default = [default_methods[type](matrix, b_sol) for matrix in matrices]

        rms_err = [rel_err_rms(exact=solutions_exact[i], approximate=solutions_default[i]) for i in
                   range(len(solutions_exact))]
        supremum_err = [rel_err_supremum(exact=solutions_exact[i], approximate=solutions_default[i]) for i in
                        range(len(solutions_exact))]
        eig_rels = [eigenvals_rel(matrix) for matrix in matrices]

        matrix_plots.show_hist(data=rms_err, columns_number=50,
                               label="Relative error using the RMS norm for {} matrices".format(type),
                               x_label="Error", name="rms_err_{}".format(type))
        matrix_plots.show_hist(data=supremum_err, columns_number=50,
                               label="Relative error using the supremum norm for {} matrices".format(type),
                               x_label="Error", name="supremum_err_{}".format(type))
        matrix_plots.show_hist(data=eig_rels, columns_number=50,
                               label="Relative max eigenvalue to min for {} matrices".format(type),
                               x_label="Relative", name="eig_rel_{}".format(type))

main()
