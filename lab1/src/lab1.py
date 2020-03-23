import numpy as np
from math import pow
from scipy.special import factorial
from scipy.misc import derivative
from labs.lab1.src.exponent_derivative import exponent_n_derivative
import labs.lab1.src.graphics as graph

SAVE_FILE = True


def l_i(i, x, x_nodes):
    numerator = 1
    for j in range(0, len(x_nodes)):
        if numerator == 0:
            break
        elif j != i:
            numerator *= x - x_nodes[j]

    if numerator == 0:
        return 0

    denominator = 1
    for j in range(0, len(x_nodes)):
        if j != i:
            denominator *= x_nodes[i] - x_nodes[j]

    return numerator / denominator


def L(x, x_nodes, y_nodes):
    if len(x_nodes) != len(y_nodes):
        print("ERROR!")

    basis = [l_i(i, x, x_nodes) for i in range(len(x_nodes))]
    return np.sum(y_nodes * basis)


def real_value(x):
    return np.exp(-(x ** 2))


def chebyshev_nodes(n):
    nodes = []
    for i in np.arange(1, n + 1, 1):
        nodes.append(np.cos(((2 * i - 1) * np.pi) / (2 * n)))

    return nodes


def lagrange_interpol(real_func, l_limit, r_limit, N, x_plot, method='evenly'):
    if method == 'evenly' or method == 'linear':
        x_nodes = np.linspace(l_limit, r_limit, N)
    elif method == 'chebyshev':
        x_nodes = chebyshev_nodes(N)
        for i in range(len(x_nodes)):
            x_nodes[i] = 0.5 * ((r_limit - l_limit) * x_nodes[i] + l_limit + r_limit)
    else:
        print("Undefined method - ", method)
        return
    y_nodes = real_func(x_nodes)

    if method == 'evenly' or method == 'chebyshev':
        lagrange_interpol_values = np.vectorize(L, excluded=['x_nodes', 'y_nodes'])
        result = lagrange_interpol_values(x_plot, x_nodes=x_nodes, y_nodes=y_nodes)
    elif method == 'linear':
        result = np.array([])
        for x in x_plot:
            h = (r_limit - l_limit) / (N - 1)
            index = int((x - l_limit) // h)
            # Handling endpoint
            if index == (N - 1):
                index -= 1

            result = np.append(result, L(x, np.array([x_nodes[index], x_nodes[index + 1]]),
                                         np.array([y_nodes[index], y_nodes[index + 1]])))
    else:
        result = None

    return result


def lagrange_residual_member_polynomial(x, x_nodes):
    multiplication = 1
    for node in x_nodes:
        multiplication *= (x - node)

    return multiplication


def scipy_der_adapter(n, x):
    result = derivative(real_value, x, n=n, dx=1e-1, order=((n // 2) * 2 + 3))
    return result


def lagrange_residual_member_max(nodes, l_limit, r_limit, func_n_deriative, x_plot_dots, method='evenly'):
    x_nodes = np.linspace(l_limit, r_limit, nodes)
    x_plot = x_plot_dots

    if 0 not in x_plot:
        x_plot = np.append(x_plot, 0)

    max_abs_derivative = 0
    max_abs_polynomial = 0
    for x in x_plot:
        if method == 'linear':
            tmp_derivative = abs(func_n_deriative(2, x))
        else:
            tmp_derivative = abs(func_n_deriative(nodes, x))
        if tmp_derivative > max_abs_derivative:
            max_abs_derivative = tmp_derivative

        if method == 'evenly':
            tmp_polynomial = abs(lagrange_residual_member_polynomial(x, x_nodes))
            if tmp_polynomial > max_abs_polynomial:
                max_abs_polynomial = tmp_polynomial

    if method == 'evenly':
        result = (max_abs_derivative * max_abs_polynomial) / factorial(nodes)
    elif method == 'chebyshev':
        result = max_abs_derivative * (pow(r_limit - l_limit, nodes) / (pow(2, 2 * nodes - 1) * factorial(nodes)))
    elif method == 'linear':
        h = (r_limit - l_limit) / (nodes - 1)
        result = (max_abs_derivative / 2) * ((h ** 2) / 4)
    else:
        print("Undefined method - ", method)
        return

    # print("TEST FOR N={}: DEVIATION = {}".format(nodes, result))
    return result


def linear_interpolation_integral(x_nodes, y_nodes):
    area = 0
    for i in range(len(x_nodes) - 1):
        area += ((y_nodes[i] + y_nodes[i + 1]) / 2) * (x_nodes[i + 1] - x_nodes[i])

    return area


def linear_inter_erf(func, x, n):
    if x > 0:
        left_erf_limit = -x
        right_erf_limit = x
    else:
        left_erf_limit = x
        right_erf_limit = -x

    x_erf_nodes = np.linspace(left_erf_limit, right_erf_limit, n)
    y_erf_nodes = lagrange_interpol(real_func=func, l_limit=left_erf_limit, r_limit=right_erf_limit,
                                    N=n, x_plot=x_erf_nodes, method='linear')

    result = linear_interpolation_integral(x_nodes=x_erf_nodes, y_nodes=y_erf_nodes) / np.sqrt(np.pi)

    return result


def main():
    print("Laboratory work #1 on the 'Computational Mathematics' course.\n Done by Kosenkov Aleksandr - RC6-64B\n")
    dots_amount_interpol = np.arange(4, 21, 1)
    dots_amount_plot = 200
    left_limit = -5
    right_limit = 5
    deviations = []
    dev_labels = []

    x_plot = np.linspace(left_limit, right_limit, dots_amount_plot)

    exponent_func = np.vectorize(real_value)
    y_real = exponent_func(x_plot)

    print("---Lagrange interpolation with evenly distributed nodes---")
    interpol_results = []
    for n in dots_amount_interpol:
        interpol_results.append(lagrange_interpol(real_func=exponent_func, l_limit=left_limit, r_limit=right_limit,
                                                  N=n, x_plot=x_plot, method='evenly'))

    interpol_plot = graph.InterpolPlots(x_plot=x_plot, y_real=y_real, interpol_results=interpol_results,
                                        dots_amount=dots_amount_interpol, color_sheme='winter',
                                        graph_name='evenly_interpol', save_file=SAVE_FILE)

    interpol_plot.show_interpol_results()
    interpol_plot.show_interpol_results_odd_even()
    evenly_deviation = interpol_plot.show_deviations()
    print("Deviations from real function:")
    for i in range(len(dots_amount_interpol)):
        print("N = {}: deviation = {}".format(dots_amount_interpol[i], evenly_deviation[i]))

    deviations.append(evenly_deviation)
    dev_labels.append("Evenly distributed nodes")

    lagrange_analysis = []
    for n in dots_amount_interpol:
        lagrange_analysis.append(lagrange_residual_member_max(nodes=n, l_limit=left_limit, r_limit=right_limit,
                                                              func_n_deriative=exponent_n_derivative,
                                                              x_plot_dots=x_plot))
    print("Residual lagrange member estimate:")
    for i in range(len(dots_amount_interpol)):
        print("N = {}: res. member = {}".format(dots_amount_interpol[i], lagrange_analysis[i]))

    interpol_plot.show_lagrange_analysis(lagrange_analysis)

    print("---Lagrange interpolation with chebyshev nodes---")
    interpol_results_cheb = []
    for n in dots_amount_interpol:
        interpol_results_cheb.append(lagrange_interpol(real_func=exponent_func, l_limit=left_limit, r_limit=right_limit,
                                                       N=n, x_plot=x_plot, method='chebyshev'))

    interpol_cheb_plot = graph.InterpolPlots(x_plot=x_plot, y_real=y_real, interpol_results=interpol_results_cheb,
                                             dots_amount=dots_amount_interpol, color_sheme='winter',
                                             graph_name='cheb_interpol', save_file=SAVE_FILE)

    interpol_cheb_plot.show_interpol_results()
    interpol_cheb_plot.show_interpol_results_odd_even()
    cheb_deviation = interpol_cheb_plot.show_deviations(legend_loc='upper right')
    print("Deviations from real function:")
    for i in range(len(dots_amount_interpol)):
        print("N = {}: deviation = {}".format(dots_amount_interpol[i], cheb_deviation[i]))

    deviations.append(cheb_deviation)
    dev_labels.append("Chebyshev nodes")

    lagrange_analysis_cheb = []
    for n in dots_amount_interpol:
        lagrange_analysis_cheb.append(lagrange_residual_member_max(nodes=n, l_limit=left_limit, r_limit=right_limit,
                                                                   func_n_deriative=exponent_n_derivative,
                                                                   x_plot_dots=x_plot, method='chebyshev'))
    print("Residual lagrange member estimate:")
    for i in range(len(dots_amount_interpol)):
        print("N = {}: res. member = {}".format(dots_amount_interpol[i], lagrange_analysis_cheb[i]))

    interpol_cheb_plot.show_lagrange_analysis(lagrange_analysis_cheb, legend_locs=('upper left', 'lower left'))

    print("---Lagrange piecewise linear interpolation---")
    interpol_results_linear = []
    for n in dots_amount_interpol:
        interpol_results_linear.append(
            lagrange_interpol(real_func=exponent_func, l_limit=left_limit, r_limit=right_limit,
                              N=n, x_plot=x_plot, method='linear'))

    interpol_linear_plot = graph.InterpolPlots(x_plot=x_plot, y_real=y_real, interpol_results=interpol_results_linear,
                                               dots_amount=dots_amount_interpol, color_sheme='winter',
                                               graph_name='linear_interpol', save_file=SAVE_FILE)

    interpol_linear_plot.show_interpol_results()
    interpol_linear_plot.show_interpol_results_odd_even()
    linear_deviation = interpol_linear_plot.show_deviations(legend_loc='upper right')
    print("Deviations from real function:")
    for i in range(len(dots_amount_interpol)):
        print("N = {}: deviation = {}".format(dots_amount_interpol[i], linear_deviation[i]))

    deviations.append(linear_deviation)
    dev_labels.append("Piecewise linear interpolation")

    lagrange_analysis_linear = []
    for n in dots_amount_interpol:
        lagrange_analysis_linear.append(lagrange_residual_member_max(nodes=n, l_limit=left_limit, r_limit=right_limit,
                                                                     func_n_deriative=exponent_n_derivative,
                                                                     x_plot_dots=x_plot, method='linear'))
    print("Residual lagrange member estimate:")
    for i in range(len(dots_amount_interpol)):
        print("N = {}: res. member = {}".format(dots_amount_interpol[i], lagrange_analysis_linear[i]))

    interpol_linear_plot.show_lagrange_analysis(lagrange_analysis_linear, legend_locs=('upper right', 'upper right'))

    graph.compare_deviations(deviations=deviations, labels=dev_labels, dots_amount_interpol=dots_amount_interpol,
                             save_file=SAVE_FILE)

    print("---Error function piecewise linear approximation---")
    erf_interpol = np.array([])
    erf_nodes = np.array([])
    for n in np.arange(3, 10, 2):
        err_f = linear_inter_erf(func=exponent_func, x=2, n=n)
        erf_interpol = np.append(erf_interpol, err_f)
        erf_nodes = np.append(erf_nodes, n)
        print("N = {}: erf = {}".format(n, err_f))

    graph.erf_compare(erf_interpol=erf_interpol, nodes=erf_nodes, x=2, save_file=SAVE_FILE)

    return


main()
