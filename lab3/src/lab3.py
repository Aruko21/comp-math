import numpy as np
import pathlib
import timeit
from scipy.optimize import root
from numpy.random import rand
import labs.lab3.src.graphics as graph

SAVE_FILE = True
ROOT_DIR = str(pathlib.Path().absolute())
PLOTS_DPI = 300

SIGMA_COEF = 10
B_COEF = 8 / 3


def get_normalize_lorentz(r, sigma, b):
    def func(x_vec):
        x = x_vec[0]
        y = x_vec[1]
        z = x_vec[2]
        result = [sigma * (y - x), x * (r - z) - y, x * y - b * z]
        return result

    return func


def euler(x_0, t_n, f, h):
    time_array = np.arange(0, t_n + h, h)
    results = [[]] * len(time_array)
    results[0] = np.array(x_0)
    for i in range(len(time_array) - 1):
        results[i + 1] = np.array(results[i]) + h * np.array(f(results[i]))

    return results


def implicit_euler(x_0, t_n, f, h):
    time_array = np.arange(0, t_n + h, h)
    results = [[]] * len(time_array)
    results[0] = np.array(x_0)

    for i in range(len(time_array) - 1):
        def func_to_solve(w_vec):
            return w_vec - np.array(results[i]) - h * np.array(f(w_vec))

        w_next = root(func_to_solve, results[i])
        results[i + 1] = w_next.x

    return results


def adams_bashforth_moulton(x_0, t_n, f, h):
    time_array = np.arange(0, t_n + h, h)
    results = [[]] * len(time_array)

    initial_w = implicit_euler(x_0, time_array[3], f, h)
    for i in range(len(initial_w)):
        results[i] = initial_w[i]

    for i in range(3, len(time_array) - 1):
        predictor_w = results[i] + (h / 24) * (55 * np.array(f(results[i])) - 59 * np.array(f(results[i - 1])) +
                                               37 * np.array(f(results[i - 2])) - 9 * np.array(f(results[i - 3])))
        corrector_w = results[i] + (h / 24) * (9 * np.array(f(predictor_w)) + 19 * np.array(f(results[i])) -
                                               5 * np.array(f(results[i - 1])) + np.array(f(results[i - 2])))
        results[i + 1] = corrector_w
    return results


def get_random_point():
    return [rand() * 100 - 50, rand() * 100 - 50, rand() * 70]


def stationary_lorentz(b_coef, r_coef):
    if r_coef == 0:
        return [[0, 0, 0]]

    first_point = [np.sqrt(b_coef * (r_coef - 1)), np.sqrt(b_coef * (r_coef - 1)), r_coef - 1]
    second_point = [-np.sqrt(b_coef * (r_coef - 1)), -np.sqrt(b_coef * (r_coef - 1)), r_coef - 1]
    return [first_point, second_point]


def main():
    print("Laboratory work #3 on the 'Computational Mathematics' course.\n Done by Kosenkov Aleksandr - RC6-64B\n")
    t_time = 100
    x_start = [10, 10, 10]
    h = 0.005
    r = 24.3
    func = get_normalize_lorentz(r, SIGMA_COEF, B_COEF)

    stat_points_initial = stationary_lorentz(b_coef=B_COEF, r_coef=r)
    euler_results = euler(x_0=x_start, t_n=t_time, f=func, h=h)
    imp_euler_results = implicit_euler(x_0=x_start, t_n=t_time, f=func, h=h)
    abm_results = adams_bashforth_moulton(x_0=x_start, t_n=t_time, f=func, h=h)

    lorentz_plots = graph.LorentzPlots(dpi=PLOTS_DPI, save_file=SAVE_FILE)
    lorentz_plots.show_results(data=[euler_results], stationary=stat_points_initial,
                               labels=["Phase portrait with Euler method\n$r={}$, $h={}$".format(r, h)],
                               name="2_euler_example_005")
    lorentz_plots.show_results(data=[imp_euler_results], stationary=stat_points_initial,
                               labels=["Phase portrait with Implicit Euler method\n$r={}$, $h={}$".format(r, h)],
                               name="2_impicit_euler_example_005")
    lorentz_plots.show_results(data=[abm_results], stationary=stat_points_initial,
                               labels=["Phase portrait with Adams Bashfourth Moulton method\n$r={}$, $h={}$".format(r, h)],
                               name="2_adm_example_005")

    x_0_randoms = [get_random_point() for i in range(100)]

    r_variations = [0, 10, 20, 30]
    for r_var in r_variations:
        func_var = get_normalize_lorentz(r_var, SIGMA_COEF, B_COEF)
        stat_points_stochastic = stationary_lorentz(b_coef=B_COEF, r_coef=r_var)
        stochastic_trajectories = []
        for i in range(10):
            stochastic_trajectories.append(adams_bashforth_moulton(x_0=x_0_randoms[i], t_n=t_time, f=func_var, h=h))

        lorentz_plots.show_results(data=stochastic_trajectories, stationary=stat_points_stochastic,
                                   name="stochastic_r_{}".format(r_var))
        if r_var == 0:
            lorentz_plots.show_projection(data=stochastic_trajectories, projection="xy",
                                          name="XY_projection_r_0")

    r_special = [24.06, 28, 100]
    for r_var in r_special:
        func_var = get_normalize_lorentz(r_var, SIGMA_COEF, B_COEF)
        stochastic_trajectories = []
        stat_points_stochastic = stationary_lorentz(b_coef=B_COEF, r_coef=r_var)
        for i in range(10):
            stochastic_trajectories.append(adams_bashforth_moulton(x_0=x_0_randoms[i], t_n=t_time, f=func_var, h=h))

        lorentz_plots.show_results(data=stochastic_trajectories, stationary=stat_points_stochastic,
                                   name="stochastic_spec__r_{}".format(r_var))

    r = 30
    func = get_normalize_lorentz(r, SIGMA_COEF, B_COEF)
    h_values = [0.1, 0.05, 0.01, 0.005]
    labels = ["Euler method", "Implicit Euler method", "ABM method"]
    euler_times = []
    imp_euler_times = []
    abm_times = []
    for h_step in h_values:
        print("Time test for {} random x_0 values, r={}, with h={}".format(100, r, h_step))
        stochastic_trajectories = []
        t_start = timeit.default_timer()
        for i in range(100):
            stochastic_trajectories.append(euler(x_0=x_0_randoms[i], t_n=t_time, f=func, h=h_step))
        t_end = timeit.default_timer()
        res_time = t_end - t_start
        euler_times.append(res_time)
        print("Euler method: ", t_end - t_start)

        stochastic_trajectories = []
        t_start = timeit.default_timer()
        for i in range(100):
            stochastic_trajectories.append(implicit_euler(x_0=x_0_randoms[i], t_n=t_time, f=func, h=h_step))
        t_end = timeit.default_timer()
        res_time = t_end - t_start
        imp_euler_times.append(res_time)
        print("Implicit Euler method: ", t_end - t_start)

        stochastic_trajectories = []
        t_start = timeit.default_timer()
        for i in range(100):
            stochastic_trajectories.append(adams_bashforth_moulton(x_0=x_0_randoms[i], t_n=t_time, f=func, h=h_step))
        t_end = timeit.default_timer()
        res_time = t_end - t_start
        abm_times.append(res_time)
        print("ABM method: ", t_end - t_start)

    lorentz_plots.show_times(data=[euler_times, imp_euler_times, abm_times], h_steps=h_values, labels=labels)


main()
