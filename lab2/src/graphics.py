import matplotlib.pyplot as plt
import numpy as np
import datetime

SAVE_DPI = 300
SAVE_IMG_LOCATION = '../graphics'
FIGSIZE = (8, 6)


def l_infinity_error_absolute(approximation, real):
    max_deviation = 0
    for i in range(len(real)):
        tmp_deviation = abs(real[i] - approximation[i])
        if tmp_deviation > max_deviation:
            max_deviation = tmp_deviation

    return max_deviation


def l_infinity_error_relative(abs_error, real):
    max_norm = 0
    for i in range(len(real)):
        tmp_norm = abs(real[i])
        if tmp_norm > max_norm:
            max_norm = tmp_norm
    return abs_error / max_norm


def l_2_error_absolute(approximation, real):
    deviations = 0
    for i in range(len(real)):
        deviations += ((real[i] - approximation[i]) ** 2)

    return np.sqrt((1 / len(real)) * deviations)


def l_2_error_relative(abs_error, real):
    real_norm = np.sqrt((1 / len(real)) * np.sum(real ** 2))
    return abs_error / real_norm


class InfectionsPlots:
    def __init__(self, data, initial_date, country='untitled', dpi=200, save_file=False):
        if not isinstance(initial_date, datetime.date):
            raise ValueError("initial_date must be instance of datetime")

        self.initial_date = initial_date

        offset = 0
        for cases in data:
            if cases == 0:
                offset += 1
            else:
                break

        self.date_offset = offset
        self.infections_cases = data[offset:]
        self.days = np.arange(0, len(self.infections_cases), 1)
        self.country = country
        self.dpi = dpi
        self.save_file = save_file

    def show_infection_cases(self):
        print("First case of COVID-19 infection in {} was {}".format(self.country,
                                                                     self.initial_date + datetime.timedelta(
                                                                         days=self.date_offset)))
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=FIGSIZE, dpi=self.dpi)
        axes.plot(self.days, self.infections_cases, '.', color="blue", label='Infections in ' + self.country)
        axes.set_ylabel('Cases of COVID-19 infection')
        axes.set_xlabel('Days')
        axes.legend(loc='best')
        axes.grid()
        plt.show()

        if self.save_file:
            fig.savefig(SAVE_IMG_LOCATION + "/2_" + self.country + "_covid_cartesian.png", dpi=SAVE_DPI)

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=FIGSIZE, dpi=self.dpi)
        axes.semilogy(self.days, self.infections_cases, '.', color="blue", label='Infections in ' + self.country)

        axes.set_ylabel('Cases of COVID-19 infection')
        axes.set_xlabel('Days')
        axes.legend(loc='best')
        axes.grid()
        plt.show()

        if self.save_file:
            fig.savefig(SAVE_IMG_LOCATION + "/2_" + self.country + "_covid_semilogy.png", dpi=SAVE_DPI)

        return

    def get_exp_cases(self, left, right=None):
        if not right:
            return np.array(self.infections_cases[left:])
        return np.array(self.infections_cases[left:right])

    def get_date_by_offset(self, offset):
        return self.initial_date + datetime.timedelta(days=self.date_offset + offset)

    def compare_real_and_model(self, model_data, start, model_name, graph_name=""):
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=FIGSIZE, dpi=self.dpi)
        axes.semilogy(self.days[start:], self.infections_cases[start:], '.', color="blue",
                      label='Infections in ' + self.country)
        axes.semilogy(self.days[start:], model_data, linestyle='-', color="red",
                      label=model_name + ' of infection in ' + self.country)

        axes.set_ylabel('Cases of COVID-19 infection')
        axes.set_xlabel('Days')
        axes.legend(loc='best')
        axes.grid()
        plt.show()

        if self.save_file:
            fig.savefig(
                SAVE_IMG_LOCATION + "/3_{}_covid_compare_with_{}_{}.png".format(self.country, model_name, graph_name),
                dpi=SAVE_DPI)

    def compare_with_models_cartesian(self, model_data, start, model_names, graph_name=""):
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=FIGSIZE, dpi=self.dpi)
        axes.plot(self.days[start:], self.infections_cases[start:], '.', color="blue",
                      label='Infections in ' + self.country)

        colors = ["red", "green"]
        for i in range(len(model_data)):
            axes.plot(np.arange(start, len(model_data[i]) + start, 1), model_data[i], linestyle='-', color=colors[i],
                      label=model_names[i] + ' of infection in ' + self.country)

        axes.set_ylabel('Cases of COVID-19 infection')
        axes.set_xlabel('Days')
        axes.legend(loc='best')
        axes.grid()
        plt.show()

        if self.save_file:
            fig.savefig(
                SAVE_IMG_LOCATION + "/4_{}_covid_{}.png".format(self.country, graph_name),
                dpi=SAVE_DPI)

    def get_errors(self, model_data, start):
        l_inf_abs = l_infinity_error_absolute(approximation=model_data, real=np.array(self.infections_cases[start:]))
        print("Absolute error in L_inf norm: ", l_inf_abs)
        print("Relative error in L_inf norm: ",
              l_infinity_error_relative(l_inf_abs, np.array(self.infections_cases[start:])))
        l_2_abs = l_2_error_absolute(approximation=model_data, real=np.array(self.infections_cases[start:]))
        print("Absolute error in L_2 norm: ", l_2_abs)
        print("Relative error in L_2 norm: ", l_2_error_relative(l_2_abs, np.array(self.infections_cases[start:])))
