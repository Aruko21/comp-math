import numpy as np
import pathlib
import datetime
import pandas as pd
from scipy.optimize import minimize
import labs.lab2.src.graphics as graph

SAVE_FILE = True
ROOT_DIR = str(pathlib.Path().absolute())
PLOTS_DPI = 300
COUNTRY = "Canada"
I0 = 20
LEFT_LIMIT_EXP = 35
RIGHT_LIMIT_EXP = 75


def norm_eq_solver(x, y):
    # np.c_ - преобразование в столбец
    X = np.c_[x]
    A = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    return A


def exp_func(i_0, hi, t):
    return i_0 * np.exp(hi * t)


def sis_model(t, hi, I_0, I_inf):
    return I_inf / (1 + ((I_inf / I_0) - 1) * np.exp(-hi * t))


def get_lsm_func(y, t, I_0, hi):
    def lsm_sis(I_inf):
        deviations = 0
        for i in range(len(y)):
            deviations += (y[i] - sis_model(t=t[i], hi=hi, I_0=I_0, I_inf=I_inf)) ** 2
        return deviations

    return lsm_sis


def sis_prediction(hi, I_0, I_inf, start=0):
    days = start
    while sis_model(days + 1, hi, I_0, I_inf) >= 1.01 * sis_model(days, hi, I_0, I_inf):
        days += 1

    return days + 1


def applicability_exp_model(exp_model, real, start=1):
    days = start
    deviation_days = 1
    for i in range(start, len(real)):
        if deviation_days > 5:
            break
        if exp_model[i] > real[i] * 1.1:
            deviation_days += 1
        else:
            deviation_days = 0
        days += 1
    return days - deviation_days


def main():
    print("Laboratory work #2 on the 'Computational Mathematics' course.\n Done by Kosenkov Aleksandr - RC6-64B\n")
    csv_data = pd.read_csv(ROOT_DIR + '/../data/total_cases.csv', delimiter=',', usecols=[COUNTRY])

    covid_canada_graph = graph.InfectionsPlots(data=csv_data[COUNTRY],
                                               initial_date=datetime.datetime(year=2019, month=12, day=31),
                                               country=COUNTRY, dpi=PLOTS_DPI, save_file=SAVE_FILE)

    exp_cases = covid_canada_graph.get_exp_cases(left=LEFT_LIMIT_EXP, right=RIGHT_LIMIT_EXP)
    sis_cases = covid_canada_graph.get_exp_cases(left=LEFT_LIMIT_EXP)
    epidemy_days = np.arange(0, len(sis_cases), 1)
    print("Cases for start start of exp model: ", exp_cases[0])

    covid_canada_graph.show_infection_cases()
    hi = norm_eq_solver(x=np.arange(0, len(exp_cases), 1), y=np.log(exp_cases / I0))
    exponent_model = [exp_func(I0, hi, t) for t in epidemy_days]
    print("Days of epidemy: ", len(exponent_model))
    print("'hi' coefficient value for exponential model is: ", hi)
    covid_canada_graph.compare_real_and_model(model_data=exponent_model, start=LEFT_LIMIT_EXP,
                                              model_name="Exp model")

    optimized = minimize(get_lsm_func(y=sis_cases, t=epidemy_days, I_0=I0, hi=hi), (1e5,))
    I_inf_opt = optimized.x[0]
    print("Optimized I_inf: ", I_inf_opt)
    sis_model_data = [sis_model(t, hi=hi, I_0=I0, I_inf=I_inf_opt) for t in epidemy_days]
    covid_canada_graph.compare_real_and_model(model_data=sis_model_data, start=LEFT_LIMIT_EXP,
                                              model_name="SIS model")

    covid_canada_graph.get_errors(model_data=sis_model_data, start=LEFT_LIMIT_EXP)

    applic_of_exp = applicability_exp_model(exponent_model, exp_cases, 20)
    print("Applicability of exponent model: {} days from the start of epidemic".format(applic_of_exp))

    end_of_epidemic = sis_prediction(hi, I0, I_inf_opt, start=40)
    print("Prediction of epidemic end in Canada using SIS model is on the {} day from start of epidemic ({})".format(
        end_of_epidemic, covid_canada_graph.get_date_by_offset(LEFT_LIMIT_EXP + end_of_epidemic)))

    epidemy_days_final = np.arange(0, len(sis_cases) + 30, 1)
    sis_model_data_final = [sis_model(t, hi=hi, I_0=I0, I_inf=I_inf_opt) for t in epidemy_days_final]

    covid_canada_graph.compare_with_models_cartesian(model_data=[sis_model_data_final], start=LEFT_LIMIT_EXP,
                                                     model_names=["SIS model"], graph_name="compare_3_models")


main()
