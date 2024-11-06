from functions import quadratic, my_f3 as f3, my_f7 as f7
from solver import solver as ES_solver
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.stats import wilcoxon
from gradient_descent import solver as SGD_solver

matplotlib.use("Agg")


def comparison_test(f, start_points, sig, t_max, repetitions, alpha, eps, logy=False):
    ES_values = []
    SGD_values = []
    for i, point in enumerate(start_points, start=1):
        average_values = np.zeros(t_max + 1)
        for _ in range(repetitions):
            result = ES_solver(f, point, sig, 5, t_max)
            values = result[2]
            average_values += values
        average_values /= 10
        ES_values.append(average_values[-1])
    for i, point in enumerate(start_points, start=1):
        result = SGD_solver(f, point, alpha, eps, max_it=t_max)
        values = result[1]
        SGD_values.append(values[-1])

    average_diff = (sum(ES_values) - sum(SGD_values)) / len(ES_values)
    ES_wins = 0
    for es, sgd in zip(ES_values, SGD_values):
        if es < sgd:
            ES_wins += 1
        if es > sgd:
            ES_wins -= 1

    statistic, p_value = wilcoxon(ES_values, SGD_values)
    return ES_wins, average_diff, p_value


DOMAIN = 100
DIM = 30
START_POINTS_NUM = 100
random_start_points = [
    [DOMAIN * (np.random.rand() * 2 - 1) for _ in range(DIM)]
    for __ in range(START_POINTS_NUM)
]


print(comparison_test(quadratic, random_start_points[:100], 1e2, 500, 1, 1e-1, 1e-12))
print(comparison_test(f3, random_start_points[:100], 1e0, 500, 1, 1e-8, 1e-12, True))
print(comparison_test(f7, random_start_points[:100], 1e2, 500, 1, 1e-3, 1e-12, True))
