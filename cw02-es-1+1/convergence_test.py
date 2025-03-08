from functions import quadratic, my_f3 as f3, my_f7 as f7
from solver import solver as ES_solver
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.stats import wilcoxon
from gradient_descent import solver as SGD_solver

matplotlib.use("Agg")


def different_start_point_test(
    f,
    start_points,
    sig,
    mut_freq,
    t_max,
    repetitions,
    logy=False,
    mul_up=1.22,
    mul_down=0.82,
):
    for i, point in enumerate(start_points, start=1):
        average_values = np.zeros(t_max + 1)
        for _ in range(repetitions):
            result = ES_solver(f, point, sig, mut_freq, t_max, mul_up, mul_down)
            values = result[2]
            average_values += values
            average_values /= 10
        steps_num = t_max + 1
        t = range(steps_num)
        label = "x" + str(i)
        plt.scatter(t, average_values, label=label, s=1)
    if logy:
        plt.yscale("log")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.xlabel("Numer iteracji (k)", fontsize=18, fontweight="bold")
    plt.ylabel("Wartość funkcji f(k)", fontsize=18, fontweight="bold")
    plt.title(
        f"Funkcja f7 σ={sig:.0e}, mnożniki {mul_up} i {mul_down}",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig("convergence_test.png")


DOMAIN = 100
DIM = 30
START_POINTS_NUM = 5
random_start_points = [
    [DOMAIN * (np.random.rand() * 2 - 1) for _ in range(DIM)]
    for __ in range(START_POINTS_NUM)
]


different_start_point_test(f7, random_start_points, 1e2, 5, 500, 10, False, 1.44, 0.64)
