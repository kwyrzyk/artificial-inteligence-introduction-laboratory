from functions import quadratic_fun, my_f3 as f3, my_f12 as f12
from gradient_descent import solver
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from random import random


def different_start_point_test(f, start_points, alpha, eps, max_it=1e3, logy=False):
    for i, point in enumerate(start_points, start=1):
        result = solver(f, point, alpha, eps, max_it=max_it)
        q = result[1]
        steps_num = len(q)
        t = range(steps_num)
        label = "x" + str(i)
        if logy:
            plt.semilogy(t, q, label=label)
        else:
            plt.plot(t, q, label=label)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.xlabel("Numer iteracji (k)", fontsize=18, fontweight="bold")
    plt.ylabel("Wartość funkcj f(k)", fontsize=18, fontweight="bold")
    plt.title(
        "Przebieg zmian wartości\n funkcji kwadratowej, α=1e-1",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig("quadratic.png")


random_start_points = [[random() * 200 - 100 for _ in range(10)] for _ in range(5)]

different_start_point_test(
    quadratic_fun, random_start_points, 1e-1, 1e-3, max_it=1e5, logy=True
)
