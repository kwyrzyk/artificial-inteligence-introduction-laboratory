import numpy as np
from data_generator import DataGenerator
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def invastigated_function(x_vec):
    x = x_vec[:, 0]
    result = x**2 * np.sin(x) + 100 * np.sin(x) * np.cos(x)
    return result.reshape(-1, 1)


def three_dim_fun(x_vec):
    x = x_vec[:, 0]
    y = x_vec[:, 1]
    z = x_vec[:, 2]

    result = np.array([x**2 - y, x*y - z])
    result = result.transpose()

    return result


if __name__ == "__main__":
    DOMAIN = (-10, 10)
    DIM = 1
    SAMPLES_DENSITY = 100

    gen = DataGenerator()
    x, y = gen.get_data(DOMAIN, DIM, invastigated_function, SAMPLES_DENSITY)

    for x, y in zip(x,y):
        print(x, y)


