import numpy as np
from data_generator import DataGenerator
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from multilayer_perceptron import *
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


class Sigmoid(ActivationFunc):
    def __call__(self, X):
        return 1 / (1 + np.exp(-X))

    def derivative(self, X):
        return sigmoid(X) * (1 - sigmoid(X))


def investigated_function(x_vec):
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

    sigmoid = Sigmoid()

    gen = DataGenerator()
    x, y = gen.get_data(DOMAIN, DIM, investigated_function, SAMPLES_DENSITY)

    scaler = MinMaxScaler(feature_range=(0, 1))
    norm_y = scaler.fit_transform(y)
    # denorm_y = scaler.inverse_transform(norm_y)

    mlp = MLP([1, 10, 10, 1], sigmoid)  # Sieć z dwoma ukrytymi warstwami
    mlp.train_es(x, norm_y)

    y_pred = mlp.forward(x)
    denorm_y_pred = scaler.inverse_transform(y_pred)
    print(denorm_y_pred)
    print("Final Loss:", mse(y, denorm_y_pred))

