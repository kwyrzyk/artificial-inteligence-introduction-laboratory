import numpy as np
from data_generator import DataGenerator
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from multilayer_perceptron import *
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from dataclasses import dataclass
import copy


@dataclass
class TestParams:
    domain: tuple
    dim: int
    samples_density: int
    test_size: int
    activation_fun: ActivationFun
    epochs: int
    learning_rate: float

def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

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

def test_MLP(layer_sizes_list, params):
    gen = DataGenerator()
    x_train, y_train = gen.get_train_data(params.domain, params.dim, investigated_function, params.samples_density)
    x_test, y_test = gen.get_test_data(params.domain, params.dim, investigated_function, params.test_size)

    x_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    x_scaler.fit(x_train)
    y_scaler.fit(y_train)

    norm_x_train = x_scaler.transform(x_train)
    norm_x_test = x_scaler.transform(x_test)
    norm_y_train = y_scaler.transform(y_train)
    norm_y_test = y_scaler.transform(y_test)

    for layer_sizes in layer_sizes_list:
        mlp = MLP(layer_sizes, params.activation_fun)
        mlp.train_gradient(norm_x_train, norm_y_train, params.epochs, params.learning_rate)

        y_pred = mlp.forward(norm_x_test)
        denorm_y_pred = y_scaler.inverse_transform(y_pred)
        
        accuracy = mse(y_test, denorm_y_pred)
        print(f"Layer sizes: {layer_sizes}, MSE: {accuracy}")

if __name__ == "__main__":

    test_params = TestParams(
        domain=(-10, 10),
        dim=1,
        samples_density=100,
        test_size=100,
        activation_fun=ActivationFun.SIGMOID,
        epochs=1000,  # Increase the number of epochs
        learning_rate=0.2  # Lower the learning rate
    )

    layer_sizes_list = [
        [1, 10, 10, 1],
    ]

    # test_MLP(layer_sizes_list, test_params)

    x_train = np.random.rand(100, 1)
    y_train = 2 * x_train + 1  # Funkcja liniowa: y = 2x + 1
    mlp = MLP(layer_sizes=[1, 10, 1], activation_fun=ActivationFun.RELU)
    mlp.train_gradient(x_train, y_train, epochs=1000, learning_rate=0.01)

    print(mlp.forward([1]))
    print(mlp.weights)





