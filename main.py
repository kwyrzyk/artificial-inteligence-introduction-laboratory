import numpy as np
from data_generator import DataGenerator
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')
from multilayer_perceptron import *
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


class Sigmoid(ActivationFunc):
    def __call__(self, X):
        return 1 / (1 + np.exp(-X))

    def derivative(self, X):
        return sigmoid(X) * (1 - sigmoid(X))


class ReLU(ActivationFunc):
    def __call__(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return (x > 0).astype(float)


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
    # Ustawienia
    DOMAIN = (-10, 10)
    DIM = 1
    SAMPLES_DENSITY = 100
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    HIDDEN_LAYERS = [10, 10]
    ACTIVATION = ReLU()
    EPOCHS = 1000
    LEARNING_RATE = 0.01
    BATCH_SIZE = 32

    gen = DataGenerator()
    x, y = gen.get_data(DOMAIN, DIM, investigated_function, SAMPLES_DENSITY)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    x_train = scaler_x.fit_transform(x_train)
    x_test = scaler_x.transform(x_test)
    y_train = scaler_y.fit_transform(y_train)
    y_test = scaler_y.transform(y_test)

    mlp = MLP([DIM] + HIDDEN_LAYERS + [1], ACTIVATION)
    mlp.train_gradient(x_train, y_train, epochs=EPOCHS, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE)

    y_pred_train = mlp.forward(x_train)
    y_pred_test = mlp.forward(x_test)

    # y_pred_train = scaler_y.inverse_transform(y_pred_train)
    # y_pred_test = scaler_y.inverse_transform(y_pred_test)

    # y_train = scaler_y.inverse_transform(y_train)
    # y_test = scaler_y.inverse_transform(y_test)

    train_loss = mse(y_train, y_pred_train)
    test_loss = mse(y_test, y_pred_test)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    plt.scatter(x_test, y_test, label="Prawdziwe", color="blue")
    plt.scatter(x_test, y_pred_test, label="Przewidywane", color="red", alpha=0.6)
    plt.legend()
    plt.show()
