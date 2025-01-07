import numpy as np
from typing import Protocol, Sequence


# Definicja protokołu dla funkcji aktywacji
class ActivationFunc(Protocol):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass

    def derivative(self, x: np.ndarray) -> np.ndarray:
        pass


# Implementacje funkcji aktywacji
class ReLU:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)


class Sigmoid:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        s = self(x)
        return s * (1 - s)


class Tanh:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2


# Funkcja kosztu (MSE) i jej pochodna
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)


def mse_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return y_pred - y_true


# Klasa wielowarstwowego perceptronu
class MLP:
    def __init__(self, layer_sizes: Sequence[int], activation_fun: ActivationFunc):
        self.layer_sizes = layer_sizes
        self.layers_num = len(layer_sizes)
        self.activation_fun = activation_fun
        self.activation_derivative = activation_fun.derivative

        # Inicjalizacja wag i biasów
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i + 1]) for i in range(self.layers_num - 1)]
        self.biases = [np.zeros((1, layer_sizes[i + 1])) for i in range(self.layers_num - 1)]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Propagacja wprzód"""
        self.a = [X]  # Lista aktywacji dla każdej warstwy
        for w, b in zip(self.weights, self.biases):
            z = np.dot(self.a[-1], w) + b  # Obliczanie sumy wag i biasu
            self.a.append(self.activation_fun(z))  # Zastosowanie funkcji aktywacji
        return self.a[-1]  # Zwrócenie wyjścia z ostatniej warstwy

    def backward(self, Y: np.ndarray, learning_rate: float = 0.01):
        """Propagacja wsteczna (backpropagation)"""
        # Obliczanie błędu w warstwie wyjściowej
        output_error = self.a[-1] - Y
        deltas = [output_error * self.activation_derivative(self.a[-1])]

        # Propagacja błędu przez warstwy ukryte
        for i in range(self.layers_num - 2, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * self.activation_derivative(self.a[i])
            deltas.insert(0, delta)  # Dodanie delty na początek listy

        # Aktualizacja wag i biasów
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * np.dot(self.a[i].T, deltas[i])
            self.biases[i] -= learning_rate * np.sum(deltas[i], axis=0, keepdims=True)

    def train_gradient(self, X: np.ndarray, Y: np.ndarray, epochs: int = 1000, learning_rate: float = 0.01):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            self.backward(Y, learning_rate)
            if epoch % 10 == 0:  # Loguj co 10 epok
                print(f"Epoch {epoch}, Loss: {mse(Y, y_pred)}")


