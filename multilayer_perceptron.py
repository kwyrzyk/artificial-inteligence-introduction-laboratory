import numpy as np
from enum import Enum
from typing import Protocol, Sequence


class ActivationFunc(Protocol):
    def __call__(self, X: Sequence[float]) -> Sequence[float]:
        pass

    def derivative(self, X: Sequence[float]) -> Sequence[float]:
        pass


class LossFunc(Protocol):
    def __call__(self, y_true: Sequence[float], y_pred: Sequence[float]) -> Sequence[float]:
        pass

    def derivative(self, y_true: Sequence[float], y_pred: Sequence[float]) -> Sequence[float]:
        pass


def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    return y_pred - y_true


class MLP():
    def __init__(self, layer_sizes, activation_fun: ActivationFunc, loss: LossFunc):
        self.layer_size = layer_sizes
        self.layers_num = len(layer_sizes)
        self.activation_fun = activation_fun
        self.loss = loss

        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) for i in range(self.layers_num - 1)]
        self.biases = [np.random.randn(1, layer_sizes[i+1]) for i in range(self.layers_num - 1)]

    def _set_weights(self, flat_weights):
        start = 0
        for i in range(len(self.weights)):
            weight_shape = self.weights[i].shape
            bias_shape = self.biases[i].shape

            weight_size = np.prod(weight_shape)
            bias_size = np.prod(bias_shape)

            self.weights[i] = flat_weights[start:start + weight_size].reshape(weight_shape)
            start += weight_size

            self.biases[i] = flat_weights[start:start + bias_size].reshape(bias_shape)
            start += bias_size

    def _get_weights(self):
        flat_weights = []
        for w, b in zip(self.weights, self.biases):
            flat_weights.extend(w.flatten())
            flat_weights.extend(b.flatten())
        return np.array(flat_weights)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return sigmoid(x) * (1 - sigmoid(x))

    def hyperbolic(self, z):
        return np.tanh(z)

    def hyperbolic_derivative(self, z):
        return 1 - np.tanh(z)**2

    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def mse_derivative(y_true, y_pred):
        return y_pred - y_true

    def train_gradient(self, x, y, epochs=1000, learning_rate=0.01, batch_size=32):
        num_samples = len(x)
        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            x_shuffled = x[indices]
            y_shuffled = y[indices]
            epoch_loss = 0
            for i in range(0, num_samples, batch_size):
                x_batch = x_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                y_pred_batch = self.forward(x_batch)
                batch_loss = self.compute_loss(x_batch, y_batch)
                epoch_loss += batch_loss
                self.backward(y_batch, learning_rate)
            epoch_loss /= (num_samples / batch_size)
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs} Loss: {epoch_loss}")

    def forward(self, X):
        """Propagacja wprzód"""
        self.a = [X]  # Lista aktywacji (dla każdej warstwy)
        for w, b in zip(self.weights, self.biases):
            z = np.dot(self.a[-1], w) + b  # Obliczanie sumy wag i biasu
            self.a.append(self.activation_fun(z))  # Zastosowanie funkcji aktywacji
        return self.a[-1]  # Zwrócenie wyniku końcowego (wynik ostatniej warstwy)

    def backward(self, y, learning_rate, lambda_reg=0.01):
        """Propagacja wsteczna (backpropagation)"""
        # Obliczanie błędu na wyjściu (gradient MSE)
        deltas = [self.loss.derivative(y, self.a[-1]) * self.activation_fun.derivative(self.a[-1])]

        # Obliczanie deltas dla warstw ukrytych
        for i in range(self.layers_num - 2, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * self.activation_fun.derivative(self.a[i])
            deltas.insert(0, delta)

        # Aktualizacja wag i biasów
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * (np.dot(self.a[i].T, deltas[i]) + lambda_reg * self.weights[i])  # Waga dla całego zbioru danych
            self.biases[i] -= learning_rate * np.sum(deltas[i], axis=0, keepdims=True)  # Bias dla całego zbioru danych

    def compute_loss(self, X, y):
        y_pred = self.forward(X)
        return self.loss(y, y_pred)

    def train_es(self, X: Sequence[float], y: Sequence[float], iterations:int=100, sigma:float=0.1):
        weights = self._get_weights()
        dim = len(weights)
        success_count = 0
        success_threshold = 1 / 5

        for i in range(iterations):
            mutation = np.random.randn(dim) * sigma
            new_weights = weights + mutation

            self._set_weights(weights)
            loss = self.compute_loss(X, y)
            self._set_weights(new_weights)
            new_loss = self.compute_loss(X, y)

            if new_loss < loss:
                weights = new_weights
                success_count += 1
            else:
                self._set_weights(weights)

            if (i + 1) % 10 == 0:
                success_rate = success_count / 10
                if success_rate > success_threshold:
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                success_count = 0

            if (i + 1) % 10 == 0:
                print(f"Iteracja {i+1}: Loss = {loss:.4f}, Sigma = {sigma:.4f}")

        self._set_weights(weights)
