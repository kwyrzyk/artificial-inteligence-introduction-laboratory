import numpy as np
from enum import Enum
from typing import Protocol, Sequence


class ActivationFunc(Protocol):
    def __call__(self, X: Sequence[float]) -> Sequence[float]:
        pass

    def derivative(self, X: Sequence[float]) -> Sequence[float]:
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
    def __init__(self, layer_sizes, activation_fun: ActivationFunc):
        self.layer_size = layer_sizes
        self.layers_num = len(layer_sizes)
        self.activation_fun = activation_fun
        self.activation_derivative = activation_fun.derivative

        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) for i in range(self.layers_num - 1)]
        self.biases = [np.zeros((1, layer_sizes[i+1])) for i in range(self.layers_num - 1)]
        # self.biases = [np.random.randn(1, layer_sizes[i+1]) for i in range(self.layers_num - 1)]
                
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

    def train_gradient(self, x, y, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            y_pred = self.forward(x)
            self.backward(y, learning_rate)
            if epoch % 10 == 0:  # Loguj co 100 epok
                print(f"Epoch {epoch}, Loss: {mse(y, y_pred)}")
    
    def forward(self, X):
        """Propagacja wprzód"""
        self.a = [X]  # Lista aktywacji (dla każdej warstwy)
        for w, b in zip(self.weights, self.biases):
            z = np.dot(self.a[-1], w) + b  # Obliczanie sumy wag i biasu
            self.a.append(self.activation_fun(z))  # Zastosowanie funkcji aktywacji
        return self.a[-1]  # Zwrócenie wyniku końcowego (wynik ostatniej warstwy)

    def backward(self, Y, learning_rate=0.01):
        """Backward propagation (backpropagation)"""
        # Compute the error at the output layer
        output_error = self.a[-1] - Y
        deltas = [output_error * self.activation_derivative(self.a[-1])]

        # Backpropagate the error through the hidden layers
        for i in range(self.layers_num - 2, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * self.activation_derivative(self.a[i])
            deltas.insert(0, delta)  # Insert at the beginning to reverse the order

        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * np.dot(self.a[i].T, deltas[i])
            self.biases[i] -= learning_rate * np.sum(deltas[i], axis=0, keepdims=True)
