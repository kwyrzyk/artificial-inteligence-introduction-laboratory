import numpy as np
from enum import Enum

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

class ActivationFun(Enum):
    RELU = 1
    SIGMOID = 2
    HYPERBOLIC = 3

class MLP():
    def __init__(self, layer_sizes, activation_fun):
        self.layer_size = layer_sizes
        self.layers_num = len(layer_sizes)
        match activation_fun:
            case ActivationFun.RELU:
                self.activation_fun = self.relu
                self.activation_derivative = self.relu_derivative
            case ActivationFun.SIGMOID:
                self.activation_fun = self.sigmoid
                self.activation_derivative = self.sigmoid_derivative
            case ActivationFun.HYPERBOLIC:
                self.activation_fun = self.hyperbolic
                self.activation_derivative = self.hyperbolic_derivative
        
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) for i in range(self.layers_num - 1)]
        self.biases = [np.random.randn(1, layer_sizes[i+1]) for i in range(self.layers_num - 1)]
                
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
        for _ in range(epochs):
            y_pred = self.forward(x)
            self.backward(y, learning_rate)
    
    def forward(self, X):
        """Propagacja wprzód"""
        self.a = [X]  # Lista aktywacji (dla każdej warstwy)
        for w, b in zip(self.weights, self.biases):
            z = np.dot(self.a[-1], w) + b  # Obliczanie sumy wag i biasu
            self.a.append(self.activation_fun(z))  # Zastosowanie funkcji aktywacji
        return self.a[-1]  # Zwrócenie wyniku końcowego (wynik ostatniej warstwy)

    def backward(self, X, Y, learning_rate=0.01):
        """Propagacja wsteczna (backpropagation)"""
        # Obliczanie błędu na wyjściu (gradient MSE)
        deltas = [mse_derivative(Y, self.a[-1]) * self.activation_derivative(self.a[-1])]
        
        # Obliczanie deltas dla warstw ukrytych
        for i in range(self.layers_num - 2, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * self.activation_derivative(self.a[i])
            deltas.insert(0, delta)
        
        # Aktualizacja wag i biasów
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * np.dot(self.a[i].T, deltas[i])  # Waga dla całego zbioru danych
            self.biases[i] -= learning_rate * np.sum(deltas[i], axis=0, keepdims=True)  # Bias dla całego zbioru danych