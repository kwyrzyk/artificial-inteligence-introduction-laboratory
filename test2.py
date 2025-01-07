import numpy as np
import math
import matplotlib.pyplot as plt
import numpy as np
from data_generator import DataGenerator
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Użycie backendu nieinteraktywnego
from multilayer_perceptron import MLP, ActivationFunc, ReLU, Sigmoid, Tanh
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import MinMaxScaler
from dataclasses import dataclass
import copy
import math



def function(x):
    return math.sin(x * math.sqrt(7)) + math.cos(2 * x)

# def function(x):
#     """Badana funkcja"""
#     result = x**2 * np.sin(x) + 100 * np.sin(x) * np.cos(x)
#     return result


class NeuralNetwork:
    def __init__(self, sample_size, hidden_layer_size=10):
        self.sample_size = sample_size
        np.random.seed(42)
        self.input_size = 1
        self.output_size = 1
        self.hidden_layer_size = hidden_layer_size
        self.weights_hidden = np.random.randn(self.hidden_layer_size, self.input_size) * np.sqrt(2 / self.input_size)
        self.biases_hidden = np.zeros((self.hidden_layer_size, 1))
        self.weights_output = np.random.randn(self.output_size, self.hidden_layer_size) * np.sqrt(2 / self.hidden_layer_size)
        self.bias_output = np.zeros((self.output_size, 1))
        self.x_data = np.random.uniform(-5, 5, sample_size)  # 1000 points between -5 and 5
        self.y_data = np.array([function(x) for x in self.x_data])

        self.x_data = self.x_data

        # Split into training and testing data
        self.x_train = self.x_data[:self.sample_size * 8 // 10]
        self.y_train = np.array([function(x) for x in self.x_train])
        self.x_test = self.x_data[self.sample_size * 8 // 10:]
        self.y_test = np.array([function(x) for x in self.x_test])

    def tanh(self, x):
        return np.tanh(x)

    def forward_propagation(self, x):
        x = x * 0.2
        z_hidden = np.dot(self.weights_hidden, x) + self.biases_hidden
        a_hidden = self.tanh(z_hidden)
        output = np.dot(self.weights_output, a_hidden) + self.bias_output
        return z_hidden, a_hidden, output

    def backwards_propagation(self, x, learning_rate=0.01):
        z1, a1, z2 = self.forward_propagation(x)

        error = z2 - function(x)
        dz2 = error
        dw2 = dz2 @ a1.T
        db2 = dz2

        dz1 = (self.weights_output.T @ dz2) * (1 - np.tanh(z1) ** 2)
        dw1 = dz1 @ x.T
        db1 = dz1

        self.weights_hidden -= learning_rate * dw1
        self.biases_hidden -= learning_rate * db1
        self.weights_output -= learning_rate * dw2
        self.bias_output -= learning_rate * db2

        return self.weights_hidden, self.biases_hidden,  self.weights_output, self.bias_output, float((error ** 2).mean())

    def train(self, epochs):
        lowest_loss = float('inf')
        for epoch in range(epochs):
            epoch_loss = 0
            for x, y in zip(self.x_train, self.y_train):
                x = np.array([[x]])
                y = np.array([[y]])
                w1, b1, w2, b2, loss = self.backwards_propagation(x)
                epoch_loss += loss
            avg_loss = epoch_loss / len(self.x_train)

            if avg_loss < lowest_loss:
                lowest_loss = avg_loss
                self.weights_hidden = w1.copy()
                self.biases_hidden = b1.copy()
                self.weights_output = w2.copy()
                self.bias_output = b2.copy()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")


def plot_true_values(neurons, epochs):
    plt.figure()
    network = NeuralNetwork(10000, neurons)
    network.train(epochs)
    x = network.x_test
    x = sorted(x)
    y = [function(x_) for x_ in x]
    y = np.array(y).flatten()
    preds = [network.forward_propagation(x_)[2] for x_ in x]
    preds = np.array(preds).flatten()
    plt.plot(x, y, label='Real')
    plt.plot(x, preds, label='Predicted')
    plt.legend()
    plt.title(f"{neurons} Neurons Network trained for {epochs} epochs")
    plt.savefig(f'{neurons}Neurons-{epochs}Epochs.png')
    plt.close()


def plot_true_values_multiplot(neurons_list, epochs_list):
    fig, axs = plt.subplots(4, 4, figsize=(16, 16))

    for i, neurons in enumerate(neurons_list):
        for j, epochs in enumerate(epochs_list):
            ax = axs[i, j]
            network = NeuralNetwork(10000, neurons)
            network.train(epochs)
            x = sorted(network.x_test)
            y = np.array([function(x_) for x_ in x]).flatten()
            preds = np.array([network.forward_propagation(x_)[2] for x_ in x]).flatten()
            ax.plot(x, y, label='Real')
            ax.plot(x, preds, label='Predicted')
            ax.set_title(f"{neurons} Neurons, {epochs} Epochs")
            ax.legend()

    fig.tight_layout()
    plt.savefig('16_subplots.png')
    plt.close()


# PLOTS
# Plot for [5, 10, 15, 20] neurons
# each with [10, 50, 100, 500] epochs

neurons = [20]
epochs = [10, 50, 100, 500]


def plots(neurons, epochs):
    for value in neurons:
        for epoch in epochs:
            plot_true_values(value, epoch)


plot_true_values_multiplot(neurons, epochs)
