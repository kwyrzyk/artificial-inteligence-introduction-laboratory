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

@dataclass
class TestParams:
    domain: tuple
    dim: int
    samples_density: int
    test_size: int
    activation_fun: ActivationFunc
    epochs: int
    learning_rate: float


def investigated_function(x_vec):
    """Badana funkcja"""
    x = x_vec[:, 0]
    result = x**2 * np.sin(x) + 100 * np.sin(x) * np.cos(x)
    return result.reshape(-1, 1)

# def investigated_function(x_vec):
#     """Badana funkcja"""
#     x = x_vec[:, 0]  # Wyciągnięcie pierwszej kolumny (wektor wejściowy)
#     result = np.sin(x * np.sqrt(7)) + np.cos(2 * x)  # Użycie funkcji NumPy
#     return result.reshape(-1, 1)  # Dopasowanie wymiarów wyjściowych




def plot_results(x_test, y_test, y_pred, domain, layer_sizes, title="Model vs. Investigated Function"):
    """Funkcja rysująca wykres porównawczy i zapisująca go do pliku"""
    plt.figure(figsize=(10, 6))
    
    # Rysowanie badanej funkcji
    x_domain = np.linspace(domain[0], domain[1], 500).reshape(-1, 1)
    y_investigated = investigated_function(x_domain)
    plt.plot(x_domain, y_investigated, label="Investigated Function", color="blue", linewidth=2)
    
    # Rysowanie punktów testowych
    plt.scatter(x_test, y_test, label="Test Points", color="green", alpha=0.7)
    
    # Rysowanie przewidywań modelu
    plt.scatter(x_test, y_pred, label="Model Predictions", color="red", alpha=0.7)
    
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Generowanie nazwy pliku na podstawie rozmiarów warstw
    filename = f"mlp_plot_layers_{'_'.join(map(str, layer_sizes))}.png"
    plt.savefig(filename)
    plt.close()  # Zamknięcie wykresu, aby uniknąć problemów z pamięcią
    print(f"Plot saved as {filename}")


def test_MLP(layer_sizes_list, params):
    """Testowanie modelu MLP"""
    gen = DataGenerator()
    x_train, y_train = gen.get_train_data(params.domain, params.dim, investigated_function, params.samples_density)
    x_test, y_test = gen.get_test_data(params.domain, params.dim, investigated_function, params.test_size)

    # Skalowanie danych
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    x_scaler.fit(x_train)
    y_scaler.fit(y_train)

    norm_x_train = x_scaler.transform(x_train)
    norm_x_test = x_scaler.transform(x_test)
    norm_y_train = y_scaler.transform(y_train)
    norm_y_test = y_scaler.transform(y_test)

    # Testowanie dla różnych rozmiarów warstw
    for layer_sizes in layer_sizes_list:
        mlp = MLP(layer_sizes, params.activation_fun)
        mlp.train_gradient(norm_x_train, norm_y_train, params.epochs, params.learning_rate)

        y_pred = mlp.forward(norm_x_test)
        denorm_y_pred = y_scaler.inverse_transform(y_pred)

        accuracy = mse(y_test, denorm_y_pred)
        print(f"Layer sizes: {layer_sizes}, MSE: {accuracy}")
        # Rysowanie wyników i zapisywanie wykresu
        plot_results(x_test, y_test, denorm_y_pred, params.domain, layer_sizes, title=f"Layer sizes: {layer_sizes}")


if __name__ == "__main__":
    # Parametry testowe
    test_params = TestParams(
        domain=(-10, 10),
        dim=1,
        samples_density=100,
        test_size=100,
        activation_fun=Tanh(),  # Użycie funkcji Sigmoid
        epochs=10000,  # Liczba epok
        learning_rate=0.001  # Współczynnik uczenia
    )

    layer_sizes_list = [
        [1, 10, 10, 1],
    ]

    # Uruchamianie testów
    test_MLP(layer_sizes_list, test_params)
