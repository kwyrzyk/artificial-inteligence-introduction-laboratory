from svm import SVM, SVMFitParams, SVMPredictParams, KernelType
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import PercentFormatter
matplotlib.use('Agg')


def comparison_test(X, y, C_values, split_count):
    C_values_count = len(C_values)
    linear_results = np.zeros(C_values_count)
    rbf_results = np.zeros(C_values_count)
    svm = SVM()
    
    for _ in tqdm(range(split_count), desc="Split Iterations"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        fit_params = SVMFitParams(X_train, y_train)
        predict_params = SVMPredictParams(X_test, None, None, X_train, y_train)

        for id, C in tqdm(enumerate(C_values), desc="C values", total=len(C_values)):
            fit_params.C = C

            fit_params.kernel = KernelType.LINEAR_KERNEL
            predict_params.kernel = KernelType.LINEAR_KERNEL

            test_result = test(svm, fit_params, predict_params, y_test)
            linear_results[id] += test_result

            fit_params.kernel = KernelType.RBF_KERNEL
            predict_params.kernel = KernelType.RBF_KERNEL

            test_result = test(svm, fit_params, predict_params, y_test)
            rbf_results[id] += test_result

    linear_results /= split_count
    rbf_results /= split_count
    return linear_results, rbf_results


def sigma_test(X, y, sig_values, C_value, split_count):
    sig_values_count = len(sig_values)
    results = np.zeros(sig_values_count)
    svm = SVM()
    
    for _ in tqdm(range(split_count), desc="Split Iterations"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        fit_params = SVMFitParams(X_train, y_train, C_value)
        fit_params.kernel = KernelType.RBF_KERNEL
        predict_params = SVMPredictParams(X_test, None, None, X_train, y_train, KernelType.RBF_KERNEL)

        for id, sig in tqdm(enumerate(sig_values), desc="Sigma values", total=len(sig_values)):
            fit_params.kernel_params = [sig]
            predict_params.kernel_params = [sig]
            test_result = test(svm, fit_params, predict_params, y_test)
            results[id] += test_result

    results /= split_count
    return results


def test(svm, fit_params, predict_params, y_test):
    alpha, b = svm.fit(fit_params)
    predict_params.alpha = alpha
    predict_params.b = b
    predictions = svm.predict(predict_params)
    correct_predictions = np.sum(predictions == y_test)
    test_size = len(y_test)
    accuracy = correct_predictions / test_size
    return accuracy

def C_comparison_plot(C_values, linear_results, rbf_results):
    plt.figure(figsize=(10, 5))
    plt.plot(C_values, linear_results, label="Linear Kernel", marker='o', linestyle='-')
    plt.plot(C_values, rbf_results, label="RBF Kernel", marker='o', linestyle='-')
    plt.xscale('log')  # Skala logarytmiczna dla osi X
    plt.xlabel('Wartość współczynnika C')
    plt.ylabel('Dokładność')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.title('Dokładność algorytmu SVM, dla jądra liniowego i jądra RBF w zależności od wspołczynnika C')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig("plots.png")

def sig_comparison_plot(sig_values, results):
    plt.figure(figsize=(10, 5))
    plt.plot(sig_values, results, label="RBF Kernel", marker='o', linestyle='-')
    plt.xscale('log')  
    plt.xlabel('Wartość parametru sigma')
    plt.ylabel('Dokładność')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.title('Dokładność algorytmu SVM, dla jądra RBF w zależności od parametru sigma')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig("sigma.png")

if __name__ == "__main__":
    DATA_SIZE = 200
    SPLIT_COUNT = 5

    X_file = 'wine_quality_features.csv'
    y_file = 'wine_quality_targets.csv'

    X = pd.read_csv(X_file).head(DATA_SIZE).values
    y = pd.read_csv(y_file).head(DATA_SIZE).values.flatten()

    X = (X - X.mean(axis=0)) / X.std(axis=0)

    y = np.where(y < 6, -1, 1)

    C_values = np.logspace(-4, 4, 9)
    sig_values = np.logspace(-4, 4, 9)
    C_VALUE = 1e1
    linear_results, rbf_results = comparison_test(X, y, C_values, SPLIT_COUNT)
    C_comparison_plot(C_values, linear_results, rbf_results)
    results = sigma_test(X, y, sig_values, C_VALUE, SPLIT_COUNT)
    sig_comparison_plot(sig_values, results)

    