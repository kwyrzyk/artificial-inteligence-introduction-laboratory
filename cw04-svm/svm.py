import numpy as np
from enum import Enum
from dataclasses import dataclass, field


class KernelType(Enum):
    LINEAR_KERNEL = 1
    RBF_KERNEL = 2


@dataclass
class SVMFitParams:
    X: np.ndarray
    y: np.ndarray
    C: float = 1.0
    learn_rate: float = 0.01
    kernel: KernelType = KernelType.LINEAR_KERNEL
    kernel_params: list = field(default_factory=list)
    max_iter: int = 1000
    eps: float = 1e-5


@dataclass
class SVMPredictParams:
    X: np.ndarray
    alpha: np.ndarray
    b: float
    X_train: np.ndarray
    y_train: np.ndarray
    kernel: KernelType = KernelType.LINEAR_KERNEL
    kernel_params: list = field(default_factory=list)


class SVM:
    def __init__(self):
        pass

    def linear_kernel(self, u, v, params):
        return np.dot(u, v)

    def rbf_kernel(self, u, v, params):
        sig = params[0]
        squared_distance = np.sum((u - v) ** 2)
        return np.exp(-squared_distance / (2 * sig ** 2))

    def gradient(self, X, y, alpha, kernel, kernel_params):
        N = len(X)
        grad = np.zeros(N)
        for i in range(N):
            grad[i] = 1 - y[i] * np.sum(alpha * y * np.array([kernel(X[i], X[j], kernel_params) for j in range(N)]))
        return grad

    def fit(self, params: SVMFitParams):
        X = params.X
        y = params.y
        C = params.C
        learn_rate = params.learn_rate
        kernel = params.kernel
        kernel_params = params.kernel_params
        max_iter = params.max_iter
        eps = params.eps

        if kernel == KernelType.LINEAR_KERNEL:
            kernel_fun = self.linear_kernel
        elif kernel == KernelType.RBF_KERNEL:
            kernel_fun = self.rbf_kernel

        N = X.shape[0]
        alpha = np.random.uniform(0, C, N)
        b = 0

        for _ in range(max_iter):
            grad = self.gradient(X, y, alpha, kernel_fun, kernel_params)
            if np.all(np.abs(grad) < eps):
                break
            alpha = alpha + learn_rate * grad
            alpha = np.clip(alpha, 0, C)

        support_vectors = np.where(alpha > 0)[0]
        b = np.mean([
            y[i] - np.sum(alpha * y * np.array([kernel_fun(X[i], X[j], kernel_params) for j in range(N)]))
            for i in support_vectors
        ])

        return alpha, b

    def predict(self, params: SVMPredictParams):
        X = params.X
        alpha = params.alpha
        b = params.b
        X_train = params.X_train
        y_train = params.y_train
        kernel = params.kernel
        kernel_params = params.kernel_params

        if kernel == KernelType.LINEAR_KERNEL:
            kernel_fun = self.linear_kernel
        elif kernel == KernelType.RBF_KERNEL:
            kernel_fun = self.rbf_kernel

        predictions = []
        for x in X:
            decision = np.sum(alpha * y_train * np.array([kernel_fun(x, x_train, kernel_params) for x_train in X_train])) + b
            predictions.append(np.sign(decision))

        return np.array(predictions)
