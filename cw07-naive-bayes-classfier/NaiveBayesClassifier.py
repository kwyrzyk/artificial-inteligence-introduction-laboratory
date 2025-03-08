import numpy as np
from scipy.stats import norm
from typing import Protocol

class PDFProtocol(Protocol):
    def __call__(self, x: np.ndarray, X: np.ndarray) -> np.ndarray:
        pass


class GaussianPDF(PDFProtocol):
    def __call__(self, x: np.ndarray, X: np.ndarray) -> np.ndarray:
        mean = np.mean(X, axis=0)
        std_dev = np.std(X, axis=0)
        return norm.pdf(x, loc=mean, scale=std_dev)


class NaiveBayesClassifier:
    def __init__(self, pdf: PDFProtocol = None):
        self.priors = {}
        self.pdf = pdf if pdf is not None else GaussianPDF()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y.flatten()

        self.classes = np.unique(y)

        for cls in self.classes:
            X_cls = X[y == cls]
            self.priors[cls] = X_cls.shape[0] / X.shape[0]

    def _class_likelihood(self, x: np.ndarray, cls: int) -> float:
        log_likelihoods = np.log(self.pdf(x, self.X[self.y == cls]))
        return np.sum(log_likelihoods, axis=1) + np.log(self.priors[cls])

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        posteriors = [self._class_likelihood(X, cls) for cls in self.classes]
        predictions.append(self.classes[np.argmax(posteriors, axis=0)])
        predictions = np.array(predictions)
        return predictions.T
