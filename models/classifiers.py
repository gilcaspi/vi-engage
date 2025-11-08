import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from base_classes.interfaces import Classifier


class SklearnLogisticRegression(Classifier):
    def __init__(self, max_iter: int = 1000, class_weight: str | dict | None = "balanced") -> None:
        self.model = make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=max_iter, class_weight=class_weight),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnLogisticRegression":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class PyTorchClassifier(Classifier):
    def __init__(self, net: object, optimizer_ctor: object, loss_fn: object,
                 epochs: int = 10, batch_size: int = 256, device: str = "cpu") -> None:
        self.net = net
        self.optimizer_ctor = optimizer_ctor
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PyTorchClassifier":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.stack([1 - 0.5 * np.ones(len(X)), 0.5 * np.ones(len(X))], axis=1)
