from typing import Protocol
import pandas as pd
import numpy as np


class Preprocessor(Protocol):
    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        ...

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        ...


class Classifier(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Classifier":
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ...


class UpliftModel(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray, t: np.ndarray) -> "UpliftModel":
        ...

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        ...


class Matcher(Protocol):
    def match(self, X: pd.DataFrame, t: pd.Series, y: pd.Series) -> dict:
        ...


class Evaluator(Protocol):
    def evaluate(self, y_true: np.ndarray, y_score: np.ndarray, **kwargs) -> dict:
        ...
