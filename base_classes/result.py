from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SupervisedMetrics:
    auc: float


@dataclass(frozen=True)
class UpliftMetrics:
    c_for_benefit: float
    pairs: int


class SupervisedResult:
    def __init__(self, metrics: SupervisedMetrics, proba_fn: Callable[[pd.DataFrame], np.ndarray]) -> None:
        self.metrics: SupervisedMetrics = metrics
        self._proba_fn: Callable[[pd.DataFrame], np.ndarray] = proba_fn

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._proba_fn(X)


class UpliftResult:
    def __init__(self, metrics: UpliftMetrics, uplift_fn: Callable[[pd.DataFrame], np.ndarray]) -> None:
        self.metrics: UpliftMetrics = metrics
        self._uplift_fn: Callable[[pd.DataFrame], np.ndarray] = uplift_fn

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        return self._uplift_fn(X)
