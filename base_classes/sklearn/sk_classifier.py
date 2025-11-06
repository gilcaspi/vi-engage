from abc import abstractmethod
from typing import Optional

from sklearn.base import ClassifierMixin, BaseEstimator

from base_classes.sklearn.base_step import BaseStep, XType, YType


class SKClassifier(BaseEstimator, ClassifierMixin, BaseStep[XType, YType, YType]):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def fit(self, x: XType, y: YType) -> "SKClassifier":
        ...

    @abstractmethod
    def predict(self, x: XType) -> YType:
        ...

    def run(self, x: XType, y: Optional[YType] = None) -> YType:
        self.fit(x, y)
        return self.predict(x)
