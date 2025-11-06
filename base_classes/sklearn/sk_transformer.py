from typing import Optional

from sklearn.base import BaseEstimator, TransformerMixin

from base_classes.sklearn.base_step import BaseStep, XType, YType


class SKTransformer(BaseEstimator, TransformerMixin, BaseStep[XType, YType, XType]):
    @property
    def name(self) -> str:
        return self.__class__.__name__

    def fit(self, x: XType, y: Optional[YType] = None) -> "SKTransformer":
        return self

    def transform(self, x: XType) -> XType:
        return x

    def run(self, x: XType, y: Optional[YType] = None) -> XType:
        self.fit(x, y)
        return self.transform(x)