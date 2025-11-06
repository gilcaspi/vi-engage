from abc import ABC, abstractmethod
from typing import Dict, Any

from base_classes.sklearn.base_step import YType


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, y_true: YType, y_pred: YType) -> Dict[str, Any]:
        """Returns metrics like accuracy, RMSE, etc."""
