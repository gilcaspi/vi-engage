from typing import Dict

import pandas as pd
from sklearn.metrics import accuracy_score

from base_classes.sklearn.base_evaluator import BaseEvaluator


class AccuracyEvaluator(BaseEvaluator):
    def evaluate(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        return {"accuracy": accuracy_score(y_true, y_pred)}
