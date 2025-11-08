from typing import Dict, Any
import numpy as np
from sklearn.metrics import roc_auc_score

from base_classes.interfaces import Evaluator
from utils.metrics import c_for_benefit_from_pairs


class AucEvaluator(Evaluator):
    @staticmethod
    def score(y_true: np.ndarray, y_score: np.ndarray, **kwargs: Any) -> Dict[str, float]:
        return {"auc": float(roc_auc_score(y_true, y_score))}


class CForBenefitEvaluator(Evaluator):
    @staticmethod
    def score(y_true: np.ndarray, y_score: np.ndarray, **kwargs: Any) -> Dict[str, float]:
        pairs = kwargs.get("pairs")
        val: float = float(c_for_benefit_from_pairs(y=y_true, uplift=y_score, pairs=pairs))
        return {"c_for_benefit": val}
