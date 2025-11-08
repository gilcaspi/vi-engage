from typing import Optional

import numpy as np
from sklift.models import TwoModels
from xgboost import XGBClassifier

from base_classes.interfaces import UpliftModel


class TwoModelsSklearn(UpliftModel):
    def __init__(
            self,
            treatment_estimator: Optional[object] = None,
            control_estimator: Optional[object] = None,
            method: str = "vanilla"
    ) -> None:
        if treatment_estimator is None:
            treatment_estimator = XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=42, n_jobs=1, eval_metric="logloss"
            )

        if control_estimator is None:
            control_estimator = XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=42, n_jobs=1, eval_metric="logloss"
            )

        self.model: TwoModels = TwoModels(
            estimator_trmnt=treatment_estimator,
            estimator_ctrl=control_estimator,
            method=method
        )

    def fit(self, X: np.ndarray, y: np.ndarray, t: np.ndarray) -> "TwoModelsSklearn":
        self.model.fit(X, 1 - y, treatment=t)
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
