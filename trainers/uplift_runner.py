from typing import Optional

import pandas as pd

from base_classes.result import UpliftResult
from models.pipeline import UpliftPipeline


class UpliftRunner:
    def __init__(self, pipeline: UpliftPipeline) -> None:
        self.pipeline: UpliftPipeline = pipeline

    def fit_eval(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            t_train: pd.Series,
            X_test: pd.DataFrame,
            y_test: pd.Series,
            pairs_test: Optional = None,
            sign: int = -1
    ) -> UpliftResult:
        self.pipeline.fit(X_train, y_train, t_train)
        metrics = self.pipeline.evaluate(X_test, y_test, pairs=pairs_test, sign=sign)
        return UpliftResult(metrics=metrics, uplift_fn=self.pipeline.predict_uplift)
