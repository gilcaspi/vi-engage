import pandas as pd

from base_classes.result import SupervisedResult
from models.pipeline import SupervisedPipeline


class SupervisedRunner:
    def __init__(self, pipeline: SupervisedPipeline) -> None:
        self.pipeline: SupervisedPipeline = pipeline

    def fit_eval(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_test: pd.DataFrame,
            y_test: pd.Series
    ) -> SupervisedResult:
        self.pipeline.fit(X_train, y_train)
        metrics = self.pipeline.evaluate(X_test, y_test)
        return SupervisedResult(metrics=metrics, proba_fn=self.pipeline.predict_proba)
