from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

from base_classes.interfaces import Preprocessor, Classifier, Evaluator, UpliftModel, Matcher
from base_classes.result import SupervisedMetrics, UpliftMetrics


class SupervisedPipeline:
    def __init__(self, pre: Preprocessor, clf: Classifier, evaluator: Evaluator) -> None:
        self.pre: Preprocessor = pre
        self.clf: Classifier = clf
        self.evaluator: Evaluator = evaluator
        self._fitted: bool = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SupervisedPipeline":
        Xp: np.ndarray = self.pre.fit_transform(X)
        self.clf.fit(Xp, y.to_numpy())
        self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xp: np.ndarray = self.pre.transform(X)
        return self.clf.predict_proba(Xp)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> SupervisedMetrics:
        p: np.ndarray = self.predict_proba(X)[:, 1]
        m: Dict[str, Any] = self.evaluator.evaluate(y_true=y.to_numpy(), y_score=p)
        return SupervisedMetrics(auc=float(m["auc"]))


class UpliftPipeline:
    def __init__(self, pre: Preprocessor, model: UpliftModel, matcher: Optional[Matcher], evaluator: Evaluator) -> None:
        self.pre: Preprocessor = pre
        self.model: UpliftModel = model
        self.matcher: Optional[Matcher] = matcher
        self.evaluator: Evaluator = evaluator
        self.match_artifacts: Optional[Dict[str, Any]] = None
        self._fitted: bool = False

    def fit(self, X: pd.DataFrame, y: pd.Series, t: pd.Series) -> "UpliftPipeline":
        if self.matcher:
            self.match_artifacts = self.matcher.match(X, t, y)
            Xm: pd.DataFrame = self.match_artifacts["X"]
            ym: pd.Series = self.match_artifacts["y"]
            tm: pd.Series = self.match_artifacts["t"]
        else:
            Xm, ym, tm = X, y, t
        Xp: np.ndarray = self.pre.fit_transform(Xm)
        self.model.fit(Xp, ym.to_numpy(), tm.to_numpy())
        self._fitted = True
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        Xp: np.ndarray = self.pre.transform(X)
        return self.model.predict_uplift(Xp)

    def evaluate(self, X: pd.DataFrame, y: pd.Series, pairs=None, sign: int = -1) -> UpliftMetrics:
        u: np.ndarray = self.predict_uplift(X) * sign
        m: Dict[str, Any] = self.evaluator.evaluate(y_true=y.to_numpy(), y_score=u, pairs=pairs)
        pairs_count: int = int(len(pairs) if pairs is not None else 0)
        return UpliftMetrics(c_for_benefit=float(m["c_for_benefit"]), pairs=pairs_count)
