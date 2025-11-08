from typing import List, Optional, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluation.evaluators import CForBenefitEvaluator
from models.matching import PropensityNearestMatcher
from models.pipeline import UpliftPipeline
from models.uplift import TwoModelsSklearn
from preprocessing.preprocessor import SklearnTabularPreprocessor


def build_supervised_pipeline(
        estimator: Optional[object] = None,
) -> Pipeline:
    if estimator is None:
        estimator = LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("logisticregression", estimator),
        ]
    )


def build_uplift_pipeline(
        num_cols: Optional[List[str]] = None,
        cat_cols: Optional[List[str]] = None,
) -> UpliftPipeline:
    num_cols, cat_cols = _validate_num_cols_and_cat_cols(num_cols, cat_cols)

    pre = SklearnTabularPreprocessor(num_cols=num_cols, cat_cols=cat_cols, scale=True)
    pm = LogisticRegression(max_iter=1000, class_weight="balanced")
    matcher = PropensityNearestMatcher(propensity_model=pm, caliper="auto", replace=False, k=1)
    model = TwoModelsSklearn()
    ev = CForBenefitEvaluator()
    return UpliftPipeline(pre, model, matcher, ev)


def _validate_num_cols_and_cat_cols(
        num_cols: Optional[List[str]],
        cat_cols: Optional[List[str]]
) -> Tuple[List[str], List[str]]:
    if num_cols is None and cat_cols is None:
        raise ValueError("At least one of num_cols or cat_cols must be provided.")

    if num_cols is None:
        num_cols = []

    if cat_cols is None:
        cat_cols = []

    return num_cols, cat_cols
