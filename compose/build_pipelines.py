import pandas as pd
from sklearn.linear_model import LogisticRegression

from evaluation.evaluators import AucEvaluator, CForBenefitEvaluator
from models.classifiers import SklearnLogisticRegression
from models.matching import PropensityNearestMatcher
from models.pipeline import SupervisedPipeline, UpliftPipeline
from models.uplift import TwoModelsSklearn
from preprocessing.preprocessor import SklearnTabularPreprocessor


def build_supervised_pipeline(X_train: pd.DataFrame) -> SupervisedPipeline:
    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]
    pre = SklearnTabularPreprocessor(num_cols=num_cols, cat_cols=cat_cols, scale=True)
    clf = SklearnLogisticRegression()
    ev = AucEvaluator()
    return SupervisedPipeline(pre, clf, ev)

def build_uplift_pipeline(X_train: pd.DataFrame) -> UpliftPipeline:
    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]
    pre = SklearnTabularPreprocessor(num_cols=num_cols, cat_cols=cat_cols, scale=True)
    pm = LogisticRegression(max_iter=1000, class_weight="balanced")
    matcher = PropensityNearestMatcher(propensity_model=pm, caliper="auto", replace=False, k=1)
    model = TwoModelsSklearn()
    ev = CForBenefitEvaluator()
    return UpliftPipeline(pre, model, matcher, ev)
