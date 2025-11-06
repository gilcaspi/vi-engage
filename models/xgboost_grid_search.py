from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from base_classes.sklearn.base_step import BaseStep


BEST_MODEL_PARAMS = {
    'colsample_bytree': 1,
    'gamma': 0.1,
    'learning_rate': 0.05,
    'max_depth': None,
    'min_child_weight': 1,
    'n_estimators': 150,
    'scale_pos_weight': np.float64(0.6036036036036037),
    'subsample': 1
}


class XGBGridSearchStep(BaseStep[pd.DataFrame, pd.Series, Any]):
    def __init__(self):
        self.best_model = None
        self.best_params = None
        self.best_score = None
        self.cv_results = None

    @property
    def name(self) -> str:
        return "XGBGridSearch"

    def run(self, x: pd.DataFrame, y: Optional[pd.Series] = None) -> Any:
        pos_ratio = y.value_counts(normalize=False).get(1, 0) / max(y.value_counts(normalize=False).get(0, 1), 1)

        xgb_grid = {
            'n_estimators': [50, 100, 150, 200, None],
            'max_depth': [3, 4, 5, None],
            'learning_rate': [0.01, 0.05, 0.1, 0.2, None],
            'subsample': [0.6, 0.7, 0.8, 1, None],
            'colsample_bytree': [0.7, 1, None],
            'scale_pos_weight': [1, pos_ratio, None],
            'gamma': [0, 0.1, 0.2, 1, 5, None],
            'min_child_weight': [1, 2, 3, None],
        }

        stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        search = GridSearchCV(
            XGBClassifier(eval_metric='auc', random_state=42),
            xgb_grid,
            cv=stratified_cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=3,
        )

        search.fit(x, y)

        self.best_model = search.best_estimator_
        self.best_params = search.best_params_
        self.best_score = search.best_score_
        self.cv_results = search.cv_results_
        return self.best_model
