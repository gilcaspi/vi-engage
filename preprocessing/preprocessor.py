from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from base_classes.interfaces import Preprocessor


class SklearnTabularPreprocessor(Preprocessor):
    def __init__(self, num_cols: List[str], cat_cols: List[str], scale: bool = True) -> None:
        num_steps: list[Tuple[str, object]] = [("impute", SimpleImputer(strategy="median"))]
        if scale:
            num_steps.append(("scale", StandardScaler()))
        num_pipe: Pipeline = Pipeline(steps=num_steps)

        cat_pipe: Pipeline = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ])

        self.ct: ColumnTransformer = ColumnTransformer(
            transformers=[
                ("num", num_pipe, num_cols),
                ("cat", cat_pipe, cat_cols),
            ],
            remainder="drop",
        )

    def fit(self, X: pd.DataFrame) -> "SklearnTabularPreprocessor":
        self.ct.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.ct.transform(X)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.ct.fit_transform(X)
