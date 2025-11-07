import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def calculate_slope(values: pd.Series) -> float:
    n = len(values)
    if n < 2:
        return 0.0

    model = LinearRegression()

    X = np.arange(n).reshape(-1, 1)
    y = values.values.reshape(-1, 1)
    model.fit(X, y)

    slope = model.coef_[0][0]

    return slope