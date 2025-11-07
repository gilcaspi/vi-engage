import numpy as np
import pandas as pd


def c_for_benefit_from_pairs(y: pd.Series, uplift: np.ndarray, pairs: list[tuple[int,int]]) -> float:
    if len(pairs) == 0:
        return np.nan
    concordant = 0.0
    valid = 0
    for pair in pairs:
        i, j = pair[:2]
        # observed benefit: lower churn under treatment vs control is good ⇒ y_j - y_i
        delta_obs  = y.iloc[j] - y.iloc[i]
        delta_pred = uplift[i] - uplift[j]
        if delta_obs == 0 and delta_pred == 0:
            concordant += 0.5
            valid += 1
        elif delta_obs == 0 or delta_pred == 0:
            concordant += 0.5
            valid += 1
        else:
            valid += 1
            concordant += 1.0 if (delta_obs * delta_pred) > 0 else 0.0
    return concordant / valid if valid > 0 else np.nan