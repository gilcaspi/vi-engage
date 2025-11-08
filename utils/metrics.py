from typing import List, Tuple

import numpy as np
import pandas as pd


def c_for_benefit_from_pairs(y: pd.Series, uplift: np.ndarray, pairs: List[Tuple[int, int]]) -> float:
    y = np.asarray(y)
    uplift = np.asarray(uplift)

    concordant = 0
    discordant = 0
    ties = 0

    for i, j in pairs:
        uplift_i, uplift_j = uplift[i], uplift[j]
        if np.isnan(uplift_i) or np.isnan(uplift_j):
            continue  # unmatched / missing predictions

        # Observed benefit = control outcome - treated outcome (on churn labels)
        # Positive delta_obs means treat helped (control churned more than treated)
        observed_benefit = y[j] - y[i]

        # Predicted relative benefit order between i and j
        predicted_benefit = uplift_i - uplift_j

        if observed_benefit > 0 and predicted_benefit > 0:
            concordant += 1
        elif observed_benefit < 0 and predicted_benefit < 0:
            concordant += 1
        elif predicted_benefit == 0 or observed_benefit == 0:
            ties += 1
        else:
            discordant += 1

    total = concordant + discordant + ties
    if total == 0:
        return np.nan

    # Count ties as half, common in concordance-style indices
    return (concordant + 0.5 * ties) / total
