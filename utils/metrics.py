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
        ui, uj = uplift[i], uplift[j]
        if np.isnan(ui) or np.isnan(uj):
            continue  # unmatched / missing predictions

        # Observed benefit: control outcome - treated outcome (on churn labels)
        # Positive delta_obs means treat helped (control churned more than treated)
        delta_obs = y[j] - y[i]

        # Predicted relative benefit order between i and j
        delta_hat = ui - uj

        if delta_obs > 0 and delta_hat > 0:
            concordant += 1
        elif delta_obs < 0 and delta_hat < 0:
            concordant += 1
        elif delta_hat == 0 or delta_obs == 0:
            ties += 1
        else:
            discordant += 1

    total = concordant + discordant + ties
    if total == 0:
        return np.nan
    # Count ties as half, common in concordance-style indices
    return (concordant + 0.5 * ties) / total