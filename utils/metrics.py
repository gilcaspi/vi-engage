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


def _qini_curve(y: np.ndarray, t: np.ndarray, uplift_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build Qini-style cumulative uplift curve on a holdout set.
    y: binary outcome (1=churn, 0=retain)  |  t: treatment indicator (1=treated)  |  uplift_scores: model scores (higher=more benefit from outreach)
    Returns (x, qini) where x is 1..N prefix index and qini is cumulative uplift vs. random baseline proxy.
    """
    order = np.argsort(-uplift_scores)
    y_ord = y[order]
    t_ord = t[order]

    cum_treat = (t_ord == 1).astype(int).cumsum()
    cum_ctrl  = (t_ord == 0).astype(int).cumsum()

    # Outcomes counted as "bad" (churn=1). For retention uplift we still measure the *difference* in bad outcomes.
    cum_y_t = ((t_ord == 1) & (y_ord == 1)).astype(int).cumsum()
    # baseline: global control churn rate among the evaluated prefix
    ctrl_rate = ((t_ord == 0) & (y_ord == 1)).sum() / max((t_ord == 0).sum(), 1)

    qini = cum_y_t - ctrl_rate * cum_treat
    x = np.arange(1, len(qini) + 1)
    return x, qini


def qini_auc(y: np.ndarray, t: np.ndarray, uplift_scores: np.ndarray) -> float:
    x, q = _qini_curve(y, t, uplift_scores)
    # Trapezoidal area under the Qini curve
    return float(np.trapz(q, x))
