from typing import Tuple, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from causallib.utils.stat_utils import calc_weighted_standardized_mean_differences as calc_smd


EPSILON = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPSILON, 1 - EPSILON)
    return np.log(p / (1 - p))


def fit_propensity_model(X: pd.DataFrame, t: pd.Series) -> Tuple[Any, np.ndarray]:
    """
    :return
        propensity_score_pipeline: the pipeline used to compute propensity scores
        propensity_score: e = P(T=1|X)
    """
    propensity_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight=None)
    )

    propensity_model.fit(X, t)

    treatment_probabilities = propensity_model.predict_proba(X)[:, 1]
    propensity_scores = np.clip(treatment_probabilities, EPSILON, 1 - EPSILON)

    return propensity_model, propensity_scores


def match_on_propensity(e: np.ndarray, t:np.ndarray, caliper='auto', replace=False):
    z = _logit(e)

    treat_idx = np.where(t == 1)[0]
    ctrl_idx  = np.where(t == 0)[0]
    z_t = z[treat_idx].reshape(-1, 1)
    z_c = z[ctrl_idx].reshape(-1, 1)

    if caliper == 'auto':
        caliper = 0.2 * np.std(z)

    nearest_neighbor = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nearest_neighbor.fit(z_c)
    dists, nn = nearest_neighbor.kneighbors(z_t, return_distance=True)

    used_ctrl = set()
    pairs = []
    for i, (d, j_rel) in enumerate(zip(dists.ravel(), nn.ravel())):
        if d > caliper:
            continue

        c_abs = ctrl_idx[j_rel]

        if not replace and c_abs in used_ctrl:
            continue

        used_ctrl.add(c_abs)
        pairs.append((treat_idx[i], c_abs))

    matched_idx = np.array([k for pr in pairs for k in pr], dtype=int)
    return matched_idx, pairs


def matching_members(X_train: pd.DataFrame, t_train: pd.Series, y_train: pd.Series):
    ps_model, e_train = fit_propensity_model(X_train, t_train)
    matched_idx, pairs = match_on_propensity(
        e=np.array(e_train),
        t=np.array(t_train),
        caliper='auto',
        replace=False
    )

    X_train_m = X_train.iloc[matched_idx]
    y_train_m = y_train.iloc[matched_idx]
    t_train_m = t_train.iloc[matched_idx]


    print(f"Matched train size: {len(X_train_m)} "
          f"(pairs: {len(pairs)}, treated: {sum(1 == t_train_m)}, control: {sum(0 == t_train_m)})")

    return X_train_m, t_train_m, y_train_m, matched_idx, pairs


def validate_matching_quality(
        X_train: pd.DataFrame,
        t_train: pd.Series,
        X_train_m: pd.DataFrame,
        t_train_m: pd.Series
) -> Tuple[pd.Series, pd.Series]:

    X_treat, X_ctrl = X_train[t_train == 1], X_train[t_train == 0]
    X_treat_m, X_ctrl_m = X_train_m[t_train_m == 1], X_train_m[t_train_m == 0]

    smd_before = X_train.columns.to_series().apply(
        lambda c: calc_smd(X_treat[c], X_ctrl[c],
                           np.ones(len(X_treat)), np.ones(len(X_ctrl)),
                           weighted_var=False)
    )

    smd_after = X_train.columns.to_series().apply(
        lambda c: calc_smd(X_treat_m[c], X_ctrl_m[c],
                           np.ones(len(X_treat_m)), np.ones(len(X_ctrl_m)),
                           weighted_var=False)
    )

    print("Mean |SMD| Before:", smd_before.abs().mean().round(3))
    print("Mean |SMD| After :", smd_after.abs().mean().round(3))

    return smd_before, smd_after
