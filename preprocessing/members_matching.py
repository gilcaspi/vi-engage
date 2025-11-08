import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from causallib.utils.stat_utils import calc_weighted_standardized_mean_differences as calc_smd


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def fit_propensity_model(X, t):
    """Return fitted pipeline and propensity scores e(x)=P(T=1|X)."""
    ps_pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight=None)
    )
    ps_pipeline.fit(X, t)
    e = ps_pipeline.predict_proba(X)[:, 1]
    e = np.clip(e, 1e-6, 1 - 1e-6)
    return ps_pipeline, e


def match_on_propensity(e, t, caliper='auto', replace=False):
    """
    1:1 greedy nearest-neighbor matching on the logit of propensity.
    Returns indices of matched rows (treated+control), and list of pairs (ti, ci).
    """
    # work on logits (Rosenbaum–Rubin caliper rule)
    z = _logit(e)

    treat_idx = np.where(t == 1)[0]
    ctrl_idx  = np.where(t == 0)[0]
    z_t = z[treat_idx].reshape(-1, 1)
    z_c = z[ctrl_idx].reshape(-1, 1)

    # auto-caliper: 0.2 * SD(logit)
    if caliper == 'auto':
        caliper = 0.2 * np.std(z)

    nbrs = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nbrs.fit(z_c)
    dists, nn = nbrs.kneighbors(z_t, return_distance=True)

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


def matching_members(X_train, t_train, y_train):
    ps_model, e_train = fit_propensity_model(X_train, t_train)
    matched_idx, pairs = match_on_propensity(
        e=np.array(e_train),
        t=np.array(t_train),
        caliper='auto',
        replace=False
    )

    # Build matched TRAIN subset
    X_train_m = X_train.iloc[matched_idx]
    y_train_m = y_train.iloc[matched_idx]
    t_train_m = t_train.iloc[matched_idx]


    print(f"Matched train size: {len(X_train_m)} "
          f"(pairs: {len(pairs)}, treated: {sum(t_train_m == 1)}, control: {sum(t_train_m == 0)})")

    return X_train_m, t_train_m, y_train_m, matched_idx, pairs


def validate_matching_quality(X_train, t_train, X_train_m, t_train_m):
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