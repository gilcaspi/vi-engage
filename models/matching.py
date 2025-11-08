from __future__ import annotations
from typing import Dict, Any, Union
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

from base_classes.interfaces import Matcher, ProbabilisticEstimator


class PropensityNearestMatcher(Matcher):
    def __init__(
            self,
            propensity_model: ProbabilisticEstimator,
            caliper: Union[str,  float] = "auto",
            replace: bool = False,
            k: int = 1
    ) -> None:
        self.pm = propensity_model
        self.caliper = caliper
        self.replace = replace
        self.k = k

    def match(self, X: pd.DataFrame, t: pd.Series, y: pd.Series) -> Dict[str, Any]:
        e: np.ndarray = self.pm.fit(X, t).predict_proba(X)[:, 1]
        idx_t: np.ndarray = np.where(t.values == 1)[0]
        idx_c: np.ndarray = np.where(t.values == 0)[0]
        e_t: np.ndarray = e[idx_t].reshape(-1, 1)
        e_c: np.ndarray = e[idx_c].reshape(-1, 1)
        nn = NearestNeighbors(n_neighbors=self.k).fit(e_c)
        dist, neigh = nn.kneighbors(e_t, return_distance=True)
        cap: float = 0.2 * float(np.std(e)) if self.caliper == "auto" else float(self.caliper)
        mask: np.ndarray = dist[:, 0] <= cap
        t_sel: np.ndarray = idx_t[mask]
        c_sel: np.ndarray = idx_c[neigh[mask, 0]]
        sel: np.ndarray = np.concatenate([t_sel, c_sel])
        Xm: pd.DataFrame = X.iloc[sel].reset_index(drop=True)
        ym: pd.Series = y.iloc[sel].reset_index(drop=True)
        tm: pd.Series = t.iloc[sel].reset_index(drop=True)
        pairs = list(zip(t_sel.tolist(), c_sel.tolist()))
        return {"X": Xm, "y": ym, "t": tm, "pairs": pairs, "idx_matched": sel}
