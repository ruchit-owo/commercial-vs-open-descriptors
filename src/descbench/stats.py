"""Scoring, bootstrap CIs over molecules, and the paired cross-endpoint tests.

Scores carry a molecule-level bootstrap interval. The cross-endpoint comparison
uses the per-endpoint alvaDesc-minus-open difference, summarised with a median
interval and a Wilcoxon signed-rank test (low power at this sample size).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon
from sklearn.metrics import accuracy_score, roc_auc_score, root_mean_squared_error


def score(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    if metric == "roc_auc":
        return float(roc_auc_score(y_true, y_pred))
    if metric == "spearman":
        return float(spearmanr(y_true, y_pred).statistic)
    if metric == "rmse":
        return float(root_mean_squared_error(y_true, y_pred))
    if metric == "accuracy":
        return float(accuracy_score(y_true, (y_pred >= 0.5).astype(int)))
    raise ValueError(f"unknown metric {metric!r}")


def secondary_metric(task: str) -> str:
    return "accuracy" if task == "classification" else "rmse"


def primary_metric(task: str) -> str:
    return "roc_auc" if task == "classification" else "spearman"


def trivial_baseline(y_true: np.ndarray, task: str) -> float:
    """The score a no-information model reaches on the primary metric: 0.5 AUC for
    classification, and for regression a mean predictor gives Spearman rho of 0."""
    return 0.5 if task == "classification" else 0.0


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, metric: str,
                 n_resamples: int, ci: float, seed: int) -> tuple[float, float]:
    """Percentile bootstrap over molecules. Resamples that become degenerate for
    a metric (a single class drawn, for AUC) are skipped rather than counted."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        if metric == "roc_auc" and len(np.unique(yt)) < 2:
            continue
        try:
            stats.append(score(yt, yp, metric))
        except ValueError:
            continue
    lo = float(np.percentile(stats, 100 * (1 - ci) / 2))
    hi = float(np.percentile(stats, 100 * (1 + ci) / 2))
    return lo, hi


def paired_differences(main: pd.DataFrame, model: str = "linear") -> pd.DataFrame:
    """Per-endpoint difference between alvaDesc and each open representation on the
    primary metric, signed so positive means alvaDesc did better."""
    rows = []
    sub = main[main["model"] == model]
    for endpoint, grp in sub.groupby("endpoint"):
        by_rep = grp.set_index("representation")["primary_score"]
        if "alvadesc" not in by_rep.index:
            continue
        alva = by_rep["alvadesc"]
        for rep, value in by_rep.items():
            if rep == "alvadesc":
                continue
            rows.append({"endpoint": endpoint, "open_representation": rep,
                         "alvadesc": alva, "open": value, "diff": alva - value})
    return pd.DataFrame(rows)


def paired_summary(diffs: pd.DataFrame, n_resamples: int, ci: float,
                   seed: int) -> pd.DataFrame:
    """For each open representation, summarise the paired differences across
    endpoints: median difference with a bootstrap interval and a Wilcoxon
    signed-rank test."""
    rng = np.random.default_rng(seed)
    rows = []
    for rep, grp in diffs.groupby("open_representation"):
        d = grp["diff"].to_numpy()
        medians = [np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(n_resamples)]
        lo = float(np.percentile(medians, 100 * (1 - ci) / 2))
        hi = float(np.percentile(medians, 100 * (1 + ci) / 2))
        # Wilcoxon is undefined when all differences are zero; guard for it.
        if np.allclose(d, 0):
            pvalue = 1.0
        else:
            pvalue = float(wilcoxon(d, zero_method="wilcox").pvalue)
        rows.append({
            "open_representation": rep, "n_endpoints": len(d),
            "median_diff": float(np.median(d)), "ci_low": lo, "ci_high": hi,
            "wilcoxon_p": pvalue, "n_alvadesc_better": int((d > 0).sum()),
        })
    return pd.DataFrame(rows)


def mean_rank_table(main: pd.DataFrame, model: str = "linear") -> pd.DataFrame:
    """Mean rank of each representation across endpoints (rank 1 = best on the
    primary metric). Higher primary_score is always better, since both AUC and
    Spearman rho are oriented that way."""
    sub = main[main["model"] == model]
    ranks = []
    for _, grp in sub.groupby("endpoint"):
        s = grp.set_index("representation")["primary_score"]
        r = pd.Series(rankdata(-s.to_numpy()), index=s.index)  # negate: high score = rank 1
        ranks.append(r)
    table = pd.DataFrame(ranks).mean().sort_values()
    return table.rename("mean_rank").reset_index().rename(columns={"index": "representation"})
