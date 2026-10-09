"""Phase 4b: effective feature widths at each matched budget.

[Added after the alvaDesc-developer review.]

The matched-budget run (04) passes a target k to SelectKBest, but the pipeline runs
a VarianceThreshold first, and SelectKBest then keeps min(k, surviving columns). So
the budget is a target, not a guaranteed effective width: a representation with fewer
usable columns than k keeps all of them (MACCS has 166 usable bits; RDKit has 217).

This script records, per (endpoint, representation), how many columns survive median
imputation and the zero-variance filter on the full matrix, and the resulting
effective width at each configured budget (min(k, survivors)). The full-matrix count
is an upper bound on the in-fold width: inside a smaller training fold the variance
filter can drop a few more. Writes results/effective_widths.csv.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from descbench.data import ROOT, load_config, processed_path  # noqa: E402
from descbench.features import ALL_REPRESENTATIONS, features_path, load_features  # noqa: E402


def main() -> int:
    cfg = load_config()
    budgets = list(cfg["matched_budget_k"])
    endpoints = [ep["name"] for ep in cfg["endpoints"]
                 if not ep.get("pending") and processed_path(cfg, ep["name"]).exists()]

    rows = []
    for name in endpoints:
        for rep in ALL_REPRESENTATIONS:
            if not features_path(cfg, name, rep).exists():
                continue
            X, _, _ = load_features(cfg, name, rep)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Xi = SimpleImputer(strategy="median").fit_transform(X)
            vt = VarianceThreshold(0.0).fit(Xi)
            survivors = int(vt.get_support().sum())
            row = {"endpoint": name, "representation": rep,
                   "orig_cols": X.shape[1], "after_variance": survivors}
            for k in budgets:
                row[f"eff_k{k}"] = min(k, survivors)
            rows.append(row)
            print(f"{name:14s} {rep:10s} orig {X.shape[1]:>5}  survivors {survivors:>5}")

    out = ROOT / cfg["paths"]["results"] / "effective_widths.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
