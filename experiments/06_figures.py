"""Phase 6: the paired cross-endpoint summary and the two figures.

Reads results/main_comparison.csv and produces:
  - results/paired_summary.csv : per open representation, the median paired
    difference from alvaDesc across endpoints, its bootstrap interval, and a
    Wilcoxon signed-rank p-value;
  - tables/mean_rank.csv        : mean rank of each representation;
  - figures/paired_difference.png : the lead figure (alvaDesc minus the best open
    representation per endpoint, zero marked);
  - figures/per_endpoint.png      : every representation per endpoint with CIs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from descbench.data import ROOT, load_config  # noqa: E402
from descbench import plots, stats  # noqa: E402


def best_open_difference(main: pd.DataFrame, model: str = "linear") -> pd.DataFrame:
    """Per endpoint: alvaDesc minus the best open representation on the primary
    metric. This is what the lead figure shows."""
    sub = main[main["model"] == model]
    rows = []
    for endpoint, grp in sub.groupby("endpoint"):
        by_rep = grp.set_index("representation")["primary_score"]
        if "alvadesc" not in by_rep.index:
            continue
        opens = by_rep.drop(index="alvadesc", errors="ignore")
        if opens.empty:
            continue
        rows.append({"endpoint": endpoint, "diff": by_rep["alvadesc"] - opens.max(),
                     "best_open": opens.idxmax()})
    return pd.DataFrame(rows)


def main() -> int:
    cfg = load_config()
    results = ROOT / cfg["paths"]["results"]
    figures = ROOT / cfg["paths"]["figures"]
    tables = ROOT / cfg["paths"]["tables"]
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    main_df = pd.read_csv(results / "main_comparison.csv")

    diffs = stats.paired_differences(main_df, model="linear")
    summary = stats.paired_summary(diffs, cfg["bootstrap"]["n_resamples"],
                                   cfg["bootstrap"]["ci"], cfg["seed"])
    summary.to_csv(results / "paired_summary.csv", index=False)
    print(summary.to_string(index=False))

    ranks = stats.mean_rank_table(main_df, model="linear")
    ranks.to_csv(tables / "mean_rank.csv", index=False)

    best = best_open_difference(main_df, model="linear")
    plots.paired_difference_plot(best, figures / "paired_difference.png")
    plots.per_endpoint_plot(main_df, figures / "per_endpoint.png")
    print(f"\nWrote figures to {figures.relative_to(ROOT)} and paired_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
