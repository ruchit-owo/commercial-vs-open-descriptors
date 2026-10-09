"""Two figures: the paired per-endpoint difference (alvaDesc minus best open) and
the per-endpoint comparison of every representation with its bootstrap interval.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def paired_difference_plot(diffs_best: pd.DataFrame, out: Path) -> None:
    """diffs_best: one row per endpoint with column 'diff' = alvaDesc minus the
    best open representation on the primary metric."""
    d = diffs_best.sort_values("diff")
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(d) + 1.5))
    colours = ["#4477aa" if v >= 0 else "#cc6677" for v in d["diff"]]
    ax.barh(d["endpoint"], d["diff"], color=colours)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("alvaDesc - best open representation (primary metric)")
    ax.set_title("Paired difference per endpoint (linear model)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def per_endpoint_plot(main: pd.DataFrame, out: Path, model: str = "linear") -> None:
    sub = main[main["model"] == model]
    endpoints = sorted(sub["endpoint"].unique())
    reps = sorted(sub["representation"].unique())
    fig, ax = plt.subplots(figsize=(1.6 * len(endpoints) + 2, 4.5))
    width = 0.8 / len(reps)
    x = np.arange(len(endpoints))
    for j, rep in enumerate(reps):
        vals, los, his = [], [], []
        for ep in endpoints:
            row = sub[(sub["endpoint"] == ep) & (sub["representation"] == rep)]
            if row.empty:
                vals.append(np.nan); los.append(0); his.append(0); continue
            row = row.iloc[0]
            vals.append(row["primary_score"])
            los.append(row["primary_score"] - row["ci_low"])
            his.append(row["ci_high"] - row["primary_score"])
        ax.bar(x + j * width, vals, width, yerr=[los, his], capsize=2, label=rep)
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels(endpoints, rotation=30, ha="right")
    ax.set_ylabel("primary metric (AUC or Spearman rho)")
    ax.set_title(f"Per-endpoint comparison ({model})")
    ax.legend(fontsize=8, ncol=len(reps))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
