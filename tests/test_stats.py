"""Checks on the statistics helpers: a bootstrap interval should contain its own
point estimate, and a paired test on identical inputs should find nothing.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from descbench import stats  # noqa: E402


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    p = y * 0.6 + rng.random(300) * 0.4       # a genuinely predictive score
    point = stats.score(y, p, "roc_auc")
    lo, hi = stats.bootstrap_ci(y, p, "roc_auc", 500, 0.95, seed=1)
    assert lo <= point <= hi
    assert lo < hi


def test_paired_summary_on_identical_scores_finds_no_difference():
    # alvaDesc and one open representation scoring identically on every endpoint.
    rows = []
    for i, ep in enumerate(["a", "b", "c", "d", "e"]):
        val = 0.7 + 0.01 * i
        rows.append({"endpoint": ep, "representation": "alvadesc",
                     "model": "linear", "primary_score": val})
        rows.append({"endpoint": ep, "representation": "rdkit2d",
                     "model": "linear", "primary_score": val})
    main = pd.DataFrame(rows)
    diffs = stats.paired_differences(main)
    summary = stats.paired_summary(diffs, 500, 0.95, seed=1)
    row = summary.iloc[0]
    assert row["median_diff"] == 0
    assert row["wilcoxon_p"] == 1.0
