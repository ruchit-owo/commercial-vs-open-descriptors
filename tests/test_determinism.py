"""Two seeded runs of the evaluator must produce the same numbers. Without this
the "every number regenerable" claim in the definition of done is empty.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from descbench.evaluate import oof_predictions  # noqa: E402
from descbench.stats import score  # noqa: E402


def test_repeated_runs_are_identical(small_cfg):
    rng = np.random.default_rng(2)
    n = 150
    X = pd.DataFrame(rng.standard_normal((n, 20)), columns=[f"f{i}" for i in range(20)])
    signal = X["f0"] + 0.5 * X["f1"]
    y = (signal + rng.standard_normal(n) * 0.5 > signal.median()).astype(int).to_numpy()

    a = oof_predictions(X, y, "classification", "rdkit2d", "linear", small_cfg)
    b = oof_predictions(X, y, "classification", "rdkit2d", "linear", small_cfg)
    assert score(a.y_true, a.y_pred, "roc_auc") == score(b.y_true, b.y_pred, "roc_auc")
    assert np.allclose(a.y_pred, b.y_pred)
