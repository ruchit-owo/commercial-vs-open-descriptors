"""The leakage guard. This is the test the credibility of the whole comparison
rests on, so it is written to be unambiguous.

On pure-noise features nothing can genuinely predict the target. The leak-free
evaluator, which fits its feature selection inside each training fold, should sit
near the 0.5 AUC baseline. A deliberately leaky variant, which chooses the "best"
features on the whole dataset before cross-validating, manufactures a large,
entirely spurious advantage. If the leak-free path ever drifts up towards the
leaky one, preprocessing has escaped the fold and every result is suspect.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

from descbench.evaluate import oof_predictions
from descbench.stats import score


def test_leakfree_selection_stays_near_baseline(small_cfg, noise_classification):
    X, y = noise_classification
    Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    # Averaging over a handful of repeats settles the out-of-fold estimate; with a
    # single repeat on this little data it is too noisy to pin near 0.5.
    cfg = {**small_cfg, "cv": {**small_cfg["cv"], "n_splits": 4, "n_repeats": 5}}

    # Leak-free: selection happens inside the pipeline, per fold.
    oof = oof_predictions(Xdf, y, "classification", "rdkit2d", "linear",
                          cfg, select_k=5)
    leakfree_auc = score(oof.y_true, oof.y_pred, "roc_auc")

    # Leaky: pick the five features most associated with y on ALL the data first,
    # then evaluate. The selector has seen the test folds' labels.
    selector = SelectKBest(f_classif, k=5).fit(X, y)
    Xleak = pd.DataFrame(selector.transform(X),
                         columns=[f"s{i}" for i in range(5)])
    oof_leak = oof_predictions(Xleak, y, "classification", "rdkit2d", "linear", cfg)
    leaky_auc = score(oof_leak.y_true, oof_leak.y_pred, "roc_auc")

    assert abs(leakfree_auc - 0.5) < 0.08, (
        f"leak-free AUC {leakfree_auc:.3f} is too far from chance on noise")
    assert leaky_auc - leakfree_auc > 0.1, (
        f"leaky ({leaky_auc:.3f}) should beat leak-free ({leakfree_auc:.3f}) clearly")
