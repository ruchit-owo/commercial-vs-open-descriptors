"""Shared fixtures. The tests use a deliberately small CV configuration so the
leakage and determinism checks run in seconds rather than reproducing the full
5x5 protocol; the code paths exercised are the same ones the real run uses.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def small_cfg():
    return {
        "seed": 20260910,
        "cv": {"n_splits": 3, "n_repeats": 1, "scaffold_for": ["classification"]},
        "bootstrap": {"n_resamples": 300, "ci": 0.95},
        "models": {
            "linear": {"classification": {"C": [0.1, 1]}, "regression": {"alpha": [1, 10]}},
            "random_forest": {"n_estimators": [50], "max_depth": [None], "max_features": ["sqrt"]},
            "hist_gradient_boosting": {"learning_rate": [0.1], "max_depth": [None],
                                        "max_iter": 50, "early_stopping": True},
        },
    }


@pytest.fixture
def noise_classification():
    """Many pure-noise features and a balanced binary target with no real
    relationship. A leak-free evaluator cannot beat chance here; a globally
    fitted feature selection can, which is exactly the trap the test sets."""
    rng = np.random.default_rng(0)
    n, p = 300, 200
    X = rng.standard_normal((n, p))
    y = rng.integers(0, 2, n)
    return X, y
