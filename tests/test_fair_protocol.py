"""No representation may be tuned harder than another. The grid a model uses must
be identical whatever the representation, so the only thing that varies across a
row of the comparison is the features themselves.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from descbench.evaluate import MODELS, param_grid  # noqa: E402
from descbench.features import ALL_REPRESENTATIONS  # noqa: E402


def test_grid_is_identical_across_representations(small_cfg):
    for model in MODELS:
        for task in ("classification", "regression"):
            grids = [param_grid(model, task, small_cfg) for _ in ALL_REPRESENTATIONS]
            first = grids[0]
            assert all(g == first for g in grids), (
                f"{model}/{task}: grid differs between representations")


def test_grid_matches_config(small_cfg):
    # The grid is derived from config, not hard-coded, so a change to the plan's
    # grid flows through to every representation at once.
    grid = param_grid("linear", "classification", small_cfg)
    assert grid == {"model__C": small_cfg["models"]["linear"]["classification"]["C"]}
