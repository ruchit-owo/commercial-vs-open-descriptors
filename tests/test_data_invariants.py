"""Invariants every processed endpoint must satisfy. These run against whatever
endpoints have been curated; if none are present yet the test skips rather than
failing, so the suite is useful both before and after phase 1.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from descbench.data import load_config, processed_path  # noqa: E402
import pandas as pd  # noqa: E402


def _available_endpoints():
    cfg = load_config()
    return [ep["name"] for ep in cfg["endpoints"] if processed_path(cfg, ep["name"]).exists()]


@pytest.mark.parametrize("name", _available_endpoints() or [pytest.param("none", marks=pytest.mark.skip)])
def test_processed_endpoint_is_clean(name):
    cfg = load_config()
    df = pd.read_csv(processed_path(cfg, name))

    assert list(df.columns) == ["id", "smiles", "target", "task"]
    assert df["id"].is_unique
    assert df["smiles"].is_unique, "duplicate canonical SMILES within an endpoint"
    assert df["smiles"].map(lambda s: Chem.MolFromSmiles(s) is not None).all()

    task = df["task"].iloc[0]
    if task == "classification":
        assert set(df["target"].unique()) <= {0, 1}
    else:
        assert np.isfinite(df["target"].to_numpy()).all()
