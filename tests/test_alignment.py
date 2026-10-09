"""The feature loader must never quietly return fewer rows than there are targets.
A missing molecule is either raised (open representations, which should not fail)
or kept as an all-NaN row and counted (alvaDesc). Both behaviours are checked
here on a synthetic table so the test does not depend on a full feature build.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from descbench import features  # noqa: E402


def _tiny_endpoint(tmp_path, monkeypatch):
    """Point the package at a temporary processed/feature layout with three
    molecules and a feature table missing the middle one."""
    proc = pd.DataFrame({
        "id": ["m0", "m1", "m2"],
        "smiles": ["CCO", "c1ccccc1", "CCN"],
        "target": [0.0, 1.0, 2.0],
        "task": ["regression"] * 3,
    })
    feat = pd.DataFrame({"a": [1.0, 3.0], "b": [4.0, 6.0]}, index=["m0", "m2"])
    feat.index.name = "id"

    cfg = {"paths": {"processed": "processed", "features": "features"}}
    (tmp_path / "processed").mkdir()
    (tmp_path / "features").mkdir()
    proc.to_csv(tmp_path / "processed" / "toy.csv", index=False)
    feat.to_parquet(tmp_path / "features" / "toy__rdkit2d.parquet")
    feat.to_parquet(tmp_path / "features" / "toy__alvadesc.parquet")

    monkeypatch.setattr(features, "ROOT", tmp_path)
    # data.load_processed and features share ROOT via the same module import.
    monkeypatch.setattr(features, "load_processed",
                        lambda cfg, name: pd.read_csv(tmp_path / "processed" / f"{name}.csv"))
    return cfg


def test_open_representation_raises_on_missing(tmp_path, monkeypatch):
    cfg = _tiny_endpoint(tmp_path, monkeypatch)
    with pytest.raises(KeyError):
        features.load_features(cfg, "toy", "rdkit2d")


def test_alvadesc_keeps_missing_as_nan_row(tmp_path, monkeypatch):
    cfg = _tiny_endpoint(tmp_path, monkeypatch)
    X, y, info = features.load_features(cfg, "toy", "alvadesc")
    assert X.shape[0] == 3                      # no row dropped
    assert list(X.index) == ["m0", "m1", "m2"]  # order matches the target vector
    assert np.isnan(X.loc["m1"].to_numpy(dtype=float)).all()
    assert info.n_missing_molecules == 1
