"""Practical-cost accounting (RQ4): compute time per 1000 molecules, failure rate
and missing-cell fraction per representation, timed on a fixed subset. Licence and
install friction are qualitative and live in the report.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .data import load_processed
from .features import compute_alvadesc, compute_open_representation


def time_representation(cfg: dict, endpoint: str, rep: str, subset: int) -> dict:
    """Time one representation on the first `subset` molecules of an endpoint and
    report descriptors produced, failure rate and missing-cell fraction. Timing
    uses the real compute path, not the cache, so it reflects genuine cost."""
    proc = load_processed(cfg, endpoint).head(subset)
    smiles = proc["smiles"].tolist()
    ids = proc["id"].tolist()
    n = len(smiles)

    t0 = time.perf_counter()
    if rep == "alvadesc":
        table, n_failed = compute_alvadesc(cfg, smiles, ids)
        table = table.reindex(ids)
    else:
        table = compute_open_representation(rep, smiles)
        n_failed = 0
    elapsed = time.perf_counter() - t0

    missing = float(np.isnan(table.to_numpy(dtype=float)).mean())
    return {
        "representation": rep,
        "endpoint": endpoint,
        "n_molecules_timed": n,
        "n_descriptors": table.shape[1],
        "seconds_per_1000": round(elapsed / n * 1000, 2),
        "failure_rate": round(n_failed / n, 4),
        "missing_cell_fraction": round(missing, 4),
    }


def cost_table(cfg: dict, endpoint: str, representations: list[str]) -> pd.DataFrame:
    subset = cfg["cost_subset"]
    rows = []
    for rep in representations:
        try:
            rows.append(time_representation(cfg, endpoint, rep, subset))
        except Exception as exc:  # noqa: BLE001 - a failed install is itself the cost
            rows.append({"representation": rep, "endpoint": endpoint,
                         "n_molecules_timed": 0, "n_descriptors": 0,
                         "seconds_per_1000": None, "failure_rate": None,
                         "missing_cell_fraction": None, "note": str(exc)[:120]})
    return pd.DataFrame(rows)
