"""Phase 2: compute and cache every representation for every endpoint.

alvaDesc is computed first for all endpoints, because it needs the time-limited
commercial licence; everything downstream reads the cached parquet tables and
does not. The open representations follow. A manifest of columns produced,
molecules failed and missing-cell fraction is written to
tables/feature_manifest.csv (the raw material for the RQ4 cost accounting).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from descbench.data import ROOT, load_config, processed_path  # noqa: E402
from descbench.features import (  # noqa: E402
    ALL_REPRESENTATIONS, build_representation, load_features,
)


def _endpoints(cfg: dict) -> list[str]:
    return [ep["name"] for ep in cfg["endpoints"]
            if not ep.get("pending") and processed_path(cfg, ep["name"]).exists()]


def _run(cfg: dict, reps: list[str], force: bool) -> list[dict]:
    manifest = []
    for rep in reps:                       # alvaDesc first (order of ALL_REPRESENTATIONS)
        for name in _endpoints(cfg):
            t0 = time.perf_counter()
            try:
                build_representation(cfg, name, rep, force=force)
            except Exception as exc:        # noqa: BLE001 - want the reason recorded, not a crash
                print(f"{rep:10s} {name:20s} FAILED: {exc}")
                manifest.append({"representation": rep, "endpoint": name,
                                 "status": "failed", "detail": str(exc)[:120]})
                continue
            _, _, info = load_features(cfg, name, rep)
            dt = time.perf_counter() - t0
            manifest.append({
                "representation": rep, "endpoint": name, "status": "ok",
                "n_columns": info.n_columns, "n_molecules": info.n_molecules,
                "n_missing_molecules": info.n_missing_molecules,
                "missing_cell_fraction": round(info.missing_cell_fraction, 5),
            })
            print(f"{rep:10s} {name:20s} cols {info.n_columns:>5}  "
                  f"missing_mol {info.n_missing_molecules:>3}  "
                  f"missing_cells {info.missing_cell_fraction:6.3f}  ({dt:5.1f}s)")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", nargs="+", help="limit to these representations")
    args = parser.parse_args()

    cfg = load_config()
    reps = list(ALL_REPRESENTATIONS)
    if args.only:
        reps = [r for r in ALL_REPRESENTATIONS if r in set(args.only)]

    manifest = _run(cfg, reps, args.force)
    if manifest:
        out = ROOT / cfg["paths"]["tables"] / "feature_manifest.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(manifest).to_csv(out, index=False)
        print(f"\nWrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
