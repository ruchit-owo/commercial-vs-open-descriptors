"""Phase 2b: build the open 3D representations for the pre-registered 3D arm (9).

One conformer per molecule (recipe fixed in plan section 9.1) is generated and reused for
rdkit3d and mordred3d, and written to an SDF so the alvaDesc 3D arm can later use
the same geometry. Nothing here needs the commercial licence.

The melting-point endpoint (10k molecules) is skipped by default: embedding and
Mordred-3D on that many molecules is slow and it is not one of the endpoints where
a 3D effect is expected. Pass --include-bradley to run it too.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from descbench.data import ROOT, load_config, processed_path  # noqa: E402
from descbench.features import load_features  # noqa: E402
from descbench.features_3d import compute_3d_features  # noqa: E402

REPS_3D = ("rdkit3d", "mordred3d")


def _endpoints(cfg: dict, include_bradley: bool) -> list[str]:
    names = [ep["name"] for ep in cfg["endpoints"]
             if not ep.get("pending") and processed_path(cfg, ep["name"]).exists()]
    if not include_bradley:
        names = [n for n in names if n != "bradley_mp"]
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--include-bradley", action="store_true",
                        help="also build the 10k melting-point set (slow)")
    args = parser.parse_args()

    cfg = load_config()
    manifest = []
    for name in _endpoints(cfg, args.include_bradley):
        t0 = time.perf_counter()
        compute_3d_features(cfg, name, force=args.force)
        for rep in REPS_3D:
            _, _, info = load_features(cfg, name, rep, allow_missing=True)
            manifest.append({
                "endpoint": name, "representation": rep,
                "n_columns": info.n_columns, "n_molecules": info.n_molecules,
                "missing_cell_fraction": round(info.missing_cell_fraction, 5),
            })
            print(f"{rep:10s} {name:20s} cols {info.n_columns:>5}  "
                  f"missing_cells {info.missing_cell_fraction:6.3f}")
        print(f"  ({time.perf_counter() - t0:.0f}s total for {name})")

    if manifest:
        out = ROOT / cfg["paths"]["tables"] / "feature_manifest_3d.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(manifest).to_csv(out, index=False)
        print(f"\nWrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
