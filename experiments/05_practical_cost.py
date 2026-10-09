"""Phase 5: practical cost (RQ4).

Times each representation on a fixed subset and records descriptors produced,
compute time per 1000 molecules, molecule failure rate and missing-cell fraction.
Timing is deliberately kept out of the comparison result files (it varies run to
run); it belongs here, where it is the measurement. Failure and missing-value
rates for the full endpoints are also pulled from the feature manifest so the
table reflects the real datasets, not just the timing subset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from descbench.data import ROOT, load_config, processed_path  # noqa: E402
from descbench.cost import cost_table  # noqa: E402
from descbench.features import ALL_REPRESENTATIONS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="lipophilicity",
                        help="endpoint to time on (needs at least cost_subset molecules)")
    args = parser.parse_args()

    cfg = load_config()
    if not processed_path(cfg, args.endpoint).exists():
        print(f"{args.endpoint} not curated; run 01_curate.py first.")
        return 1

    timing = cost_table(cfg, args.endpoint, list(ALL_REPRESENTATIONS))

    # Bring in full-dataset failure and missingness from the feature manifest, so
    # the cost table describes the real endpoints rather than only the subset.
    manifest_path = ROOT / cfg["paths"]["tables"] / "feature_manifest.csv"
    if manifest_path.exists():
        man = pd.read_csv(manifest_path)
        full = (man[man["status"] == "ok"]
                .groupby("representation")
                .agg(full_missing_molecules=("n_missing_molecules", "sum"),
                     full_mean_missing_cells=("missing_cell_fraction", "mean"))
                .reset_index())
        timing = timing.merge(full, on="representation", how="left")

    out = ROOT / cfg["paths"]["results"] / "practical_cost.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    timing.to_csv(out, index=False)
    print(timing.to_string(index=False))
    print(f"\nWrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
