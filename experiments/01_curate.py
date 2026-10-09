"""Phase 1: download and standardise every endpoint.

Writes data/processed/<endpoint>.csv, prints a summary table of counts and the
trivial baselines, and records raw-file hashes and per-endpoint curation counts
into PROVENANCE.md. Endpoints marked pending in config (nucleoside gelation,
whose data is supplied separately) are skipped with a note.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from descbench.data import (  # noqa: E402
    ROOT, curate_endpoint, endpoint_config, file_sha256, load_config, processed_path,
)


def _baseline(df: pd.DataFrame, task: str) -> str:
    y = df["target"].to_numpy()
    if task == "classification":
        p = y.mean()
        return f"majority {max(p, 1 - p):.3f} (prevalence {p:.3f})"
    return f"mean predictor RMSE {np.std(y):.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="recurate even if the CSV exists")
    args = parser.parse_args()

    cfg = load_config()
    rows, hashes = [], []
    for ep in cfg["endpoints"]:
        name = ep["name"]
        out = processed_path(cfg, name)
        if ep.get("pending"):
            print(f"{name:20s} pending - data supplied separately, skipped")
            continue
        if out.exists() and not args.force:
            df = pd.read_csv(out)
            print(f"{name:20s} cached ({len(df)} rows)")
            rows.append({"endpoint": name, "task": ep["task"], "rows_out": len(df),
                         "baseline": _baseline(df, ep["task"])})
            continue
        try:
            df, report = curate_endpoint(cfg, name)
        except FileNotFoundError as exc:
            print(f"{name:20s} SKIPPED - {exc}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        info = report.as_row()
        info["task"] = ep["task"]
        info["baseline"] = _baseline(df, ep["task"])
        rows.append(info)
        print(f"{name:20s} in {report.rows_in:>6}  parse_fail {report.parse_failures:>4}  "
              f"dups {report.duplicates_collapsed:>5}  conflicts {report.conflicts_dropped:>4}  "
              f"out {report.rows_out:>6}  | {info['baseline']}")

        raw_file = ROOT / cfg["paths"]["raw"] / endpoint_config(cfg, name)["source"]
        if raw_file.exists():
            hashes.append((name, raw_file.name, file_sha256(raw_file)))

    if rows:
        summary = pd.DataFrame(rows)
        (ROOT / cfg["paths"]["tables"]).mkdir(parents=True, exist_ok=True)
        summary.to_csv(ROOT / cfg["paths"]["tables"] / "curation_summary.csv", index=False)
        print("\nWrote tables/curation_summary.csv")
    _update_provenance(hashes)
    return 0


def _update_provenance(hashes: list[tuple[str, str, str]]) -> None:
    """Fill the SHA256 column of the manifest in place, so the table stays a real
    record rather than a set of placeholders."""
    if not hashes:
        return
    prov = ROOT / "PROVENANCE.md"
    text = prov.read_text(encoding="utf-8")
    for name, _, digest in hashes:
        # Match the row by its leading endpoint cell and replace the placeholder.
        marker = f"| {name} |"
        for line in text.splitlines():
            if line.startswith(marker) and "_filled by 01_curate_" in line:
                text = text.replace(line, line.replace("_filled by 01_curate_", digest[:16]))
    prov.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
