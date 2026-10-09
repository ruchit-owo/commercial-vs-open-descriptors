"""Phase 4: the matched-feature-budget rerun (RQ3).

alvaDesc brings thousands of columns and the fingerprints bring a couple of
thousand more, while RDKit brings a couple of hundred. Some of any apparent
advantage could be nothing more than a fold-internal selector having more columns
to choose from. To separate "better descriptors" from "simply more columns", the
comparison is rerun with a univariate selector inside each training fold cutting
every representation down to the same k. The linear model is used, as the primary
model of the study, at each k in config's matched_budget_k.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from descbench.data import ROOT, load_config, processed_path  # noqa: E402
from descbench.evaluate import oof_predictions  # noqa: E402
from descbench.features import ALL_REPRESENTATIONS, features_path, load_features  # noqa: E402
from descbench import stats  # noqa: E402

OUT = "matched_budget.csv"


def _endpoints(cfg):
    return [ep for ep in cfg["endpoints"]
            if not ep.get("pending") and processed_path(cfg, ep["name"]).exists()]


def _done(path: Path) -> set:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return set(zip(df["endpoint"], df["representation"], df["k"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="linear")
    parser.add_argument("--reps", nargs="+", default=list(ALL_REPRESENTATIONS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    out = ROOT / cfg["paths"]["results"] / OUT
    if args.force:
        out.unlink(missing_ok=True)
    done = _done(out)

    for ep in _endpoints(cfg):
        name, task = ep["name"], ep["task"]
        metric = stats.primary_metric(task)
        for rep in args.reps:
            if not features_path(cfg, name, rep).exists():
                continue
            X, y, _ = load_features(cfg, name, rep)
            for k in cfg["matched_budget_k"]:
                if (name, rep, k) in done:
                    continue
                t0 = time.perf_counter()
                oof = oof_predictions(X, y, task, rep, args.model, cfg, select_k=k)
                lo, hi = stats.bootstrap_ci(oof.y_true, oof.y_pred, metric,
                                            cfg["bootstrap"]["n_resamples"],
                                            cfg["bootstrap"]["ci"], cfg["seed"])
                row = {"endpoint": name, "representation": rep, "model": args.model,
                       "k": k, "task": task, "primary_metric": metric,
                       "primary_score": round(stats.score(oof.y_true, oof.y_pred, metric), 4),
                       "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                       "n_molecules": len(y)}
                out.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
                print(f"{name:20s} {rep:10s} k={k:<4} {metric} "
                      f"{row['primary_score']:.3f} [{row['ci_low']:.3f},{row['ci_high']:.3f}]"
                      f"  ({time.perf_counter()-t0:.0f}s)")

    if out.exists():
        df = pd.read_csv(out).sort_values(["endpoint", "k", "representation"])
        df.to_csv(out, index=False)
        print(f"\nWrote {out.relative_to(ROOT)} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
