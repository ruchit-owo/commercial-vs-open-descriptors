"""Phase 3b: the free 3D pre-check (plan 9.3-9.4).

Does adding 3D descriptors help at all on these endpoints? For each endpoint the
primary linear model is run, under the same leak-free protocol as the 2D study, on:

  rdkit2d              2D descriptors (open)
  mordred              2D descriptors (open, ignore_3D=True)
  rdkit3d              rdkit 3D descriptors only
  rdkit_2d3d           rdkit 2D + 3D concatenated
  mordred3d            mordred with 3D enabled (this already contains the 2D set)

For each endpoint it compares the best 2D-only representation against the best
2D+3D representation, with a paired bootstrap over molecules for the lift, and
applies the pre-registered decision rule: 3D is worth testing with alvaDesc only
if some endpoint shows a lift of at least 0.02 whose interval excludes zero.

No commercial licence is used. Writes results/pre_check_3d.csv (per representation)
and results/pre_check_3d_summary.csv (per endpoint, with the decision).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from descbench.data import ROOT, load_config, processed_path  # noqa: E402
from descbench.evaluate import oof_predictions  # noqa: E402
from descbench.features import features_path, load_features  # noqa: E402
from descbench import stats  # noqa: E402

LIFT_THRESHOLD = 0.02        # plan 9.4


def _have(cfg, endpoint, rep) -> bool:
    return features_path(cfg, endpoint, rep).exists()


def _load_combined(cfg, endpoint, reps):
    """Concatenate several cached representations by column, aligned on id."""
    frames, y = [], None
    for rep in reps:
        X, y, _ = load_features(cfg, endpoint, rep, allow_missing=True)
        frames.append(X)
    combined = pd.concat(frames, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    return combined, y


def _paired_lift_ci(y, pred_2d, pred_2d3d, metric, n_resamples, ci, seed):
    """Bootstrap the metric difference (2D+3D minus 2D) over molecules. Both
    predictions cover the same molecules in the same order, so the difference is
    paired and the interval reflects the lift, not each model's own spread."""
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, n, n)
        diffs[b] = (stats.score(y[idx], pred_2d3d[idx], metric)
                    - stats.score(y[idx], pred_2d[idx], metric))
    alpha = (1 - ci) / 2
    return float(np.quantile(diffs, alpha)), float(np.quantile(diffs, 1 - alpha))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoints", nargs="+", help="limit to these endpoints")
    args = parser.parse_args()

    cfg = load_config()
    seed = cfg["seed"]
    nres, ci = cfg["bootstrap"]["n_resamples"], cfg["bootstrap"]["ci"]

    endpoints = [ep for ep in cfg["endpoints"]
                 if not ep.get("pending") and processed_path(cfg, ep["name"]).exists()]
    if args.endpoints:
        endpoints = [ep for ep in endpoints if ep["name"] in set(args.endpoints)]

    variants = {
        "rdkit2d":   ("2d",   ["rdkit2d"]),
        "mordred":   ("2d",   ["mordred"]),
        "rdkit3d":   ("3d",   ["rdkit3d"]),
        "rdkit_2d3d": ("2d3d", ["rdkit2d", "rdkit3d"]),
        "mordred3d": ("2d3d", ["mordred3d"]),   # already 2D+3D
    }

    rows, summary = [], []
    for ep in endpoints:
        name, task = ep["name"], ep["task"]
        metric = stats.primary_metric(task)
        preds, scores = {}, {}
        for rep, (kind, parts) in variants.items():
            if not all(_have(cfg, name, p) for p in parts):
                continue
            X, y = _load_combined(cfg, name, parts)
            oof = oof_predictions(X, y, task, rep, "linear", cfg)
            s = stats.score(oof.y_true, oof.y_pred, metric)
            lo, hi = stats.bootstrap_ci(oof.y_true, oof.y_pred, metric, nres, ci, seed)
            preds[rep] = oof
            scores[rep] = s
            rows.append({"endpoint": name, "representation": rep, "kind": kind,
                         "primary_metric": metric, "primary_score": round(s, 4),
                         "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                         "baseline": round(stats.trivial_baseline(y, task), 4)})
            print(f"{name:20s} {rep:11s} {metric} {s:.3f}  [{lo:.3f}, {hi:.3f}]")

        two_d = {r: scores[r] for r in ("rdkit2d", "mordred") if r in scores}
        two_d3d = {r: scores[r] for r in ("rdkit_2d3d", "mordred3d") if r in scores}
        if not two_d or not two_d3d:
            continue
        best_2d = max(two_d, key=two_d.get)
        best_2d3d = max(two_d3d, key=two_d3d.get)
        lift = scores[best_2d3d] - scores[best_2d]
        lo, hi = _paired_lift_ci(preds[best_2d].y_true, preds[best_2d].y_pred,
                                 preds[best_2d3d].y_pred, metric, nres, ci, seed)
        decisive = lift >= LIFT_THRESHOLD and lo > 0
        summary.append({
            "endpoint": name, "best_2d": best_2d, "best_2d_score": round(scores[best_2d], 4),
            "best_2d3d": best_2d3d, "best_2d3d_score": round(scores[best_2d3d], 4),
            "lift": round(lift, 4), "lift_ci_low": round(lo, 4), "lift_ci_high": round(hi, 4),
            "helps_by_rule": bool(decisive),
        })
        print(f"  -> {name}: lift {lift:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
              f"{'HELPS' if decisive else 'no'}\n")

    res = ROOT / cfg["paths"]["results"]
    res.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(res / "pre_check_3d.csv", index=False)
    sm = pd.DataFrame(summary)
    sm.to_csv(res / "pre_check_3d_summary.csv", index=False)

    any_help = bool(sm["helps_by_rule"].any()) if len(sm) else False
    print("=" * 64)
    if any_help:
        hits = ", ".join(sm.loc[sm["helps_by_rule"], "endpoint"])
        print(f"VERDICT: 3D descriptors help on: {hits}.")
        print("The alvaDesc 3D arm is worth running on these (needs the trial).")
    else:
        print("VERDICT: no endpoint shows a 3D lift >= 0.02 with an interval above zero.")
        print("Per plan 9.4, do not spend trial time on the alvaDesc 3D arm; report the null.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
