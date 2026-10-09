"""Phase 3c: the symmetric commercial-vs-open 3D comparison.

The free pre-check (03b) decided, under the pre-registered rule, that the alvaDesc
3D arm was not worth the trial. That decision assumed the commercial descriptors
cost money and time. With alvaDesc available at no cost the assumption no longer
holds, so the arm is run after all (see the 2026-09-27 amendment in the plan) to
give the 3D question the same two-sided treatment the 2D study had.

alvaDesc 3D descriptors are computed from the very same conformer used for the open
3D descriptors, so no package gets a better geometry. Three alvaDesc variants are
evaluated under the identical leak-free protocol and placed beside the open results
already computed in 03b:

  alvadesc        alvaDesc 2D descriptors
  alvadesc3d      alvaDesc 3D descriptors only
  alvadesc_2d3d   alvaDesc 2D + 3D concatenated

Two questions are answered per endpoint: does 3D help alvaDesc (2D+3D vs 2D, with a
paired interval), and does alvaDesc 3D beat the open 3D descriptors (commercial vs
open, both on their own and as 2D+3D). Writes results/alvadesc_3d.csv and
results/alvadesc_3d_summary.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from descbench.data import ROOT, load_config, processed_path  # noqa: E402
from descbench.evaluate import oof_predictions  # noqa: E402
from descbench.features import features_path, load_features  # noqa: E402
from descbench import stats  # noqa: E402

ALVA_VARIANTS = {
    "alvadesc":      ("2d",   ["alvadesc"]),
    "alvadesc3d":    ("3d",   ["alvadesc3d"]),
    "alvadesc_2d3d": ("2d3d", ["alvadesc", "alvadesc3d"]),
}


def _combined(cfg, endpoint, reps):
    frames, y = [], None
    for rep in reps:
        X, y, _ = load_features(cfg, endpoint, rep, allow_missing=True)
        frames.append(X)
    combined = pd.concat(frames, axis=1)
    return combined.loc[:, ~combined.columns.duplicated()], y


def _paired_lift_ci(y, pred_a, pred_b, metric, n, ci, seed):
    """Bootstrap metric(b) - metric(a) over molecules (paired)."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    m = len(y)
    for i in range(n):
        idx = rng.integers(0, m, m)
        diffs[i] = stats.score(y[idx], pred_b[idx], metric) - stats.score(y[idx], pred_a[idx], metric)
    a = (1 - ci) / 2
    return float(np.quantile(diffs, a)), float(np.quantile(diffs, 1 - a))


def main() -> int:
    cfg = load_config()
    seed = cfg["seed"]
    nres, ci = cfg["bootstrap"]["n_resamples"], cfg["bootstrap"]["ci"]

    open_path = ROOT / cfg["paths"]["results"] / "pre_check_3d.csv"
    open_df = pd.read_csv(open_path) if open_path.exists() else pd.DataFrame()

    endpoints = [ep for ep in cfg["endpoints"]
                 if not ep.get("pending") and processed_path(cfg, ep["name"]).exists()
                 and features_path(cfg, ep["name"], "alvadesc3d").exists()]

    rows, summary = [], []
    for ep in endpoints:
        name, task = ep["name"], ep["task"]
        metric = stats.primary_metric(task)
        preds, score = {}, {}

        for rep, (kind, parts) in ALVA_VARIANTS.items():
            X, y = _combined(cfg, name, parts)
            oof = oof_predictions(X, y, task, rep, "linear", cfg)
            s = stats.score(oof.y_true, oof.y_pred, metric)
            lo, hi = stats.bootstrap_ci(oof.y_true, oof.y_pred, metric, nres, ci, seed)
            preds[rep], score[rep] = oof, s
            rows.append({"endpoint": name, "representation": rep, "kind": kind,
                         "primary_metric": metric, "primary_score": round(s, 4),
                         "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
            print(f"{name:14s} {rep:13s} {metric} {s:.3f}  [{lo:.3f}, {hi:.3f}]")

        # Open scores for this endpoint from the pre-check (same protocol/seed).
        o = open_df[open_df["endpoint"] == name].set_index("representation")["primary_score"] \
            if len(open_df) else pd.Series(dtype=float)

        def best(cands):
            cands = {r: o[r] for r in cands if r in o.index}
            return (max(cands, key=cands.get), cands[max(cands, key=cands.get)]) if cands else (None, np.nan)

        best_open_2d = best(["rdkit2d", "mordred"])
        best_open_2d3d = best(["rdkit_2d3d", "mordred3d"])
        open_3d_alone = ("rdkit3d", o["rdkit3d"]) if "rdkit3d" in o.index else (None, np.nan)

        # Does 3D help alvaDesc? (paired, its own 2D vs 2D+3D)
        lift = score["alvadesc_2d3d"] - score["alvadesc"]
        llo, lhi = _paired_lift_ci(preds["alvadesc"].y_true, preds["alvadesc"].y_pred,
                                   preds["alvadesc_2d3d"].y_pred, metric, nres, ci, seed)

        summary.append({
            "endpoint": name,
            "alvadesc_2d": round(score["alvadesc"], 4),
            "alvadesc_3d_only": round(score["alvadesc3d"], 4),
            "alvadesc_2d3d": round(score["alvadesc_2d3d"], 4),
            "alva_3d_lift": round(lift, 4), "lift_ci_low": round(llo, 4), "lift_ci_high": round(lhi, 4),
            "alva_3d_helps": bool(lift >= 0.02 and llo > 0),
            "open_3d_only": round(open_3d_alone[1], 4),
            "best_open_2d3d": best_open_2d3d[0], "best_open_2d3d_score": round(best_open_2d3d[1], 4),
            "alva_2d3d_minus_best_open_2d3d": round(score["alvadesc_2d3d"] - best_open_2d3d[1], 4),
            "alva3d_minus_open3d": round(score["alvadesc3d"] - open_3d_alone[1], 4),
        })
        print(f"  -> {name}: alva 3D lift {lift:+.3f} [{llo:+.3f}, {lhi:+.3f}]; "
              f"alva(2D+3D)-bestOpen(2D+3D) {score['alvadesc_2d3d']-best_open_2d3d[1]:+.3f}; "
              f"alva3D-open3D {score['alvadesc3d']-open_3d_alone[1]:+.3f}\n")

    res = ROOT / cfg["paths"]["results"]
    pd.DataFrame(rows).to_csv(res / "alvadesc_3d.csv", index=False)
    sm = pd.DataFrame(summary)
    sm.to_csv(res / "alvadesc_3d_summary.csv", index=False)

    print("=" * 66)
    helps = sm[sm["alva_3d_helps"]]["endpoint"].tolist() if len(sm) else []
    if helps:
        print(f"alvaDesc's own 3D descriptors lift it by >= 0.02 on: {', '.join(helps)}.")
    else:
        print("Adding alvaDesc 3D to alvaDesc 2D does not lift any endpoint by >= 0.02.")
    if len(sm):
        wins = int((sm["alva_2d3d_minus_best_open_2d3d"] > 0).sum())
        print(f"alvaDesc 2D+3D beats the best open 2D+3D on {wins}/{len(sm)} endpoints "
              f"(median diff {sm['alva_2d3d_minus_best_open_2d3d'].median():+.3f}).")
        w3 = int((sm["alva3d_minus_open3d"] > 0).sum())
        print(f"alvaDesc 3D-only beats open (RDKit) 3D-only on {w3}/{len(sm)} endpoints "
              f"(median diff {sm['alva3d_minus_open3d'].median():+.3f}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
