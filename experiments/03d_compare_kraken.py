"""Phase 3d: the Kraken test - a property that really is driven by 3D shape.

The six pre-registered endpoints, and the 3D pre-check on them, all said the same
thing: 3D descriptors add nothing over 2D, so alvaDesc's 3D advantage is real at the
descriptor level but does not matter. That was on properties 2D descriptors already
handle well. This phase is the fair counter-test: a property where 2D descriptors
should struggle by construction.

Kraken is a library of monodentate phosphine ligands. Buried volume (%Vbur) is the
canonical 3D steric parameter used throughout organophosphorus catalysis; HOMO
energy is an electronic property on the same molecules, included as a contrast. If
3D descriptors, or alvaDesc, are ever going to earn their place, buried volume is
where it should show. Every representation is run through the identical leak-free
protocol, and 2D, 3D and combined sets are compared for the open and commercial
packages side by side.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from descbench.data import ROOT, load_config, processed_path  # noqa: E402
from descbench.evaluate import oof_predictions  # noqa: E402
from descbench.features import features_path, load_features  # noqa: E402
from descbench import stats  # noqa: E402

# label -> (parts to concatenate, kind)
VARIANTS = {
    "rdkit2d":       (["rdkit2d"], "2d-open"),
    "mordred":       (["mordred"], "2d-open"),
    "ecfp4":         (["ecfp4"], "fp-open"),
    "alvadesc":      (["alvadesc"], "2d-commercial"),
    "rdkit3d":       (["rdkit3d"], "3d-open"),
    "alvadesc3d":    (["alvadesc3d"], "3d-commercial"),
    "rdkit_2d3d":    (["rdkit2d", "rdkit3d"], "2d3d-open"),
    "mordred3d":     (["mordred3d"], "2d3d-open"),
    "alvadesc_2d3d": (["alvadesc", "alvadesc3d"], "2d3d-commercial"),
}


def _combined(cfg, endpoint, reps):
    frames, y = [], None
    for rep in reps:
        X, y, _ = load_features(cfg, endpoint, rep, allow_missing=True)
        frames.append(X)
    combined = pd.concat(frames, axis=1)
    return combined.loc[:, ~combined.columns.duplicated()], y


def main() -> int:
    cfg = load_config()
    seed = cfg["seed"]
    nres, ci = cfg["bootstrap"]["n_resamples"], cfg["bootstrap"]["ci"]

    endpoints = [ep["name"] for ep in cfg["endpoints"]
                 if ep.get("exploratory") and processed_path(cfg, ep["name"]).exists()
                 and features_path(cfg, ep["name"], "alvadesc3d").exists()]

    rows, summary = [], []
    for name in endpoints:
        task = "regression"
        metric = stats.primary_metric(task)
        score = {}
        for rep, (parts, kind) in VARIANTS.items():
            if not all(features_path(cfg, name, p).exists() for p in parts):
                continue
            X, y = _combined(cfg, name, parts)
            oof = oof_predictions(X, y, task, rep, "linear", cfg)
            s = stats.score(oof.y_true, oof.y_pred, metric)
            lo, hi = stats.bootstrap_ci(oof.y_true, oof.y_pred, metric, nres, ci, seed)
            score[rep] = s
            rows.append({"endpoint": name, "representation": rep, "kind": kind,
                         "primary_metric": metric, "primary_score": round(s, 4),
                         "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
            print(f"{name:13s} {rep:14s} {kind:16s} {metric} {s:.3f}  [{lo:.3f}, {hi:.3f}]")

        # The comparisons that matter for this test.
        s = score
        summary.append({
            "endpoint": name,
            "best_2d_open": round(max(s.get("rdkit2d", 0), s.get("mordred", 0)), 4),
            "alvadesc_2d": round(s.get("alvadesc", float("nan")), 4),
            "rdkit3d": round(s.get("rdkit3d", float("nan")), 4),
            "alvadesc3d": round(s.get("alvadesc3d", float("nan")), 4),
            "best_2d3d_open": round(max(s.get("rdkit_2d3d", 0), s.get("mordred3d", 0)), 4),
            "alvadesc_2d3d": round(s.get("alvadesc_2d3d", float("nan")), 4),
            # does 3D help, for each package?
            "open_3d_gain": round(max(s.get("rdkit_2d3d", 0), s.get("mordred3d", 0))
                                  - max(s.get("rdkit2d", 0), s.get("mordred", 0)), 4),
            "alva_3d_gain": round(s.get("alvadesc_2d3d", float("nan")) - s.get("alvadesc", float("nan")), 4),
            # commercial vs open at each level
            "alva_minus_open_3d_only": round(s.get("alvadesc3d", float("nan")) - s.get("rdkit3d", float("nan")), 4),
            "alva_minus_open_2d3d": round(s.get("alvadesc_2d3d", float("nan"))
                                          - max(s.get("rdkit_2d3d", 0), s.get("mordred3d", 0)), 4),
        })
        r = summary[-1]
        print(f"  -> {name}: open 3D gain {r['open_3d_gain']:+.3f}, alva 3D gain {r['alva_3d_gain']:+.3f}; "
              f"alva3D-open3D {r['alva_minus_open_3d_only']:+.3f}; "
              f"alva(2D+3D)-bestOpen(2D+3D) {r['alva_minus_open_2d3d']:+.3f}\n")

    res = ROOT / cfg["paths"]["results"]
    pd.DataFrame(rows).to_csv(res / "kraken_comparison.csv", index=False)
    pd.DataFrame(summary).to_csv(res / "kraken_summary.csv", index=False)
    print(f"Wrote kraken_comparison.csv and kraken_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
