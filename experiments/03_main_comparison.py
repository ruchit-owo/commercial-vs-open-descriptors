"""Phase 3: the leak-free nested-CV comparison.

For every (endpoint, representation, model) it pools out-of-fold predictions,
scores the primary and secondary metrics once, and attaches a bootstrap interval
over molecules and the trivial baseline. Rows are written to
results/main_comparison.csv as they finish, so the run is resumable: a cell that
already has a row is skipped unless --force is given. The classification
endpoints additionally get a scaffold-split pass into results/scaffold_comparison.csv.

The linear model is the primary one and is quick; the two tree models are heavier.
Use --models linear to get the headline table first.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from descbench.data import ROOT, load_config, load_processed, processed_path  # noqa: E402
from descbench.evaluate import MODELS, oof_predictions, scaffold_oof  # noqa: E402
from descbench.features import ALL_REPRESENTATIONS, features_path, load_features  # noqa: E402
from descbench import stats  # noqa: E402

MAIN = "main_comparison.csv"
SCAFFOLD = "scaffold_comparison.csv"


def _endpoints(cfg):
    return [ep for ep in cfg["endpoints"]
            if not ep.get("pending") and processed_path(cfg, ep["name"]).exists()]


def _model_settings(cfg, model):
    """Repeats and in-fold feature cap for a model. The primary linear model runs
    at the full protocol on all features; the tree models use the lighter settings
    from config's tree_models block (see the plan's 2026-09-11 amendments)."""
    tm = cfg.get("tree_models", {})
    if model in tm.get("models", []):
        return tm.get("n_repeats"), tm.get("select_k")
    return None, None


def _done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return set(zip(df["endpoint"], df["representation"], df["model"]))


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    pd.DataFrame([row]).to_csv(path, mode="a", header=header, index=False)


def _score_row(cfg, endpoint, rep, model, task, oof) -> dict:
    metric = stats.primary_metric(task)
    sec = stats.secondary_metric(task)
    primary = stats.score(oof.y_true, oof.y_pred, metric)
    lo, hi = stats.bootstrap_ci(oof.y_true, oof.y_pred, metric,
                                cfg["bootstrap"]["n_resamples"], cfg["bootstrap"]["ci"],
                                cfg["seed"])
    return {
        "endpoint": endpoint, "representation": rep, "model": model, "task": task,
        "primary_metric": metric, "primary_score": round(primary, 4),
        "ci_low": round(lo, 4), "ci_high": round(hi, 4),
        "baseline": round(stats.trivial_baseline(oof.y_true, task), 4),
        "secondary_metric": sec,
        "secondary_score": round(stats.score(oof.y_true, oof.y_pred, sec), 4),
        "n_molecules": len(oof.y_true),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--reps", nargs="+", default=list(ALL_REPRESENTATIONS))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-scaffold", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    main_path = ROOT / cfg["paths"]["results"] / MAIN
    scaffold_path = ROOT / cfg["paths"]["results"] / SCAFFOLD
    if args.force:
        main_path.unlink(missing_ok=True)
        scaffold_path.unlink(missing_ok=True)
    done = _done_keys(main_path)
    done_scaffold = _done_keys(scaffold_path)

    for ep in _endpoints(cfg):
        name, task = ep["name"], ep["task"]
        proc = load_processed(cfg, name)
        for rep in args.reps:
            if not features_path(cfg, name, rep).exists():
                print(f"{name}/{rep}: no features cached, skipping")
                continue
            X, y, _ = load_features(cfg, name, rep)
            for model in args.models:
                if (name, rep, model) in done:
                    continue
                t0 = time.perf_counter()
                n_repeats, select_k = _model_settings(cfg, model)
                oof = oof_predictions(X, y, task, rep, model, cfg,
                                      select_k=select_k, n_repeats=n_repeats)
                row = _score_row(cfg, name, rep, model, task, oof)
                _append(main_path, row)
                print(f"{name:20s} {rep:10s} {model:22s} "
                      f"{row['primary_metric']} {row['primary_score']:.3f} "
                      f"[{row['ci_low']:.3f},{row['ci_high']:.3f}]  "
                      f"base {row['baseline']:.3f}  ({time.perf_counter()-t0:.0f}s)")

                if task == "classification" and not args.no_scaffold \
                        and (name, rep, model) not in done_scaffold:
                    s_oof = scaffold_oof(X, y, proc["smiles"].tolist(), task, rep, model,
                                         cfg, select_k=select_k)
                    s_row = _score_row(cfg, name, rep, model, task, s_oof)
                    s_row["split"] = "scaffold"
                    _append(scaffold_path, s_row)

    if main_path.exists():
        df = pd.read_csv(main_path).sort_values(["endpoint", "model", "representation"])
        df.to_csv(main_path, index=False)
        print(f"\nWrote {main_path.relative_to(ROOT)} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
