"""Leak-free evaluator shared by every representation.

All preprocessing and the hyperparameter search sit inside one sklearn Pipeline
fitted on the training fold only, so the comparison is fair and leak-free. OOF
predictions are averaged over repeats and scored once (see descbench.stats).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.feature_selection import (
    SelectKBest, VarianceThreshold, f_classif, f_regression,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import (
    GridSearchCV, KFold, RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FINGERPRINTS = {"ecfp4", "maccs"}
MODELS = ("linear", "random_forest", "hist_gradient_boosting")
INNER_CV = 3


# --- Estimators and grids ----------------------------------------------------
# Grids are read from config and turned into GridSearchCV param dicts keyed on the
# pipeline's "model__" prefix. The same dict is used for every representation; the
# fair-protocol test relies on that.
def _estimator(model: str, task: str, seed: int):
    if model == "linear":
        if task == "classification":
            # penalty defaults to l2; passing it explicitly is deprecated in
            # sklearn 1.8+, so we rely on the default with the liblinear solver.
            return LogisticRegression(solver="liblinear", max_iter=1000)
        return Ridge()
    if model == "random_forest":
        cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
        # Trees are built with a thread backend that shares the training matrix, so
        # n_jobs here parallelises across cores without copying the data - unlike
        # parallelising the grid search, which would clone the matrix per worker.
        return cls(random_state=seed, n_jobs=-1)
    if model == "hist_gradient_boosting":
        cls = (HistGradientBoostingClassifier if task == "classification"
               else HistGradientBoostingRegressor)
        return cls(random_state=seed)
    raise ValueError(f"unknown model {model!r}")


def param_grid(model: str, task: str, cfg: dict) -> dict:
    m = cfg["models"]
    if model == "linear":
        if task == "classification":
            return {"model__C": list(m["linear"]["classification"]["C"])}
        return {"model__alpha": list(m["linear"]["regression"]["alpha"])}
    if model == "random_forest":
        rf = m["random_forest"]
        return {
            "model__n_estimators": list(rf["n_estimators"]),
            "model__max_depth": list(rf["max_depth"]),
            "model__max_features": list(rf["max_features"]),
        }
    if model == "hist_gradient_boosting":
        hg = m["hist_gradient_boosting"]
        return {
            "model__learning_rate": list(hg["learning_rate"]),
            "model__max_depth": list(hg["max_depth"]),
        }
    raise ValueError(f"unknown model {model!r}")


def _fixed_params(model: str, cfg: dict) -> dict:
    if model == "hist_gradient_boosting":
        hg = cfg["models"]["hist_gradient_boosting"]
        return {"model__max_iter": hg["max_iter"],
                "model__early_stopping": hg["early_stopping"]}
    return {}


def build_pipeline(model: str, task: str, rep: str, cfg: dict, seed: int,
                   select_k: int | None = None) -> Pipeline:
    """One pipeline for a (model, representation). Fingerprints skip scaling;
    descriptors are imputed and scaled. When select_k is given (the matched-budget
    run) a univariate selector is inserted so every representation is cut to the
    same width inside the fold."""
    steps: list[tuple] = [
        ("impute", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(0.0)),
    ]
    if rep not in FINGERPRINTS:
        steps.append(("scale", StandardScaler()))
    if select_k is not None:
        score_func = f_classif if task == "classification" else f_regression
        # select_k is a TARGET budget, not a guaranteed effective width. The
        # VarianceThreshold above runs first and drops constant columns, and
        # SelectKBest selects min(k, columns surviving the variance filter). So a
        # representation with fewer usable columns than k keeps all of them: MACCS
        # (166 usable bits) never reaches a 200 target, and RDKit (217) falls a
        # little short on some folds. The effective widths are recorded by
        # experiments/04b_effective_widths.py -> results/effective_widths.csv.
        # [added after the alvaDesc-developer review: do not claim an exact k here.]
        steps.append(("select", SelectKBest(score_func=score_func, k=select_k)))
    est = _estimator(model, task, seed)
    est.set_params(**{k.replace("model__", ""): v
                      for k, v in _fixed_params(model, cfg).items()})
    steps.append(("model", est))
    return Pipeline(steps)


def _outer_splits(y: np.ndarray, task: str, cfg: dict, n_repeats: int | None = None):
    n_splits = cfg["cv"]["n_splits"]
    if n_repeats is None:
        n_repeats = cfg["cv"]["n_repeats"]
    seed = cfg["seed"]
    if task == "classification":
        rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    else:
        rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    # Group the flat split stream back into repeats so predictions can be averaged
    # within a molecule across repeats.
    splits = list(rkf.split(np.zeros(len(y)), y if task == "classification" else None))
    return [splits[i * n_splits:(i + 1) * n_splits] for i in range(n_repeats)]


def _inner_cv(task: str, seed: int):
    if task == "classification":
        return StratifiedKFold(n_splits=INNER_CV, shuffle=True, random_state=seed)
    return KFold(n_splits=INNER_CV, shuffle=True, random_state=seed)


@dataclass
class OOFResult:
    y_true: np.ndarray
    y_pred: np.ndarray        # averaged over repeats; probability for classification
    task: str


def oof_predictions(X: pd.DataFrame, y: np.ndarray, task: str, rep: str,
                    model: str, cfg: dict, select_k: int | None = None,
                    n_repeats: int | None = None) -> OOFResult:
    """Nested cross-validation. For each repeat, every molecule is predicted once
    from a model tuned only on the other folds; predictions are then averaged over
    the repeats. n_repeats and select_k allow the caller to give the tree models a
    lighter repeat count and an in-fold feature cap (see the plan's amendments),
    while the linear model runs at the full protocol on all features."""
    seed = cfg["seed"]
    grid = param_grid(model, task, cfg)
    scoring = "roc_auc" if task == "classification" else "neg_root_mean_squared_error"
    repeats = _outer_splits(y, task, cfg, n_repeats=n_repeats)
    preds = np.full((len(repeats), len(y)), np.nan)

    for r, folds in enumerate(repeats):
        for train_idx, test_idx in folds:
            pipe = build_pipeline(model, task, rep, cfg, seed, select_k=select_k)
            k = _guarded_k(select_k, X.shape[1])
            if k is not None:
                pipe.set_params(select__k=k)
            search = GridSearchCV(pipe, grid, scoring=scoring, cv=_inner_cv(task, seed),
                                  n_jobs=1, refit=True)
            search.fit(X.iloc[train_idx], y[train_idx])
            if task == "classification":
                preds[r, test_idx] = search.predict_proba(X.iloc[test_idx])[:, 1]
            else:
                preds[r, test_idx] = search.predict(X.iloc[test_idx])
    return OOFResult(y_true=y, y_pred=np.nanmean(preds, axis=0), task=task)


def _guarded_k(select_k: int | None, n_cols: int) -> int | None:
    if select_k is None:
        return None
    return min(select_k, n_cols)


# --- Scaffold split (harder generalisation test) -----------------------------
def scaffold_oof(X: pd.DataFrame, y: np.ndarray, smiles: list[str], task: str,
                 rep: str, model: str, cfg: dict, select_k: int | None = None) -> OOFResult:
    """A single scaffold-grouped k-fold pass. Molecules sharing a Bemis-Murcko
    scaffold are kept together, so the test folds contain chemistry the training
    folds have not seen. Reported alongside the random-split numbers."""
    from .splits import scaffold_folds

    folds = scaffold_folds(smiles, n_splits=cfg["cv"]["n_splits"], seed=cfg["seed"])
    grid = param_grid(model, task, cfg)
    scoring = "roc_auc" if task == "classification" else "neg_root_mean_squared_error"
    seed = cfg["seed"]
    preds = np.full(len(y), np.nan)
    for train_idx, test_idx in folds:
        pipe = build_pipeline(model, task, rep, cfg, seed, select_k=select_k)
        k = _guarded_k(select_k, X.shape[1])
        if k is not None:
            pipe.set_params(select__k=k)
        search = GridSearchCV(pipe, grid, scoring=scoring, cv=_inner_cv(task, seed),
                              n_jobs=1, refit=True)
        search.fit(X.iloc[train_idx], y[train_idx])
        if task == "classification":
            preds[test_idx] = search.predict_proba(X.iloc[test_idx])[:, 1]
        else:
            preds[test_idx] = search.predict(X.iloc[test_idx])
    return OOFResult(y_true=y, y_pred=preds, task=task)
