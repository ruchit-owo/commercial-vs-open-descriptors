"""Download, standardise and cache the endpoint datasets.

One curation path for every endpoint: standardise SMILES, dedupe, and write
data/processed/<name>.csv with columns id, smiles, target, task. Bradley melting
point has its own branch (quality flag, replicate averaging, seeded subsample).
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

# RDKit is very chatty about sanitisation on messy public data. We count and
# report failures ourselves, so the per-molecule warnings are just noise.
RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def endpoint_config(cfg: dict, name: str) -> dict:
    for ep in cfg["endpoints"]:
        if ep["name"] == name:
            return ep
    raise KeyError(f"Endpoint {name!r} is not in config.yaml")


# --- SMILES standardisation --------------------------------------------------
# Built once and reused; these objects are stateless choosers.
_largest_fragment = rdMolStandardize.LargestFragmentChooser()
_uncharger = rdMolStandardize.Uncharger()


def standardise_smiles(smiles: str) -> str | None:
    """Canonical SMILES for the largest, neutralised fragment, or None if RDKit
    cannot parse it. Salts and counterions fall away with the smaller fragments;
    neutralising makes protonation states comparable across sources."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = _largest_fragment.choose(mol)
        mol = _uncharger.uncharge(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


@dataclass
class CurationReport:
    endpoint: str
    rows_in: int
    parse_failures: int
    duplicates_collapsed: int
    conflicts_dropped: int
    rows_out: int

    def as_row(self) -> dict:
        return self.__dict__.copy()


def _dedupe(df: pd.DataFrame, task: str) -> tuple[pd.DataFrame, int, int]:
    """Collapse duplicate canonical SMILES. Regression targets are averaged
    (replicate measurements of a continuous property); classification duplicates
    with disagreeing labels are dropped, since we cannot know which is right."""
    n_before = len(df)
    conflicts_dropped = 0
    if task == "regression":
        agg = df.groupby("smiles", as_index=False, sort=False)["target"].mean()
    else:
        # Keep SMILES where every duplicate agrees on the label; drop the rest.
        label_counts = df.groupby("smiles")["target"].nunique()
        conflicted = label_counts[label_counts > 1].index
        conflicts_dropped = int(df["smiles"].isin(conflicted).sum())
        clean = df[~df["smiles"].isin(conflicted)]
        agg = clean.groupby("smiles", as_index=False, sort=False)["target"].first()
    duplicates_collapsed = n_before - len(agg) - conflicts_dropped
    return agg, duplicates_collapsed, conflicts_dropped


def _standardise_frame(raw: pd.DataFrame, smiles_col: str, target_col: str,
                       task: str) -> tuple[pd.DataFrame, int]:
    df = raw[[smiles_col, target_col]].rename(
        columns={smiles_col: "raw_smiles", target_col: "target"}
    )
    df = df.dropna(subset=["raw_smiles", "target"])
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df = df.dropna(subset=["target"])
    df["smiles"] = df["raw_smiles"].map(standardise_smiles)
    parse_failures = int(df["smiles"].isna().sum())
    df = df.dropna(subset=["smiles"])
    return df[["smiles", "target"]], parse_failures


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    urllib.request.urlretrieve(url, dest)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_raw(cfg: dict, ep: dict) -> pd.DataFrame:
    raw_dir = ROOT / cfg["paths"]["raw"]
    source = ep["source"]
    path = raw_dir / source
    if ep.get("local"):
        # An endpoint supplied as a local file rather than downloaded (e.g. the
        # Kraken ligand library). Read by extension.
        if not path.exists():
            raise FileNotFoundError(
                f"{source} not found in {raw_dir}; place it there by hand."
            )
        return pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") \
            else pd.read_csv(path)
    if ep["name"] == "bradley_mp":
        if not path.exists():
            raise FileNotFoundError(
                f"Place {source} in {raw_dir} by hand (see PROVENANCE.md)."
            )
        return pd.read_excel(path)
    if ep["name"] == "nucleoside_gelation":
        if not path.exists():
            raise FileNotFoundError(
                f"{source} not found in {raw_dir}; this endpoint is supplied "
                "separately (see config: pending)."
            )
        return pd.read_csv(path)
    # MoleculeNet mirror files.
    _download(cfg["deepchem_base"] + source, path)
    return pd.read_csv(path)


def curate_endpoint(cfg: dict, name: str) -> tuple[pd.DataFrame, CurationReport]:
    """Return the tidy (id, smiles, target, task) frame for one endpoint plus a
    report of how many rows were lost at each step."""
    ep = endpoint_config(cfg, name)
    task = ep["task"]
    raw = _load_raw(cfg, ep)
    rows_in = len(raw)

    if name == "bradley_mp":
        # The dataset ships its own quality flag; honour it before anything else.
        flagged = raw["donotuse"].astype(str).str.strip().str.lower().eq("x")
        raw = raw[~flagged]

    tidy, parse_failures = _standardise_frame(
        raw, ep["smiles_col"], ep["target_col"], task
    )
    deduped, duplicates_collapsed, conflicts_dropped = _dedupe(tidy, task)

    if ep.get("subsample") and len(deduped) > ep["subsample"]:
        # Seeded so the drawn set is reproducible; the plan commits to recording
        # which molecules were kept, which the stable id below makes possible.
        deduped = deduped.sample(
            n=ep["subsample"], random_state=cfg["seed"]
        ).reset_index(drop=True)

    deduped = deduped.sort_values("smiles").reset_index(drop=True)
    deduped.insert(0, "id", [f"{name}_{i:06d}" for i in range(len(deduped))])
    deduped["task"] = task

    report = CurationReport(
        endpoint=name,
        rows_in=rows_in,
        parse_failures=parse_failures,
        duplicates_collapsed=duplicates_collapsed,
        conflicts_dropped=conflicts_dropped,
        rows_out=len(deduped),
    )
    return deduped[["id", "smiles", "target", "task"]], report


def processed_path(cfg: dict, name: str) -> Path:
    return ROOT / cfg["paths"]["processed"] / f"{name}.csv"


def load_processed(cfg: dict, name: str) -> pd.DataFrame:
    path = processed_path(cfg, name)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run 01_curate.py first.")
    return pd.read_csv(path)
