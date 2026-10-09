"""Compute, cache and load the five representations.

Each representation is cached id-indexed under data/features/<endpoint>__<rep>.parquet.
load_features returns (X, y) aligned to the processed ids. It does not drop rows: a
missing molecule raises, except for alvaDesc where failures are kept as NaN and
imputed in-fold. alvaDesc needs the commercial CLI; the rest are RDKit/Mordred.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

from .data import ROOT, load_processed

RDLogger.DisableLog("rdApp.*")

OPEN_REPRESENTATIONS = ("rdkit2d", "mordred", "ecfp4", "maccs")
ALL_REPRESENTATIONS = ("alvadesc",) + OPEN_REPRESENTATIONS


def features_path(cfg: dict, endpoint: str, rep: str) -> Path:
    return ROOT / cfg["paths"]["features"] / f"{endpoint}__{rep}.parquet"


# --- Open representations ----------------------------------------------------
_rdkit_names = [name for name, _ in Descriptors._descList]
_rdkit_calc = MolecularDescriptorCalculator(_rdkit_names)
_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _rdkit2d(mols: list[Chem.Mol]) -> pd.DataFrame:
    rows = []
    for mol in mols:
        # CalcDescriptors returns a plain tuple; bad values come back as NaN/inf
        # and are dealt with by the in-fold imputer, not here.
        rows.append(_rdkit_calc.CalcDescriptors(mol))
    arr = np.array(rows, dtype=float)
    return pd.DataFrame(arr, columns=_rdkit_names)


def _ecfp4(mols: list[Chem.Mol]) -> pd.DataFrame:
    arr = np.vstack([_morgan_gen.GetFingerprintAsNumPy(mol) for mol in mols])
    cols = [f"ecfp4_{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr.astype(np.uint8), columns=cols)


def _maccs(mols: list[Chem.Mol]) -> pd.DataFrame:
    # MACCSkeys returns 167 bits (bit 0 is unused but kept for the standard layout).
    arr = np.vstack([np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.uint8) for mol in mols])
    cols = [f"maccs_{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=cols)


def _mordred(mols: list[Chem.Mol]) -> pd.DataFrame:
    # Imported lazily: the open stack still works without it, and its absence is a
    # recordable cost. We use the maintained mordredcommunity fork because the
    # original mordred does not install on current Python.
    from mordred import Calculator, descriptors

    calc = Calculator(descriptors, ignore_3D=True)
    df = calc.pandas(mols, quiet=True, nproc=1)
    # Mordred marks failures with a Missing/Error object; coerce the whole table
    # to numeric so those become NaN and can be imputed in the fold.
    df = df.apply(pd.to_numeric, errors="coerce")
    df.columns = [str(c) for c in df.columns]
    return df.reset_index(drop=True)


_OPEN_FUNCS = {
    "rdkit2d": _rdkit2d,
    "ecfp4": _ecfp4,
    "maccs": _maccs,
    "mordred": _mordred,
}


def compute_open_representation(rep: str, smiles: list[str]) -> pd.DataFrame:
    if rep not in _OPEN_FUNCS:
        raise ValueError(f"{rep!r} is not an open representation")
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    if any(m is None for m in mols):
        # Curation already canonicalised every SMILES, so a parse failure here is
        # a real bug rather than dirty input.
        bad = [s for s, m in zip(smiles, mols) if m is None]
        raise ValueError(f"{rep}: {len(bad)} processed SMILES failed to parse, e.g. {bad[:3]}")
    return _OPEN_FUNCS[rep](mols)


# --- alvaDesc ----------------------------------------------------------------
def _alvadesc_cli(cfg: dict) -> str:
    return os.environ.get("ALVADESC_CLI", cfg["alvadesc"]["cli"])


def _load_2d_descriptor_names(cfg: dict, cli: str) -> list[str]:
    """The 2D descriptor names, i.e. every descriptor flagged 3D=No. Cached
    next to the features so the exact set used is recorded with the run."""
    cache = ROOT / cfg["paths"]["features"] / "alvadesc_2d_names.txt"
    if cache.exists():
        return cache.read_text(encoding="utf-8").split()
    listing = subprocess.run(
        [cli, "--descriptorlist"], capture_output=True, text=False, check=True
    ).stdout
    # The descriptor list carries accented characters in its descriptions.
    table = pd.read_csv(pd.io.common.BytesIO(listing), sep="\t", encoding="latin-1")
    is_2d = table["3D"].astype(str).str.strip().str.lower().eq("no")
    names = table.loc[is_2d, "Name"].astype(str).tolist()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("\n".join(names), encoding="utf-8")
    return names


def compute_alvadesc(cfg: dict, smiles: list[str], ids: list[str]) -> tuple[pd.DataFrame, int]:
    """Run alvaDescCLI on the given molecules and return the 2D descriptor table
    (id-indexed) together with the number of molecules alvaDesc dropped.

    All 2D blocks, no 3D: we request every descriptor and keep only the columns
    flagged 3D=No. On SMILES input the 3D descriptors are all NaN anyway, so this
    is faithful to "2D only" and avoids the Windows command-line length limit.
    """
    cli = _alvadesc_cli(cfg)
    names_2d = set(_load_2d_descriptor_names(cfg, cli))

    with tempfile.TemporaryDirectory() as tmp:
        smi_path = Path(tmp) / "input.smi"
        out_path = Path(tmp) / "output.txt"
        # SMILES <tab> id, one per line; the id lands in alvaDesc's NAME column so
        # the table aligns by id rather than fragile row order.
        smi_path.write_text(
            "".join(f"{s}\t{i}\n" for s, i in zip(smiles, ids)), encoding="ascii"
        )
        subprocess.run(
            [cli, f"--input={smi_path}", "--inputtype=SMILES",
             "--descriptors=ALL", f"--output={out_path}", "--labels"],
            check=True, capture_output=True,
        )
        table = pd.read_csv(out_path, sep="\t", encoding="latin-1")

    table = table.rename(columns={"NAME": "id"}).drop(columns=["No."], errors="ignore")
    keep = ["id"] + [c for c in table.columns if c in names_2d]
    table = table[keep].set_index("id")
    table = table.apply(pd.to_numeric, errors="coerce")
    n_failed = len(ids) - table.index.isin(ids).sum()
    return table, int(n_failed)


# --- Cache + aligned loading -------------------------------------------------
def build_representation(cfg: dict, endpoint: str, rep: str, force: bool = False) -> Path:
    """Compute one (endpoint, representation) table and cache it, id-indexed."""
    out = features_path(cfg, endpoint, rep)
    if out.exists() and not force:
        return out
    proc = load_processed(cfg, endpoint)
    out.parent.mkdir(parents=True, exist_ok=True)

    if rep == "alvadesc":
        table, _ = compute_alvadesc(cfg, proc["smiles"].tolist(), proc["id"].tolist())
    else:
        table = compute_open_representation(rep, proc["smiles"].tolist())
        table.index = proc["id"].to_numpy()
    table.index.name = "id"
    table.to_parquet(out)
    return out


@dataclass
class FeatureInfo:
    endpoint: str
    representation: str
    n_columns: int
    n_molecules: int
    n_missing_molecules: int      # rows absent from the table entirely
    missing_cell_fraction: float  # NaNs across the aligned matrix


def load_features(cfg: dict, endpoint: str, rep: str,
                  allow_missing: bool | None = None) -> tuple[pd.DataFrame, np.ndarray, FeatureInfo]:
    """Return (X, y, info) with X's rows in the processed frame's id order.

    Rows are never silently dropped. If a molecule is missing from the cached
    table the loader raises, unless allow_missing is set - which defaults to True
    only for alvaDesc, where a handful of failures is expected and imputed."""
    if allow_missing is None:
        allow_missing = rep == "alvadesc"

    proc = load_processed(cfg, endpoint)
    path = features_path(cfg, endpoint, rep)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run 02_features.py first.")
    table = pd.read_parquet(path)

    ids = proc["id"].to_numpy()
    missing = [i for i in ids if i not in table.index]
    if missing and not allow_missing:
        raise KeyError(
            f"{rep}/{endpoint}: {len(missing)} molecules missing from the feature "
            f"table (e.g. {missing[:3]}); refusing to drop rows."
        )
    X = table.reindex(ids)
    y = proc["target"].to_numpy()

    info = FeatureInfo(
        endpoint=endpoint,
        representation=rep,
        n_columns=X.shape[1],
        n_molecules=len(ids),
        n_missing_molecules=len(missing),
        missing_cell_fraction=float(np.isnan(X.to_numpy(dtype=float)).mean()),
    )
    return X, y, info
