"""3D representations for the 3D arm (plan section 9).

One conformer per molecule (fixed recipe), reused for every 3D representation and
written to SDF so alvaDesc can use the same geometry. Produces rdkit3d and mordred3d
(and alvadesc3d via the CLI), cached id-indexed like the 2D features; a molecule with
no conformer keeps a NaN row, imputed in-fold.
"""

from __future__ import annotations

import io
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors3D, rdMolDescriptors

from .data import ROOT, load_processed
from .features import features_path

RDLogger.DisableLog("rdApp.*")

N_EMBED = 10                 # ensemble size before keeping the lowest-energy one
_SEED_DEFAULT = 20260910


# --- conformer generation ----------------------------------------------------

def make_conformer(smiles: str, seed: int) -> Chem.Mol | None:
    """One low-energy 3D conformer with explicit hydrogens, or None on failure.

    The recipe is fixed in plan section 9.1: ETKDGv3, a 10-conformer ensemble, MMFF94
    with a UFF fallback, keep the lowest-energy conformer.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=N_EMBED, params=params))
    if not cids:
        # a single fixed attempt sometimes succeeds where the batch did not
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None
        cids = [0]

    energies = []
    for cid in cids:
        try:
            ff = AllChem.MMFFGetMoleculeForceField(
                mol, AllChem.MMFFGetMoleculeProperties(mol), confId=cid)
            if ff is None:
                raise ValueError("no MMFF params")
        except Exception:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
        if ff is None:
            energies.append((np.inf, cid))
            continue
        ff.Minimize(maxIts=500)
        energies.append((ff.CalcEnergy(), cid))

    best = min(energies, key=lambda t: t[0])[1]
    keep = Chem.Mol(mol)
    keep.RemoveAllConformers()
    keep.AddConformer(mol.GetConformer(best), assignId=True)
    return keep


def _conf_worker(arg: tuple[str, int]) -> Chem.Mol | None:
    """Module-level wrapper so make_conformer can run in a process pool. Each
    molecule is embedded from the same fixed seed, so the parallel result is
    identical to the serial one, molecule for molecule."""
    smiles, seed = arg
    return make_conformer(smiles, seed)


def build_conformers(smiles: list[str], seed: int) -> list[Chem.Mol | None]:
    """One conformer per molecule, generated in parallel across CPU cores.

    Embedding is by far the slowest part of the 3D arm and is independent per
    molecule, so it is spread over a process pool; order is preserved. Falls back
    to a serial pass if the pool cannot start (e.g. an odd Windows environment)."""
    n_workers = min(os.cpu_count() or 4, 12)
    args = [(s, seed) for s in smiles]
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            return list(ex.map(_conf_worker, args, chunksize=8))
    except Exception:
        return [make_conformer(s, seed) for s in smiles]


# --- rdkit 3D descriptor block ----------------------------------------------

def _rdkit3d_blocks(mol: Chem.Mol) -> dict[str, float]:
    """Every rdkit 3D descriptor for a conformer-bearing molecule, as a flat dict.

    The vector descriptors (WHIM etc.) return fixed-length lists, so the column
    set is identical for every molecule that has a conformer.
    """
    out: dict[str, float] = {}
    for name in ("Asphericity", "Eccentricity", "InertialShapeFactor", "NPR1",
                 "NPR2", "PMI1", "PMI2", "PMI3", "RadiusOfGyration",
                 "SpherocityIndex"):
        out[f"shape_{name}"] = float(getattr(Descriptors3D, name)(mol))
    for label, fn in (("WHIM", rdMolDescriptors.CalcWHIM),
                      ("GETAWAY", rdMolDescriptors.CalcGETAWAY),
                      ("AUTOCORR3D", rdMolDescriptors.CalcAUTOCORR3D),
                      ("RDF", rdMolDescriptors.CalcRDF),
                      ("MORSE", rdMolDescriptors.CalcMORSE)):
        vals = fn(mol)
        for i, v in enumerate(vals):
            out[f"{label}_{i}"] = float(v)
    return out


# The column template, computed once from a reference molecule so that a molecule
# whose conformer failed can be given an all-NaN row of the right width.
def _template_columns() -> list[str]:
    ref = make_conformer("c1ccccc1", _SEED_DEFAULT)
    return list(_rdkit3d_blocks(ref).keys())


_RDKIT3D_COLUMNS = _template_columns()


def rdkit3d(conformers: list[Chem.Mol | None]) -> pd.DataFrame:
    rows = []
    for mol in conformers:
        if mol is None:
            rows.append({c: np.nan for c in _RDKIT3D_COLUMNS})
            continue
        try:
            rows.append(_rdkit3d_blocks(mol))
        except Exception:
            rows.append({c: np.nan for c in _RDKIT3D_COLUMNS})
    return pd.DataFrame(rows, columns=_RDKIT3D_COLUMNS)


# --- mordred 3D --------------------------------------------------------------

def mordred3d(conformers: list[Chem.Mol | None]) -> pd.DataFrame:
    """Mordred with 3D enabled, on the conformers. Molecules without a conformer
    are passed as an empty mol so Mordred returns NaNs for the whole row rather
    than crashing the batch; the column set stays aligned.

    NOTE [added after the alvaDesc-developer review]: ignore_3D=False enables the
    3D descriptors *within the full Mordred catalogue*; it does not return a
    3D-only block. This table (about 1826 columns) is therefore Mordred 2D+3D, and
    it is used only as a 2D+3D representation in 03b/03c, never as a "3D-only" set.
    The only true 3D-only open block compared head-to-head with alvaDesc 3D is the
    RDKit one built in rdkit3d() below (shape + WHIM/GETAWAY/AUTOCORR3D/RDF/MORSE).
    A Mordred 3D-only subset was not isolated; doing so is left as follow-up."""
    from mordred import Calculator, descriptors

    calc = Calculator(descriptors, ignore_3D=False)
    safe = [m if m is not None else Chem.Mol() for m in conformers]
    df = calc.pandas(safe, quiet=True, nproc=1)
    df = df.apply(pd.to_numeric, errors="coerce")
    df.columns = [str(c) for c in df.columns]
    return df.reset_index(drop=True)


# --- build + cache all three 3D artefacts for one endpoint -------------------

def compute_3d_features(cfg: dict, endpoint: str, force: bool = False) -> dict[str, Path]:
    """Generate conformers once and cache rdkit3d, mordred3d and an SDF.

    Returns the paths written. Re-running is cheap because it skips artefacts that
    already exist unless force is set.
    """
    seed = cfg.get("seed", _SEED_DEFAULT)
    feat_dir = ROOT / cfg["paths"]["features"]
    feat_dir.mkdir(parents=True, exist_ok=True)
    rdkit_path = features_path(cfg, endpoint, "rdkit3d")
    mordred_path = features_path(cfg, endpoint, "mordred3d")
    sdf_path = feat_dir / f"{endpoint}__conformers.sdf"

    done = {p for p in (rdkit_path, mordred_path, sdf_path) if p.exists()}
    if len(done) == 3 and not force:
        return {"rdkit3d": rdkit_path, "mordred3d": mordred_path, "sdf": sdf_path}

    proc = load_processed(cfg, endpoint)
    ids = proc["id"].to_numpy()
    smiles = proc["smiles"].tolist()

    conformers = build_conformers(smiles, seed)
    n_failed = sum(m is None for m in conformers)
    print(f"  {endpoint}: {len(conformers) - n_failed}/{len(conformers)} conformers "
          f"built ({n_failed} failed, kept as NaN rows)")

    # SDF for the alvaDesc arm: one record per molecule, titled with its id, so the
    # geometry alvaDesc sees is exactly the one used here. Failed molecules are
    # written as an empty record placeholder to keep the record order aligned.
    writer = Chem.SDWriter(str(sdf_path))
    for mol_id, mol in zip(ids, conformers):
        m = mol if mol is not None else Chem.Mol()
        m.SetProp("_Name", str(mol_id))
        writer.write(m)
    writer.close()

    rk = rdkit3d(conformers); rk.index = ids; rk.index.name = "id"
    rk.to_parquet(rdkit_path)
    md = mordred3d(conformers); md.index = ids; md.index.name = "id"
    md.to_parquet(mordred_path)

    return {"rdkit3d": rdkit_path, "mordred3d": mordred_path, "sdf": sdf_path}


# --- alvaDesc 3D (the commercial arm, computed from the same conformers) ------

def _alvadesc_3d_names(cfg: dict, cli: str) -> list[str]:
    """The 3D descriptor names (every descriptor flagged 3D=Yes), cached beside
    the features. These are the blocks the 2D study switched off."""
    cache = ROOT / cfg["paths"]["features"] / "alvadesc_3d_names.txt"
    if cache.exists():
        return cache.read_text(encoding="utf-8").split()
    listing = subprocess.run([cli, "--descriptorlist"], capture_output=True,
                             check=True).stdout
    table = pd.read_csv(io.BytesIO(listing), sep="\t", encoding="latin-1")
    is_3d = table["3D"].astype(str).str.strip().str.lower().eq("yes")
    names = table.loc[is_3d, "Name"].astype(str).tolist()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("\n".join(names), encoding="utf-8")
    return names


def compute_alvadesc3d(cfg: dict, endpoint: str, force: bool = False) -> Path:
    """Run alvaDesc on the endpoint's conformer SDF and cache the 3D descriptor
    columns, id-indexed. The geometry is the same one the open 3D representations
    used, so alvaDesc gets no better a conformer than RDKit or Mordred did. A
    molecule whose conformer failed (an empty SDF record) is absent from alvaDesc's
    output and is left as a NaN row, matching how failures are handled elsewhere.
    """
    out = features_path(cfg, endpoint, "alvadesc3d")
    if out.exists() and not force:
        return out
    cli = os.environ.get("ALVADESC_CLI", cfg["alvadesc"]["cli"])
    names_3d = set(_alvadesc_3d_names(cfg, cli))
    sdf = ROOT / cfg["paths"]["features"] / f"{endpoint}__conformers.sdf"
    if not sdf.exists():
        raise FileNotFoundError(f"{sdf} missing; run 02b_features_3d.py first.")
    ids = load_processed(cfg, endpoint)["id"].to_numpy()

    with tempfile.TemporaryDirectory() as tmp:
        out_txt = Path(tmp) / "alva3d.txt"
        subprocess.run(
            [cli, f"--input={sdf}", "--inputtype=MDL", "--descriptors=ALL",
             f"--output={out_txt}", "--labels"],
            check=True, capture_output=True,
        )
        table = pd.read_csv(out_txt, sep="\t", encoding="latin-1", low_memory=False)

    table = table.rename(columns={"NAME": "id"}).drop(columns=["No."], errors="ignore")
    keep = ["id"] + [c for c in table.columns if c in names_3d]
    table = table[keep].set_index("id").apply(pd.to_numeric, errors="coerce")
    table = table.reindex(ids)
    table.index.name = "id"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out)
    return out
