"""Scaffold-based folds for the harder generalisation test. Molecules are grouped
by Bemis-Murcko scaffold and whole groups dealt into folds (largest first, into the
smallest fold), so a test fold holds scaffolds the training folds did not.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


def murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def scaffold_folds(smiles: list[str], n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    scaffolds: dict[str, list[int]] = {}
    for i, smi in enumerate(smiles):
        scaffolds.setdefault(murcko_scaffold(smi), []).append(i)

    # Largest scaffold groups first; ties broken deterministically by the seed so
    # the assignment is reproducible without depending on dict order.
    rng = np.random.default_rng(seed)
    groups = list(scaffolds.values())
    order = sorted(range(len(groups)),
                   key=lambda g: (len(groups[g]), rng.random()), reverse=True)

    fold_members: list[list[int]] = [[] for _ in range(n_splits)]
    for g in order:
        target = min(range(n_splits), key=lambda f: len(fold_members[f]))
        fold_members[target].extend(groups[g])

    all_idx = np.arange(len(smiles))
    folds = []
    for f in range(n_splits):
        test = np.array(sorted(fold_members[f]), dtype=int)
        train = np.array(sorted(set(all_idx) - set(test)), dtype=int)
        folds.append((train, test))
    return folds
