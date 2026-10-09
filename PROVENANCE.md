# Provenance

Where each dataset comes from, and what curation was applied. The raw files
themselves are git-ignored (see `.gitignore`); this file is the manifest that
makes them recoverable. Curation is implemented in `experiments/01_curate.py`
and `src/descbench/data.py`; the counts below are filled in by that script and
should match what it prints.

## Sources

Five endpoints download from the DeepChem/MoleculeNet mirror
(`https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/`). Bradley melting
point comes from figshare. Nucleoside gelation comes from earlier work in this
portfolio and is supplied as a small CSV.

| Endpoint | File | Origin | SHA256 (raw) |
|---|---|---|---|
| esol | delaney-processed.csv | MoleculeNet mirror | 8c06a76f0c6487d2 |
| freesolv | SAMPL.csv | MoleculeNet mirror | ab5895d914ee87cb |
| lipophilicity | Lipophilicity.csv | MoleculeNet mirror | aed41590cb30609d |
| bradley_mp | BradleyMeltingPointDataset.xlsx | figshare doi:10.6084/m9.figshare.1031637 | e8a66818609bbad6 |
| bace | bace.csv | MoleculeNet mirror | f3fb9ce90bada3e2 |
| bbbp | BBBP.csv | MoleculeNet mirror | d07a38487aeac5ce |
| nucleoside_gelation | nucleoside_gelation.csv | own prior work (nucleoside-hydrogel-repro) | _filled by 01_curate_ |

Bradley melting point must be placed in `data/raw/` by hand
(`BradleyMeltingPointDataset.xlsx`); it is not downloaded automatically.

## Curation summary

Standardisation is identical for every endpoint (see `data.py`): keep the largest
organic fragment, strip salts, neutralise where sensible, canonicalise with RDKit,
drop what will not parse, and deduplicate by canonical SMILES. For regression
duplicates the target is averaged; for classification duplicates with conflicting
labels the molecule is dropped. Per-endpoint counts (rows in, parse failures,
duplicates collapsed, rows out) are written by `01_curate.py` to the table below.

| Endpoint | Rows in | Parse failures | Duplicates collapsed | Conflicts dropped | Rows out |
|---|---|---|---|---|---|
| esol | 1128 | 0 | 11 | 0 | 1117 |
| freesolv | 642 | 0 | 0 | 0 | 642 |
| lipophilicity | 4200 | 0 | 0 | 0 | 4200 |
| bradley_mp | 28645 | 292 | 8038 | 0 | 10000 |
| bace | 1513 | 0 | 1 | 0 | 1512 |
| bbbp | 2050 | 11 | 62 | 22 | 1955 |

Bradley melting point drops 377 rows flagged in its own `donotuse` column before
these counts, and the 10 000 figure is the seeded subsample of the roughly 19 900
molecules that remain after curation. The processed counts sit a little below the
raw parseable counts recorded in the plan because deduplication removes a further
handful per endpoint; BBBP in particular loses 22 molecules whose duplicate SMILES
carried disagreeing labels. Regenerate this table with `python
experiments/01_curate.py`.
