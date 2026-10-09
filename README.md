# Commercial vs open molecular descriptors

This repository compares a commercial molecular-descriptor package (alvaDesc)
against open-source alternatives (RDKit 2D descriptors, Mordred, and the ECFP4 and
MACCS fingerprints) across six public small-molecule property endpoints, under one
identical leak-free evaluation and a decision rule fixed in advance. The question is
practical: for the kind of two-dimensional QSPR a materials or soft-matter lab does,
do the commercial descriptors predict better than free alternatives, and is any gain
a matter of descriptor quality or of having more columns?

## Summary of findings (provisional, six endpoints)

Across six public endpoints under one leak-free protocol, alvaDesc had the best mean
rank, but its median advantage over the best open representation (Mordred) was 0.004
on a scale-free metric, with a confidence interval whose lower end sits at zero and a
Wilcoxon signed-rank test that does not reject at the 0.05 level. When every
representation was cut to a common target feature budget inside each fold, the
advantage did not persist, and on the two hardest regression endpoints the RDKit set
was the most accurate. A pre-registered 3D extension found no endpoint where adding 3D
descriptors (open or alvaDesc's own) cleared the pre-set 0.02 threshold; taken on
their own, alvaDesc's 3D descriptors scored higher than RDKit's 3D descriptors on all
six endpoints, which is a descriptive comparison without a paired interval. On a
deliberately 3D-driven test (phosphine-ligand buried volume, from the Kraken library)
3D did help, and the free RDKit shape descriptors matched alvaDesc there.

These are observations from a small panel, which is a limited sample for a paired
test, and are not a general claim about commercial versus open descriptors. The
pre-registration, metrics and decision rule are in `ANALYSIS_PLAN.md`.

## What is here

- `ANALYSIS_PLAN.md` — the pre-registration. Endpoints, metrics, models and the
  decision rule were all fixed before any result was computed, so the conclusion
  cannot have been chosen after the fact. It is not edited once results exist;
  changes go in its amendments table.
- `src/descbench/` — the library: curation, the five representations, the single
  leak-free evaluator, the statistics, and the cost accounting.
- `experiments/` — the numbered scripts that run the 2D study end to end
  (`01`–`06`), plus `02b`/`03b` for the pre-registered 3D pre-check, `03c` for the
  alvaDesc 3D head-to-head, `03d` for the Kraken exploratory test, and `04b` for the
  effective-width check on the matched budget (plan §9 and its amendments).
- `results/`, `tables/`, `figures/` — the generated outputs.

## Reproducing

```bash
python -m venv .venv && .venv/Scripts/activate    # or source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

`run_all.py` runs the phases in order: curate, features, main comparison, matched
budget, practical cost, figures. One input is not downloaded automatically and must
be placed by hand: `BradleyMeltingPointDataset.xlsx` in `data/raw/` (from figshare,
doi:10.6084/m9.figshare.1031637). The other five endpoints download from the
MoleculeNet mirror on first run. A seventh endpoint (nucleoside gelation) is left
pending and is not part of the results. Provenance and file hashes are in
`PROVENANCE.md`.

alvaDesc is needed only for the feature-export phase and only if you want to
regenerate its descriptor tables; everything downstream reads the cached feature
files. The path to `alvaDescCLI.exe` is set in `config.yaml` (or the `ALVADESC_CLI`
environment variable). Without a licence, the four open representations still run and
the comparison degrades gracefully to those.

## Method in one paragraph

Each representation is turned into a feature matrix and passed through the same
scikit-learn pipeline, in which every data-dependent step (imputation, variance
filtering, scaling, any feature selection, and the hyperparameter search) is fitted
inside the training fold only. The same repeated five-by-five folds, seeded once, are
reused for every representation, so the comparison is paired at the molecule level;
predictions are pooled out of fold and scored once, with a bootstrap interval taken
over molecules. The primary metric is scale-free (ROC-AUC for classification,
Spearman's rho for regression) so the endpoints can be compared, and every score is
printed next to its trivial baseline. The full protocol, and why each choice was
made, is in the analysis plan.

## Tests

```bash
pip install pytest
pytest
```

The suite includes a leakage guard, the test the whole comparison depends on. It
builds a dataset of pure-noise features and checks that the leak-free evaluator
scores at chance while a deliberately leaky variant does not, so a preprocessing step
that escaped the fold would fail the build.
