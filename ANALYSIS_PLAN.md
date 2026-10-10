# Analysis plan — commercial vs open molecular descriptors

This plan is fixed in advance. With only a handful of endpoints it would
otherwise be easy to arrive at whichever conclusion looked better by choosing
endpoints or metrics after seeing the numbers, so the endpoint list, the metrics,
the models and the decision rule are all committed here before any result is
computed.

Written: 2026-09-10
Author: Ruchit Ambilduke, Indian Institute of Science, Bangalore
Status: pre-registered before any results were computed.

## 1. Question

Do commercial molecular descriptors (alvaDesc) predict better than free
open-source alternatives (RDKit 2D, Mordred 2D, ECFP4, MACCS), across a range of
small-molecule property endpoints, under one identical leak-free evaluation?

Secondary: if there is an advantage, is it explained simply by alvaDesc producing
far more features; and what does the package cost in compute time, failure rate,
and the licence it requires?

## 2. Endpoints (fixed list — this is the pre-registration)

Seven endpoints. All will be run and reported, including any where the result is
inconvenient. Nothing is added or removed after results exist except through the
amendments section (§8).

Sources marked with the base URL were checked on 2026-09-04: each returned HTTP
200 and every SMILES was parsed with RDKit at the count shown. Base URL:
`https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/`

| # | Endpoint | Task | File / source | SMILES col | Target col | Usable *n* | Primary metric |
|---|---|---|---|---|---|---|---|
| 1 | ESOL / Delaney aqueous solubility | regression | `delaney-processed.csv` | `smiles` | `measured log solubility in mols per litre` | 1128 / 1128 | Spearman ρ |
| 2 | FreeSolv hydration free energy | regression | `SAMPL.csv` | `smiles` | `expt` | 642 / 642 | Spearman ρ |
| 3 | Lipophilicity (ChEMBL logD 7.4) | regression | `Lipophilicity.csv` | `smiles` | `exp` | 4200 / 4200 | Spearman ρ |
| 4 | Bradley melting point | regression | `BradleyMeltingPointDataset.xlsx` (obtained) | `smiles` | `mpC` | 10,000 (subsampled from 19,945) | Spearman ρ |
| 5 | BACE | classification | `bace.csv` | `mol` | `Class` | 1513 / 1513 | ROC-AUC |
| 6 | BBBP | classification | `BBBP.csv` | `smiles` | `p_np` | 2039 / 2050 | ROC-AUC |
| 7 | Nucleoside gelation | classification | own prior work (`nucleoside-hydrogel-repro`) | (deposited) | gelator/non-gelator | 71 | ROC-AUC |

Four regression, three classification. Sizes span from 71 to 10,000, so the
comparison is not dominated by any one regime.

**Bradley melting point — obtained and inspected** (figshare
doi:10.6084/m9.figshare.1031637, file `BradleyMeltingPointDataset.xlsx`, 28,645
rows). Columns confirmed: `smiles` and `mpC` (melting point in °C). Curation,
fixed here:

- drop the 377 rows flagged in the `donotuse` column (the dataset's own
  quality flag), and the rows whose SMILES RDKit cannot parse;
- average `mpC` across duplicate canonical SMILES rather than dropping them, since
  these are replicate measurements of a continuous property;
- subsample to **10,000** molecules with seed 20260910, per the large-set rule
  below, and record the drawn keys.

Melting point spans roughly −205 to 517 °C. It is a hard, crystal-packing
dependent target that 2D descriptors predict only moderately; that is expected and
still informative for the comparison.

**Endpoints considered and not used, recorded here so the choice is not silently
made after the fact:**

- **AqSolDB** (aqueous solubility, ~10k; Sorkun et al., *Sci. Data* 2019,
  doi:10.1038/s41597-019-0151-1; Harvard Dataverse doi:10.7910/DVN/OVHAW8).
  Excluded because it aggregates the ESOL/Delaney data and would place two
  strongly correlated aqueous-solubility endpoints into the paired analysis, which
  double-weights one property and breaks the independence the signed-rank test
  assumes. ESOL is kept as the single aqueous-solubility endpoint.
- **Tripeptide aggregation propensity** (Frederix et al., *Nat. Chem.* 2015,
  doi:10.1038/nchem.2122). Excluded because the ~8000 scores exist only inside a
  21 MB supplementary PDF with no machine-readable release, so extraction is not
  worth the effort or the transcription risk for one endpoint. The nucleoside
  gelation set remains as the soft-matter anchor. If a machine-readable version is
  obtained later it can be added as an exploratory endpoint through §8.

**Notes:**

- Subsampling: only Bradley melting point, capped at 10,000 molecules with seed
  20260910 after curation (see endpoint 4). Every other endpoint is used in full.
- No continuous target is binarised: the three classification endpoints ship with
  binary labels, and the melting-point endpoint is treated as regression.
- BBBP curation: a small number of the 2050 SMILES fail to parse in RDKit (a
  documented quirk of that file). Drop them and log the count rather than patching
  them by hand.

**Why these.** The set is weighted towards physicochemical properties (aqueous
solubility, hydration free energy, lipophilicity, melting point) rather than
bioactivity, because the question is being asked from a materials standpoint.
Melting point is included as a genuinely hard, packing-dependent property that
2D descriptors predict poorly; a property where nothing works well is still an
informative test of whether the commercial set helps. The nucleoside gelation set
keeps the comparison connected to the soft-matter work this project sits beside.

## 3. Representations compared

- alvaDesc — all 2D blocks, all 3D blocks disabled, variable reduction off.
- RDKit 2D descriptors — the full `rdkit.Chem.Descriptors` set.
- Mordred 2D — included if it installs cleanly; if not, its absence is recorded as
  a practical cost rather than quietly skipped.
- ECFP4 fingerprints, 2048 bits.
- MACCS keys, 167 bits.

## 4. Models and tuning

Identical for every representation and every endpoint. No representation receives
a larger grid, extra seeds, or additional preprocessing.

- Regularised linear model — **primary**. Logistic regression
  (`penalty='l2'`, `solver='liblinear'`) for classification; ridge for regression.
- Random forest.
- Gradient boosting (`HistGradientBoostingClassifier`/`Regressor`).

Hyperparameter grid, fixed here before any result:

```
linear (LR / Ridge):
    C     in {0.01, 0.1, 1, 10}        # classification (logistic)
    alpha in {0.1, 1, 10, 100}         # regression (ridge)

random forest:
    n_estimators in {300}
    max_depth    in {None, 8}
    max_features in {"sqrt", 0.3}

hist gradient boosting:
    learning_rate in {0.05, 0.1}
    max_depth     in {3, None}
    max_iter = 300, early_stopping = True
```

The same three models and this same grid are applied to alvaDesc, RDKit, Mordred,
ECFP4 and MACCS without exception.

## 5. Evaluation protocol

- All preprocessing (imputation of missing descriptor values, variance filter,
  scaling, and any feature selection) and the hyperparameter choice are fitted
  **inside the training fold only**, through one scikit-learn `Pipeline` wrapped in
  `GridSearchCV`. Fingerprints skip scaling; descriptors are scaled inside the
  pipeline.
- Splits: repeated stratified k-fold for classification and repeated k-fold for
  regression, **5 folds × 5 repeats**, with fixed seed **20260910**. The same fold
  assignments are reused for every representation so the comparison is paired at
  the molecule level.
- A **scaffold split** is additionally run for the three classification endpoints
  as a harder generalisation test, reported alongside the random-split numbers.
- Predictions are pooled out of fold and scored once. A **bootstrap 95% confidence
  interval over molecules** (not folds) accompanies every score, and the trivial
  baseline (majority class for classification, mean predictor for regression) is
  printed beside it.

**Cross-endpoint comparison metric.** RMSE is not comparable across endpoints with
different units, so the paired analysis uses a scale-free primary metric:
**ROC-AUC** for classification and **Spearman ρ** for regression. RMSE and accuracy
are reported per endpoint as secondary numbers.

## 6. Primary comparison

For each of the seven endpoints, compute the difference between alvaDesc and each
open representation on the primary metric, signed so that positive always means
alvaDesc did better. Across endpoints, report:

- the median paired difference with a bootstrap confidence interval,
- a Wilcoxon signed-rank test,
- a mean-rank table over representations.

With seven endpoints this test has low power. A non-significant result will be
reported as "no difference detected at this sample size", not as proof of
equivalence.

## 7. Decision rule (stated before seeing results)

alvaDesc will be described as **giving a practically meaningful advantage for this
kind of work** only if all three hold:

1. the median paired improvement is at least **0.02** in AUC/ρ over the best open
   representation, and
2. the signed-rank test rejects at α = 0.05, and
3. the advantage survives the matched-feature-budget rerun (every representation
   reduced to the same number of features inside each fold), showing it is not
   merely a consequence of having more columns.

If the median difference is below 0.02 and the confidence interval spans zero, the
conclusion will be reported as **no practical difference on these endpoints**.

The 0.02 margin is a judgement, chosen before seeing any result: on datasets of a
few thousand molecules a couple of hundredths of AUC or ρ is within ordinary
run-to-run noise, so a smaller gap would not count as a practical advantage.

Anything decided or noticed after results exist is **exploratory** and will be
labelled as such in the report, kept separate from the confirmatory analysis above.

## 8. Amendments

Any deviation from this plan is recorded here with a date and a reason. An empty
table at submission is expected; an honest entry is much better than a silent
change.

| Date | Change | Reason |
|---|---|---|
| 2026-09-11 | Tree models (random forest, HistGradientBoosting) evaluated at 5 folds × 1 repeat instead of 5 × 5. The primary linear model keeps the full 5 × 5. | The full protocol with both tree models on the wide descriptor matrices (alvaDesc ≈ 4200 columns, plus the 10k-molecule melting-point set) was not feasible on the available 16 GB machine — a single random-forest cell ran to many hours. The linear model is the primary one and is unaffected. |
| 2026-09-11 | In the main comparison, the tree models additionally receive an in-fold univariate feature cap of k = 500 (fitted inside the training fold, as with every other data-dependent step). The linear model still sees all features. | Same compute constraint. The cap is applied identically to every representation, so it does not favour any of them; if anything it removes part of alvaDesc's column-count advantage, which the matched-budget analysis examines directly. |
| 2026-09-11 | Mordred is provided by the maintained `mordredcommunity` fork rather than the original `mordred` package. | The original package no longer installs on current Python (3.13); the community fork is a drop-in with the same descriptors. Recorded as part of the open-stack install friction under RQ4. |
| 2026-09-26 | Added a **3D descriptor arm** as a pre-registered extension (§9), decided after the 2D results existed. It is exploratory and secondary: the confirmatory 2D conclusion in §7 stands unchanged. The conformer recipe, the 3D representations, and the decision rule below were all fixed before any 3D result was computed. | The 2D study deliberately excluded 3D descriptors, which are the main thing a commercial package like alvaDesc offers over free 2D descriptors. Testing them is needed before any general statement about the package's worth. |
| 2026-09-27 | Ran the alvaDesc 3D arm after all, although the free pre-check (§9.3) had shown no 3D benefit and §9.4 therefore said to skip it. alvaDesc 3D descriptors were computed on the same exported conformers and compared directly against the open 3D descriptors (`experiments/03c_compare_3d_alvadesc.py`). | The decision rule in §9.4 was written to avoid spending a paid, time-limited licence on an arm the free pre-check suggested was pointless. alvaDesc turned out to be available at no cost, so the reason for gating disappeared and the symmetric commercial-vs-open comparison — data from both sides, as in the 2D study — was worth having. This is exploratory and does not change the confirmatory 2D result. |
| 2026-09-27 | Added two exploratory endpoints from the Kraken phosphine-ligand library (`experiments/03d_compare_kraken.py`): buried volume (%Vbur, a 3D steric property) and HOMO energy (electronic, as a contrast). These are outside the pre-registered set and are labelled exploratory. | The six confirmatory endpoints are all properties that 2D descriptors already predict well, so they cannot show whether 3D descriptors or alvaDesc help where 3D genuinely matters. Buried volume is a property driven by 3D shape, chosen as a deliberately favourable test for the 3D and commercial descriptors — the place their advantage should appear if it appears anywhere. |
| 2026-10-09 | Clarification of the §7 matched-budget criterion after external review: `k` is a **target** feature budget, not a guaranteed effective width. The pipeline applies a zero-variance filter before `SelectKBest`, so each representation keeps `min(k, surviving columns)`. At `k = 200`, alvaDesc/Mordred/ECFP4 reach 200, RDKit reaches ~185–200, and MACCS reaches ~143–155 (it has 166 usable bits); at `k = 50` all reach 50. Effective widths are recorded in `results/effective_widths.csv` (`experiments/04b_effective_widths.py`). No result changes; the wording "same number of features" is corrected to "common target budget". | A reviewer of the write-up noted that the original phrasing overstated the match. The budget is matched in count where the columns exist, but MACCS and (on some folds) RDKit cannot supply 200; the matched-budget finding that RDKit is strongest on melting point and lipophilicity is a genuine 200-vs-200 comparison there. |
| 2026-10-09 | Clarification of the §9 3D comparison after external review: `mordred3d` (`ignore_3D=False`) is Mordred's **full 2D+3D catalogue** (~1826 columns), used only as a 2D+3D representation, not a 3D-only block. The 3D-only head-to-head is alvaDesc-3D-only (1584) vs RDKit-3D-only (911) specifically; a Mordred 3D-only subset was not isolated. The alvaDesc-vs-RDKit 3D-only difference (median ~0.03) and the alvaDesc-vs-best-open 2D+3D difference (median ~0.004) are **point estimates without a paired confidence interval**, so they are described as descriptive comparisons, not significance tests. | The reviewer asked that the 3D-only claim not be presented as a statistically significant or general superiority over open 3D descriptors. The pre-registered §9.4 lift test (which does carry paired intervals) is unaffected: no endpoint cleared 0.02. |
| 2026-10-09 | **Endpoint 7 (nucleoside gelation) was not completed.** Its curated dataset was not finalised in time, so six endpoints were run and reported, not the seven committed in §2. All §6–§7 analyses are over the six completed endpoints. | The gelation set depends on separate prior work that was not ready, and it was left pending rather than rushed or substituted. Its absence is recorded here rather than silently dropping it; it can be added later as an exploratory endpoint if the data is completed. |
| 2026-10-09 | The §7 outcome label was reworded from "worth its licence for this kind of work" to "giving a practically meaningful advantage", and the 0.02 rationale was stated in neutral terms (removing the clause about justifying cost and reproducibility). No endpoint, metric, model, threshold, or decision condition changed. | Keeping the plan and the write-up neutral and tied to what was measured, rather than stating an economic verdict. |


## 9. 3D descriptor arm (pre-registered extension, 2026-09-26)

This section was written and committed before any 3D result was computed, so the
same pre-registration discipline that governs the 2D study governs this arm. It is
**exploratory and secondary**: nothing here changes the confirmatory 2D result
(§6, §7), which was already fixed and reported. The question is narrower and
separate: **do 3D descriptors help on these endpoints at all, and if so does the
commercial 3D set do better than an open 3D set?**

### 9.1 Conformer recipe (fixed here)

Every molecule is given one 3D conformer, generated identically:

- start from the curated canonical SMILES; add explicit hydrogens;
- embed with RDKit ETKDG version 3, using the global seed (20260910), as a small
  ensemble of 10 conformers;
- optimise each conformer with MMFF94; if MMFF parameters are unavailable for a
  molecule, fall back to UFF; record which was used;
- keep the single lowest-energy conformer.

If embedding fails for a molecule, it is kept in the table with an all-NaN 3D row
(imputed inside the fold, exactly as alvaDesc failures are handled) and the count
is reported. Conformer generation is itself a source of variability; fixing the
recipe here means the comparison is between descriptor packages, not between
conformer generators. The **same conformer** is used for every 3D representation,
including alvaDesc, so no package is advantaged by a better geometry.

### 9.2 Representations

- **Open 3D (free, run now):** the RDKit 3D descriptor set (WHIM, GETAWAY,
  AUTOCORR3D, RDF, MoRSE, and the whole-molecule shape descriptors of
  `Descriptors3D`), and the Mordred 3D descriptors (`ignore_3D=False`) on the same
  conformer.
- **Commercial 3D (needs the alvaDesc trial):** the alvaDesc 3D blocks — the ones
  the 2D study switched off ([13]–[20], [26], [30], [31], [34]) — computed from
  the exported conformer SDF. Only run when the trial is active.

### 9.3 The free pre-check (does 3D help at all?)

Run before, and independently of, any alvaDesc 3D work, using only free tools:

- for each endpoint, evaluate the open 3D representations under the identical
  leak-free protocol (same folds, same seed, primary linear model) used for the 2D
  representations;
- compare each endpoint's **best 2D open representation** against **that same set
  concatenated with the 3D descriptors**, and against the 3D set alone;
- report the per-endpoint change in the primary metric with a bootstrap 95%
  confidence interval over molecules, and the trivial baseline beside it.

### 9.4 Decision rule (fixed before the result)

The alvaDesc 3D arm is worth spending trial time on **only if** the open 3D
descriptors raise the primary metric by at least **0.02** over the best 2D open
representation on at least one endpoint, with the confidence interval not spanning
zero. If no endpoint shows a lift of that size, the honest conclusion is that 3D
descriptors do not help on this panel, the alvaDesc 3D arm is not run, and that
null result is reported. The expectation, stated in advance: the mostly
topological endpoints (aqueous solubility, logD, BBBP) are unlikely to benefit,
while hydration free energy (FreeSolv) and melting point are the plausible places
a genuine 3D effect could appear. The arm is reported per endpoint for this reason,
not pooled.
