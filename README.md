# Psychosocial Factors Associated With Past-Year Thoughts of Wanting to Die in Korean Adults

**Analysis code repository — public release v2.0 (2026-09-10)**

This repository contains the analysis and reporting code for the manuscript **“Psychosocial Factors Associated With Past-Year Thoughts of Wanting to Die in Korean Adults.”** The finalized analytic sample included **6,605 Korean adults aged 20–69 years**, with **1,850 (28.0%)** reporting past-year thoughts of wanting to die.

The study integrates three complementary analytic perspectives:

1. **Adjusted association** — multivariable logistic regression with restricted cubic splines and sensitivity analyses.
2. **Predictive contribution** — six classifiers evaluated with repeated nested cross-validation, calibration metrics, bootstrap comparisons, and held-out SHAP.
3. **Conditional network connectivity** — regularized partial-correlation networks with centrality, bridge expected influence, and bootstrap accuracy/stability assessment.

> The machine-learning analyses are cross-sectional classification analyses, not prediction of future suicidal behavior. SHAP values are predictive-attribution measures rather than independent or causal effects. Network edges are conditional, nondirectional associations.

## Repository structure

```text
analysis/
  01_descriptive/             Table 1 descriptive statistics
  02_regression/              Primary regression + sensitivity analyses
  03_machine_learning_shap/   Nested CV, performance, calibration, SHAP
  04_network/                 EBICglasso networks + bootstrap assessment
reporting/
  figures/                    Figure 3 and supplementary network figures
  tables/                     Supplementary Table S4 generator
data_templates/               Header-only input templates (no records)
environment/                  Python/R environment information
provenance/                   Run manifests, package versions, settings
verification/                 Aggregate non-participant-level ML outputs
docs/                         Run guides, variable dictionary, code map
extras/                       Optional manuscript-workbook formatting helpers
```

## Data availability and privacy

**No participant-level data are included.** The study data cannot be externally transferred or publicly shared under the applicable ethics approval and institutional data-governance requirements. Header-only CSV templates are provided in `data_templates/`.

The internal machine-learning checkpoint archive is also excluded because it contains participant-level outcomes, predicted probabilities, row identifiers, and split indices. The files under `verification/` are aggregate summaries only. See `docs/DATA_AVAILABILITY.md`.

## Quick start

The complete run instructions are in `docs/RUN_GUIDE.md` (English) and `docs/RUN_GUIDE_KO.md` (Korean).

### Table 1

```bash
python analysis/01_descriptive/table1_descriptive_statistics.py \
  --input primary_cc.csv \
  --output table1_statistics.csv
```

### Regression

```bash
python analysis/02_regression/regression_analysis.py \
  --primary primary_cc.csv \
  --phq8 phq8_cc.csv \
  --categorical categorical_sensitivity.csv \
  --outdir regression_results
```

### Machine learning + SHAP

Run `analysis/03_machine_learning_shap/machine_learning_nested_cv_shap.ipynb`.

### Network analysis

Run `analysis/04_network/network_analysis.R`. Use `TEST_MODE <- TRUE` for a short code check and `FALSE` for the finalized computation.

## Analysis settings preserved in this release

| Component | Final setting |
|---|---|
| Primary sample | N = 6,605 |
| Outcome | `py_thoughts_wanting_to_die` |
| ML outer CV | 5 folds × 3 repeats |
| ML inner CV | 5 folds |
| ML performance bootstrap | 2,000 participant-level resamples |
| Network estimator | EBICglasso, gamma = 0.50 |
| Edge-weight bootstrap | 2,000 |
| Centrality case-dropping bootstrap | 1,000 |
| Publication Figure 3 layout | Common fixed coordinates from the total-sample layout across all panels |

## Reproducing manuscript outputs

`docs/MANUSCRIPT_CODE_MAP.md` maps the manuscript tables, figures, and sensitivity analyses to the corresponding public code files.

## Environment

The analysis used **Python 3.10.9** for regression/machine learning and **R 4.5.2** for network analysis. Exact ML/SHAP package versions are preserved in `provenance/ml_run_manifest.json`; R session/package information is under `provenance/`. See `environment/README.md` before recreating environments.

## Release integrity

This v2.0 release reorganizes filenames, documentation, and reporting labels for public readability. **No statistical model, estimate, result, or inferential decision was changed.** See `CHANGELOG.md`.

## Citation

Citation metadata for this code release are provided in `CITATION.cff`.

## License

No software license is currently granted. All rights are reserved unless the authors add a license in a future release.
