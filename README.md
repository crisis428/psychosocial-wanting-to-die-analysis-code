# Psychosocial correlates of past-year thoughts of wanting to die in Korean adults across regression, machine learning and network analysis

**Analysis code — version 2.1 (2026-09-28)**

This repository contains the analysis and reporting code for the manuscript above. The analytic sample included **6,605 Korean adults aged 20–69 years**, with **1,850 (28.0%)** reporting past-year thoughts of wanting to die.

The study compares three analytic perspectives using the same participants and predictor set:

1. **Adjusted association** — multivariable logistic regression with a GAD-7 restricted cubic spline, White (HC0) sandwich covariance, marginal standardization, and sensitivity analyses.
2. **Classification and attribution** — six classifiers evaluated with repeated nested cross-validation, calibration metrics, paired bootstrap comparisons, and held-out SHAP.
3. **Conditional network connectivity** — regularized partial-correlation networks, centrality, predictor-only bridge expected influence, and bootstrap accuracy and stability.

These are cross-sectional analyses. Classification performance does not establish prediction of future suicidal behavior; SHAP values describe model attribution, and network edges describe conditional, nondirectional associations.

## Repository structure

```text
analysis/
  01_descriptive/             Table 1 descriptive statistics
  02_regression/              Primary regression and sensitivity analyses
  03_machine_learning_shap/   Nested CV, performance, calibration, SHAP
  04_network/                 EBICglasso networks and bootstrap assessment
reporting/
  figures/                    Figure 3 and supplementary network figures
  tables/                     Supplementary Table S4 generator
data_templates/               Header-only input templates
environment/                  Python and R dependency information
provenance/                   Run manifests, package versions, settings
verification/                 Aggregate ML outputs
tests/                        Synthetic-data software tests
docs/                         Run guides, variable dictionary, code map
```

The ML and network workflows are split into numbered `steps/` files. Their top-level runner scripts execute the steps sequentially in one shared analysis environment.

## Data availability

Individual participant data are not publicly available because the IRB-approved data-use period has ended and external transfer or release is prohibited by the study's ethics approval and institutional data-governance requirements. The IRB approval requires deletion of participant-level data at the end of the approved data-use period.

**No participant-level data are included.** The repository contains code, header-only templates, computational settings, and aggregate outputs. Participant-level outcomes, predictions, identifiers, split indices, and checkpoint archives are not distributed. See `docs/DATA_AVAILABILITY.md`.

## Running the code

Use the scripts only with data that you are authorized to process and that follow the headers in `data_templates/`. The commands below illustrate the input structure; they do not imply that the original study data are available. Full instructions are in `docs/RUN_GUIDE.md` and `docs/RUN_GUIDE_KO.md`.

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

### Machine learning and SHAP

```bash
python analysis/03_machine_learning_shap/machine_learning_nested_cv_shap.py
```

### Network analysis

```bash
Rscript analysis/04_network/network_analysis.R
```

Use `TEST_MODE <- TRUE` for a short network run and `FALSE` for the full computation.

### Synthetic-data tests

```bash
python -m unittest discover -s tests -v
```

The covariance test uses generated data and requires no study records.

## Analysis settings

| Component | Setting |
|---|---|
| Primary sample | N = 6,605 |
| Outcome | `py_thoughts_wanting_to_die` |
| Regression covariance | White (HC0) sandwich |
| GAD-7 spline knots | 0, 2, 5, 14 |
| ML outer CV | 5 folds × 3 repeats |
| ML inner CV | 5 folds |
| ML performance bootstrap | 2,000 participant-level resamples |
| Network estimator | EBICglasso, gamma = 0.50 |
| Edge-weight bootstrap | 2,000 |
| Centrality case-dropping bootstrap | 1,000 |
| Figure 3 layout | Common fixed coordinates from the total-sample layout |

`docs/MANUSCRIPT_CODE_MAP.md` links manuscript components to code. Some tables and multi-panel figures were assembled from numerical outputs and component plots. Response labels follow the questionnaire translations in Supplementary Table S7; numeric coding and reference categories are unchanged.

## Software

Regression models use Python and statsmodels. The ML run used Python 3.10.9; its package versions are recorded in `provenance/ml_run_manifest.json`. Network analyses used R 4.5.2; R session and package information is under `provenance/`. The ML manifest is not a regression run manifest. See `environment/README.md` for dependency information.

## Version history and citation

Version 2.1 aligns the manuscript title, response labels, covariance terminology, and documentation. See `CHANGELOG.md` for the changes. Citation metadata are in `CITATION.cff`.

## License

No software license is currently granted. All rights are reserved unless the authors add a license.
