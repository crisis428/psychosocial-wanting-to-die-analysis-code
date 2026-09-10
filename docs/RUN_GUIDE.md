# Run guide

The code assumes that restricted local CSV files are available in your working directory and follow the headers in `data_templates/`. No study data are included here.

## 1. Table 1 descriptive statistics

```bash
python analysis/01_descriptive/table1_descriptive_statistics.py \
  --input primary_cc.csv \
  --output table1_statistics.csv
```

This reproduces mean (SD), No. (%), Welch two-sample t tests, and Pearson chi-square tests used in Table 1.

## 2. Regression and sensitivity analyses

```bash
python analysis/02_regression/regression_analysis.py \
  --primary primary_cc.csv \
  --phq8 phq8_cc.csv \
  --categorical categorical_sensitivity.csv \
  --outdir regression_results
```

## 3. Machine learning and SHAP

Run:

`analysis/03_machine_learning_shap/machine_learning_nested_cv_shap.py`

The workflow evaluates logistic regression, decision tree, random forest, SVM, XGBoost, and CatBoost using 5-fold × 3-repeat nested stratified cross-validation. The internal checkpoint archive created by this workflow is deliberately excluded from the public repository because it contains participant-level outputs.

## 4. Network analysis

Run `analysis/04_network/network_analysis.R` from the directory containing `primary_cc.csv`, or set `WORK_DIR` in the script. Use `TEST_MODE <- TRUE` for a short code check and `FALSE` for the final computation.

## 5. Reporting helpers

Supplementary network figures from saved network results:

```r
source("reporting/figures/supplementary_network_figures.R")
```

Table S4 from a locally generated ML checkpoint archive:

```bash
python reporting/tables/table_s4_ml_performance.py \
  --results-zip full_results_checkpointed.zip \
  --output-dir table_s4_ml_performance
```

Publication Figure 3 fixed-layout panels:

```bash
python reporting/figures/figure3_network_fixed_layout.py \
  --results-zip network_results_FINAL_v1.zip \
  --output-dir figure3_network_panels
```

The publication Figure 3 helper intentionally uses the total-sample coordinates for all three panels to support direct visual comparison; node distances are not interpreted statistically.
