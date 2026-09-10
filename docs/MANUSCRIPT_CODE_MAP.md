# Manuscript-to-code map

| Manuscript component | Public code | Notes |
|---|---|---|
| Table 1 | `analysis/01_descriptive/table1_descriptive_statistics.py` | Descriptive statistics, Welch t tests, Pearson chi-square tests |
| Table 2 / primary adjusted model | `analysis/02_regression/regression_analysis.py` | HC3 logistic regression; GAD-7 restricted cubic spline |
| GAD-7 standardized prevalence | `analysis/02_regression/` | Marginal standardization from the primary regression model |
| PHQ-8 sensitivity | `analysis/02_regression/` | Simultaneous GAD-7 and PHQ-8 splines |
| Categorical-coding sensitivity | `analysis/02_regression/` | Marital status and ordinal predictors as categorical indicators |
| Sex interactions | `analysis/02_regression/` | Prespecified interactions with Holm correction |
| ROC/model performance | `analysis/03_machine_learning_shap/machine_learning_nested_cv_shap.py` | Repeated nested CV and aggregate out-of-fold evaluation |
| SHAP importance / beeswarm | `analysis/03_machine_learning_shap/machine_learning_nested_cv_shap.py` | Held-out SHAP; fold-rank stability |
| Network estimation / centrality / bootstrap | `analysis/04_network/network_analysis.R` | Mixed correlations, EBICglasso, centrality, bridge EI, accuracy/stability bootstrap |
| Figure 3 | `reporting/figures/figure3_network_fixed_layout.py` | Common total-sample node coordinates across panels |
| Supplementary Figures S5-S7 | `reporting/figures/supplementary_network_figures.R` | Plotting from saved network results; no bootstrap re-estimation |
| Supplementary Table S4 | `reporting/tables/table_s4_ml_performance.py` | Detailed ML performance from local checkpoint archive |
