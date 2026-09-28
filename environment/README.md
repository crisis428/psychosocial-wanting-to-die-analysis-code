# Software environments

## Regression

Regression models are fitted in Python using statsmodels and patsy, with White (HC0) sandwich covariance. The code explicitly requests `cov_type='HC0'`. The consolidated requirements file below provides dependency information; it is not a separate record of the original regression execution environment.

## Machine learning and SHAP

`provenance/ml_run_manifest.json` records the ML run: Python 3.10.9, statsmodels 0.14.4, scikit-learn 1.6.1, CatBoost 1.2.8, SHAP 0.48.0, and the remaining package versions. These versions describe the ML run, not all regression scripts.

## Networks

Network analyses used R 4.5.2. `r_environment.txt`, `provenance/network_package_versions.csv`, and the R session records under `provenance/` contain the network package information.

## Dependencies and tests

`requirements_reference_python3109.txt` is the consolidated Python dependency snapshot from the code collection. Its existing package pins are unchanged. Use the component-specific records above when interpreting this reference file.

From the repository root, run the synthetic-data covariance test with:

```bash
python -m unittest discover -s tests -v
```

The test compares the fitted binomial-logit HC0 covariance with the direct White sandwich formula and uses no participant data.
