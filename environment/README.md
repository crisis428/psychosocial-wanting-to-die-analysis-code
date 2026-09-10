# Software environments

The manuscript reports **Python 3.10.9** for regression/machine learning and **R 4.5.2** for network analysis.

- `requirements_reference_python3109.txt` is the consolidated Python package snapshot retained with the finalized code collection.
- `r_environment.txt` and `provenance/network_package_versions.csv` preserve the R version and key package versions used for network analysis.
- The exact ML/SHAP package versions recorded at the finalized run are also stored in `provenance/ml_run_manifest.json`.

Because the original regression and ML code components were finalized in separate working environments, treat the consolidated requirements file as a reproducibility reference rather than a guarantee that every historical helper script was executed in one identical environment.
