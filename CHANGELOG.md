# Changelog

## v2.1 — 2026-09-28

- Updated the manuscript title in the English and Korean READMEs and citation metadata.
- Matched descriptive and categorical-regression response labels to Supplementary Table S7. Numeric codes, category order, and reference categories are unchanged.
- Replaced `cov_type='HC3'` with explicit `cov_type='HC0'` in the regression script and corrected the generated estimator label. The earlier GLM calls used statsmodels' uncorrected White sandwich path, not leverage-adjusted HC3. See the [statsmodels 0.14.6 implementation](https://github.com/statsmodels/statsmodels/blob/v0.14.6/statsmodels/base/covtype.py).
- Added a synthetic-data test of the covariance against the direct White sandwich formula. A synthetic-data comparison in statsmodels 0.14.6 gave identical coefficients and covariance matrices for the old and updated primary, PHQ-8, categorical, and interaction specifications. This test did not use study data.
- Separated regression dependency information from the ML-specific Python version and run manifest.
- Aligned data-availability documentation with the manuscript and the IRB requirement for deletion at the end of the approved data-use period. Participant-level data remain excluded.
- Preserved the existing aggregate result files, ML and network code, numerical model specifications, and prior version history. No study-data analysis was rerun for this update.

## v2.0 — 2026-09-10

Repository-presentation and portability release. **No statistical model, estimate, result, or inferential decision was changed.**

- Reorganized code into `analysis/`, `reporting/`, `provenance/`, `verification/`, and `docs/`.
- Renamed files to remove legacy journal-specific and development-time names.
- Added English/Korean run guides, a manuscript-to-code map, analytic variable dictionary, and `CITATION.cff`.
- Relabeled legacy `eTable 4`/`eFigure 5-7` reporting helpers as `Table S4`/`Figure S5-S7` to match journal-neutral supplement naming.
- Clarified that the publication Figure 3 uses the total-sample coordinates as a common fixed layout across all three panels.
- Removed duplicate/reference notebooks and development-time workbook-formatting helpers from the lean public repository.
- Retained compact aggregate verification summaries while excluding bulky fold-level checkpoint exports.
- Continued to exclude participant-level data, participant-level predictions, split indices, and internal checkpoint archives.

## v1.1 — 2026-09-10

- Added Table 1 descriptive-statistics code.
- Sanitized local paths and checkpoint references.
- Added aggregate verification outputs.
