# Changelog

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
