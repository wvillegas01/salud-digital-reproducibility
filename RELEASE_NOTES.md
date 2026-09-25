# v0.2.0

Technical audit update for the primary 24-hour landmark reproducibility workflow.

This update adds:

- Canonical restricted prediction generation for the primary 24-hour diagnostics, with Table 7, Figure 5 calibration bins, confusion-matrix counts, ECE values, and the public prediction summary derived from the same saved predictions.
- Sequential eICU cohort-flow audit separating local patient-table input, raw APACHE result rows, one-APACHE-result-per-ICU-stay selection, outcome availability, and 24-hour landmark eligibility.
- Controlled-access eICU input manifest reporting source-table names, byte sizes, and SHA-256 checksums for the local eICU files used in the revised analysis.
- ICU-stay-level MIMIC laboratory aggregation for the primary 24-hour landmark extraction.
- Explicit MIMIC item-id and eICU lab-name feature mappings.
- eICU mortality labeling from `apachePatientResult.actualhospitalmortality`.
- Landmark eligibility excluding episodes with death, ICU discharge, or hospital discharge before completion of the 24-hour observation window.
- Primary missingness-rule audit showing exclusion of temperature descriptors from the shared-feature matrix because of high eICU missingness.
- Updated primary cohort counts, transfer metrics, bootstrap/exact-binomial uncertainty intervals, calibration diagnostics, random-seed sensitivity, class-weight sensitivity, and patient-grouped internal validation.
- Regenerated secondary full-feature 24-hour landmark metric output so undefined precision is represented as missing/NA when TP+FP=0.
- Regenerated manuscript figures and revised manuscript/PDF aligned with the audited primary analysis, including a revised Figure 2 in which all observed decision behaviors are available to both internal and transfer evaluation settings.
- Added manuscript Ethics Statement and Generative AI Statement for Frontiers front-matter consistency.

Patient-level MIMIC/eICU raw or derived records remain excluded from the public archive. The package contains only scripts, documentation, schema-only examples, manuscript artifacts, and aggregate non-patient-level outputs.

# v0.1.1

Manuscript-alignment update for the peer-review revision.

This update adds:

- Revised redline manuscript and compiled PDF reflecting the primary 24-hour shared-feature interpretation.
- Primary source-domain versus transfer diagnostics, including average precision, Brier score, ECE, calibration intercept, and calibration slope.
- Repeated internal MIMIC cross-validation summary to contextualize the MIMIC-to-eICU transfer direction.
- Regenerated Figure 3 using only primary shared-feature variables, replacing the previous mean-arterial-pressure panel with systolic blood pressure.
- Scripts used to generate the primary diagnostics and revised Figure 3.

# v0.1.0

Initial reproducibility release for peer-review revision.

This release includes:

- Original pipeline scripts used for MIMIC/eICU extraction and modeling.
- Revision-analysis scripts for bootstrap confidence intervals, calibration/threshold evaluation, random-seed sensitivity, class-weight sensitivity, 24-hour landmark analysis, and strict shared-feature transfer.
- Aggregate, non-patient-level results generated for manuscript revision.
- Schema-only data examples, variable dictionary, data-access instructions, run order, environment files, and reproducibility checklist.
- Revised LaTeX manuscript and bibliography files.

Patient-level MIMIC/eICU raw or derived records are not included. Users must obtain authorized access to the relevant PhysioNet resources and reconstruct local input files according to the documentation.
