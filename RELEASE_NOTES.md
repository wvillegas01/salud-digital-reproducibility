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
