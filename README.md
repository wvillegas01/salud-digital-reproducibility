# Salud Digital ICU Mortality Transfer Reproducibility Package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22802875.svg)](https://doi.org/10.5281/zenodo.22802875)

This package contains the code, aggregate outputs, variable documentation, and manuscript files needed to reproduce the revised analyses for the manuscript on cross-domain ICU mortality prediction across MIMIC and eICU.

## What is included

- Original extraction/modeling scripts in `scripts/original_pipeline/`.
- Peer-review revision analyses in `scripts/revision_analyses/`.
- Aggregate, non-patient-level result files in `data/aggregate_outputs/`.
- Schema-only examples in `data/schema_examples/`.
- Variable and data-access documentation in `docs/`.
- A results manifest mapping manuscript tables and figures to scripts and aggregate outputs in `docs/RESULTS_MANIFEST.md`.
- Revised LaTeX manuscript files, compiled redline PDF, and revised manuscript figures in `manuscript/`.

## What is not included

Patient-level MIMIC/eICU raw or derived tables are not included in this public package. MIMIC and eICU are PhysioNet resources subject to credentialing, data-use agreements, and access controls. Users must obtain authorized access and place the required files locally before running the complete pipeline.

## Reproducibility focus

The revised manuscript distinguishes four related analyses:

1. Primary twenty-four-hour shared-feature analysis.
2. Full-episode/full-feature retrospective analysis.
3. Secondary strict shared-feature transfer analysis.
4. Calibration, threshold, patient-cluster bootstrap, seed, and class-weight sensitivity analyses.

The primary analysis and supporting sensitivity analyses address methodological concerns about temporal leakage, fixed thresholds, calibration, uncertainty, repeated ICU stays within patients, class imbalance, random seeds, and imperfect feature equivalence.

The v0.2.0 technical-audit update adds a full primary 24-hour audit. The MIMIC extraction now aggregates laboratory measurements by ICU stay rather than hospital admission, uses explicit MIMIC item identifiers and eICU lab-name mappings, applies death or discharge landmark eligibility consistently, obtains the eICU outcome from `apachePatientResult.actualhospitalmortality`, excludes high-missingness temperature descriptors from the primary shared-feature set, adds patient-grouped internal validation, and reports patient-cluster bootstrap intervals for cross-domain transfer uncertainty. It also regenerates Table 7 and Figure 5 from the same restricted canonical prediction file so the calibration bins, ECE values, confusion-matrix counts, and threshold-dependent diagnostics are internally consistent. The eICU sequential-flow audit now includes `primary_24h_eicu_input_manifest.json`, which identifies the controlled-access local eICU input files by source-table name, byte size, and SHA-256 checksum without redistributing patient-level records.

## Software environment

The regenerated revised outputs were run with Python 3.8.10, scikit-learn 1.3.2, pandas 2.0.3, and NumPy 1.24.3, matching `environment.yml` and `requirements.txt`.

## Citation

Please cite this archived reproducibility package using DOI: https://doi.org/10.5281/zenodo.22802875.

