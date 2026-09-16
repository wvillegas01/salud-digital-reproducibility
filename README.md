# Salud Digital ICU Mortality Transfer Reproducibility Package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22779700.svg)](https://doi.org/10.5281/zenodo.22779700)

This package contains the code, aggregate outputs, variable documentation, and manuscript files needed to reproduce the revised analyses for the manuscript on cross-domain ICU mortality prediction across MIMIC and eICU.

## What is included

- Original extraction/modeling scripts in `scripts/original_pipeline/`.
- Peer-review revision analyses in `scripts/revision_analyses/`.
- Aggregate, non-patient-level result files in `data/aggregate_outputs/`.
- Schema-only examples in `data/schema_examples/`.
- Variable and data-access documentation in `docs/`.
- Revised LaTeX manuscript files, compiled redline PDF, and revised manuscript figures in `manuscript/`.

## What is not included

Patient-level MIMIC/eICU raw or derived tables are not included in this public package. MIMIC and eICU are PhysioNet resources subject to credentialing, data-use agreements, and access controls. Users must obtain authorized access and place the required files locally before running the complete pipeline.

## Reproducibility focus

The revised manuscript distinguishes four related analyses:

1. Primary twenty-four-hour shared-feature analysis.
2. Full-episode/full-feature retrospective analysis.
3. Secondary strict shared-feature transfer analysis.
4. Calibration, threshold, bootstrap, exact-binomial interval, seed, and class-weight sensitivity analyses.

The primary analysis and supporting sensitivity analyses address methodological concerns about temporal leakage, fixed thresholds, calibration, uncertainty, class imbalance, random seeds, and imperfect feature equivalence.

The v0.1.1 manuscript-alignment update adds a full primary 24-hour audit. The MIMIC extraction now aggregates laboratory measurements by ICU stay rather than hospital admission, uses explicit MIMIC item identifiers and eICU lab-name mappings, applies death or discharge landmark eligibility consistently, obtains the eICU outcome from `apachePatientResult.actualhospitalmortality`, excludes high-missingness temperature descriptors from the primary shared-feature set, and adds patient-grouped internal validation.

## Software environment

The final regenerated primary outputs were run with Python 3.12, scikit-learn 1.8.0, pandas 2.2.2, and NumPy 2.0.2. Earlier secondary outputs were generated under Python 3.8.10, scikit-learn 1.3.2, pandas 2.0.3, and NumPy 1.24.3.

## Citation

Please cite this archived reproducibility package using DOI: https://doi.org/10.5281/zenodo.22779700.

