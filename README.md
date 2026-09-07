# Salud Digital ICU Mortality Transfer Reproducibility Package

This package contains the code, aggregate outputs, variable documentation, and manuscript files needed to reproduce the revised analyses for the manuscript on cross-domain ICU mortality prediction across MIMIC and eICU.

## What is included

- Original extraction/modeling scripts in `scripts/original_pipeline/`.
- Peer-review revision analyses in `scripts/revision_analyses/`.
- Aggregate, non-patient-level result files in `data/aggregate_outputs/`.
- Schema-only examples in `data/schema_examples/`.
- Variable and data-access documentation in `docs/`.
- Revised LaTeX manuscript files in `manuscript/`.

## What is not included

Patient-level MIMIC/eICU raw or derived tables are not included in this public package. MIMIC and eICU are PhysioNet resources subject to credentialing, data-use agreements, and access controls. Users must obtain authorized access and place the required files locally before running the complete pipeline.

## Reproducibility focus

The revised manuscript distinguishes four related analyses:

1. Full-episode/full-feature retrospective analysis.
2. Twenty-four-hour landmark analysis.
3. Strict shared-feature transfer analysis.
4. Calibration, threshold, bootstrap, seed, and class-weight sensitivity analyses.

The third and fourth analyses were added to address reviewer concerns about temporal leakage, fixed thresholds, calibration, uncertainty, class imbalance, random seeds, and imperfect feature equivalence.

## Software environment

The analyses were run with Python 3.8.10, scikit-learn 1.3.2, pandas 2.0.3, and NumPy 1.24.3.
