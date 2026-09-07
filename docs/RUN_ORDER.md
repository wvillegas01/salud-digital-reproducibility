# Suggested Run Order

1. Confirm authorized access to MIMIC/eICU source files.
2. Update path constants in the scripts to match the local data directory.
3. Run `scripts/original_pipeline/MIMIC+eICU.py` to reconstruct source-level clinical tables.
4. Run `scripts/original_pipeline/variables-faltantes.py` to audit missingness and build the final modeling matrix.
5. Run `scripts/original_pipeline/dataset_clinico_final_mimic_eicu.py` for the original full-episode/full-feature analysis.
6. Run `scripts/revision_analyses/landmark_24h_analysis.py` for the 24-hour landmark analysis.
7. Run `scripts/revision_analyses/shared_feature_sensitivity.py` for the strict shared-feature transfer analysis.
8. Run `scripts/revision_analyses/revision_analyses.py` for bootstrap transfer intervals and Brier summaries.
9. Run `scripts/revision_analyses/threshold_calibration_audit.py` for calibration-gap and threshold-sensitivity outputs.
10. Run `scripts/revision_analyses/seed_and_class_weight_sensitivity.py` for seed and class-weight checks.
