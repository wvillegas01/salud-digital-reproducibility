# Suggested Run Order

1. Confirm authorized access to MIMIC/eICU source files.
2. Update path constants in the scripts to match the local data directory.
3. Run `scripts/original_pipeline/MIMIC+eICU.py` to reconstruct source-level clinical tables.
4. Run `scripts/original_pipeline/variables-faltantes.py` to audit missingness and build the final modeling matrix.
5. Run `scripts/original_pipeline/dataset_clinico_final_mimic_eicu.py` for the original full-episode/full-feature analysis.
6. Run `scripts/revision_analyses/landmark_24h_analysis.py` to reconstruct the 24-hour landmark dataset.
7. Run `scripts/revision_analyses/landmark_24h_shared_feature_analysis.py` for the primary 24-hour shared-feature analysis.
8. Run `scripts/revision_analyses/landmark_24h_shared_uncertainty.py` for the primary transfer uncertainty analysis.
9. Run `scripts/revision_analyses/plot_primary_24h_shared_probabilities.py` to regenerate the primary transfer probability-distribution figure.
10. Run `scripts/revision_analyses/shared_feature_sensitivity.py` for the secondary full-episode strict shared-feature transfer analysis.
11. Run `scripts/revision_analyses/revision_analyses.py` for secondary full-episode bootstrap transfer intervals and Brier summaries.
12. Run `scripts/revision_analyses/threshold_calibration_audit.py` for secondary calibration-gap and oracle/post hoc threshold-sensitivity outputs.
13. Run `scripts/revision_analyses/seed_and_class_weight_sensitivity.py` for seed and class-weight checks.
