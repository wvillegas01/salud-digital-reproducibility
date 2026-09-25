# Suggested Run Order

1. Confirm authorized access to MIMIC/eICU source files.
2. Update path constants in the scripts to match the local data directory.
3. Run `scripts/original_pipeline/MIMIC+eICU.py` to reconstruct source-level clinical tables.
4. Run `scripts/original_pipeline/variables-faltantes.py` to audit missingness and build the final modeling matrix.
5. Run `scripts/original_pipeline/dataset_clinico_final_mimic_eicu.py` for the original full-episode/full-feature analysis.
6. Run `scripts/revision_analyses/landmark_24h_analysis.py` to reconstruct the 24-hour landmark dataset.
7. Run `scripts/revision_analyses/landmark_24h_shared_feature_analysis.py` for the primary 24-hour shared-feature analysis.
8. Run `scripts/revision_analyses/landmark_24h_shared_uncertainty.py` for the primary transfer uncertainty analysis.
9. Run `scripts/revision_analyses/primary_24h_shared_diagnostics.py` for primary calibration, source-domain internal validation, MIMIC repeated cross-validation, and patient-grouped internal validation. This script writes a restricted local canonical prediction file and derives Table 7 diagnostics, Figure 5 calibration bins, and the public canonical prediction summary from the same saved predictions.
10. Run `scripts/revision_analyses/seed_and_class_weight_sensitivity.py` for primary 24-hour random-seed and class-weight checks.
11. Run `scripts/revision_analyses/cohort_feature_audit.py` for cohort, mapping, missingness, outcome-source, and patient-dependence audit outputs.
12. Run `scripts/revision_analyses/eicu_sequential_flow_audit.py` to generate the sequential eICU APACHE selection, outcome-availability, and 24-hour landmark flow, plus the controlled-access eICU input manifest with file sizes and SHA-256 checksums.
13. Run `scripts/revision_analyses/generate_fig3_primary_shared.py`, `scripts/revision_analyses/plot_primary_24h_shared_probabilities.py`, and `scripts/revision_analyses/generate_revision_figures.py` to regenerate the primary manuscript figures. `generate_revision_figures.py` reads `data/aggregate_outputs/primary_24h_calibration_plot_bins.csv` for Figure 5 and does not refit models.
14. Run `scripts/revision_analyses/shared_feature_sensitivity.py` for the secondary full-episode strict shared-feature transfer analysis.
15. Run `scripts/revision_analyses/revision_analyses.py` for secondary full-episode bootstrap transfer intervals and Brier summaries.
16. Run `scripts/revision_analyses/threshold_calibration_audit.py` for secondary calibration-gap and oracle/post hoc threshold-sensitivity outputs.
