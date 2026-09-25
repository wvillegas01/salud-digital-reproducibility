# Results manifest

This manifest maps each manuscript table or figure to the script and aggregate output used to generate or verify the reported result. Patient-level MIMIC/eICU records and row-level predictions are not included in the public archive.

Files with the `primary_24h_` prefix are the authoritative outputs for the corrected primary 24-hour shared-feature analysis. In particular, Table 7 and Figure 5 are generated from `primary_24h_shared_diagnostics.py` and the same restricted canonical prediction source; older non-prefixed calibration files retained for secondary or historical full-episode diagnostics are not used to generate the primary calibration figure or primary diagnostic table.

| Manuscript item | Main generating or verification script | Public aggregate output or artifact |
| --- | --- | --- |
| Table 1. Explicit source mappings | `scripts/revision_analyses/cohort_feature_audit.py`; `scripts/revision_analyses/landmark_24h_analysis.py` | `data/aggregate_outputs/primary_24h_feature_mapping_audit.csv`; `docs/VARIABLE_DICTIONARY.md` |
| Table 2. Cohort flow | `scripts/revision_analyses/landmark_24h_analysis.py`; `scripts/revision_analyses/cohort_feature_audit.py`; `scripts/revision_analyses/eicu_sequential_flow_audit.py` | `data/aggregate_outputs/landmark_24h_cohort_summary.csv`; `data/aggregate_outputs/primary_24h_cohort_feature_audit_summary.json`; `data/aggregate_outputs/primary_24h_group_dependence_summary.csv`; `data/aggregate_outputs/primary_24h_eicu_sequential_flow.csv`; `data/aggregate_outputs/primary_24h_eicu_input_manifest.json` |
| Table 3. Secondary versus corrected primary workflow | `scripts/revision_analyses/landmark_24h_shared_feature_analysis.py`; `scripts/revision_analyses/shared_feature_sensitivity.py`; `scripts/revision_analyses/revision_analyses.py` | `data/aggregate_outputs/landmark_24h_shared_feature_metrics.csv`; `data/aggregate_outputs/shared_feature_transfer_sensitivity.csv`; `data/aggregate_outputs/transfer_bootstrap_ci.csv` |
| Table 4. Intra-domain and integrated CV performance | `scripts/revision_analyses/landmark_24h_shared_feature_analysis.py` | `data/aggregate_outputs/landmark_24h_shared_feature_metrics.csv` |
| Table 5. Cross-domain transfer performance | `scripts/revision_analyses/landmark_24h_shared_feature_analysis.py` | `data/aggregate_outputs/landmark_24h_shared_feature_metrics.csv` |
| Table 6. Transfer uncertainty | `scripts/revision_analyses/landmark_24h_shared_uncertainty.py` | `data/aggregate_outputs/landmark_24h_shared_transfer_uncertainty.csv` |
| Table 7. Source-domain and transfer diagnostics | `scripts/revision_analyses/primary_24h_shared_diagnostics.py` | `data/aggregate_outputs/primary_24h_shared_calibration_transfer_internal.csv`; `data/aggregate_outputs/primary_24h_canonical_prediction_summary.csv` |
| Table 8. Patient-grouped internal validation | `scripts/revision_analyses/primary_24h_shared_diagnostics.py` | `data/aggregate_outputs/primary_24h_patient_grouped_internal_validation.csv` |
| Table 9. Full-feature 24-hour landmark sensitivity | `scripts/revision_analyses/landmark_24h_analysis.py` | `data/aggregate_outputs/landmark_24h_metrics.csv` |
| Table 10. Full-episode strict shared-feature sensitivity | `scripts/revision_analyses/shared_feature_sensitivity.py` | `data/aggregate_outputs/shared_feature_transfer_sensitivity.csv` |
| Table 11. Confusion-matrix decision summary | `scripts/revision_analyses/landmark_24h_shared_feature_analysis.py`; `scripts/revision_analyses/primary_24h_shared_diagnostics.py` | `data/aggregate_outputs/landmark_24h_shared_feature_metrics.csv`; `data/aggregate_outputs/primary_24h_shared_calibration_transfer_internal.csv` |
| Figure 1. Multi-environment protocol | Manuscript schematic | `manuscript/Fig1.jpg` |
| Figure 2. Decision-level diagnostic framework | `scripts/revision_analyses/generate_revision_figures.py` | `manuscript/Fig2.jpg` |
| Figure 3. Shared clinical-variable distributions | `scripts/revision_analyses/generate_fig3_primary_shared.py` | `manuscript/Fig3_primary_24h_shared.jpg`; `data/aggregate_outputs/landmark_24h_missingness_by_source.csv` |
| Figure 4. Transfer probability distributions | `scripts/revision_analyses/plot_primary_24h_shared_probabilities.py` | `manuscript/Fig4_primary_24h_shared.jpg`; `data/aggregate_outputs/landmark_24h_shared_feature_metrics.csv` |
| Figure 5. Calibration curves | `scripts/revision_analyses/primary_24h_shared_diagnostics.py`; `scripts/revision_analyses/generate_revision_figures.py` | `manuscript/Fig5.jpg`; `data/aggregate_outputs/primary_24h_calibration_plot_bins.csv`; `data/aggregate_outputs/primary_24h_canonical_prediction_summary.csv` |
