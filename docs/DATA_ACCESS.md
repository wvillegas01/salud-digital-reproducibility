# Data Access and Redistribution

This study uses MIMIC-III Clinical Database Demo v1.4 and the eICU Collaborative Research Database through PhysioNet. These resources contain deidentified but real patient data and are subject to PhysioNet access conditions, credentialing, training, and data-use agreements.

For this reason, this public reproducibility package does not redistribute raw or patient-level derived MIMIC/eICU records, labels, or predictions. Instead, it provides:

- scripts required to reconstruct the cohorts from authorized local copies;
- schema-only examples;
- aggregate result files used in the revised manuscript;
- documentation of software versions, seeds, model settings, and sensitivity analyses.

Authorized users should place the required source data in a local directory and update the path constants at the top of the scripts.

The primary regenerated analysis expects the following source tables locally: MIMIC `PATIENTS.csv`, `ADMISSIONS.csv`, `ICUSTAYS.csv`, `LABEVENTS.csv`, `CHARTEVENTS.csv`, `D_LABITEMS.csv`, and `D_ITEMS.csv`; eICU `patient.csv.gz`, `apachePatientResult.csv.gz`, `vitalPeriodic.csv.gz`, `vitalAperiodic.csv.gz`, and `lab.csv.gz`. The public package intentionally includes only aggregate outputs and schema-only examples.

The eICU analyses use the controlled-access local eICU subset available in the authorized analysis directory. The revision scripts do not apply an additional hospital-level or date-range sampling rule before cohort construction. To make this input auditable without redistributing controlled-access records, `data/aggregate_outputs/primary_24h_eicu_input_manifest.json` reports the source-table names, byte sizes, and SHA-256 checksums for the local eICU files used to generate the revised outputs. Authorized researchers can compare their local files against this manifest when attempting to identify the same input population.

The primary diagnostics script writes a restricted local canonical prediction file containing one row per evaluated episode, scenario, fold where applicable, model, target label, predicted probability, and fixed-threshold prediction. This restricted file is not redistributed publicly because it is episode-level derived clinical data. Public aggregate outputs derived from the same file include `primary_24h_shared_calibration_transfer_internal.csv`, `primary_24h_calibration_plot_bins.csv`, and `primary_24h_canonical_prediction_summary.csv`.

Primary source citations:

- Johnson AEW et al. MIMIC-III, a freely accessible critical care database. Scientific Data. 2016.
- Johnson A, Pollard T, Mark R. MIMIC-III Clinical Database Demo (version 1.4). PhysioNet. 2019. DOI: 10.13026/C2HM2Q.
- Pollard TJ et al. The eICU Collaborative Research Database, a freely available multi-center database for critical care research. Scientific Data. 2018.
