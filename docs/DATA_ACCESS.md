# Data Access and Redistribution

This study uses MIMIC-III Clinical Database Demo v1.4 and the eICU Collaborative Research Database through PhysioNet. These resources contain deidentified but real patient data and are subject to PhysioNet access conditions, credentialing, training, and data-use agreements.

For this reason, this public reproducibility package does not redistribute raw or patient-level derived MIMIC/eICU records, labels, or predictions. Instead, it provides:

- scripts required to reconstruct the cohorts from authorized local copies;
- schema-only examples;
- aggregate result files used in the revised manuscript;
- documentation of software versions, seeds, model settings, and sensitivity analyses.

Authorized users should place the required source data in a local directory and update the path constants at the top of the scripts.

Primary source citations:

- Johnson AEW et al. MIMIC-III, a freely accessible critical care database. Scientific Data. 2016.
- Johnson A, Pollard T, Mark R. MIMIC-III Clinical Database Demo (version 1.4). PhysioNet. 2019. DOI: 10.13026/C2HM2Q.
- Pollard TJ et al. The eICU Collaborative Research Database, a freely available multi-center database for critical care research. Scientific Data. 2018.
