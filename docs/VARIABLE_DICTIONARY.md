# Variable Dictionary

## Identifiers and provenance

- `case_id`: ICU episode identifier. MIMIC uses `icustay_id`; eICU uses `patientunitstayid`.
- `patient_group_id`: patient-level grouping identifier used only for internal grouped validation. It is not included as a predictor.
- `admission_group_id`: admission or hospital-system-stay grouping identifier used for cohort auditing. It is not included as a predictor.
- `source_dataset`: source label (`MIMIC` or `eICU`).
- `environment_type`: modeling environment label.

## Outcome

- `target_mortality`: in-hospital mortality. MIMIC uses `hospital_expire_flag`; eICU uses `actualhospitalmortality` converted to a binary label.

## Primary 24-hour shared clinical predictors

The primary model retains variables only when the corresponding descriptor has no more than 60% missingness in each source. Temperature descriptors are extracted and audited but excluded from the primary shared-feature matrix because eICU temperature missingness is 91.8% in the corrected 24-hour landmark cohort.

- `age`
- `gender`
- `hr_mean`, `hr_min`, `hr_max`
- `respiration_mean`, `respiration_min`, `respiration_max`
- `bp_sys_mean`, `bp_sys_min`, `bp_sys_max`
- `bp_dia_mean`, `bp_dia_min`, `bp_dia_max`
- `bp_mean_mean`, `bp_mean_min`, `bp_mean_max`
- `glucose_mean`, `glucose_min`, `glucose_max`
- `creatinine_mean`, `creatinine_min`, `creatinine_max`
- `wbc_mean`, `wbc_min`, `wbc_max`
- `hematocrit_mean`, `hematocrit_min`, `hematocrit_max`
- `sodium_mean`, `sodium_min`, `sodium_max`
- `bun_mean`, `bun_min`, `bun_max`

## Secondary full-episode strict shared clinical predictors

These variables are available in both domains and define the strict shared-feature sensitivity analysis:

- `age`
- `gender`
- `hr_mean`, `hr_std`
- `respiration_mean`, `respiration_std`
- `glucose_mean`, `glucose_std`
- `creatinine_mean`, `creatinine_std`
- `wbc_mean`, `wbc_std`
- `sodium_mean`, `sodium_std`
- `bun_mean`, `bun_std`

## Source-specific or non-shared predictors excluded from strict shared-feature sensitivity

- `hospitalid`: eICU-specific hospital identifier.
- `apachescore`: eICU severity score.
- `predictedhospitalmortality`: eICU APACHE-derived mortality prediction.
- arterial or pressure summary variables that are completely absent in one extracted domain.

These variables are retained only for the original full-feature retrospective analysis and are not treated as clinically harmonized cross-domain predictors.

## Explicit source mappings added in review

MIMIC laboratory variables are selected by item identifier rather than broad label matching: glucose `50809, 50931`; creatinine `50912`; white blood cell count `51301`; hematocrit `50810, 51221`; sodium `50824, 50983`; blood urea nitrogen `51006`. MIMIC vital signs are selected from explicit `CHARTEVENTS` item identifiers listed in `data/aggregate_outputs/primary_24h_feature_mapping_audit.csv`. eICU laboratory variables are selected by exact lower-case lab names: `glucose`, `creatinine`, `wbc x 1000`, `hct`, `sodium`, and `bun`.
