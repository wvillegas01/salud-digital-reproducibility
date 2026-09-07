# Variable Dictionary

## Identifiers and provenance

- `case_id`: ICU episode identifier. MIMIC uses `icustay_id`; eICU uses `patientunitstayid`.
- `source_dataset`: source label (`MIMIC` or `eICU`).
- `environment_type`: modeling environment label.

## Outcome

- `target_mortality`: in-hospital mortality. MIMIC uses `hospital_expire_flag`; eICU uses `actualhospitalmortality` converted to a binary label.

## Strict shared clinical predictors

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
