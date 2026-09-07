from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

BASE_PATH = Path(r"C:\Users\wilop\Documents\Datos-generales\Clinicos")

MIMIC_PATH = BASE_PATH / "mimic-iii-clinical-database-demo-1.4"
EICU_PATH = BASE_PATH / "eICU"

OUTPUT_RAW = BASE_PATH / "dataset_clinico_integrado_mimic_eicu_raw.csv"
OUTPUT_CLEAN = BASE_PATH / "dataset_clinico_integrado_mimic_eicu_clean.csv"
OUTPUT_AUDIT = BASE_PATH / "auditoria_dataset_integrado_mimic_eicu.xlsx"


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def normalize_gender(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip().lower()
    if value in ["m", "male"]:
        return "male"
    if value in ["f", "female"]:
        return "female"
    return "unknown"


def mortality_to_binary(x):
    if pd.isna(x):
        return np.nan

    x = str(x).strip().lower()

    if x in ["alive", "0", "false", "no"]:
        return 0

    if x in ["expired", "dead", "1", "true", "yes"]:
        return 1

    return np.nan


def aggregate_numeric(df, group_col, value_col, prefix):
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    out = df.groupby(group_col)[value_col].agg(
        mean="mean",
        min="min",
        max="max",
        std="std"
    ).reset_index()

    out.columns = [
        group_col,
        f"{prefix}_mean",
        f"{prefix}_min",
        f"{prefix}_max",
        f"{prefix}_std"
    ]

    return out


# ============================================================
# PROCESAMIENTO MIMIC
# ============================================================

def build_mimic_dataset():
    print("\nProcesando MIMIC...")

    patients = pd.read_csv(MIMIC_PATH / "PATIENTS.csv", low_memory=False)
    admissions = pd.read_csv(MIMIC_PATH / "ADMISSIONS.csv", low_memory=False)
    icustays = pd.read_csv(MIMIC_PATH / "ICUSTAYS.csv", low_memory=False)
    labevents = pd.read_csv(MIMIC_PATH / "LABEVENTS.csv", low_memory=False)
    chartevents = pd.read_csv(MIMIC_PATH / "CHARTEVENTS.csv", low_memory=False)
    d_labitems = pd.read_csv(MIMIC_PATH / "D_LABITEMS.csv", low_memory=False)
    d_items = pd.read_csv(MIMIC_PATH / "D_ITEMS.csv", low_memory=False)

    base = icustays.merge(
        admissions[[
            "subject_id",
            "hadm_id",
            "admittime",
            "hospital_expire_flag",
            "ethnicity"
        ]],
        on=["subject_id", "hadm_id"],
        how="left"
    )

    base = base.merge(
        patients[["subject_id", "gender", "dob"]],
        on="subject_id",
        how="left"
    )

    base["admittime"] = pd.to_datetime(base["admittime"], errors="coerce")
    base["dob"] = pd.to_datetime(base["dob"], errors="coerce")

    base["admit_year"] = base["admittime"].dt.year
    base["dob_year"] = base["dob"].dt.year
    base["age"] = base["admit_year"] - base["dob_year"]

    base.loc[base["age"] > 120, "age"] = 90
    base.loc[base["age"] < 0, "age"] = np.nan

    base["gender"] = base["gender"].apply(normalize_gender)

    # ---------------------------
    # LABORATORIOS MIMIC
    # ---------------------------

    labs = labevents.merge(
        d_labitems[["itemid", "label"]],
        on="itemid",
        how="left"
    )

    labs["label_lower"] = labs["label"].astype(str).str.lower()
    labs["valuenum"] = pd.to_numeric(labs["valuenum"], errors="coerce")

    lab_map = {
        "glucose": ["glucose"],
        "creatinine": ["creatinine"],
        "wbc": ["white blood cells", "wbc"],
        "hematocrit": ["hematocrit"],
        "sodium": ["sodium"],
        "bun": ["urea nitrogen", "bun"]
    }

    for feature, keywords in lab_map.items():
        mask = labs["label_lower"].apply(lambda x: any(k in x for k in keywords))
        tmp = aggregate_numeric(
            labs.loc[mask].copy(),
            group_col="hadm_id",
            value_col="valuenum",
            prefix=feature
        )
        base = base.merge(tmp, on="hadm_id", how="left")

    # ---------------------------
    # SIGNOS VITALES MIMIC
    # ---------------------------

    chart = chartevents.merge(
        d_items[["itemid", "label"]],
        on="itemid",
        how="left"
    )

    chart["label_lower"] = chart["label"].astype(str).str.lower()
    chart["valuenum"] = pd.to_numeric(chart["valuenum"], errors="coerce")

    vital_map = {
        "hr": ["heart rate"],
        "temperature": ["temperature"],
        "respiration": ["respiratory rate"],
        "bp_sys": ["systolic"],
        "bp_dia": ["diastolic"],
        "bp_mean": ["mean arterial pressure", "arterial pressure mean"]
    }

    for feature, keywords in vital_map.items():
        mask = chart["label_lower"].apply(lambda x: any(k in x for k in keywords))
        tmp = aggregate_numeric(
            chart.loc[mask].copy(),
            group_col="icustay_id",
            value_col="valuenum",
            prefix=feature
        )
        base = base.merge(tmp, on="icustay_id", how="left")

    base["source_dataset"] = "MIMIC"
    base["environment_type"] = "single_center_icu"
    base["case_id"] = base["icustay_id"]
    base["target_mortality"] = base["hospital_expire_flag"]

    keep_cols = [
        "source_dataset", "environment_type", "case_id",
        "age", "gender", "ethnicity",
        "hr_mean", "hr_min", "hr_max", "hr_std",
        "temperature_mean", "temperature_min", "temperature_max", "temperature_std",
        "respiration_mean", "respiration_min", "respiration_max", "respiration_std",
        "bp_sys_mean", "bp_sys_min", "bp_sys_max", "bp_sys_std",
        "bp_dia_mean", "bp_dia_min", "bp_dia_max", "bp_dia_std",
        "bp_mean_mean", "bp_mean_min", "bp_mean_max", "bp_mean_std",
        "glucose_mean", "glucose_min", "glucose_max", "glucose_std",
        "creatinine_mean", "creatinine_min", "creatinine_max", "creatinine_std",
        "wbc_mean", "wbc_min", "wbc_max", "wbc_std",
        "hematocrit_mean", "hematocrit_min", "hematocrit_max", "hematocrit_std",
        "sodium_mean", "sodium_min", "sodium_max", "sodium_std",
        "bun_mean", "bun_min", "bun_max", "bun_std",
        "target_mortality"
    ]

    keep_cols = [c for c in keep_cols if c in base.columns]
    return base[keep_cols]


# ============================================================
# PROCESAMIENTO eICU
# ============================================================

def build_eicu_dataset():
    print("\nProcesando eICU...")

    patient = pd.read_csv(EICU_PATH / "patient.csv.gz", compression="gzip", low_memory=False)
    vital = pd.read_csv(EICU_PATH / "vitalPeriodic.csv.gz", compression="gzip", low_memory=False)
    vital_ap = pd.read_csv(EICU_PATH / "vitalAperiodic.csv.gz", compression="gzip", low_memory=False)
    lab = pd.read_csv(EICU_PATH / "lab.csv.gz", compression="gzip", low_memory=False)
    apache = pd.read_csv(EICU_PATH / "apachePatientResult.csv.gz", compression="gzip", low_memory=False)

    base = patient.copy()

    base["age"] = pd.to_numeric(base["age"], errors="coerce")
    base["gender"] = base["gender"].apply(normalize_gender)

    # ---------------------------
    # VITALES PERIÓDICOS eICU
    # ---------------------------

    vital_cols = {
        "heartrate": "hr",
        "temperature": "temperature",
        "respiration": "respiration",
        "systemicsystolic": "bp_sys",
        "systemicdiastolic": "bp_dia",
        "systemicmean": "bp_mean"
    }

    for original_col, feature in vital_cols.items():
        if original_col in vital.columns:
            tmp = aggregate_numeric(
                vital[["patientunitstayid", original_col]].copy(),
                group_col="patientunitstayid",
                value_col=original_col,
                prefix=feature
            )
            base = base.merge(tmp, on="patientunitstayid", how="left")

    # ---------------------------
    # VITALES APERIÓDICOS eICU
    # ---------------------------

    ap_cols = {
        "noninvasivesystolic": "bp_sys_ap",
        "noninvasivediastolic": "bp_dia_ap",
        "noninvasivemean": "bp_mean_ap"
    }

    for original_col, feature in ap_cols.items():
        if original_col in vital_ap.columns:
            tmp = aggregate_numeric(
                vital_ap[["patientunitstayid", original_col]].copy(),
                group_col="patientunitstayid",
                value_col=original_col,
                prefix=feature
            )
            base = base.merge(tmp, on="patientunitstayid", how="left")

    # ---------------------------
    # LABORATORIOS eICU
    # ---------------------------

    lab["labname_lower"] = lab["labname"].astype(str).str.lower()
    lab["labresult"] = pd.to_numeric(lab["labresult"], errors="coerce")

    lab_map = {
        "glucose": ["glucose"],
        "creatinine": ["creatinine"],
        "wbc": ["wbc", "white blood cell"],
        "hematocrit": ["hematocrit"],
        "sodium": ["sodium"],
        "bun": ["bun", "blood urea nitrogen"]
    }

    for feature, keywords in lab_map.items():
        mask = lab["labname_lower"].apply(lambda x: any(k in x for k in keywords))
        tmp = aggregate_numeric(
            lab.loc[mask, ["patientunitstayid", "labresult"]].copy(),
            group_col="patientunitstayid",
            value_col="labresult",
            prefix=feature
        )
        base = base.merge(tmp, on="patientunitstayid", how="left")

    # ---------------------------
    # TARGET / APACHE eICU
    # ---------------------------

    apache_cols = [
        "patientunitstayid",
        "apachescore",
        "predictedicumortality",
        "actualicumortality",
        "predictedhospitalmortality",
        "actualhospitalmortality"
    ]

    available_apache_cols = [c for c in apache_cols if c in apache.columns]

    base = base.merge(
        apache[available_apache_cols],
        on="patientunitstayid",
        how="left"
    )

    base["target_mortality"] = base["actualhospitalmortality"].apply(mortality_to_binary)

    base["source_dataset"] = "eICU"
    base["environment_type"] = "multicenter_icu"
    base["case_id"] = base["patientunitstayid"]

    keep_cols = [
        "source_dataset", "environment_type", "case_id",
        "hospitalid",
        "age", "gender", "ethnicity",
        "hr_mean", "hr_min", "hr_max", "hr_std",
        "temperature_mean", "temperature_min", "temperature_max", "temperature_std",
        "respiration_mean", "respiration_min", "respiration_max", "respiration_std",
        "bp_sys_mean", "bp_sys_min", "bp_sys_max", "bp_sys_std",
        "bp_dia_mean", "bp_dia_min", "bp_dia_max", "bp_dia_std",
        "bp_mean_mean", "bp_mean_min", "bp_mean_max", "bp_mean_std",
        "bp_sys_ap_mean", "bp_sys_ap_min", "bp_sys_ap_max", "bp_sys_ap_std",
        "bp_dia_ap_mean", "bp_dia_ap_min", "bp_dia_ap_max", "bp_dia_ap_std",
        "bp_mean_ap_mean", "bp_mean_ap_min", "bp_mean_ap_max", "bp_mean_ap_std",
        "glucose_mean", "glucose_min", "glucose_max", "glucose_std",
        "creatinine_mean", "creatinine_min", "creatinine_max", "creatinine_std",
        "wbc_mean", "wbc_min", "wbc_max", "wbc_std",
        "hematocrit_mean", "hematocrit_min", "hematocrit_max", "hematocrit_std",
        "sodium_mean", "sodium_min", "sodium_max", "sodium_std",
        "bun_mean", "bun_min", "bun_max", "bun_std",
        "apachescore",
        "predictedicumortality",
        "predictedhospitalmortality",
        "target_mortality"
    ]

    keep_cols = [c for c in keep_cols if c in base.columns]
    return base[keep_cols]


# ============================================================
# AUDITORÍA DEL DATASET FINAL
# ============================================================

def build_audit_report(raw_df, clean_df):
    file_summary = pd.DataFrame({
        "metric": [
            "raw_rows",
            "raw_columns",
            "clean_rows",
            "clean_columns",
            "excluded_missing_target"
        ],
        "value": [
            raw_df.shape[0],
            raw_df.shape[1],
            clean_df.shape[0],
            clean_df.shape[1],
            raw_df["target_mortality"].isna().sum()
        ]
    })

    source_distribution = raw_df["source_dataset"].value_counts(dropna=False).reset_index()
    source_distribution.columns = ["source_dataset", "records"]

    target_distribution = (
        raw_df.groupby("source_dataset")["target_mortality"]
        .value_counts(dropna=False)
        .reset_index(name="records")
    )

    missing_raw = raw_df.isnull().mean().sort_values(ascending=False).reset_index()
    missing_raw.columns = ["variable", "missing_ratio_raw"]

    missing_clean = clean_df.isnull().mean().sort_values(ascending=False).reset_index()
    missing_clean.columns = ["variable", "missing_ratio_clean"]

    variable_profile = []

    for col in clean_df.columns:
        s = clean_df[col]
        variable_profile.append({
            "variable": col,
            "dtype": str(s.dtype),
            "non_nulls": int(s.notnull().sum()),
            "nulls": int(s.isnull().sum()),
            "missing_ratio": float(s.isnull().mean()),
            "unique_values": int(s.nunique(dropna=True)),
            "sample_values": ", ".join(map(str, s.dropna().unique()[:5]))
        })

    variable_profile = pd.DataFrame(variable_profile)

    candidate_features = variable_profile[
        (variable_profile["missing_ratio"] <= 0.60)
        &
        (~variable_profile["variable"].isin([
            "source_dataset",
            "environment_type",
            "case_id",
            "target_mortality"
        ]))
    ].copy()

    return {
        "01_summary": file_summary,
        "02_source_distribution": source_distribution,
        "03_target_distribution": target_distribution,
        "04_missing_raw": missing_raw,
        "05_missing_clean": missing_clean,
        "06_variable_profile": variable_profile,
        "07_candidate_features": candidate_features
    }


# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================

if __name__ == "__main__":
    mimic_df = build_mimic_dataset()
    eicu_df = build_eicu_dataset()

    print("\nResumen inicial:")
    print("MIMIC:", mimic_df.shape)
    print("eICU:", eicu_df.shape)

    integrated_raw = pd.concat(
        [mimic_df, eicu_df],
        ignore_index=True,
        sort=False
    )

    integrated_raw.to_csv(OUTPUT_RAW, index=False)

    integrated_clean = integrated_raw.dropna(subset=["target_mortality"]).copy()
    integrated_clean["target_mortality"] = integrated_clean["target_mortality"].astype(int)

    integrated_clean.to_csv(OUTPUT_CLEAN, index=False)

    audit_sheets = build_audit_report(integrated_raw, integrated_clean)

    with pd.ExcelWriter(OUTPUT_AUDIT, engine="openpyxl") as writer:
        for sheet_name, df in audit_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print("\nArchivos generados:")
    print("RAW:", OUTPUT_RAW)
    print("CLEAN:", OUTPUT_CLEAN)
    print("AUDIT:", OUTPUT_AUDIT)

    print("\nDistribución por fuente:")
    print(integrated_raw["source_dataset"].value_counts(dropna=False))

    print("\nDistribución del target:")
    print(
        integrated_raw
        .groupby("source_dataset")["target_mortality"]
        .value_counts(dropna=False)
    )

    print("\nVariables candidatas con missing <= 60%:")
    print(audit_sheets["07_candidate_features"][["variable", "missing_ratio", "unique_values"]])

    # =========================
# SELECCIÓN FINAL DE FEATURES
# =========================

FINAL_FEATURES = [
    "source_dataset",
    "environment_type",
    "case_id",
    "hospitalid",
    "age",
    "gender",
    "hr_mean", "hr_std",
    "respiration_mean", "respiration_std",
    "bp_sys_ap_mean", "bp_sys_ap_std",
    "bp_dia_ap_mean", "bp_dia_ap_std",
    "bp_mean_ap_mean", "bp_mean_ap_std",
    "glucose_mean", "glucose_std",
    "creatinine_mean", "creatinine_std",
    "wbc_mean", "wbc_std",
    "sodium_mean", "sodium_std",
    "bun_mean", "bun_std",
    "apachescore",
    "predictedhospitalmortality",
    "target_mortality"
]

final_cols = [c for c in FINAL_FEATURES if c in integrated_clean.columns]
dataset_final = integrated_clean[final_cols].copy()

# Encoding simple para gender
dataset_final["gender"] = dataset_final["gender"].map({
    "male": 1,
    "female": 0
})

# Guardar
OUTPUT_FINAL = BASE_PATH / "dataset_clinico_final_mimic_eicu.csv"
dataset_final.to_csv(OUTPUT_FINAL, index=False)

print("\nDataset FINAL generado:")
print(OUTPUT_FINAL)
print(dataset_final.shape)

# Auditoría rápida final
print("\nMissing final:")
print(dataset_final.isnull().mean().sort_values(ascending=False))

print("\nDistribución target:")
print(dataset_final["target_mortality"].value_counts())