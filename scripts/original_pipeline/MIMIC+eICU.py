from pathlib import Path
import pandas as pd
import numpy as np

BASE_PATH = Path(r"C:\Users\wilop\Documents\Datos-generales\Clinicos")

MIMIC_PATH = BASE_PATH / "mimic-iii-clinical-database-demo-1.4"
EICU_PATH = BASE_PATH / "eICU"

OUTPUT_FILE = BASE_PATH / "dataset_clinico_integrado_mimic_eicu.csv"


# =========================
# UTILIDADES
# =========================

def safe_mean(series):
    series = pd.to_numeric(series, errors="coerce")
    return series.mean()


def normalize_gender(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip().lower()
    if value in ["m", "male"]:
        return "male"
    if value in ["f", "female"]:
        return "female"
    return "unknown"


# =========================
# MIMIC
# =========================

def build_mimic_dataset():
    print("Procesando MIMIC...")

    patients = pd.read_csv(MIMIC_PATH / "PATIENTS.csv", low_memory=False)
    admissions = pd.read_csv(MIMIC_PATH / "ADMISSIONS.csv", low_memory=False)
    icustays = pd.read_csv(MIMIC_PATH / "ICUSTAYS.csv", low_memory=False)
    labevents = pd.read_csv(MIMIC_PATH / "LABEVENTS.csv", low_memory=False)
    chartevents = pd.read_csv(MIMIC_PATH / "CHARTEVENTS.csv", low_memory=False)
    d_labitems = pd.read_csv(MIMIC_PATH / "D_LABITEMS.csv", low_memory=False)
    d_items = pd.read_csv(MIMIC_PATH / "D_ITEMS.csv", low_memory=False)

    # -------------------------
    # Base paciente-admisión-UCI
    # -------------------------
    base = icustays.merge(
        admissions[["subject_id", "hadm_id", "hospital_expire_flag", "admittime"]],
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

# Regla habitual para edades anonimizadas en MIMIC
    base.loc[base["age"] > 120, "age"] = 90
    base.loc[base["age"] < 0, "age"] = np.nan

    # -------------------------
    # Laboratorios
    # -------------------------
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

    lab_features = []

    for feature, keywords in lab_map.items():
        mask = labs["label_lower"].apply(lambda x: any(k in x for k in keywords))
        tmp = labs.loc[mask].groupby("hadm_id")["valuenum"].agg(
            [("mean", "mean"), ("min", "min"), ("max", "max")]
        ).reset_index()

        tmp.columns = ["hadm_id", f"{feature}_mean", f"{feature}_min", f"{feature}_max"]
        lab_features.append(tmp)

    for lf in lab_features:
        base = base.merge(lf, on="hadm_id", how="left")

    # -------------------------
    # Signos vitales
    # -------------------------
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

    vital_features = []

    for feature, keywords in vital_map.items():
        mask = chart["label_lower"].apply(lambda x: any(k in x for k in keywords))
        tmp = chart.loc[mask].groupby("icustay_id")["valuenum"].agg(
            [("mean", "mean"), ("min", "min"), ("max", "max")]
        ).reset_index()

        tmp.columns = ["icustay_id", f"{feature}_mean", f"{feature}_min", f"{feature}_max"]
        vital_features.append(tmp)

    for vf in vital_features:
        base = base.merge(vf, on="icustay_id", how="left")

    # -------------------------
    # Salida MIMIC
    # -------------------------
    base["source_dataset"] = "MIMIC"
    base["environment_type"] = "single_center_icu"
    base["case_id"] = base["icustay_id"]
    base["target_mortality"] = base["hospital_expire_flag"]

    keep_cols = [
        "source_dataset", "environment_type", "case_id",
        "age", "gender",
        "hr_mean", "hr_min", "hr_max",
        "temperature_mean", "temperature_min", "temperature_max",
        "respiration_mean", "respiration_min", "respiration_max",
        "bp_sys_mean", "bp_sys_min", "bp_sys_max",
        "bp_dia_mean", "bp_dia_min", "bp_dia_max",
        "bp_mean", "bp_mean_min", "bp_mean_max",
        "glucose_mean", "glucose_min", "glucose_max",
        "creatinine_mean", "creatinine_min", "creatinine_max",
        "wbc_mean", "wbc_min", "wbc_max",
        "hematocrit_mean", "hematocrit_min", "hematocrit_max",
        "sodium_mean", "sodium_min", "sodium_max",
        "bun_mean", "bun_min", "bun_max",
        "target_mortality"
    ]

    existing_cols = [c for c in keep_cols if c in base.columns]
    return base[existing_cols]


# =========================
# eICU
# =========================

def build_eicu_dataset():
    print("Procesando eICU...")

    patient = pd.read_csv(EICU_PATH / "patient.csv.gz", compression="gzip", low_memory=False)
    vital = pd.read_csv(EICU_PATH / "vitalPeriodic.csv.gz", compression="gzip", low_memory=False)
    vital_ap = pd.read_csv(EICU_PATH / "vitalAperiodic.csv.gz", compression="gzip", low_memory=False)
    lab = pd.read_csv(EICU_PATH / "lab.csv.gz", compression="gzip", low_memory=False)
    apache = pd.read_csv(EICU_PATH / "apachePatientResult.csv.gz", compression="gzip", low_memory=False)
    apache_vars = pd.read_csv(EICU_PATH / "apacheApsVar.csv.gz", compression="gzip", low_memory=False)

    base = patient.copy()

    base["age"] = pd.to_numeric(base["age"], errors="coerce")
    base["gender"] = base["gender"].apply(normalize_gender)

    # -------------------------
    # Vital periodic
    # -------------------------
    vital_cols = {
        "heartrate": "hr",
        "temperature": "temperature",
        "respiration": "respiration",
        "systemicsystolic": "bp_sys",
        "systemicdiastolic": "bp_dia",
        "systemicmean": "bp_mean"
    }

    vital_features = []

    for original, feature in vital_cols.items():
        if original in vital.columns:
            vital[original] = pd.to_numeric(vital[original], errors="coerce")
            tmp = vital.groupby("patientunitstayid")[original].agg(
                [("mean", "mean"), ("min", "min"), ("max", "max")]
            ).reset_index()
            tmp.columns = [
                "patientunitstayid",
                f"{feature}_mean",
                f"{feature}_min",
                f"{feature}_max"
            ]
            vital_features.append(tmp)

    for vf in vital_features:
        base = base.merge(vf, on="patientunitstayid", how="left")

    # -------------------------
    # Vital aperiodic
    # -------------------------
    ap_cols = {
        "noninvasivesystolic": "bp_sys_ap",
        "noninvasivediastolic": "bp_dia_ap",
        "noninvasivemean": "bp_mean_ap"
    }

    for original, feature in ap_cols.items():
        if original in vital_ap.columns:
            vital_ap[original] = pd.to_numeric(vital_ap[original], errors="coerce")
            tmp = vital_ap.groupby("patientunitstayid")[original].agg(
                [("mean", "mean"), ("min", "min"), ("max", "max")]
            ).reset_index()
            tmp.columns = [
                "patientunitstayid",
                f"{feature}_mean",
                f"{feature}_min",
                f"{feature}_max"
            ]
            base = base.merge(tmp, on="patientunitstayid", how="left")

    # -------------------------
    # Laboratorios
    # -------------------------
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
        tmp = lab.loc[mask].groupby("patientunitstayid")["labresult"].agg(
            [("mean", "mean"), ("min", "min"), ("max", "max")]
        ).reset_index()

        tmp.columns = [
            "patientunitstayid",
            f"{feature}_mean",
            f"{feature}_min",
            f"{feature}_max"
        ]
        base = base.merge(tmp, on="patientunitstayid", how="left")

    # -------------------------
    # APACHE / target
    # -------------------------
    base = base.merge(
        apache[
            [
                "patientunitstayid",
                "apachescore",
                "predictedicumortality",
                "actualicumortality",
                "predictedhospitalmortality",
                "actualhospitalmortality"
            ]
        ],
        on="patientunitstayid",
        how="left"
    )

    base = base.merge(
        apache_vars[
            [
                "patientunitstayid",
                "wbc",
                "temperature",
                "respiratoryrate",
                "sodium",
                "heartrate",
                "meanbp",
                "hematocrit",
                "creatinine",
                "bun",
                "glucose"
            ]
        ],
        on="patientunitstayid",
        how="left",
        suffixes=("", "_apache")
    )

    # Target binario
    def mortality_to_binary(x):
        if pd.isna(x):
            return np.nan
        x = str(x).strip().lower()
        if x in ["alive", "0", "false", "no"]:
            return 0
        if x in ["expired", "dead", "1", "true", "yes"]:
            return 1
        return np.nan

    base["target_mortality"] = base["actualhospitalmortality"].apply(mortality_to_binary)

    base["source_dataset"] = "eICU"
    base["environment_type"] = "multicenter_icu"
    base["case_id"] = base["patientunitstayid"]

    keep_cols = [
        "source_dataset", "environment_type", "case_id",
        "hospitalid",
        "age", "gender", "ethnicity",
        "hr_mean", "hr_min", "hr_max",
        "temperature_mean", "temperature_min", "temperature_max",
        "respiration_mean", "respiration_min", "respiration_max",
        "bp_sys_mean", "bp_sys_min", "bp_sys_max",
        "bp_dia_mean", "bp_dia_min", "bp_dia_max",
        "bp_mean", "bp_mean_min", "bp_mean_max",
        "bp_sys_ap_mean", "bp_sys_ap_min", "bp_sys_ap_max",
        "bp_dia_ap_mean", "bp_dia_ap_min", "bp_dia_ap_max",
        "bp_mean_ap_mean", "bp_mean_ap_min", "bp_mean_ap_max",
        "glucose_mean", "glucose_min", "glucose_max",
        "creatinine_mean", "creatinine_min", "creatinine_max",
        "wbc_mean", "wbc_min", "wbc_max",
        "hematocrit_mean", "hematocrit_min", "hematocrit_max",
        "sodium_mean", "sodium_min", "sodium_max",
        "bun_mean", "bun_min", "bun_max",
        "apachescore",
        "predictedicumortality",
        "predictedhospitalmortality",
        "target_mortality"
    ]

    existing_cols = [c for c in keep_cols if c in base.columns]
    return base[existing_cols]


# =========================
# EJECUCIÓN
# =========================

if __name__ == "__main__":
    mimic_df = build_mimic_dataset()
    eicu_df = build_eicu_dataset()

    print("\nMIMIC:", mimic_df.shape)
    print("eICU:", eicu_df.shape)

    integrated = pd.concat([mimic_df, eicu_df], ignore_index=True, sort=False)

    integrated.to_csv(OUTPUT_FILE, index=False)

    print("\nDataset integrado generado:")
    print(OUTPUT_FILE)
    print(integrated.head())
    print("\nDistribución por fuente:")
    print(integrated["source_dataset"].value_counts(dropna=False))
    print("\nTarget mortality:")
    print(integrated.groupby("source_dataset")["target_mortality"].value_counts(dropna=False))