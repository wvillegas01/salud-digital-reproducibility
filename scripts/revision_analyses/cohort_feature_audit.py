import json
from pathlib import Path

import pandas as pd


BASE = Path(r"C:\Users\wilop\Documents\Datos-generales\Clinicos")
MIMIC = BASE / "mimic-iii-clinical-database-demo-1.4"
EICU = BASE / "eICU"
OUT = Path(r"C:\Users\wilop\Documents\Codex\2026-09-06\ha\work")
DATA_PATH = OUT / "dataset_clinico_landmark_24h.csv"
TARGET = "target_mortality"
ID_COLUMNS = ["case_id", "patient_group_id", "admission_group_id", "source_dataset", "environment_type"]
WINDOW_HOURS = 24

LAB_ITEMIDS_MIMIC = {
    "glucose": [50809, 50931],
    "creatinine": [50912],
    "wbc": [51301],
    "hematocrit": [50810, 51221],
    "sodium": [50824, 50983],
    "bun": [51006],
}

CHART_ITEMIDS_MIMIC = {
    "hr": [211, 220045],
    "temperature_c": [676, 677, 223762, 226329],
    "temperature_f": [678, 679, 223761],
    "respiration": [618, 220210],
    "bp_sys": [51, 442, 455, 220050, 220179, 224167, 225309, 227243],
    "bp_dia": [8368, 8440, 8441, 220051, 220180, 224643, 225310, 227242],
    "bp_mean": [52, 456, 220052, 220181, 225312],
}

LAB_NAMES_EICU = {
    "glucose": ["glucose"],
    "creatinine": ["creatinine"],
    "wbc": ["wbc x 1000"],
    "hematocrit": ["hct"],
    "sodium": ["sodium"],
    "bun": ["bun"],
}


def build_mimic_eligible_base():
    patients = pd.read_csv(MIMIC / "PATIENTS.csv", low_memory=False)
    admissions = pd.read_csv(MIMIC / "ADMISSIONS.csv", low_memory=False)
    icustays = pd.read_csv(MIMIC / "ICUSTAYS.csv", low_memory=False)
    base = icustays.merge(
        admissions[["subject_id", "hadm_id", "hospital_expire_flag", "admittime", "dischtime", "deathtime"]],
        on=["subject_id", "hadm_id"],
        how="left",
    ).merge(patients[["subject_id", "gender", "dob"]], on="subject_id", how="left")
    for col in ["intime", "outtime", "deathtime", "admittime", "dischtime"]:
        base[col] = pd.to_datetime(base[col], errors="coerce")
    base["window_end"] = base["intime"] + pd.Timedelta(hours=WINDOW_HOURS)
    early_death = (base["hospital_expire_flag"] == 1) & base["deathtime"].notna() & (base["deathtime"] <= base["window_end"])
    early_icu_discharge = base["outtime"].notna() & (base["outtime"] <= base["window_end"])
    early_hospital_discharge = base["dischtime"].notna() & (base["dischtime"] <= base["window_end"])
    return base[~(early_death | early_icu_discharge | early_hospital_discharge)].copy()


def audit_mimic_lab_windows(base):
    labevents = pd.read_csv(MIMIC / "LABEVENTS.csv", low_memory=False)
    labevents["charttime"] = pd.to_datetime(labevents["charttime"], errors="coerce")
    itemids = [itemid for ids in LAB_ITEMIDS_MIMIC.values() for itemid in ids]
    labs = labevents[labevents["itemid"].isin(itemids)].merge(
        base[["subject_id", "hadm_id", "icustay_id", "intime", "window_end"]],
        on=["subject_id", "hadm_id"],
        how="inner",
    )
    outside_before_filter = int(((labs["charttime"] < labs["intime"]) | (labs["charttime"] > labs["window_end"])).sum())
    labs_24h = labs[(labs["charttime"] >= labs["intime"]) & (labs["charttime"] <= labs["window_end"])].copy()
    outside_after_filter = int(((labs_24h["charttime"] < labs_24h["intime"]) | (labs_24h["charttime"] > labs_24h["window_end"])).sum())
    return {
        "selected_mimic_lab_rows_before_window_filter": int(len(labs)),
        "selected_mimic_lab_rows_outside_own_icu_window_before_filter": outside_before_filter,
        "selected_mimic_lab_rows_after_own_icu_window_filter": int(len(labs_24h)),
        "selected_mimic_lab_rows_outside_own_icu_window_after_filter": outside_after_filter,
        "eligible_mimic_admissions_with_multiple_icu_stays": int((base.groupby("hadm_id")["icustay_id"].nunique() > 1).sum()),
    }


def main():
    df = pd.read_csv(DATA_PATH).dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)
    candidate_features = [c for c in df.columns if c not in ID_COLUMNS + [TARGET]]
    missing_by_source = df.groupby("source_dataset")[candidate_features].apply(lambda g: g.isna().mean())
    retained = [
        col
        for col in candidate_features
        if all(missing_by_source.loc[source, col] <= 0.60 for source in missing_by_source.index)
    ]
    missing_audit = missing_by_source.T.reset_index().rename(columns={"index": "variable"})
    missing_audit["retained_primary_shared_feature"] = missing_audit["variable"].isin(retained)
    missing_audit.to_csv(OUT / "primary_24h_missingness_rule_audit.csv", index=False)

    mapping_rows = []
    for feature, itemids in LAB_ITEMIDS_MIMIC.items():
        mapping_rows.append({"source": "MIMIC", "feature": feature, "table": "LABEVENTS", "mapping": ",".join(map(str, itemids))})
    for feature, itemids in CHART_ITEMIDS_MIMIC.items():
        mapping_rows.append({"source": "MIMIC", "feature": feature, "table": "CHARTEVENTS", "mapping": ",".join(map(str, itemids))})
    for feature, names in LAB_NAMES_EICU.items():
        mapping_rows.append({"source": "eICU", "feature": feature, "table": "lab", "mapping": ";".join(names)})
    pd.DataFrame(mapping_rows).to_csv(OUT / "primary_24h_feature_mapping_audit.csv", index=False)

    group_rows = []
    for source, group in df.groupby("source_dataset"):
        patient_sizes = group.groupby("patient_group_id").size()
        admission_sizes = group.groupby("admission_group_id").size()
        group_rows.append(
            {
                "source_dataset": source,
                "episodes": int(len(group)),
                "events": int(group[TARGET].sum()),
                "event_prevalence": float(group[TARGET].mean()),
                "unique_patients": int(group["patient_group_id"].nunique()),
                "unique_admissions": int(group["admission_group_id"].nunique()),
                "patients_with_multiple_episodes": int((patient_sizes > 1).sum()),
                "admissions_with_multiple_episodes": int((admission_sizes > 1).sum()),
            }
        )
    pd.DataFrame(group_rows).to_csv(OUT / "primary_24h_group_dependence_summary.csv", index=False)

    mimic_base = build_mimic_eligible_base()
    summary = {
        "analysis": "primary_24h_landmark_cohort_feature_audit",
        "landmark_hours": WINDOW_HOURS,
        "source_dataset_rows": {source: int(len(group)) for source, group in df.groupby("source_dataset")},
        "events": {source: int(group[TARGET].sum()) for source, group in df.groupby("source_dataset")},
        "eicu_outcome_source": "apachePatientResult.actualhospitalmortality; APACHE IVa preferred over IV when duplicated",
        "mimic_outcome_source": "ADMISSIONS.hospital_expire_flag",
        "mimic_lab_window_audit": audit_mimic_lab_windows(mimic_base),
        "primary_shared_feature_rule": "feature retained only when missingness is <= 60% in every source dataset",
        "primary_shared_feature_count": len(retained),
        "primary_shared_features": retained,
        "temperature_retained": any(col.startswith("temperature_") for col in retained),
    }
    (OUT / "primary_24h_cohort_feature_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
