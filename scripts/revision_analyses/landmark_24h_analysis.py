from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE = Path(r"C:\Users\wilop\Documents\Datos-generales\Clinicos")
MIMIC = BASE / "mimic-iii-clinical-database-demo-1.4"
EICU = BASE / "eICU"
OUT = Path(r"C:\Users\wilop\Documents\Codex\2026-09-06\ha\work")
OUT.mkdir(parents=True, exist_ok=True)
WINDOW_HOURS = 24
WINDOW_MINUTES = WINDOW_HOURS * 60

FEATURES = [
    "age", "gender",
    "hr_mean", "hr_min", "hr_max",
    "temperature_mean", "temperature_min", "temperature_max",
    "respiration_mean", "respiration_min", "respiration_max",
    "bp_sys_mean", "bp_sys_min", "bp_sys_max",
    "bp_dia_mean", "bp_dia_min", "bp_dia_max",
    "bp_mean_mean", "bp_mean_min", "bp_mean_max",
    "glucose_mean", "glucose_min", "glucose_max",
    "creatinine_mean", "creatinine_min", "creatinine_max",
    "wbc_mean", "wbc_min", "wbc_max",
    "hematocrit_mean", "hematocrit_min", "hematocrit_max",
    "sodium_mean", "sodium_min", "sodium_max",
    "bun_mean", "bun_min", "bun_max",
]

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


def normalize_gender(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip().lower()
    if value in ["m", "male"]:
        return "male"
    if value in ["f", "female"]:
        return "female"
    return "unknown"


def aggregate_measurements(df, group_col, value_col, prefix):
    if df.empty:
        return pd.DataFrame(columns=[group_col, f"{prefix}_mean", f"{prefix}_min", f"{prefix}_max"])
    tmp = df.groupby(group_col)[value_col].agg(["mean", "min", "max"]).reset_index()
    tmp.columns = [group_col, f"{prefix}_mean", f"{prefix}_min", f"{prefix}_max"]
    return tmp


def mortality_to_binary(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().lower()
    if x in ["alive", "0", "false", "no"]:
        return 0
    if x in ["expired", "dead", "1", "true", "yes"]:
        return 1
    return np.nan


def build_mimic_24h():
    patients = pd.read_csv(MIMIC / "PATIENTS.csv", low_memory=False)
    admissions = pd.read_csv(MIMIC / "ADMISSIONS.csv", low_memory=False)
    icustays = pd.read_csv(MIMIC / "ICUSTAYS.csv", low_memory=False)
    labevents = pd.read_csv(MIMIC / "LABEVENTS.csv", low_memory=False)
    chartevents = pd.read_csv(MIMIC / "CHARTEVENTS.csv", low_memory=False)
    d_labitems = pd.read_csv(MIMIC / "D_LABITEMS.csv", low_memory=False)
    d_items = pd.read_csv(MIMIC / "D_ITEMS.csv", low_memory=False)

    base = icustays.merge(
        admissions[["subject_id", "hadm_id", "hospital_expire_flag", "admittime", "dischtime", "deathtime"]],
        on=["subject_id", "hadm_id"],
        how="left",
    ).merge(patients[["subject_id", "gender", "dob"]], on="subject_id", how="left")

    base["intime"] = pd.to_datetime(base["intime"], errors="coerce")
    base["outtime"] = pd.to_datetime(base["outtime"], errors="coerce")
    base["deathtime"] = pd.to_datetime(base["deathtime"], errors="coerce")
    base["admittime"] = pd.to_datetime(base["admittime"], errors="coerce")
    base["dischtime"] = pd.to_datetime(base["dischtime"], errors="coerce")
    base["dob"] = pd.to_datetime(base["dob"], errors="coerce")
    base["age"] = base["admittime"].dt.year - base["dob"].dt.year
    base.loc[base["age"] > 120, "age"] = 90
    base.loc[base["age"] < 0, "age"] = np.nan
    base["gender"] = base["gender"].apply(normalize_gender)
    base["window_end"] = base["intime"] + pd.Timedelta(hours=WINDOW_HOURS)

    early_death = (base["hospital_expire_flag"] == 1) & base["deathtime"].notna() & (base["deathtime"] <= base["window_end"])
    early_icu_discharge = base["outtime"].notna() & (base["outtime"] <= base["window_end"])
    early_hospital_discharge = base["dischtime"].notna() & (base["dischtime"] <= base["window_end"])
    base = base[~(early_death | early_icu_discharge | early_hospital_discharge)].copy()

    labs = labevents.merge(d_labitems[["itemid", "label"]], on="itemid", how="left")
    labs = labs.merge(base[["subject_id", "hadm_id", "icustay_id", "intime", "window_end"]], on=["subject_id", "hadm_id"], how="inner")
    labs["charttime"] = pd.to_datetime(labs["charttime"], errors="coerce")
    labs["valuenum"] = pd.to_numeric(labs["valuenum"], errors="coerce")
    labs = labs[(labs["charttime"] >= labs["intime"]) & (labs["charttime"] <= labs["window_end"])]
    for feature, itemids in LAB_ITEMIDS_MIMIC.items():
        mask = labs["itemid"].isin(itemids)
        base = base.merge(aggregate_measurements(labs.loc[mask], "icustay_id", "valuenum", feature), on="icustay_id", how="left")

    chart = chartevents.merge(d_items[["itemid", "label"]], on="itemid", how="left")
    chart = chart.merge(base[["icustay_id", "intime", "window_end"]], on="icustay_id", how="inner")
    chart["charttime"] = pd.to_datetime(chart["charttime"], errors="coerce")
    chart["valuenum"] = pd.to_numeric(chart["valuenum"], errors="coerce")
    chart = chart[(chart["charttime"] >= chart["intime"]) & (chart["charttime"] <= chart["window_end"])]
    temp_c = chart.loc[chart["itemid"].isin(CHART_ITEMIDS_MIMIC["temperature_c"]), ["icustay_id", "valuenum"]].copy()
    temp_f = chart.loc[chart["itemid"].isin(CHART_ITEMIDS_MIMIC["temperature_f"]), ["icustay_id", "valuenum"]].copy()
    temp_f["valuenum"] = (temp_f["valuenum"] - 32) * 5 / 9
    temp = pd.concat([temp_c, temp_f], ignore_index=True)
    temp = temp[(temp["valuenum"] >= 25) & (temp["valuenum"] <= 45)]
    base = base.merge(aggregate_measurements(temp, "icustay_id", "valuenum", "temperature"), on="icustay_id", how="left")

    for feature in ["hr", "respiration", "bp_sys", "bp_dia", "bp_mean"]:
        mask = chart["itemid"].isin(CHART_ITEMIDS_MIMIC[feature])
        base = base.merge(aggregate_measurements(chart.loc[mask], "icustay_id", "valuenum", feature), on="icustay_id", how="left")

    base["source_dataset"] = "MIMIC"
    base["environment_type"] = "single_center_icu"
    base["case_id"] = base["icustay_id"]
    base["patient_group_id"] = "MIMIC_" + base["subject_id"].astype(str)
    base["admission_group_id"] = "MIMIC_" + base["hadm_id"].astype(str)
    base["target_mortality"] = base["hospital_expire_flag"]
    return base[["source_dataset", "environment_type", "case_id", "patient_group_id", "admission_group_id", "age", "gender", *[c for c in FEATURES if c not in ["age", "gender"]], "target_mortality"]]


def build_eicu_24h():
    patient = pd.read_csv(EICU / "patient.csv.gz", compression="gzip", low_memory=False)
    apache = pd.read_csv(EICU / "apachePatientResult.csv.gz", compression="gzip", low_memory=False)
    vital = pd.read_csv(EICU / "vitalPeriodic.csv.gz", compression="gzip", low_memory=False)
    vital_ap = pd.read_csv(EICU / "vitalAperiodic.csv.gz", compression="gzip", low_memory=False)
    lab = pd.read_csv(EICU / "lab.csv.gz", compression="gzip", low_memory=False)

    base = patient.copy()
    base["age"] = pd.to_numeric(base["age"], errors="coerce")
    base["gender"] = base["gender"].apply(normalize_gender)

    apache = apache[["patientunitstayid", "apacheversion", "actualhospitalmortality"]].copy()
    apache["target_mortality"] = apache["actualhospitalmortality"].apply(mortality_to_binary)
    apache["apache_priority"] = apache["apacheversion"].astype(str).str.lower().map({"iva": 0, "iv": 1}).fillna(2)
    apache = apache.sort_values(["patientunitstayid", "apache_priority"]).drop_duplicates("patientunitstayid")
    base = base.merge(apache[["patientunitstayid", "target_mortality"]], on="patientunitstayid", how="left")
    unit_discharge_offset = pd.to_numeric(base["unitdischargeoffset"], errors="coerce")
    hospital_discharge_offset = pd.to_numeric(base["hospitaldischargeoffset"], errors="coerce")
    early_unit_discharge = unit_discharge_offset.notna() & (unit_discharge_offset <= WINDOW_MINUTES)
    early_hospital_discharge = hospital_discharge_offset.notna() & (hospital_discharge_offset <= WINDOW_MINUTES)
    base = base[~(early_unit_discharge | early_hospital_discharge)].copy()

    vital = vital[(vital["observationoffset"] >= 0) & (vital["observationoffset"] <= WINDOW_MINUTES)].copy()
    vital["temperature"] = pd.to_numeric(vital["temperature"], errors="coerce")
    vital.loc[vital["temperature"] > 60, "temperature"] = (vital.loc[vital["temperature"] > 60, "temperature"] - 32) * 5 / 9
    vital.loc[~vital["temperature"].between(25, 45), "temperature"] = np.nan
    vital_cols = {
        "heartrate": "hr",
        "temperature": "temperature",
        "respiration": "respiration",
        "systemicsystolic": "bp_sys",
        "systemicdiastolic": "bp_dia",
        "systemicmean": "bp_mean",
    }
    for original, feature in vital_cols.items():
        vital[original] = pd.to_numeric(vital[original], errors="coerce")
        base = base.merge(aggregate_measurements(vital, "patientunitstayid", original, feature), on="patientunitstayid", how="left")

    vital_ap = vital_ap[(vital_ap["observationoffset"] >= 0) & (vital_ap["observationoffset"] <= WINDOW_MINUTES)].copy()
    ap_cols = {
        "noninvasivesystolic": "bp_sys",
        "noninvasivediastolic": "bp_dia",
        "noninvasivemean": "bp_mean",
    }
    for original, feature in ap_cols.items():
        original_vals = pd.to_numeric(vital_ap[original], errors="coerce")
        tmp = vital_ap.assign(_value=original_vals)
        agg = aggregate_measurements(tmp, "patientunitstayid", "_value", feature)
        for suffix in ["mean", "min", "max"]:
            col = f"{feature}_{suffix}"
            if col in base.columns:
                base[col] = base[col].fillna(base.set_index("patientunitstayid").index.map(agg.set_index("patientunitstayid")[col]).to_series(index=base.index))
            else:
                base = base.merge(agg[["patientunitstayid", col]], on="patientunitstayid", how="left")

    lab = lab[(lab["labresultoffset"] >= 0) & (lab["labresultoffset"] <= WINDOW_MINUTES)].copy()
    lab["labname_lower"] = lab["labname"].astype(str).str.lower()
    lab["labresult"] = pd.to_numeric(lab["labresult"], errors="coerce")
    for feature, names in LAB_NAMES_EICU.items():
        mask = lab["labname_lower"].isin(names)
        base = base.merge(aggregate_measurements(lab.loc[mask], "patientunitstayid", "labresult", feature), on="patientunitstayid", how="left")

    base["source_dataset"] = "eICU"
    base["environment_type"] = "multicenter_icu"
    base["case_id"] = base["patientunitstayid"]
    base["patient_group_id"] = "eICU_" + base["uniquepid"].astype(str)
    base["admission_group_id"] = "eICU_" + base["patienthealthsystemstayid"].astype(str)
    return base[["source_dataset", "environment_type", "case_id", "patient_group_id", "admission_group_id", "age", "gender", *[c for c in FEATURES if c not in ["age", "gender"]], "target_mortality"]]


def build_pipeline(model, numeric_features, categorical_features):
    return Pipeline([
        ("preprocessor", ColumnTransformer([
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
        ])),
        ("model", model),
    ])


def evaluate(model, x, y, scenario, model_name):
    score = model.predict_proba(x)[:, 1]
    pred = (score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "scenario": scenario,
        "model": model_name,
        "n_test": len(y),
        "events": int(np.sum(y == 1)),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auc_roc": roc_auc_score(y, score) if len(np.unique(y)) > 1 else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }, pd.DataFrame({"scenario": scenario, "model": model_name, "y_true": y.to_numpy(), "y_pred": pred, "y_score": score})


print("Building 24h MIMIC...")
mimic = build_mimic_24h()
print("Building 24h eICU...")
eicu = build_eicu_24h()
df = pd.concat([mimic, eicu], ignore_index=True, sort=False)
df = df.dropna(subset=["target_mortality"]).copy()
df["target_mortality"] = df["target_mortality"].astype(int)

ID_COLUMNS = ["case_id", "patient_group_id", "admission_group_id", "source_dataset", "environment_type"]
feature_cols = [c for c in df.columns if c not in ID_COLUMNS + ["target_mortality"]]
x = df[feature_cols]
y = df["target_mortality"]
cat = [c for c in feature_cols if df[c].dtype == "object"]
num = [c for c in feature_cols if c not in cat]

models = {
    "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
}

metrics = []
preds = []
df_eicu = df[df["source_dataset"] == "eICU"].copy()
df_mimic = df[df["source_dataset"] == "MIMIC"].copy()

for name, model in models.items():
    pipe = build_pipeline(model, num, cat)
    x_train, x_test, y_train, y_test = train_test_split(
        df_eicu[feature_cols], df_eicu["target_mortality"], test_size=0.25, random_state=42, stratify=df_eicu["target_mortality"]
    )
    pipe.fit(x_train, y_train)
    rec, pr = evaluate(pipe, x_test, y_test, "eICU_internal_validation_24h", name)
    metrics.append(rec); preds.append(pr)

    pipe = build_pipeline(model, num, cat)
    pipe.fit(df_eicu[feature_cols], df_eicu["target_mortality"])
    rec, pr = evaluate(pipe, df_mimic[feature_cols], df_mimic["target_mortality"], "train_eICU_test_MIMIC_24h", name)
    metrics.append(rec); preds.append(pr)

    pipe = build_pipeline(model, num, cat)
    pipe.fit(df_mimic[feature_cols], df_mimic["target_mortality"])
    rec, pr = evaluate(pipe, df_eicu[feature_cols], df_eicu["target_mortality"], "train_MIMIC_test_eICU_24h", name)
    metrics.append(rec); preds.append(pr)

    pipe = build_pipeline(model, num, cat)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(pipe, x, y, cv=cv, scoring={"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1", "auc_roc": "roc_auc"}, n_jobs=-1)
    metrics.append({
        "scenario": "integrated_5fold_cv_24h",
        "model": name,
        "n_test": len(y),
        "events": int(y.sum()),
        "accuracy": scores["test_accuracy"].mean(),
        "precision": scores["test_precision"].mean(),
        "recall": scores["test_recall"].mean(),
        "f1": scores["test_f1"].mean(),
        "auc_roc": scores["test_auc_roc"].mean(),
        "accuracy_std": scores["test_accuracy"].std(),
        "precision_std": scores["test_precision"].std(),
        "recall_std": scores["test_recall"].std(),
        "f1_std": scores["test_f1"].std(),
        "auc_roc_std": scores["test_auc_roc"].std(),
    })

metrics_df = pd.DataFrame(metrics)
pred_df = pd.concat(preds, ignore_index=True)
cohort = df.groupby("source_dataset")["target_mortality"].agg(n="count", events="sum", prevalence="mean")
missing = df.groupby("source_dataset").apply(lambda g: g.isna().mean()).T.reset_index().rename(columns={"index": "variable"})

df.to_csv(OUT / "dataset_clinico_landmark_24h.csv", index=False)
metrics_df.to_csv(OUT / "landmark_24h_metrics.csv", index=False)
pred_df.to_csv(OUT / "landmark_24h_predictions.csv", index=False)
cohort.to_csv(OUT / "landmark_24h_cohort_summary.csv")
missing.to_csv(OUT / "landmark_24h_missingness_by_source.csv", index=False)

print("\nCohort 24h:")
print(cohort.to_string())
print("\nMetrics 24h:")
print(metrics_df.round(3).to_string(index=False))
