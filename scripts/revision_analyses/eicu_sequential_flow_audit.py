from pathlib import Path
import hashlib
import json

import pandas as pd


BASE = Path(r"C:\Users\wilop\Documents\Datos-generales\Clinicos\eICU")
OUT = Path(r"C:\Users\wilop\Documents\Codex\2026-09-06\ha\work")
WINDOW_MINUTES = 24 * 60
INPUT_FILES = [
    "patient.csv.gz",
    "apachePatientResult.csv.gz",
    "vitalPeriodic.csv.gz",
    "vitalAperiodic.csv.gz",
    "lab.csv.gz",
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def mortality_to_binary(x):
    if pd.isna(x):
        return pd.NA
    x = str(x).strip().lower()
    if x in ["alive", "0", "false", "no"]:
        return 0
    if x in ["expired", "dead", "1", "true", "yes"]:
        return 1
    return pd.NA


def summarize(label, df, target_col="target_mortality"):
    out = {
        "stage": label,
        "rows": len(df),
        "unique_icu_stays": df["patientunitstayid"].nunique() if "patientunitstayid" in df else pd.NA,
    }
    if target_col in df:
        resolved = df[target_col].notna()
        out["resolved_outcomes"] = int(resolved.sum())
        out["deaths"] = int(pd.to_numeric(df.loc[resolved, target_col]).sum())
    else:
        out["resolved_outcomes"] = pd.NA
        out["deaths"] = pd.NA
    return out


def main():
    patient = pd.read_csv(BASE / "patient.csv.gz", compression="gzip", low_memory=False)
    apache_raw = pd.read_csv(BASE / "apachePatientResult.csv.gz", compression="gzip", low_memory=False)

    rows = [summarize("Local eICU patient table input", patient)]

    apache = apache_raw[["patientunitstayid", "apacheversion", "actualhospitalmortality"]].copy()
    apache["target_mortality"] = apache["actualhospitalmortality"].apply(mortality_to_binary)
    apache["apache_priority"] = apache["apacheversion"].astype(str).str.lower().map({"iva": 0, "iv": 1}).fillna(2)
    rows.append(summarize("Raw APACHE result rows before selection", apache))

    apache_selected = apache.sort_values(["patientunitstayid", "apache_priority"]).drop_duplicates("patientunitstayid")
    rows.append(summarize("After selecting one APACHE result per ICU stay", apache_selected))

    joined = patient.merge(apache_selected[["patientunitstayid", "target_mortality"]], on="patientunitstayid", how="left")
    rows.append(summarize("After joining selected APACHE result to patient table", joined))

    outcome_resolved = joined[joined["target_mortality"].notna()].copy()
    rows.append(summarize("After requiring resolvable hospital mortality outcome", outcome_resolved))

    unit_discharge_offset = pd.to_numeric(outcome_resolved["unitdischargeoffset"], errors="coerce")
    hospital_discharge_offset = pd.to_numeric(outcome_resolved["hospitaldischargeoffset"], errors="coerce")
    early_unit_discharge = unit_discharge_offset.notna() & (unit_discharge_offset <= WINDOW_MINUTES)
    early_hospital_discharge = hospital_discharge_offset.notna() & (hospital_discharge_offset <= WINDOW_MINUTES)
    eligible = outcome_resolved[~(early_unit_discharge | early_hospital_discharge)].copy()
    rows.append(summarize("After 24-hour ICU/hospital landmark eligibility", eligible))

    flow = pd.DataFrame(rows)
    flow["rows_removed_from_previous_stage"] = flow["rows"].shift(1) - flow["rows"]
    flow["unique_icu_stays_removed_from_previous_stage"] = (
        flow["unique_icu_stays"].shift(1) - flow["unique_icu_stays"]
    )
    flow["deaths_removed_from_previous_stage"] = flow["deaths"].shift(1) - flow["deaths"]
    flow.to_csv(OUT / "primary_24h_eicu_sequential_flow.csv", index=False)

    manifest = {
        "source_database": "eICU Collaborative Research Database, local authorized subset",
        "subset_note": (
            "No additional hospital-level or date-range selection rule was applied by the revision scripts. "
            "The analysis uses the eICU tables available in the authorized local eICU directory; because this "
            "is a partial local extract, hospital-level representativeness is not inferred."
        ),
        "landmark_minutes": WINDOW_MINUTES,
        "input_files": [],
    }
    for name in INPUT_FILES:
        path = BASE / name
        if path.exists():
            manifest["input_files"].append(
                {
                    "file": name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    with open(OUT / "primary_24h_eicu_input_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(flow.to_string(index=False))


if __name__ == "__main__":
    main()
