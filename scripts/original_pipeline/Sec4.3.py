from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_PATH = Path(r"C:\Users\wilop\Documents\Datos-generales\Clinicos")

INPUT_RESULTS = BASE_PATH / "resultados_modelado_multi_entorno.xlsx"

OUTPUT_TABLE_CSV = BASE_PATH / "tabla_4_2_desempeno_intra_entorno.csv"
OUTPUT_TABLE_XLSX = BASE_PATH / "tabla_4_2_desempeno_intra_entorno.xlsx"

SCENARIOS = [
    "eICU_internal_validation",
    "integrated_5fold_cv"
]

METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc_roc"
]

# ============================================================
# CARGA DE RESULTADOS
# ============================================================

df = pd.read_excel(INPUT_RESULTS, sheet_name="01_metrics")

df = df[df["scenario"].isin(SCENARIOS)].copy()

# ============================================================
# FORMATEO DE MÉTRICAS
# ============================================================

def format_metric(row, metric):
    value = row.get(metric, np.nan)
    std = row.get(f"{metric}_std", np.nan)

    if pd.isna(value):
        return ""

    if row["scenario"] == "integrated_5fold_cv" and not pd.isna(std):
        return f"{value:.3f} ± {std:.3f}"

    return f"{value:.3f}"


records = []

for _, row in df.iterrows():
    records.append({
        "Scenario": row["scenario"],
        "Model": row["model"],
        "Accuracy": format_metric(row, "accuracy"),
        "Precision": format_metric(row, "precision"),
        "Recall": format_metric(row, "recall"),
        "F1-score": format_metric(row, "f1"),
        "AUC-ROC": format_metric(row, "auc_roc")
    })

table = pd.DataFrame(records)

# Ordenar modelos y escenarios
scenario_order = {
    "eICU_internal_validation": 1,
    "integrated_5fold_cv": 2
}

model_order = {
    "LogisticRegression": 1,
    "RandomForest": 2,
    "GradientBoosting": 3
}

table["scenario_order"] = table["Scenario"].map(scenario_order)
table["model_order"] = table["Model"].map(model_order)

table = table.sort_values(["scenario_order", "model_order"])
table = table.drop(columns=["scenario_order", "model_order"])

# Renombrar escenarios para presentación
table["Scenario"] = table["Scenario"].replace({
    "eICU_internal_validation": "eICU internal validation",
    "integrated_5fold_cv": "Integrated 5-fold CV"
})

# ============================================================
# GUARDAR RESULTADOS
# ============================================================

table.to_csv(OUTPUT_TABLE_CSV, index=False)
table.to_excel(OUTPUT_TABLE_XLSX, index=False)

print("\nTabla 4.2 generada:")
print(table)

print("\nArchivos:")
print(OUTPUT_TABLE_CSV)
print(OUTPUT_TABLE_XLSX)