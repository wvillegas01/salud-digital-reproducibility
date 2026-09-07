from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_PATH = Path(r"C:\Users\wilop\Documents\Datos-generales\Clinicos")

RAW_FILE = BASE_PATH / "dataset_clinico_integrado_mimic_eicu_raw.csv"
CLEAN_FILE = BASE_PATH / "dataset_clinico_integrado_mimic_eicu_clean.csv"

OUTPUT_TABLE = BASE_PATH / "tabla_4_1_resumen_dataset.csv"
OUTPUT_FIG = BASE_PATH / "figura_4_1_distribucion_dataset.png"

TARGET = "target_mortality"

# ============================================================
# CARGA DE DATOS
# ============================================================

raw_df = pd.read_csv(RAW_FILE)
clean_df = pd.read_csv(CLEAN_FILE)

# ============================================================
# TABLA 4.1 - RESUMEN DEL DATASET
# ============================================================

records = []

for dataset in raw_df["source_dataset"].unique():
    subset_raw = raw_df[raw_df["source_dataset"] == dataset]
    subset_clean = clean_df[clean_df["source_dataset"] == dataset]

    records.append({
        "Dataset": dataset,
        "Raw samples": subset_raw.shape[0],
        "Valid samples": subset_clean.shape[0],
        "Non-mortality cases (0)": int((subset_clean[TARGET] == 0).sum()),
        "Mortality cases (1)": int((subset_clean[TARGET] == 1).sum()),
        "Missing target": int(subset_raw[TARGET].isna().sum()),
        "Retained variables": clean_df.shape[1]
    })

records.append({
    "Dataset": "Integrated dataset",
    "Raw samples": raw_df.shape[0],
    "Valid samples": clean_df.shape[0],
    "Non-mortality cases (0)": int((clean_df[TARGET] == 0).sum()),
    "Mortality cases (1)": int((clean_df[TARGET] == 1).sum()),
    "Missing target": int(raw_df[TARGET].isna().sum()),
    "Retained variables": clean_df.shape[1]
})

summary_table = pd.DataFrame(records)
summary_table.to_csv(OUTPUT_TABLE, index=False)

print("\nTabla 4.1 generada:")
print(summary_table)

# ============================================================
# FIGURA 4.1 - DISTRIBUCIÓN DEL DATASET
# ============================================================

dataset_counts = clean_df["source_dataset"].value_counts().sort_index()

target_counts = (
    clean_df
    .groupby(["source_dataset", TARGET])
    .size()
    .unstack(fill_value=0)
    .sort_index()
)

target_percent = target_counts.div(target_counts.sum(axis=1), axis=0) * 100

fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

# -------------------------
# Panel A: muestras válidas por dataset
# -------------------------

axes[0].bar(dataset_counts.index, dataset_counts.values)
axes[0].set_title("(a) Valid samples by dataset", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Dataset")
axes[0].set_ylabel("Number of valid samples")

for i, value in enumerate(dataset_counts.values):
    axes[0].text(
        i,
        value + max(dataset_counts.values) * 0.02,
        f"{value:,}",
        ha="center",
        va="bottom",
        fontsize=10
    )

# -------------------------
# Panel B: distribución del target por dataset
# -------------------------

x = np.arange(len(target_percent.index))
width = 0.35

axes[1].bar(
    x - width / 2,
    target_percent[0],
    width,
    label="Non-mortality (0)"
)

axes[1].bar(
    x + width / 2,
    target_percent[1],
    width,
    label="Mortality (1)"
)

axes[1].set_title("(b) Mortality outcome distribution", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Dataset")
axes[1].set_ylabel("Percentage (%)")
axes[1].set_xticks(x)
axes[1].set_xticklabels(target_percent.index)
axes[1].legend(frameon=False)

for i in range(len(target_percent.index)):
    axes[1].text(
        x[i] - width / 2,
        target_percent.iloc[i, 0] + 1,
        f"{target_percent.iloc[i, 0]:.1f}%",
        ha="center",
        fontsize=9
    )
    axes[1].text(
        x[i] + width / 2,
        target_percent.iloc[i, 1] + 1,
        f"{target_percent.iloc[i, 1]:.1f}%",
        ha="center",
        fontsize=9
    )

# -------------------------
# Estética general
# -------------------------

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

plt.tight_layout()
plt.savefig(OUTPUT_FIG, bbox_inches="tight")
plt.show()

print("\nFigura 4.1 generada:")
print(OUTPUT_FIG)