from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ============================================================
# RESET DE ESTILO
# ============================================================

mpl.rcParams.update(mpl.rcParamsDefault)

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0
})

# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_PATH = Path(r"C:\Users\wilop\Documents\Datos-generales\Clinicos")

INPUT_RESULTS = BASE_PATH / "resultados_modelado_multi_entorno.xlsx"

OUTPUT_TABLE_CSV = BASE_PATH / "tabla_4_3_transferencia_multi_entorno.csv"
OUTPUT_TABLE_XLSX = BASE_PATH / "tabla_4_3_transferencia_multi_entorno.xlsx"
OUTPUT_FIG = BASE_PATH / "figura_4_3_transferencia_multi_entorno.png"

SCENARIOS = [
    "train_eICU_test_MIMIC",
    "train_MIMIC_test_eICU"
]

SCENARIO_LABELS = {
    "train_eICU_test_MIMIC": "Train eICU → Test MIMIC",
    "train_MIMIC_test_eICU": "Train MIMIC → Test eICU"
}

MODEL_LABELS = {
    "LogisticRegression": "Logistic",
    "RandomForest": "RF",
    "GradientBoosting": "GB"
}

MODEL_ORDER = [
    "LogisticRegression",
    "RandomForest",
    "GradientBoosting"
]

# ============================================================
# CARGA DE RESULTADOS
# ============================================================

df = pd.read_excel(INPUT_RESULTS, sheet_name="01_metrics")

df = df[df["scenario"].isin(SCENARIOS)].copy()

df["Scenario"] = df["scenario"].map(SCENARIO_LABELS)
df["Model"] = df["model"].map(MODEL_LABELS)

df["model_order"] = df["model"].map({
    "LogisticRegression": 1,
    "RandomForest": 2,
    "GradientBoosting": 3
})

df["scenario_order"] = df["scenario"].map({
    "train_eICU_test_MIMIC": 1,
    "train_MIMIC_test_eICU": 2
})

df = df.sort_values(["scenario_order", "model_order"])

# ============================================================
# TABLA 4.3
# ============================================================

table = df[[
    "Scenario",
    "Model",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc_roc"
]].copy()

table = table.rename(columns={
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
    "auc_roc": "AUC-ROC"
})

for col in ["Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC"]:
    table[col] = table[col].map(lambda x: f"{x:.3f}")

table.to_csv(OUTPUT_TABLE_CSV, index=False)
table.to_excel(OUTPUT_TABLE_XLSX, index=False)

print("\nTabla 4.3 generada:")
print(table)

# ============================================================
# FIGURA 3 - TRANSFERENCIA MULTI-ENTORNO
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.1), dpi=400)

metrics_to_plot = ["f1", "auc_roc"]
metric_labels = {
    "f1": "F1-score",
    "auc_roc": "AUC-ROC"
}

bar_width = 0.34
x = np.arange(len(MODEL_ORDER))

for ax, scenario, panel_label in zip(
    axes,
    SCENARIOS,
    ["(a) Train eICU → Test MIMIC", "(b) Train MIMIC → Test eICU"]
):
    subset = df[df["scenario"] == scenario].set_index("model").loc[MODEL_ORDER]

    f1_values = subset["f1"].values
    auc_values = subset["auc_roc"].values

    ax.bar(
        x - bar_width / 2,
        f1_values,
        width=bar_width,
        label="F1-score",
        linewidth=0.5,
        edgecolor="black"
    )

    ax.bar(
        x + bar_width / 2,
        auc_values,
        width=bar_width,
        label="AUC-ROC",
        linewidth=0.5,
        edgecolor="black"
    )

    ax.set_title(panel_label, loc="left", fontsize=8.5, pad=2)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Metric value")
    ax.grid(axis="y", linestyle="--", linewidth=0.35, alpha=0.35)

    ax.tick_params(axis="both", labelsize=7, width=0.6, length=2.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)

    # Etiquetas numéricas compactas
    for i, value in enumerate(f1_values):
        ax.text(
            x[i] - bar_width / 2,
            value + 0.025,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.5
        )

    for i, value in enumerate(auc_values):
        ax.text(
            x[i] + bar_width / 2,
            value + 0.025,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.5
        )

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=2,
    frameon=False,
    handlelength=2.0,
    columnspacing=1.4
)

plt.subplots_adjust(
    left=0.08,
    right=0.99,
    top=0.91,
    bottom=0.24,
    wspace=0.30
)

plt.savefig(OUTPUT_FIG, dpi=400, bbox_inches="tight")
plt.close()

print("\nFigura 3 generada:")
print(OUTPUT_FIG)