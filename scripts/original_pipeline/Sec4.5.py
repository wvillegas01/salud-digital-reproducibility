from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import gaussian_kde
from sklearn.metrics import confusion_matrix

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

PRED_FILE = BASE_PATH / "predicciones_multi_entorno.csv"

OUTPUT_TABLE_CSV = BASE_PATH / "tabla_4_4_confusion_summary.csv"
OUTPUT_TABLE_XLSX = BASE_PATH / "tabla_4_4_confusion_summary.xlsx"
OUTPUT_FIG = BASE_PATH / "figura_4_4_predicted_probabilities.png"

SCENARIOS = [
    "train_eICU_test_MIMIC",
    "train_MIMIC_test_eICU"
]

SCENARIO_LABELS = {
    "train_eICU_test_MIMIC": "Train eICU → Test MIMIC",
    "train_MIMIC_test_eICU": "Train MIMIC → Test eICU"
}

MODELS = [
    "LogisticRegression",
    "RandomForest",
    "GradientBoosting"
]

MODEL_LABELS = {
    "LogisticRegression": "Logistic",
    "RandomForest": "RF",
    "GradientBoosting": "GB"
}

# ============================================================
# CARGA
# ============================================================

df = pd.read_csv(PRED_FILE)

df = df[df["scenario"].isin(SCENARIOS)].copy()
df["y_true"] = df["y_true"].astype(int)
df["y_pred"] = df["y_pred"].astype(int)
df["y_score"] = pd.to_numeric(df["y_score"], errors="coerce")

# ============================================================
# TABLA 4.4 - MATRIZ DE CONFUSIÓN RESUMIDA
# ============================================================

records = []

for scenario in SCENARIOS:
    for model in MODELS:
        subset = df[
            (df["scenario"] == scenario) &
            (df["model"] == model)
        ].copy()

        if subset.empty:
            continue

        cm = confusion_matrix(
            subset["y_true"],
            subset["y_pred"],
            labels=[0, 1]
        )

        tn, fp, fn, tp = cm.ravel()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan

        records.append({
            "Scenario": SCENARIO_LABELS[scenario],
            "Model": MODEL_LABELS[model],
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "Sensitivity": round(sensitivity, 3),
            "Specificity": round(specificity, 3)
        })

conf_table = pd.DataFrame(records)

conf_table.to_csv(OUTPUT_TABLE_CSV, index=False)
conf_table.to_excel(OUTPUT_TABLE_XLSX, index=False)

print("\nTabla 4.4 generada:")
print(conf_table)

# ============================================================
# FIGURA 4 - DISTRIBUCIÓN DE PROBABILIDADES PREDICHAS
# ============================================================

def plot_probability_kde(ax, subset, title):
    class_0 = subset[subset["y_true"] == 0]["y_score"].dropna()
    class_1 = subset[subset["y_true"] == 1]["y_score"].dropna()

    x_grid = np.linspace(0, 1, 300)

    if len(class_0) > 5 and class_0.nunique() > 1:
        kde_0 = gaussian_kde(class_0)
        ax.plot(
            x_grid,
            kde_0(x_grid),
            label="True class 0",
            linewidth=1.0
        )

    if len(class_1) > 5 and class_1.nunique() > 1:
        kde_1 = gaussian_kde(class_1)
        ax.plot(
            x_grid,
            kde_1(x_grid),
            label="True class 1",
            linewidth=1.0,
            linestyle="--"
        )

    ax.axvline(
        0.5,
        linestyle=":",
        linewidth=0.8
    )

    ax.set_title(title, loc="left", fontsize=8.5, pad=2)
    ax.set_xlabel("Predicted probability of mortality")
    ax.set_ylabel("Density")

    ax.set_xlim(0, 1)

    ax.grid(axis="y", linestyle="--", linewidth=0.35, alpha=0.35)

    ax.tick_params(axis="both", labelsize=7, width=0.6, length=2.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)


fig, axes = plt.subplots(2, 3, figsize=(8.2, 5.2), dpi=400)

panel_labels = [
    "(a)", "(b)", "(c)",
    "(d)", "(e)", "(f)"
]

panel_idx = 0

for row_idx, scenario in enumerate(SCENARIOS):
    for col_idx, model in enumerate(MODELS):
        ax = axes[row_idx, col_idx]

        subset = df[
            (df["scenario"] == scenario) &
            (df["model"] == model)
        ].copy()

        title = f"{panel_labels[panel_idx]} {MODEL_LABELS[model]}"

        plot_probability_kde(ax, subset, title)

        if col_idx == 0:
            ax.text(
                -0.28,
                0.5,
                SCENARIO_LABELS[scenario],
                transform=ax.transAxes,
                rotation=90,
                va="center",
                ha="center",
                fontsize=8
            )

        panel_idx += 1

handles, labels = axes[0, 0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=3,
    frameon=False,
    handlelength=2.0,
    columnspacing=1.4
)

plt.subplots_adjust(
    left=0.11,
    right=0.99,
    top=0.96,
    bottom=0.14,
    wspace=0.32,
    hspace=0.42
)

plt.savefig(OUTPUT_FIG, dpi=400, bbox_inches="tight")
plt.close()

print("\nFigura 4 generada:")
print(OUTPUT_FIG)