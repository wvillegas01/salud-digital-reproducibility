from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib as mpl

# ============================================================
# RESET TOTAL DE ESTILO
# ============================================================

mpl.rcParams.update(mpl.rcParamsDefault)

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0
})

BASE_PATH = Path(r"C:\Users\wilop\Documents\Datos-generales\Clinicos")

INPUT_FILE = BASE_PATH / "dataset_clinico_final_mimic_eicu.csv"
OUTPUT_FIG = BASE_PATH / "figura_4_2_domain_shift_kde_final.png"

VARIABLES = [
    ("hr_mean", "Heart rate", "(a)"),
    ("glucose_mean", "Glucose", "(b)"),
    ("creatinine_mean", "Creatinine", "(c)"),
    ("bp_mean_ap_mean", "Mean arterial pressure", "(d)")
]


def kde_curve(values, x_grid):
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) <= 5:
        return None
    return gaussian_kde(values)(x_grid)


def plot_panel(ax, df_mimic, df_eicu, var, xlabel, panel):
    mimic = pd.to_numeric(df_mimic[var], errors="coerce").dropna()
    eicu = pd.to_numeric(df_eicu[var], errors="coerce").dropna()

    combined = pd.concat([mimic, eicu])
    q1, q99 = combined.quantile([0.01, 0.99])

    mimic = mimic[(mimic >= q1) & (mimic <= q99)]
    eicu = eicu[(eicu >= q1) & (eicu <= q99)]

    x_grid = np.linspace(q1, q99, 300)

    y_mimic = kde_curve(mimic, x_grid)
    y_eicu = kde_curve(eicu, x_grid)

    if y_mimic is not None:
        ax.plot(x_grid, y_mimic, label="MIMIC", linewidth=1.0)

    if y_eicu is not None:
        ax.plot(x_grid, y_eicu, label="eICU", linewidth=1.0, linestyle="--")

    ax.set_xlabel(xlabel, labelpad=2)
    ax.set_ylabel("Density", labelpad=2)

    ax.set_title(
    f"{panel} {xlabel}",
    loc="left",
    fontsize=8.5,
    pad=2
)

    ax.tick_params(axis="both", labelsize=7, width=0.6, length=2.5)

    ax.grid(axis="y", linestyle="--", linewidth=0.35, alpha=0.35)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)


df = pd.read_csv(INPUT_FILE)

df_mimic = df[df["source_dataset"] == "MIMIC"]
df_eicu = df[df["source_dataset"] == "eICU"]

fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.4), dpi=400)
axes = axes.flatten()

for ax, (var, label, panel) in zip(axes, VARIABLES):
    plot_panel(ax, df_mimic, df_eicu, var, label, panel)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=2,
    frameon=False,
    handlelength=2.0,
    columnspacing=1.2
)

plt.subplots_adjust(
    left=0.09,
    right=0.98,
    top=0.97,
    bottom=0.15,
    wspace=0.32,
    hspace=0.42
)

plt.savefig(OUTPUT_FIG, dpi=400, bbox_inches="tight")
plt.close()

print("Figura generada:")
print(OUTPUT_FIG)