from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(r"C:\Users\wilop\Documents\Codex\2026-09-06\ha\work")
DATA = ROOT / "dataset_clinico_landmark_24h.csv"
OUT = Path(r"C:\Users\wilop\Dropbox\MPDI\2026\Salud-digital\Frontiers_LaTeX_Templates\Fig3.jpg")

PANELS = [
    ("hr_mean", "(a) Heart rate", "beats/min", (0, None)),
    ("glucose_mean", "(b) Glucose", "mg/dL", (0, None)),
    ("creatinine_mean", "(c) Creatinine", "mg/dL", (0, None)),
    ("bp_sys_mean", "(d) Systolic blood pressure", "mmHg", (0, None)),
]


def main() -> None:
    df = pd.read_csv(DATA)
    df = df[df["source_dataset"].isin(["MIMIC", "eICU"])].copy()

    sns.set_theme(style="whitegrid", context="notebook")
    palette = {"MIMIC": "#1f77b4", "eICU": "#d62728"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300)
    for ax, (column, title, unit, clip) in zip(axes.flat, PANELS):
        for source in ["MIMIC", "eICU"]:
            values = pd.to_numeric(
                df.loc[df["source_dataset"] == source, column], errors="coerce"
            ).dropna()
            sns.kdeplot(
                values,
                ax=ax,
                label=source,
                color=palette[source],
                linewidth=2.2,
                fill=False,
                common_norm=False,
                cut=0,
                clip=clip,
            )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(unit)
        ax.set_ylabel("Density")
        ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
