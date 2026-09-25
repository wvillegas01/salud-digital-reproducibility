import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PACKAGE_ROOT / "data" / "restricted_inputs" / "dataset_clinico_landmark_24h.csv"
DEFAULT_TEX_DIR = PACKAGE_ROOT / "manuscript"
DEFAULT_BINS_PATH = PACKAGE_ROOT / "data" / "aggregate_outputs" / "primary_24h_calibration_plot_bins.csv"


def generate_fig2(tex_dir):
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#f8f9fb", ec="#34495e", size=10, weight="normal"):
        rect = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, weight=weight, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.5, color="#34495e"))

    box(4.0, 6.8, 4.0, 0.8, "Train probabilistic model", fc="#eef3f8", weight="bold")
    box(0.7, 5.35, 4.6, 0.9, "(a) In-domain evaluation\nInspect probabilities and fixed operating point", fc="#f5fbf7")
    box(6.7, 5.35, 4.6, 0.9, "(b) Cross-domain evaluation\nInspect probabilities and fixed operating point", fc="#fff7ef")
    box(4.1, 4.15, 3.8, 0.75, "Apply unchanged threshold\n$\\tau = 0.5$", fc="#fbfbfb")
    box(0.35, 2.45, 2.65, 0.85, "Stable activation", fc="#eaf7ef")
    box(3.15, 2.45, 2.65, 0.85, "Conservative decisions", fc="#fff4e6")
    box(5.95, 2.45, 2.65, 0.85, "Liberal decisions", fc="#fff4e6")
    box(8.75, 2.45, 2.65, 0.85, "Degenerate decisions", fc="#fdecec")
    box(2.1, 0.85, 7.8, 0.85, "All four behaviors are empirical model-specific outcomes that can occur in internal or transfer evaluation.", fc="#f7f7f7", size=9)

    arrow(6.0, 6.8, 3.0, 6.25)
    arrow(6.0, 6.8, 9.0, 6.25)
    arrow(3.0, 5.35, 5.0, 4.9)
    arrow(9.0, 5.35, 7.0, 4.9)
    for x in [1.65, 4.45, 7.25, 10.05]:
        arrow(6.0, 4.15, x, 3.3)
    arrow(6.0, 2.45, 6.0, 1.7)

    fig.tight_layout()
    tex_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(tex_dir / "Fig2.jpg", bbox_inches="tight")
    plt.close(fig)


def generate_fig5(_data_path, tex_dir, bins_path):
    if not bins_path.exists():
        raise FileNotFoundError(
            f"Calibration bins not found: {bins_path}. "
            "Run primary_24h_shared_diagnostics.py first so Figure 5 and Table 7 use the same saved predictions."
        )
    bins_df = pd.read_csv(bins_path)
    required = {
        "scenario",
        "model",
        "ece_10bins",
        "mean_predicted",
        "observed_event_rate",
        "n",
    }
    missing = required.difference(bins_df.columns)
    if missing:
        raise ValueError(f"Calibration-bin file is missing required columns: {sorted(missing)}")

    scenarios = [
        "(a) eICU internal OOF",
        "(b) eICU -> MIMIC",
        "(c) MIMIC internal OOF",
        "(d) MIMIC -> eICU",
    ]
    models = ["LR", "RF", "GB"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300, sharex=True, sharey=True)
    colors = {"LR": "#1f77b4", "RF": "#d62728", "GB": "#2ca02c"}

    for ax, title in zip(axes.flat, scenarios):
        for model_name in models:
            rows = bins_df[(bins_df["scenario"] == title) & (bins_df["model"] == model_name)].copy()
            if rows.empty:
                continue
            ece = float(rows["ece_10bins"].iloc[0])
            rows = rows.sort_values("mean_predicted")
            ax.plot(
                rows["mean_predicted"],
                rows["observed_event_rate"],
                marker="o",
                linewidth=1.6,
                markersize=4,
                label=f"{model_name} (ECE={ece:.3f})",
                color=colors[model_name],
            )
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=7, loc="upper left")

    fig.supxlabel("Mean predicted probability")
    fig.supylabel("Observed event rate")
    fig.tight_layout()
    tex_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(tex_dir / "Fig5.jpg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate revision Figure 2 and Figure 5.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH, help="Path to the de-identified 24-hour landmark dataset.")
    parser.add_argument("--tex-dir", type=Path, default=DEFAULT_TEX_DIR, help="Directory where Fig2.jpg and Fig5.jpg are written.")
    parser.add_argument("--output-bins", type=Path, default=DEFAULT_BINS_PATH, help="CSV path for calibration-bin summaries.")
    args = parser.parse_args()

    generate_fig2(args.tex_dir)
    generate_fig5(args.input, args.tex_dir, args.output_bins)
