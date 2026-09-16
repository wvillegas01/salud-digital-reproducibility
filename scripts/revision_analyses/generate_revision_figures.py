import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PACKAGE_ROOT / "data" / "restricted_inputs" / "dataset_clinico_landmark_24h.csv"
DEFAULT_TEX_DIR = PACKAGE_ROOT / "manuscript"
DEFAULT_BINS_PATH = PACKAGE_ROOT / "data" / "aggregate_outputs" / "primary_24h_calibration_plot_bins.csv"
TARGET = "target_mortality"
ID_COLUMNS = ["case_id", "patient_group_id", "admission_group_id", "source_dataset", "environment_type"]


def build_pipeline(model, numeric_features, categorical_features):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def ece_bins(y_true, y_score, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    for i, (left, right) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (y_score >= left) & (y_score < right)
        if i == n_bins - 1:
            mask = (y_score >= left) & (y_score <= right)
        if not mask.any():
            continue
        mean_p = float(y_score[mask].mean())
        obs = float(y_true[mask].mean())
        weight = float(mask.mean())
        ece += weight * abs(mean_p - obs)
        rows.append({"mean_predicted": mean_p, "observed_event_rate": obs, "n": int(mask.sum())})
    return ece, rows


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
    box(0.5, 5.3, 4.8, 0.9, "(a) In-domain evaluation\nInspect probability distribution and fixed operating point", fc="#f5fbf7")
    box(6.7, 5.3, 4.8, 0.9, "(b) Cross-domain evaluation\nInspect probability shift and operating-point stability", fc="#fff7ef")
    box(2.0, 3.8, 3.8, 0.8, "Apply unchanged threshold\n$\\tau = 0.5$", fc="#fbfbfb")
    box(6.2, 3.8, 3.8, 0.8, "Apply unchanged threshold\n$\\tau = 0.5$", fc="#fbfbfb")
    box(0.3, 2.0, 2.6, 0.9, "Stable activation", fc="#eaf7ef")
    box(3.2, 2.0, 2.6, 0.9, "Conservative decisions", fc="#fff4e6")
    box(6.1, 2.0, 2.6, 0.9, "Liberal decisions", fc="#fff4e6")
    box(9.0, 2.0, 2.6, 0.9, "Degenerate decisions", fc="#fdecec")
    box(2.2, 0.7, 7.6, 0.8, "Decision behavior is empirical and model-specific; internal performance does not guarantee transfer stability.", fc="#f7f7f7", size=9)

    arrow(6.0, 6.8, 2.9, 6.2)
    arrow(6.0, 6.8, 9.1, 6.2)
    arrow(2.9, 5.3, 3.9, 4.6)
    arrow(9.1, 5.3, 8.1, 4.6)
    for x in [1.6, 4.5]:
        arrow(3.9, 3.8, x, 2.9)
    for x in [7.4, 10.3]:
        arrow(8.1, 3.8, x, 2.9)
    arrow(5.8, 2.0, 6.0, 1.5)

    fig.tight_layout()
    tex_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(tex_dir / "Fig2.jpg", bbox_inches="tight")
    plt.close(fig)


def generate_fig5(data_path, tex_dir, bins_path):
    if not data_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {data_path}. "
            "Provide the de-identified 24-hour landmark dataset with --input."
        )
    df = pd.read_csv(data_path).dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)
    candidate_features = [c for c in df.columns if c not in ID_COLUMNS + [TARGET]]
    missing_by_source = df.groupby("source_dataset")[candidate_features].apply(lambda g: g.isna().mean())
    shared_features = [
        col for col in candidate_features if all(missing_by_source.loc[src, col] <= 0.60 for src in missing_by_source.index)
    ]
    categorical_features = [col for col in shared_features if df[col].dtype == "object"]
    numeric_features = [col for col in shared_features if col not in categorical_features]
    models = {
        "LR": LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
        "RF": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
        "GB": GradientBoostingClassifier(random_state=42),
    }
    df_eicu = df[df["source_dataset"] == "eICU"].copy()
    df_mimic = df[df["source_dataset"] == "MIMIC"].copy()

    scenarios = [
        ("(a) eICU internal OOF", "internal", df_eicu, df_eicu),
        ("(b) eICU -> MIMIC", "transfer", df_eicu, df_mimic),
        ("(c) MIMIC internal OOF", "internal", df_mimic, df_mimic),
        ("(d) MIMIC -> eICU", "transfer", df_mimic, df_eicu),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300, sharex=True, sharey=True)
    colors = {"LR": "#1f77b4", "RF": "#d62728", "GB": "#2ca02c"}
    out_rows = []

    for ax, (title, kind, train_df, test_df) in zip(axes.flat, scenarios):
        for model_name, model in models.items():
            pipe = build_pipeline(clone(model), numeric_features, categorical_features)
            if kind == "internal":
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scores = cross_val_predict(
                    pipe,
                    train_df[shared_features],
                    train_df[TARGET],
                    cv=cv,
                    method="predict_proba",
                )[:, 1]
                y_true = train_df[TARGET].to_numpy()
            else:
                pipe.fit(train_df[shared_features], train_df[TARGET])
                scores = pipe.predict_proba(test_df[shared_features])[:, 1]
                y_true = test_df[TARGET].to_numpy()

            ece, rows = ece_bins(y_true, scores, n_bins=10)
            for row in rows:
                out_rows.append({"scenario": title, "model": model_name, "ece_10bins": ece, **row})
            ax.plot(
                [r["mean_predicted"] for r in rows],
                [r["observed_event_rate"] for r in rows],
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
    bins_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(tex_dir / "Fig5.jpg", bbox_inches="tight")
    pd.DataFrame(out_rows).to_csv(bins_path, index=False)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate revision Figure 2 and Figure 5.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA_PATH, help="Path to the de-identified 24-hour landmark dataset.")
    parser.add_argument("--tex-dir", type=Path, default=DEFAULT_TEX_DIR, help="Directory where Fig2.jpg and Fig5.jpg are written.")
    parser.add_argument("--output-bins", type=Path, default=DEFAULT_BINS_PATH, help="CSV path for calibration-bin summaries.")
    args = parser.parse_args()

    generate_fig2(args.tex_dir)
    generate_fig5(args.input, args.tex_dir, args.output_bins)
