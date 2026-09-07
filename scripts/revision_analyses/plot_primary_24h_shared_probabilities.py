from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


OUT = Path(r"C:\Users\wilop\Documents\Codex\2026-09-06\ha\work")
DATA_PATH = OUT / "dataset_clinico_landmark_24h.csv"
FIG_PATH = Path(r"C:\Users\wilop\Dropbox\MPDI\2026\Salud-digital\Frontiers_LaTeX_Templates\Fig4.jpg")
TARGET = "target_mortality"
ID_COLUMNS = ["case_id", "source_dataset", "environment_type"]


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


def main():
    df = pd.read_csv(DATA_PATH).dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)
    candidate_features = [c for c in df.columns if c not in ID_COLUMNS + [TARGET]]
    missing_by_source = df.groupby("source_dataset")[candidate_features].apply(lambda g: g.isna().mean())
    shared_features = [
        col
        for col in candidate_features
        if all(missing_by_source.loc[source, col] < 1.0 for source in missing_by_source.index)
    ]
    x = df[shared_features]
    categorical_features = [col for col in x.columns if x[col].dtype == "object"]
    numeric_features = [col for col in x.columns if col not in categorical_features]
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    df_eicu = df[df["source_dataset"] == "eICU"].copy()
    df_mimic = df[df["source_dataset"] == "MIMIC"].copy()
    scenarios = [
        ("Train eICU -> Test MIMIC", df_eicu, df_mimic),
        ("Train MIMIC -> Test eICU", df_mimic, df_eicu),
    ]

    rows = []
    for scenario_name, train_df, test_df in scenarios:
        for model_name, model in models.items():
            pipe = build_pipeline(model, numeric_features, categorical_features)
            pipe.fit(train_df[shared_features], train_df[TARGET])
            scores = pipe.predict_proba(test_df[shared_features])[:, 1]
            for y, score in zip(test_df[TARGET], scores):
                rows.append(
                    {
                        "scenario": scenario_name,
                        "model": model_name,
                        "Mortality": "Death" if y == 1 else "Survival",
                        "Predicted probability": score,
                    }
                )
    pred = pd.DataFrame(rows)
    pred.to_csv(OUT / "primary_24h_shared_probability_plot_data.csv", index=False)

    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=False)
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    for row_idx, (scenario_name, _, _) in enumerate(scenarios):
        for col_idx, model_name in enumerate(models):
            ax = axes[row_idx, col_idx]
            panel_label = panel_labels[row_idx * len(models) + col_idx]
            subset = pred[(pred["scenario"] == scenario_name) & (pred["model"] == model_name)]
            for label, color in [("Survival", "#2c7fb8"), ("Death", "#d95f0e")]:
                values = subset.loc[subset["Mortality"] == label, "Predicted probability"]
                sns.kdeplot(values, ax=ax, label=label, color=color, fill=False, linewidth=1.8, clip=(0, 1))
            ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
            ax.set_xlim(0, 1)
            ax.set_title(f"{panel_label} {scenario_name}\n{model_name}", fontsize=10)
            ax.set_xlabel("Predicted mortality probability")
            ax.set_ylabel("Density")
            if row_idx == 0 and col_idx == 2:
                ax.legend(frameon=False, loc="upper right")
            else:
                ax.legend_.remove() if ax.legend_ else None
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
