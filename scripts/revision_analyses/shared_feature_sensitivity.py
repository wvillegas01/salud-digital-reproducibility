from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path(r"C:\Users\wilop\Documents\Datos-generales\Clinicos\dataset_clinico_final_mimic_eicu.csv")
OUT_DIR = Path(r"C:\Users\wilop\Documents\Codex\2026-09-06\ha\work")
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


def evaluate(model, x_test, y_test, scenario, model_name):
    y_pred = model.predict(x_test)
    y_score = model.predict_proba(x_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    return {
        "scenario": scenario,
        "model": model_name,
        "n_test": len(y_test),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "auc_roc": roc_auc_score(y_test, y_score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "positive_predictions": int(tp + fp),
    }


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

    x = df[shared_features].copy()
    categorical_features = [col for col in x.columns if x[col].dtype == "object"]
    numeric_features = [col for col in x.columns if col not in categorical_features]
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }

    df_eicu = df[df["source_dataset"] == "eICU"].copy()
    df_mimic = df[df["source_dataset"] == "MIMIC"].copy()
    scenarios = {
        "train_eICU_test_MIMIC_shared_features": (df_eicu, df_mimic),
        "train_MIMIC_test_eICU_shared_features": (df_mimic, df_eicu),
    }
    rows = []
    for scenario, (train_df, test_df) in scenarios.items():
        for model_name, model in models.items():
            pipe = build_pipeline(model, numeric_features, categorical_features)
            pipe.fit(train_df[shared_features], train_df[TARGET])
            rows.append(evaluate(pipe, test_df[shared_features], test_df[TARGET], scenario, model_name))

    pd.DataFrame(rows).to_csv(OUT_DIR / "shared_feature_transfer_sensitivity.csv", index=False)
    pd.DataFrame({"shared_features": shared_features}).to_csv(OUT_DIR / "shared_feature_list.csv", index=False)
    pd.DataFrame(missing_by_source.T).to_csv(OUT_DIR / "shared_feature_missingness_audit.csv")


if __name__ == "__main__":
    main()
