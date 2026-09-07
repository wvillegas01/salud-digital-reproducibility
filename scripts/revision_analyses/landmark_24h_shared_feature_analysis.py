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
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


OUT = Path(r"C:\Users\wilop\Documents\Codex\2026-09-06\ha\work")
DATA_PATH = OUT / "dataset_clinico_landmark_24h.csv"
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
    precision = np.nan if (tp + fp) == 0 else precision_score(y_test, y_pred)
    return {
        "scenario": scenario,
        "model": model_name,
        "n_test": len(y_test),
        "events": int(y_test.sum()),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision,
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

    x = df[shared_features]
    categorical_features = [col for col in x.columns if x[col].dtype == "object"]
    numeric_features = [col for col in x.columns if col not in categorical_features]
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }

    df_eicu = df[df["source_dataset"] == "eICU"].copy()
    df_mimic = df[df["source_dataset"] == "MIMIC"].copy()
    rows = []

    for model_name, model in models.items():
        x_train, x_test, y_train, y_test = train_test_split(
            df_eicu[shared_features],
            df_eicu[TARGET],
            test_size=0.25,
            random_state=42,
            stratify=df_eicu[TARGET],
        )
        pipe = build_pipeline(model, numeric_features, categorical_features)
        pipe.fit(x_train, y_train)
        rows.append(evaluate(pipe, x_test, y_test, "eICU_internal_validation_24h_shared_features", model_name))

    scenarios = {
        "train_eICU_test_MIMIC_24h_shared_features": (df_eicu, df_mimic),
        "train_MIMIC_test_eICU_24h_shared_features": (df_mimic, df_eicu),
    }
    for scenario, (train_df, test_df) in scenarios.items():
        for model_name, model in models.items():
            pipe = build_pipeline(model, numeric_features, categorical_features)
            pipe.fit(train_df[shared_features], train_df[TARGET])
            rows.append(evaluate(pipe, test_df[shared_features], test_df[TARGET], scenario, model_name))

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "auc_roc": "roc_auc",
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for model_name, model in models.items():
        pipe = build_pipeline(model, numeric_features, categorical_features)
        cv_result = cross_validate(
            pipe,
            df[shared_features],
            df[TARGET],
            cv=cv,
            scoring=scoring,
            error_score=np.nan,
        )
        rows.append(
            {
                "scenario": "integrated_5fold_cv_24h_shared_features",
                "model": model_name,
                "n_test": len(df),
                "events": int(df[TARGET].sum()),
                "accuracy": np.nanmean(cv_result["test_accuracy"]),
                "precision": np.nanmean(cv_result["test_precision"]),
                "recall": np.nanmean(cv_result["test_recall"]),
                "f1": np.nanmean(cv_result["test_f1"]),
                "specificity": np.nan,
                "auc_roc": np.nanmean(cv_result["test_auc_roc"]),
                "accuracy_std": np.nanstd(cv_result["test_accuracy"]),
                "precision_std": np.nanstd(cv_result["test_precision"]),
                "recall_std": np.nanstd(cv_result["test_recall"]),
                "f1_std": np.nanstd(cv_result["test_f1"]),
                "auc_roc_std": np.nanstd(cv_result["test_auc_roc"]),
            }
        )

    pd.DataFrame(rows).to_csv(OUT / "landmark_24h_shared_feature_metrics.csv", index=False)
    pd.DataFrame({"shared_features_24h": shared_features}).to_csv(OUT / "landmark_24h_shared_feature_list.csv", index=False)


if __name__ == "__main__":
    main()
