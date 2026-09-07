from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


OUT = Path(r"C:\Users\wilop\Documents\Codex\2026-09-06\ha\work")
DATA_PATH = OUT / "dataset_clinico_landmark_24h.csv"
TARGET = "target_mortality"
ID_COLUMNS = ["case_id", "source_dataset", "environment_type"]
EPS = 1e-6


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


def ece_score(y_true, y_score, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (y_score >= left) & (y_score < right)
        if right == 1:
            mask = (y_score >= left) & (y_score <= right)
        if not mask.any():
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - y_score[mask].mean())
    return ece


def calibration_intercept_slope(y_true, y_score):
    y_score = np.clip(y_score, EPS, 1 - EPS)
    logits = np.log(y_score / (1 - y_score)).reshape(-1, 1)
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    model.fit(logits, y_true)
    return float(model.intercept_[0]), float(model.coef_[0][0])


def metrics(y_true, y_score, label, model_name):
    y_pred = (y_score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = np.nan if (tp + fp) == 0 else precision_score(y_true, y_pred)
    intercept, slope = calibration_intercept_slope(y_true, y_score)
    return {
        "scenario": label,
        "model": model_name,
        "n": len(y_true),
        "events": int(y_true.sum()),
        "event_prevalence": float(y_true.mean()),
        "precision": precision,
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_score),
        "average_precision": average_precision_score(y_true, y_score),
        "brier": brier_score_loss(y_true, y_score),
        "ece_10bins": ece_score(y_true, y_score, n_bins=10),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
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
    categorical_features = [col for col in shared_features if df[col].dtype == "object"]
    numeric_features = [col for col in shared_features if col not in categorical_features]
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    df_eicu = df[df["source_dataset"] == "eICU"].copy()
    df_mimic = df[df["source_dataset"] == "MIMIC"].copy()
    rows = []

    for scenario, train_df, test_df in [
        ("eICU to MIMIC transfer", df_eicu, df_mimic),
        ("MIMIC to eICU transfer", df_mimic, df_eicu),
    ]:
        for model_name, model in models.items():
            pipe = build_pipeline(model, numeric_features, categorical_features)
            pipe.fit(train_df[shared_features], train_df[TARGET])
            y_score = pipe.predict_proba(test_df[shared_features])[:, 1]
            rows.append(metrics(test_df[TARGET].to_numpy(), y_score, scenario, model_name))

    internal_rows = []
    for source_label, source_df in [("eICU internal 5-fold OOF", df_eicu), ("MIMIC internal 5-fold OOF", df_mimic)]:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for model_name, model in models.items():
            pipe = build_pipeline(clone(model), numeric_features, categorical_features)
            y_score = cross_val_predict(
                pipe,
                source_df[shared_features],
                source_df[TARGET],
                cv=cv,
                method="predict_proba",
            )[:, 1]
            internal_rows.append(metrics(source_df[TARGET].to_numpy(), y_score, source_label, model_name))

    repeated_rows = []
    repeated_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)
    for model_name, model in models.items():
        fold_rows = []
        for fold, (train_idx, test_idx) in enumerate(repeated_cv.split(df_mimic[shared_features], df_mimic[TARGET])):
            pipe = build_pipeline(clone(model), numeric_features, categorical_features)
            train_df = df_mimic.iloc[train_idx]
            test_df = df_mimic.iloc[test_idx]
            pipe.fit(train_df[shared_features], train_df[TARGET])
            y_score = pipe.predict_proba(test_df[shared_features])[:, 1]
            fold_rows.append(metrics(test_df[TARGET].to_numpy(), y_score, "MIMIC repeated internal fold", model_name))
            fold_rows[-1]["fold"] = fold
        folds = pd.DataFrame(fold_rows)
        summary = {"scenario": "MIMIC internal repeated 5-fold CV", "model": model_name}
        for metric in ["auc_roc", "average_precision", "recall", "specificity", "f1", "brier", "ece_10bins"]:
            summary[f"{metric}_mean"] = folds[metric].mean()
            summary[f"{metric}_std"] = folds[metric].std()
        summary["zero_positive_prediction_folds"] = int((folds["positive_predictions"] == 0).sum())
        summary["n_folds"] = len(folds)
        repeated_rows.append(summary)

    pd.DataFrame(rows + internal_rows).to_csv(OUT / "primary_24h_shared_calibration_transfer_internal.csv", index=False)
    pd.DataFrame(repeated_rows).to_csv(OUT / "mimic_internal_repeated_cv_summary.csv", index=False)


if __name__ == "__main__":
    main()
