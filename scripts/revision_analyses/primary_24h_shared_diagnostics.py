from pathlib import Path
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
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
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


OUT = Path(r"C:\Users\wilop\Documents\Codex\2026-09-06\ha\work")
DATA_PATH = OUT / "dataset_clinico_landmark_24h.csv"
TARGET = "target_mortality"
ID_COLUMNS = ["case_id", "patient_group_id", "admission_group_id", "source_dataset", "environment_type"]
EPS = 1e-6


SCENARIO_LABELS = {
    "eicu_internal_oof": "eICU internal 5-fold OOF",
    "eicu_to_mimic": "eICU to MIMIC transfer",
    "mimic_internal_oof": "MIMIC internal 5-fold OOF",
    "mimic_to_eicu": "MIMIC to eICU transfer",
}
FIGURE_LABELS = {
    "eicu_internal_oof": "(a) eICU internal OOF",
    "eicu_to_mimic": "(b) eICU -> MIMIC",
    "mimic_internal_oof": "(c) MIMIC internal OOF",
    "mimic_to_eicu": "(d) MIMIC -> eICU",
}
MODEL_ABBREVIATIONS = {
    "Logistic Regression": "LR",
    "Random Forest": "RF",
    "Gradient Boosting": "GB",
}


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
    for i, (left, right) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (y_score >= left) & (y_score < right)
        if i == n_bins - 1:
            mask = (y_score >= left) & (y_score <= right)
        if not mask.any():
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - y_score[mask].mean())
    return float(ece)


def calibration_bins(y_true, y_score, scenario_key, model_name, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = ece_score(y_true, y_score, n_bins=n_bins)
    rows = []
    for i, (left, right) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (y_score >= left) & (y_score < right)
        if i == n_bins - 1:
            mask = (y_score >= left) & (y_score <= right)
        if not mask.any():
            continue
        rows.append(
            {
                "scenario_key": scenario_key,
                "scenario": FIGURE_LABELS[scenario_key],
                "model": MODEL_ABBREVIATIONS[model_name],
                "model_full": model_name,
                "bin_index": i + 1,
                "bin_left": float(left),
                "bin_right": float(right),
                "ece_10bins": ece,
                "mean_predicted": float(y_score[mask].mean()),
                "observed_event_rate": float(y_true[mask].mean()),
                "n": int(mask.sum()),
                "events": int(y_true[mask].sum()),
                "positive_predictions": int((y_score[mask] >= 0.5).sum()),
                "true_positives_at_0_5": int(((y_score[mask] >= 0.5) & (y_true[mask] == 1)).sum()),
            }
        )
    return rows


def calibration_intercept_slope(y_true, y_score):
    y_score = np.clip(y_score, EPS, 1 - EPS)
    logits = np.log(y_score / (1 - y_score)).reshape(-1, 1)
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    model.fit(logits, y_true)
    return float(model.intercept_[0]), float(model.coef_[0][0])


def metrics_from_predictions(pred_df):
    y_true = pred_df[TARGET].to_numpy(dtype=int)
    y_score = pred_df["predicted_probability"].to_numpy(dtype=float)
    y_pred = pred_df["predicted_label"].to_numpy(dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = np.nan if (tp + fp) == 0 else precision_score(y_true, y_pred)
    intercept, slope = calibration_intercept_slope(y_true, y_score)
    return {
        "scenario": pred_df["scenario"].iloc[0],
        "model": pred_df["model"].iloc[0],
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


def prediction_rows(test_df, scores, scenario_key, scenario_kind, model_name, fold=None, train_source=None):
    rows = test_df[ID_COLUMNS + [TARGET]].copy()
    rows.insert(0, "scenario_key", scenario_key)
    rows.insert(1, "scenario", SCENARIO_LABELS[scenario_key])
    rows.insert(2, "scenario_kind", scenario_kind)
    rows.insert(3, "train_source_dataset", train_source if train_source is not None else "")
    rows.insert(4, "model", model_name)
    rows.insert(5, "model_abbrev", MODEL_ABBREVIATIONS[model_name])
    rows.insert(6, "fold", "" if fold is None else int(fold))
    rows["predicted_probability"] = scores
    rows["predicted_label"] = (rows["predicted_probability"] >= 0.5).astype(int)
    return rows


def transfer_predictions(train_df, test_df, features, numeric_features, categorical_features, model, model_name, scenario_key):
    pipe = build_pipeline(clone(model), numeric_features, categorical_features)
    pipe.fit(train_df[features], train_df[TARGET])
    scores = pipe.predict_proba(test_df[features])[:, 1]
    train_source = train_df["source_dataset"].iloc[0]
    return prediction_rows(test_df, scores, scenario_key, "transfer", model_name, train_source=train_source)


def internal_oof_predictions(source_df, features, numeric_features, categorical_features, model, model_name, scenario_key):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_frames = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(source_df[features], source_df[TARGET]), start=1):
        pipe = build_pipeline(clone(model), numeric_features, categorical_features)
        train_df = source_df.iloc[train_idx]
        test_df = source_df.iloc[test_idx]
        pipe.fit(train_df[features], train_df[TARGET])
        scores = pipe.predict_proba(test_df[features])[:, 1]
        fold_frames.append(
            prediction_rows(
                test_df,
                scores,
                scenario_key,
                "internal_stratified_oof",
                model_name,
                fold=fold,
                train_source=source_df["source_dataset"].iloc[0],
            )
        )
    return pd.concat(fold_frames, ignore_index=True)


def patient_grouped_oof_predictions(source_df, features, numeric_features, categorical_features, model, model_name, scenario_key):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    groups = source_df["patient_group_id"].astype(str)
    fold_frames = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(source_df[features], source_df[TARGET], groups=groups), start=1):
        pipe = build_pipeline(clone(model), numeric_features, categorical_features)
        train_df = source_df.iloc[train_idx]
        test_df = source_df.iloc[test_idx]
        pipe.fit(train_df[features], train_df[TARGET])
        scores = pipe.predict_proba(test_df[features])[:, 1]
        fold_frames.append(
            prediction_rows(
                test_df,
                scores,
                scenario_key,
                "internal_patient_grouped_oof",
                model_name,
                fold=fold,
                train_source=source_df["source_dataset"].iloc[0],
            )
        )
    return pd.concat(fold_frames, ignore_index=True)


def build_shared_feature_sets(df):
    candidate_features = [c for c in df.columns if c not in ID_COLUMNS + [TARGET]]
    missing_by_source = df.groupby("source_dataset")[candidate_features].apply(lambda g: g.isna().mean())
    shared_features = [
        col
        for col in candidate_features
        if all(missing_by_source.loc[source, col] <= 0.60 for source in missing_by_source.index)
    ]
    categorical_features = [col for col in shared_features if df[col].dtype == "object"]
    numeric_features = [col for col in shared_features if col not in categorical_features]
    return shared_features, numeric_features, categorical_features


def summarize_repeated_mimic(df_mimic, shared_features, numeric_features, categorical_features, models):
    repeated_rows = []
    repeated_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)
    for model_name, model in models.items():
        fold_rows = []
        for fold, (train_idx, test_idx) in enumerate(repeated_cv.split(df_mimic[shared_features], df_mimic[TARGET]), start=1):
            pipe = build_pipeline(clone(model), numeric_features, categorical_features)
            train_df = df_mimic.iloc[train_idx]
            test_df = df_mimic.iloc[test_idx]
            pipe.fit(train_df[shared_features], train_df[TARGET])
            scores = pipe.predict_proba(test_df[shared_features])[:, 1]
            pred = prediction_rows(
                test_df,
                scores,
                "mimic_internal_oof",
                "mimic_repeated_internal_fold",
                model_name,
                fold=fold,
                train_source="MIMIC",
            )
            fold_rows.append(metrics_from_predictions(pred))
            fold_rows[-1]["fold"] = fold
        folds = pd.DataFrame(fold_rows)
        summary = {"scenario": "MIMIC internal repeated 5-fold CV", "model": model_name}
        for metric in ["auc_roc", "average_precision", "recall", "specificity", "f1", "brier", "ece_10bins"]:
            summary[f"{metric}_mean"] = folds[metric].mean()
            summary[f"{metric}_std"] = folds[metric].std()
        summary["zero_positive_prediction_folds"] = int((folds["positive_predictions"] == 0).sum())
        summary["n_folds"] = len(folds)
        repeated_rows.append(summary)
    return pd.DataFrame(repeated_rows)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    df = pd.read_csv(DATA_PATH).dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)
    shared_features, numeric_features, categorical_features = build_shared_feature_sets(df)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    df_eicu = df[df["source_dataset"] == "eICU"].copy()
    df_mimic = df[df["source_dataset"] == "MIMIC"].copy()

    canonical_frames = []
    grouped_frames = []
    for model_name, model in models.items():
        canonical_frames.append(
            internal_oof_predictions(
                df_eicu, shared_features, numeric_features, categorical_features, model, model_name, "eicu_internal_oof"
            )
        )
        canonical_frames.append(
            transfer_predictions(
                df_eicu, df_mimic, shared_features, numeric_features, categorical_features, model, model_name, "eicu_to_mimic"
            )
        )
        canonical_frames.append(
            internal_oof_predictions(
                df_mimic, shared_features, numeric_features, categorical_features, model, model_name, "mimic_internal_oof"
            )
        )
        canonical_frames.append(
            transfer_predictions(
                df_mimic, df_eicu, shared_features, numeric_features, categorical_features, model, model_name, "mimic_to_eicu"
            )
        )
        grouped_frames.append(
            patient_grouped_oof_predictions(
                df_eicu,
                shared_features,
                numeric_features,
                categorical_features,
                model,
                model_name,
                "eicu_internal_oof",
            )
        )
        grouped_frames.append(
            patient_grouped_oof_predictions(
                df_mimic,
                shared_features,
                numeric_features,
                categorical_features,
                model,
                model_name,
                "mimic_internal_oof",
            )
        )

    canonical_predictions = pd.concat(canonical_frames, ignore_index=True)
    grouped_predictions = pd.concat(grouped_frames, ignore_index=True)
    canonical_path = OUT / "primary_24h_canonical_predictions_restricted.csv"
    grouped_path = OUT / "primary_24h_patient_grouped_predictions_restricted.csv"
    canonical_predictions.to_csv(canonical_path, index=False)
    grouped_predictions.to_csv(grouped_path, index=False)

    diagnostic_rows = []
    bin_rows = []
    for (scenario_key, model_name), pred_df in canonical_predictions.groupby(["scenario_key", "model"], sort=False):
        diagnostic_rows.append(metrics_from_predictions(pred_df))
        y_true = pred_df[TARGET].to_numpy(dtype=int)
        y_score = pred_df["predicted_probability"].to_numpy(dtype=float)
        bin_rows.extend(calibration_bins(y_true, y_score, scenario_key, model_name, n_bins=10))

    grouped_rows = []
    for (_scenario_key, model_name), pred_df in grouped_predictions.groupby(["scenario_key", "model"], sort=False):
        row = metrics_from_predictions(pred_df)
        source = pred_df["source_dataset"].iloc[0]
        row["scenario"] = f"{source} internal patient-grouped 5-fold OOF"
        row["model"] = model_name
        grouped_rows.append(row)

    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics.to_csv(OUT / "primary_24h_shared_calibration_transfer_internal.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(OUT / "primary_24h_calibration_plot_bins.csv", index=False)
    pd.DataFrame(grouped_rows).to_csv(OUT / "primary_24h_patient_grouped_internal_validation.csv", index=False)
    summarize_repeated_mimic(df_mimic, shared_features, numeric_features, categorical_features, models).to_csv(
        OUT / "mimic_internal_repeated_cv_summary.csv", index=False
    )

    summary = diagnostics[
        [
            "scenario",
            "model",
            "n",
            "events",
            "tn",
            "fp",
            "fn",
            "tp",
            "positive_predictions",
            "auc_roc",
            "average_precision",
            "brier",
            "ece_10bins",
        ]
    ].copy()
    summary.to_csv(OUT / "primary_24h_canonical_prediction_summary.csv", index=False)
    metadata = {
        "canonical_predictions_file": str(canonical_path),
        "canonical_predictions_sha256": sha256_file(canonical_path),
        "patient_grouped_predictions_file": str(grouped_path),
        "patient_grouped_predictions_sha256": sha256_file(grouped_path),
        "dataset_file": str(DATA_PATH),
        "dataset_sha256": sha256_file(DATA_PATH),
        "classification_rule": "predicted_probability >= 0.5",
        "ece_definition": "10 equal-width bins over [0, 1]; empty bins omitted from plotted bins and contribute zero to ECE",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "shared_features": shared_features,
    }
    with open(OUT / "primary_24h_canonical_prediction_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
