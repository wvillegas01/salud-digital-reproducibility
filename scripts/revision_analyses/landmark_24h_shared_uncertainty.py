from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


OUT = Path(r"C:\Users\wilop\Documents\Codex\2026-09-06\ha\work")
DATA_PATH = OUT / "dataset_clinico_landmark_24h.csv"
TARGET = "target_mortality"
ID_COLUMNS = ["case_id", "source_dataset", "environment_type"]
N_BOOT = 2000
SEED = 20260907


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


def exact_binomial_ci(k, n, alpha=0.05):
    if n == 0:
        return np.nan, np.nan
    low = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    high = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return low, high


def point_metrics(y_true, y_pred, y_score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = np.nan if (tp + fp) == 0 else precision_score(y_true, y_pred)
    sens_low, sens_high = exact_binomial_ci(tp, tp + fn)
    spec_low, spec_high = exact_binomial_ci(tn, tn + fp)
    return {
        "accuracy_point": accuracy_score(y_true, y_pred),
        "precision_point": precision,
        "precision_defined": int((tp + fp) > 0),
        "recall_point": recall_score(y_true, y_pred, zero_division=0),
        "recall_exact_ci_low": sens_low,
        "recall_exact_ci_high": sens_high,
        "f1_point": f1_score(y_true, y_pred, zero_division=0),
        "specificity_point": tn / (tn + fp) if (tn + fp) else np.nan,
        "specificity_exact_ci_low": spec_low,
        "specificity_exact_ci_high": spec_high,
        "auc_roc_point": roc_auc_score(y_true, y_score),
        "brier_point": brier_score_loss(y_true, y_score),
        "n": len(y_true),
        "events": int(y_true.sum()),
        "non_events": int(len(y_true) - y_true.sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "positive_predictions": int(tp + fp),
    }


def bootstrap_metrics(y_true, y_pred, y_score):
    rng = np.random.default_rng(SEED)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    values = {"accuracy": [], "f1": [], "auc_roc": [], "brier": []}
    discarded = 0
    for _ in range(N_BOOT):
        sample_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        sample_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([sample_pos, sample_neg])
        yt, yp, ys = y_true[idx], y_pred[idx], y_score[idx]
        try:
            values["auc_roc"].append(roc_auc_score(yt, ys))
        except ValueError:
            discarded += 1
            continue
        values["accuracy"].append(accuracy_score(yt, yp))
        values["f1"].append(f1_score(yt, yp, zero_division=0))
        values["brier"].append(brier_score_loss(yt, ys))
    intervals = {"bootstrap_resamples": N_BOOT, "bootstrap_discarded_single_class": discarded}
    for metric, vals in values.items():
        intervals[f"{metric}_ci_low"] = np.percentile(vals, 2.5)
        intervals[f"{metric}_ci_high"] = np.percentile(vals, 97.5)
    return intervals


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
    scenarios = {
        "train_eICU_test_MIMIC_24h_shared_features": (df_eicu, df_mimic),
        "train_MIMIC_test_eICU_24h_shared_features": (df_mimic, df_eicu),
    }
    rows = []
    for scenario, (train_df, test_df) in scenarios.items():
        for model_name, model in models.items():
            pipe = build_pipeline(model, numeric_features, categorical_features)
            pipe.fit(train_df[shared_features], train_df[TARGET])
            y_true = test_df[TARGET].to_numpy()
            y_pred = pipe.predict(test_df[shared_features])
            y_score = pipe.predict_proba(test_df[shared_features])[:, 1]
            row = {"scenario": scenario, "model": model_name}
            row.update(point_metrics(y_true, y_pred, y_score))
            row.update(bootstrap_metrics(y_true, y_pred, y_score))
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "landmark_24h_shared_transfer_uncertainty.csv", index=False)


if __name__ == "__main__":
    main()
