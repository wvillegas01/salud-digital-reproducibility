import json
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


ROOT = Path(r"C:\Users\wilop\Dropbox\graficos-python\salud-digital")
DATA = Path(r"C:\Users\wilop\Documents\Datos-generales\Clinicos\dataset_clinico_final_mimic_eicu.csv")
OUT = Path(r"C:\Users\wilop\Documents\Codex\2026-09-06\ha\work")
OUT.mkdir(parents=True, exist_ok=True)

TARGET = "target_mortality"
SOURCE = "source_dataset"
SEEDS = [1, 2, 3, 4, 5, 10, 20, 42, 100, 2026]


def make_preprocessor(x):
    numeric = x.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in x.columns if c not in numeric]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ],
        remainder="drop",
    )


def metrics(y_true, pred, prob):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "events": int(np.sum(y_true)),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "f1": f1_score(y_true, pred, zero_division=0),
        "auc": roc_auc_score(y_true, prob) if len(np.unique(y_true)) > 1 else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def fit_predict(train, test, model, sample_weight=None):
    x_train = train.drop(columns=[TARGET, SOURCE, "case_id"], errors="ignore")
    y_train = train[TARGET].astype(int)
    x_test = test.drop(columns=[TARGET, SOURCE, "case_id"], errors="ignore")
    y_test = test[TARGET].astype(int)
    pipe = Pipeline([("preprocess", make_preprocessor(x_train)), ("model", model)])
    if sample_weight is None:
        pipe.fit(x_train, y_train)
    else:
        pipe.fit(x_train, y_train, model__sample_weight=sample_weight)
    prob = pipe.predict_proba(x_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return metrics(y_test, pred, prob)


def inverse_frequency_weights(y):
    y = np.asarray(y).astype(int)
    counts = np.bincount(y, minlength=2)
    total = len(y)
    return np.array([total / (2 * counts[v]) if counts[v] else 0 for v in y])


def main():
    df = pd.read_csv(DATA)
    mimic = df[df[SOURCE].str.lower().eq("mimic")].copy()
    eicu = df[df[SOURCE].str.lower().eq("eicu")].copy()

    scenarios = {
        "train_eICU_test_MIMIC": (eicu, mimic),
        "train_MIMIC_test_eICU": (mimic, eicu),
    }

    seed_rows = []
    for scenario, (train, test) in scenarios.items():
        for seed in SEEDS:
            models = {
                "RandomForest": RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced"),
                "GradientBoosting": GradientBoostingClassifier(random_state=seed),
            }
            for model_name, model in models.items():
                row = {"scenario": scenario, "model": model_name, "seed": seed}
                row.update(fit_predict(train, test, model))
                seed_rows.append(row)

    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(OUT / "random_seed_sensitivity.csv", index=False)

    summary_metrics = ["accuracy", "precision", "recall", "specificity", "f1", "auc", "tp", "fp", "fn", "tn"]
    seed_summary = (
        seed_df.groupby(["scenario", "model"])[summary_metrics]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    seed_summary.columns = ["_".join([str(x) for x in c if x]) for c in seed_summary.columns]
    seed_summary.to_csv(OUT / "random_seed_sensitivity_summary.csv", index=False)

    weight_rows = []
    for scenario, (train, test) in scenarios.items():
        y_train = train[TARGET].astype(int)
        gb_weights = inverse_frequency_weights(y_train)
        variants = {
            "LogisticRegression_balanced": (LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"), None),
            "LogisticRegression_unweighted": (LogisticRegression(max_iter=2000, class_weight=None, solver="liblinear"), None),
            "RandomForest_balanced": (RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"), None),
            "RandomForest_unweighted": (RandomForestClassifier(n_estimators=300, random_state=42, class_weight=None), None),
            "GradientBoosting_unweighted": (GradientBoostingClassifier(random_state=42), None),
            "GradientBoosting_inverse_frequency": (GradientBoostingClassifier(random_state=42), gb_weights),
        }
        for variant, (model, sample_weight) in variants.items():
            row = {"scenario": scenario, "variant": variant}
            row.update(fit_predict(train, test, model, sample_weight=sample_weight))
            weight_rows.append(row)

    weight_df = pd.DataFrame(weight_rows)
    weight_df.to_csv(OUT / "class_weight_sensitivity.csv", index=False)

    metadata = {
        "data": str(DATA),
        "seeds": SEEDS,
        "threshold": 0.5,
        "notes": "Transfer-only sensitivity using the same common feature table and preprocessing used in the submitted manuscript.",
    }
    (OUT / "seed_and_class_weight_sensitivity_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
