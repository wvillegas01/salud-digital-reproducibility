import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PRED_PATH = Path(r"C:\Users\wilop\Documents\Datos-generales\Clinicos\predicciones_multi_entorno.csv")
OUT_DIR = Path(r"C:\Users\wilop\Documents\Codex\2026-09-06\ha\work")


def threshold_metrics(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "positive_predictions": int(tp + fp),
    }


def calibration_slope_intercept(y_true, y_score):
    eps = 1e-6
    p = np.clip(y_score, eps, 1 - eps)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    clf.fit(logit, y_true)
    return float(clf.intercept_[0]), float(clf.coef_[0][0])


def expected_calibration_error(y_true, y_score, n_bins=5):
    df = pd.DataFrame({"y": y_true, "p": y_score})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    total = len(df)
    ece = 0.0
    rows = []
    for label, group in df.groupby("bin", observed=False):
        mean_p = group["p"].mean()
        obs = group["y"].mean()
        weight = len(group) / total
        ece += weight * abs(mean_p - obs)
        rows.append(
            {
                "bin": str(label),
                "n": len(group),
                "mean_predicted": mean_p,
                "observed_event_rate": obs,
                "absolute_gap": abs(mean_p - obs),
            }
        )
    return float(ece), rows


def main():
    pred = pd.read_csv(PRED_PATH)
    transfer = pred[pred["scenario"].isin(["train_eICU_test_MIMIC", "train_MIMIC_test_eICU"])].copy()
    summary_rows = []
    threshold_rows = []
    calibration_rows = []

    for (scenario, model), group in transfer.groupby(["scenario", "model"], sort=True):
        y_true = group["y_true"].astype(int).to_numpy()
        y_score = group["y_score"].astype(float).to_numpy()
        auprc = average_precision_score(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        intercept, slope = calibration_slope_intercept(y_true, y_score)
        ece, bins = expected_calibration_error(y_true, y_score)

        default = threshold_metrics(y_true, y_score, 0.5)
        unique_scores = np.unique(y_score)
        candidate_thresholds = np.r_[0.0, unique_scores, 1.0]
        all_metrics = [threshold_metrics(y_true, y_score, t) for t in candidate_thresholds]
        best_f1 = max(all_metrics, key=lambda row: (row["f1"], row["recall"], -row["threshold"]))

        summary_rows.append(
            {
                "scenario": scenario,
                "model": model,
                "auc_roc": auc,
                "average_precision": auprc,
                "calibration_intercept": intercept,
                "calibration_slope": slope,
                "ece_5bins": ece,
                "default_threshold": 0.5,
                "default_precision": default["precision"],
                "default_recall": default["recall"],
                "default_f1": default["f1"],
                "default_specificity": default["specificity"],
                "best_f1_threshold": best_f1["threshold"],
                "best_f1_precision": best_f1["precision"],
                "best_f1_recall": best_f1["recall"],
                "best_f1": best_f1["f1"],
                "best_f1_specificity": best_f1["specificity"],
                "best_f1_positive_predictions": best_f1["positive_predictions"],
            }
        )

        threshold_rows.append({"scenario": scenario, "model": model, "rule": "fixed_0.5", **default})
        threshold_rows.append({"scenario": scenario, "model": model, "rule": "best_f1_in_test", **best_f1})
        for row in bins:
            calibration_rows.append({"scenario": scenario, "model": model, **row})

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "threshold_calibration_summary.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(OUT_DIR / "threshold_sensitivity_summary.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(OUT_DIR / "calibration_gap_bins.csv", index=False)
    metadata = {
        "source_predictions": str(PRED_PATH),
        "note": "Threshold optimization is post hoc on the test set and is reported only as a sensitivity analysis, not as a deployable tuning strategy.",
    }
    (OUT_DIR / "threshold_calibration_audit_metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
