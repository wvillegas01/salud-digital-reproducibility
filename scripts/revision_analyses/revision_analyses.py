from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
)


BASE = Path(r"C:\Users\wilop\Documents\Datos-generales\Clinicos")
OUT = Path(r"C:\Users\wilop\Documents\Codex\2026-09-06\ha\work")
OUT.mkdir(parents=True, exist_ok=True)

PRED_FILE = BASE / "predicciones_multi_entorno.csv"
DATA_FILE = BASE / "dataset_clinico_final_mimic_eicu.csv"


def metric_bundle(y_true, y_pred, y_score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": len(y_true),
        "events": int(np.sum(y_true == 1)),
        "non_events": int(np.sum(y_true == 0)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "auc_roc": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan,
        "brier": brier_score_loss(y_true, y_score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def bootstrap_ci(sub, n_boot=500, seed=20260906):
    rng = np.random.default_rng(seed)
    y_true = sub["y_true"].to_numpy(dtype=int)
    y_pred = sub["y_pred"].to_numpy(dtype=int)
    y_score = sub["y_score"].to_numpy(dtype=float)
    point = metric_bundle(y_true, y_pred, y_score)
    rows = []
    n = len(sub)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals = metric_bundle(y_true[idx], y_pred[idx], y_score[idx])
        rows.append(vals)
    boot = pd.DataFrame(rows)
    rec = {
        "scenario": sub["scenario"].iloc[0],
        "model": sub["model"].iloc[0],
    }
    for metric in ["accuracy", "precision", "recall", "f1", "specificity", "auc_roc", "brier"]:
        rec[f"{metric}_point"] = point[metric]
        rec[f"{metric}_ci_low"] = boot[metric].quantile(0.025)
        rec[f"{metric}_ci_high"] = boot[metric].quantile(0.975)
    rec.update({k: point[k] for k in ["n", "events", "non_events", "tn", "fp", "fn", "tp"]})
    return rec


def calibration_summary(sub, bins=5):
    s = sub.copy()
    s["bin"] = pd.qcut(s["y_score"], q=bins, duplicates="drop")
    rows = []
    for interval, g in s.groupby("bin", observed=True):
        rows.append({
            "scenario": s["scenario"].iloc[0],
            "model": s["model"].iloc[0],
            "bin": str(interval),
            "n": len(g),
            "mean_predicted": g["y_score"].mean(),
            "observed_event_rate": g["y_true"].mean(),
        })
    return rows


pred = pd.read_csv(PRED_FILE)
pred = pred[pred["scenario"].isin(["train_eICU_test_MIMIC", "train_MIMIC_test_eICU"])].copy()
pred["y_true"] = pred["y_true"].astype(int)
pred["y_pred"] = pred["y_pred"].astype(int)
pred["y_score"] = pd.to_numeric(pred["y_score"], errors="coerce")

ci_rows = []
cal_rows = []
for (scenario, model), sub in pred.groupby(["scenario", "model"]):
    print(f"Bootstrapping {scenario} / {model} (n={len(sub)})", flush=True)
    ci_rows.append(bootstrap_ci(sub))
    cal_rows.extend(calibration_summary(sub))

ci = pd.DataFrame(ci_rows)
cal = pd.DataFrame(cal_rows)

data = pd.read_csv(DATA_FILE)
cohort = data.groupby("source_dataset")["target_mortality"].agg(
    n="count",
    events="sum",
    prevalence="mean",
)

missing = data.groupby("source_dataset").apply(lambda g: g.isna().mean()).T
missing = missing.reset_index().rename(columns={"index": "variable"})

ci.to_csv(OUT / "transfer_bootstrap_ci.csv", index=False)
cal.to_csv(OUT / "transfer_calibration_bins.csv", index=False)
cohort.to_csv(OUT / "cohort_summary.csv")
missing.to_csv(OUT / "missingness_by_source.csv", index=False)

print("Wrote:")
print(OUT / "transfer_bootstrap_ci.csv")
print(OUT / "transfer_calibration_bins.csv")
print(OUT / "cohort_summary.csv")
print(OUT / "missingness_by_source.csv")
print("\nBootstrap CI summary:")
print(ci.round(3).to_string(index=False))
