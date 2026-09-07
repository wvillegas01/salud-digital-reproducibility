from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_PATH = Path(r"C:\Users\wilop\Documents\Datos-generales\Clinicos")

INPUT_FILE = BASE_PATH / "dataset_clinico_final_mimic_eicu.csv"

OUTPUT_METRICS = BASE_PATH / "resultados_modelado_multi_entorno.xlsx"
OUTPUT_PREDICTIONS = BASE_PATH / "predicciones_multi_entorno.csv"

TARGET = "target_mortality"

ID_COLUMNS = [
    "case_id",
    "source_dataset",
    "environment_type"
]


# ============================================================
# FUNCIONES
# ============================================================

def evaluate_model(model, X_test, y_test, scenario, model_name):
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = y_pred

    metrics = {
        "scenario": scenario,
        "model": model_name,
        "n_test": len(y_test),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_test, y_score) if len(np.unique(y_test)) > 1 else np.nan
    }

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        columns=["pred_0", "pred_1"],
        index=["true_0", "true_1"]
    )

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()
    report_df["scenario"] = scenario
    report_df["model"] = model_name

    pred_df = pd.DataFrame({
        "scenario": scenario,
        "model": model_name,
        "y_true": y_test.values,
        "y_pred": y_pred,
        "y_score": y_score
    })

    return metrics, cm_df, report_df, pred_df


def build_pipeline(model, numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline


# ============================================================
# CARGA DE DATOS
# ============================================================

df = pd.read_csv(INPUT_FILE)

df = df.dropna(subset=[TARGET]).copy()
df[TARGET] = df[TARGET].astype(int)

print("\nDataset cargado:")
print(df.shape)

print("\nDistribución por fuente:")
print(df["source_dataset"].value_counts())

print("\nDistribución del target:")
print(df[TARGET].value_counts())


# ============================================================
# DEFINICIÓN DE VARIABLES
# ============================================================

feature_columns = [
    col for col in df.columns
    if col not in ID_COLUMNS + [TARGET]
]

X = df[feature_columns].copy()
y = df[TARGET].copy()

categorical_features = [
    col for col in X.columns
    if X[col].dtype == "object"
]

numeric_features = [
    col for col in X.columns
    if col not in categorical_features
]

print("\nVariables numéricas:")
print(numeric_features)

print("\nVariables categóricas:")
print(categorical_features)


# ============================================================
# MODELOS
# ============================================================

models = {
    "LogisticRegression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear"
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ),
    "GradientBoosting": GradientBoostingClassifier(
        random_state=42
    )
}


# ============================================================
# ESCENARIOS DE EVALUACIÓN
# ============================================================

metrics_records = []
confusion_matrices = []
classification_reports = []
prediction_records = []


# ------------------------------------------------------------
# Escenario 1: validación interna eICU
# ------------------------------------------------------------

df_eicu = df[df["source_dataset"] == "eICU"].copy()

X_eicu = df_eicu[feature_columns]
y_eicu = df_eicu[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X_eicu,
    y_eicu,
    test_size=0.25,
    random_state=42,
    stratify=y_eicu
)

for model_name, model in models.items():
    print(f"\nEntrenando {model_name} - eICU internal validation")

    pipeline = build_pipeline(
        model,
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )

    pipeline.fit(X_train, y_train)

    metrics, cm_df, report_df, pred_df = evaluate_model(
        pipeline,
        X_test,
        y_test,
        scenario="eICU_internal_validation",
        model_name=model_name
    )

    metrics_records.append(metrics)

    cm_df["scenario"] = "eICU_internal_validation"
    cm_df["model"] = model_name
    confusion_matrices.append(cm_df.reset_index(names="true_class"))

    classification_reports.append(report_df)

    pred_df["source_test"] = "eICU"
    prediction_records.append(pred_df)


# ------------------------------------------------------------
# Escenario 2: entrenamiento eICU -> prueba MIMIC
# ------------------------------------------------------------

df_mimic = df[df["source_dataset"] == "MIMIC"].copy()

X_train = df_eicu[feature_columns]
y_train = df_eicu[TARGET]

X_test = df_mimic[feature_columns]
y_test = df_mimic[TARGET]

for model_name, model in models.items():
    print(f"\nEntrenando {model_name} - Train eICU / Test MIMIC")

    pipeline = build_pipeline(
        model,
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )

    pipeline.fit(X_train, y_train)

    metrics, cm_df, report_df, pred_df = evaluate_model(
        pipeline,
        X_test,
        y_test,
        scenario="train_eICU_test_MIMIC",
        model_name=model_name
    )

    metrics_records.append(metrics)

    cm_df["scenario"] = "train_eICU_test_MIMIC"
    cm_df["model"] = model_name
    confusion_matrices.append(cm_df.reset_index(names="true_class"))

    classification_reports.append(report_df)

    pred_df["source_test"] = "MIMIC"
    prediction_records.append(pred_df)


# ------------------------------------------------------------
# Escenario 3: entrenamiento MIMIC -> prueba eICU
# ------------------------------------------------------------

X_train = df_mimic[feature_columns]
y_train = df_mimic[TARGET]

X_test = df_eicu[feature_columns]
y_test = df_eicu[TARGET]

for model_name, model in models.items():
    print(f"\nEntrenando {model_name} - Train MIMIC / Test eICU")

    pipeline = build_pipeline(
        model,
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )

    pipeline.fit(X_train, y_train)

    metrics, cm_df, report_df, pred_df = evaluate_model(
        pipeline,
        X_test,
        y_test,
        scenario="train_MIMIC_test_eICU",
        model_name=model_name
    )

    metrics_records.append(metrics)

    cm_df["scenario"] = "train_MIMIC_test_eICU"
    cm_df["model"] = model_name
    confusion_matrices.append(cm_df.reset_index(names="true_class"))

    classification_reports.append(report_df)

    pred_df["source_test"] = "eICU"
    prediction_records.append(pred_df)


# ------------------------------------------------------------
# Escenario 4: validación cruzada sobre dataset integrado
# ------------------------------------------------------------

for model_name, model in models.items():
    print(f"\nValidación cruzada integrada - {model_name}")

    pipeline = build_pipeline(
        model,
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "auc_roc": "roc_auc"
    }

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    metrics_records.append({
        "scenario": "integrated_5fold_cv",
        "model": model_name,
        "n_test": len(y),
        "accuracy": scores["test_accuracy"].mean(),
        "precision": scores["test_precision"].mean(),
        "recall": scores["test_recall"].mean(),
        "f1": scores["test_f1"].mean(),
        "auc_roc": scores["test_auc_roc"].mean(),
        "accuracy_std": scores["test_accuracy"].std(),
        "precision_std": scores["test_precision"].std(),
        "recall_std": scores["test_recall"].std(),
        "f1_std": scores["test_f1"].std(),
        "auc_roc_std": scores["test_auc_roc"].std()
    })


# ============================================================
# GUARDADO DE RESULTADOS
# ============================================================

df_metrics = pd.DataFrame(metrics_records)
df_confusion = pd.concat(confusion_matrices, ignore_index=True)
df_reports = pd.concat(classification_reports, ignore_index=True)
df_predictions = pd.concat(prediction_records, ignore_index=True)

df_predictions.to_csv(OUTPUT_PREDICTIONS, index=False)

with pd.ExcelWriter(OUTPUT_METRICS, engine="openpyxl") as writer:
    df_metrics.to_excel(writer, sheet_name="01_metrics", index=False)
    df_confusion.to_excel(writer, sheet_name="02_confusion_matrices", index=False)
    df_reports.to_excel(writer, sheet_name="03_classification_reports", index=True)
    df_predictions.to_excel(writer, sheet_name="04_predictions", index=False)

print("\nModelado multi-entorno finalizado.")
print("Resultados:", OUTPUT_METRICS)
print("Predicciones:", OUTPUT_PREDICTIONS)

print("\nMétricas principales:")
print(df_metrics)