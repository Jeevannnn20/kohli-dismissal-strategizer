"""Train temporal baseline and boosted dismissal-risk models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
TARGET_COLUMN = "is_dismissal"
ID_COLUMNS = {"match_id", "date", TARGET_COLUMN}


def load_config(config_path: Path | None = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def project_root_from_config(config_path: Path | None = None) -> Path:
    return Path(config_path).resolve().parent if config_path else PROJECT_ROOT


def feature_columns(features: pd.DataFrame) -> list[str]:
    return [column for column in features.columns if column not in ID_COLUMNS]


def temporal_split(
    features: pd.DataFrame, split_date: str = "2021-01-01"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sorted_features = features.copy()
    sorted_features["date"] = pd.to_datetime(sorted_features["date"])
    sorted_features = sorted_features.sort_values("date")
    split_timestamp = pd.Timestamp(split_date)
    train = sorted_features[sorted_features["date"] < split_timestamp].copy()
    test = sorted_features[sorted_features["date"] >= split_timestamp].copy()
    return train, test


def evaluate_predictions(y_true: pd.Series, probabilities: Any) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "auc_roc": float(roc_auc_score(y_true, probabilities)),
        "auc_pr": float(average_precision_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }


def train_logistic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[Pipeline, dict[str, float]]:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    return model, evaluate_predictions(y_test, probabilities)


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int,
) -> tuple[XGBClassifier, dict[str, float]]:
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    scale_pos_weight = negatives / positives if positives else 1
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        early_stopping_rounds=30,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    probabilities = model.predict_proba(X_test)[:, 1]
    return model, evaluate_predictions(y_test, probabilities)


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int,
) -> tuple[lgb.LGBMClassifier, dict[str, float]]:
    model = lgb.LGBMClassifier(
        objective="binary",
        is_unbalance=True,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    probabilities = model.predict_proba(X_test)[:, 1]
    return model, evaluate_predictions(y_test, probabilities)


def print_split_summary(train: pd.DataFrame, test: pd.DataFrame) -> None:
    train_rate = float(train[TARGET_COLUMN].mean() * 100) if len(train) else 0
    test_rate = float(test[TARGET_COLUMN].mean() * 100) if len(test) else 0
    print(f"[trainer] train size: {len(train)}, positive rate: {train_rate:.2f}%")
    print(f"[trainer] test size: {len(test)}, positive rate: {test_rate:.2f}%")


def print_comparison(results: dict[str, dict[str, float]]) -> None:
    print("[trainer] model comparison:")
    print("Model | AUC-ROC | AUC-PR | Precision | Recall | F1")
    for model_name, metrics in results.items():
        print(
            f"{model_name} | {metrics['auc_roc']:.4f} | {metrics['auc_pr']:.4f} | "
            f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f}"
        )


def run_trainer(
    config_path: Path | None = None,
    enabled_models: tuple[str, ...] = ("logistic_regression", "xgboost", "lightgbm"),
) -> dict[str, Any]:
    config = load_config(config_path)
    project_root = project_root_from_config(config_path)
    features_path = project_root / Path(config["paths"]["kohli_features"])
    results_path = project_root / Path(config["paths"]["model_results"])
    models_dir = project_root / Path(config["paths"]["models_dir"])
    xgb_path = project_root / Path(config["paths"]["xgboost_model"])
    lgb_path = project_root / Path(config["paths"]["lightgbm_model"])
    logistic_path = project_root / Path(config["paths"]["logistic_model"])
    split_date = config["features"]["temporal_train_end_date"]
    random_state = int(config["project"].get("random_state", 42))

    features = pd.read_parquet(features_path)
    train, test = temporal_split(features, split_date)
    print_split_summary(train, test)
    if train.empty or test.empty:
        raise ValueError("Temporal split produced an empty train or test set")

    columns = feature_columns(features)
    X_train, y_train = train[columns], train[TARGET_COLUMN]
    X_test, y_test = test[columns], test[TARGET_COLUMN]

    models_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, float]] = {}

    if "logistic_regression" in enabled_models:
        logistic, metrics = train_logistic(X_train, y_train, X_test, y_test)
        joblib.dump(logistic, logistic_path)
        results["logistic_regression"] = metrics

    if "xgboost" in enabled_models:
        xgb_model, metrics = train_xgboost(X_train, y_train, X_test, y_test, random_state)
        xgb_model.save_model(xgb_path)
        results["xgboost"] = metrics

    if "lightgbm" in enabled_models:
        lgb_model, metrics = train_lightgbm(X_train, y_train, X_test, y_test, random_state)
        lgb_model.booster_.save_model(lgb_path)
        results["lightgbm"] = metrics

    best_model = max(results, key=lambda name: results[name]["auc_pr"])
    output = {
        "best_model": best_model,
        "feature_columns": columns,
        "split_date": split_date,
        "train_size": int(len(train)),
        "test_size": int(len(test)),
        "train_positive_rate": float(y_train.mean()),
        "test_positive_rate": float(y_test.mean()),
        "models": results,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print_comparison(results)
    print(f"[trainer] best model by AUC-PR: {best_model}")
    print(f"[trainer] wrote {results_path}")
    print(f"[trainer] feature matrix shape: {features.shape}")
    return output


def main() -> None:
    run_trainer()


if __name__ == "__main__":
    main()
