"""Train independent XGBoost dismissal models per cricket format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from src.models.trainer import TARGET_COLUMN, feature_columns, temporal_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
FORMATS = ["tests", "odis", "t20is", "ipl"]
COMBINED_AUC_PR_BASELINE = 0.0433


def load_config(config_path: Path | None = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def project_root_from_config(config_path: Path | None = None) -> Path:
    return Path(config_path).resolve().parent if config_path else PROJECT_ROOT


def safe_auc_roc(y_true: pd.Series, probabilities) -> float:
    return float(roc_auc_score(y_true, probabilities)) if y_true.nunique() > 1 else 0.5


def safe_auc_pr(y_true: pd.Series, probabilities) -> float:
    return float(average_precision_score(y_true, probabilities)) if y_true.nunique() > 1 else float(y_true.mean())


def train_format_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int,
    params: dict[str, Any] | None = None,
) -> XGBClassifier:
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    scale_pos_weight = negatives / positives if positives else 1.0
    default_params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "early_stopping_rounds": 30,
        "scale_pos_weight": scale_pos_weight,
        "random_state": random_state,
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)
    model = XGBClassifier(**default_params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return model


def run_format_trainer(
    config_path: Path | None = None,
    model_params: dict[str, Any] | None = None,
    min_test_positives: int = 20,
) -> dict[str, dict[str, Any]]:
    config = load_config(config_path)
    project_root = project_root_from_config(config_path)
    features_path = project_root / Path(config["paths"]["kohli_features"])
    results_path = project_root / Path(config["paths"]["format_model_results"])
    models_dir = project_root / Path(config["paths"]["models_dir"])
    split_date = config["features"]["temporal_train_end_date"]
    format_encoding = config["features"]["format_encoding"]
    random_state = int(config["project"].get("random_state", 42))

    features = pd.read_parquet(features_path)
    columns = feature_columns(features)
    results: dict[str, dict[str, Any]] = {}
    models_dir.mkdir(parents=True, exist_ok=True)

    for format_name in FORMATS:
        encoded = int(format_encoding[format_name])
        subset = features[features["format_encoded"].eq(encoded)].copy()
        train, test = temporal_split(subset, split_date)
        y_train = train[TARGET_COLUMN] if not train.empty else pd.Series(dtype=int)
        y_test = test[TARGET_COLUMN] if not test.empty else pd.Series(dtype=int)
        test_positives = int(y_test.sum()) if not y_test.empty else 0
        model_path = models_dir / f"xgboost_{format_name}.json"
        entry: dict[str, Any] = {
            "train_size": int(len(train)),
            "test_size": int(len(test)),
            "test_positives": test_positives,
            "test_positive_rate": float(y_test.mean()) if len(y_test) else 0.0,
            "auc_roc": None,
            "auc_pr": None,
            "vs_combined": None,
            "model_path": str(model_path),
            "skipped": False,
        }

        if test_positives < min_test_positives or y_train.nunique() < 2 or y_test.nunique() < 2:
            print(
                f"[format_trainer] warning: {format_name} has insufficient positives/classes; "
                f"test positives={test_positives}"
            )
            entry["skipped"] = True
            results[format_name] = entry
            continue

        model = train_format_model(
            train[columns], y_train, test[columns], y_test, random_state, model_params
        )
        probabilities = model.predict_proba(test[columns])[:, 1]
        auc_roc = safe_auc_roc(y_test, probabilities)
        auc_pr = safe_auc_pr(y_test, probabilities)
        model.save_model(model_path)
        entry.update(
            {
                "auc_roc": auc_roc,
                "auc_pr": auc_pr,
                "vs_combined": auc_pr - COMBINED_AUC_PR_BASELINE,
            }
        )
        results[format_name] = entry

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("Format | Train size | Test size | Test positives | AUC-ROC | AUC-PR | vs Combined")
    for format_name, entry in results.items():
        print(
            f"{format_name} | {entry['train_size']} | {entry['test_size']} | "
            f"{entry['test_positives']} | {entry['auc_roc']} | {entry['auc_pr']} | "
            f"{entry['vs_combined']}"
        )
    print(f"[format_trainer] wrote {results_path}")
    return results


def main() -> None:
    run_format_trainer()


if __name__ == "__main__":
    main()
