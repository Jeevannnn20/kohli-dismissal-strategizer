from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from src.models.trainer import run_trainer, temporal_split


def trainer_fixture() -> pd.DataFrame:
    rows = []
    for i in range(20):
        rows.append(
            {
                "match_id": f"m{i}",
                "date": "2020-01-01" if i < 12 else "2021-02-01",
                "format_encoded": i % 4,
                "feature_a": float(i),
                "feature_b": float(i % 3),
                "is_dismissal": 1 if i in {2, 8, 14, 18} else 0,
            }
        )
    return pd.DataFrame(rows)


def test_temporal_split_partitions_by_date() -> None:
    train, test = temporal_split(trainer_fixture(), "2021-01-01")

    assert train["date"].max() < pd.Timestamp("2021-01-01")
    assert test["date"].min() >= pd.Timestamp("2021-01-01")
    assert len(train) == 12
    assert len(test) == 8


def test_trainer_writes_model_results_with_required_keys(tmp_path: Path) -> None:
    config = {
        "project": {"random_state": 42},
        "paths": {
            "kohli_features": "data/processed/kohli_features.parquet",
            "model_results": "data/processed/model_results.json",
            "models_dir": "models",
            "xgboost_model": "models/xgboost_kohli.json",
            "lightgbm_model": "models/lightgbm_kohli.txt",
            "logistic_model": "models/logistic_kohli.joblib",
        },
        "features": {"temporal_train_end_date": "2021-01-01"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    features_path = tmp_path / "data/processed/kohli_features.parquet"
    features_path.parent.mkdir(parents=True)
    trainer_fixture().to_parquet(features_path, compression="snappy", index=False)

    output = run_trainer(config_path, enabled_models=("logistic_regression",))

    results_path = tmp_path / "data/processed/model_results.json"
    assert results_path.exists()
    results = json.loads(results_path.read_text(encoding="utf-8"))
    assert results["best_model"] == output["best_model"]
    assert {"best_model", "models", "train_size", "test_size", "feature_columns"} <= set(results)
    assert "auc_pr" in results["models"]["logistic_regression"]
