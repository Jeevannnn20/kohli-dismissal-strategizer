from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from src.models.format_trainer import run_format_trainer
from src.models.trainer import temporal_split


def format_features_fixture() -> pd.DataFrame:
    rows = []
    for fmt_code in range(4):
        for i in range(24):
            rows.append(
                {
                    "match_id": f"f{fmt_code}_{i}",
                    "date": "2020-01-01" if i < 12 else "2021-02-01",
                    "format_encoded": fmt_code,
                    "feature_a": float(i),
                    "feature_b": float(fmt_code),
                    "feature_c": float((i + fmt_code) % 5),
                    "is_dismissal": 1 if i in {2, 8, 14, 20} else 0,
                }
            )
    return pd.DataFrame(rows)


def test_format_trainer_writes_results_and_models(tmp_path: Path) -> None:
    config = {
        "project": {"random_state": 42},
        "paths": {
            "kohli_features": "data/processed/kohli_features.parquet",
            "format_model_results": "data/processed/format_model_results.json",
            "models_dir": "models",
        },
        "features": {
            "temporal_train_end_date": "2021-01-01",
            "format_encoding": {"tests": 0, "odis": 1, "t20is": 2, "ipl": 3},
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    features_path = tmp_path / "data/processed/kohli_features.parquet"
    features_path.parent.mkdir(parents=True)
    format_features_fixture().to_parquet(features_path, compression="snappy", index=False)

    run_format_trainer(
        config_path,
        model_params={"n_estimators": 8, "early_stopping_rounds": 2, "max_depth": 2},
        min_test_positives=1,
    )

    results_path = tmp_path / "data/processed/format_model_results.json"
    assert results_path.exists()
    results = json.loads(results_path.read_text(encoding="utf-8"))
    assert set(results) == {"tests", "odis", "t20is", "ipl"}
    for fmt in results:
        assert {"auc_roc", "auc_pr"} <= set(results[fmt])
        assert (tmp_path / f"models/xgboost_{fmt}.json").exists()


def test_format_trainer_uses_temporal_split() -> None:
    train, test = temporal_split(format_features_fixture(), "2021-01-01")
    assert train["date"].max() < pd.Timestamp("2021-01-01")
    assert test["date"].min() >= pd.Timestamp("2021-01-01")
