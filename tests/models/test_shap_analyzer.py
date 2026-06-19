from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.models.shap_analyzer import run_shap_analyzer


def test_shap_analyzer_writes_payload_with_required_keys(monkeypatch, tmp_path: Path) -> None:
    config = {
        "paths": {
            "kohli_features": "data/processed/kohli_features.parquet",
            "model_results": "data/processed/model_results.json",
            "shap_results": "data/processed/shap_results.pkl",
            "plots_dir": "data/processed/plots",
            "xgboost_model": "models/xgboost_kohli.json",
            "lightgbm_model": "models/lightgbm_kohli.txt",
            "logistic_model": "models/logistic_kohli.joblib",
        },
        "features": {"temporal_train_end_date": "2021-01-01"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    features_path = tmp_path / "data/processed/kohli_features.parquet"
    results_path = tmp_path / "data/processed/model_results.json"
    features_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "match_id": ["m1", "m2", "m3"],
            "date": ["2020-01-01", "2021-01-02", "2021-01-03"],
            "feature_a": [0.1, 0.2, 0.3],
            "feature_b": [1.0, 2.0, 3.0],
            "is_dismissal": [0, 1, 0],
        }
    ).to_parquet(features_path, compression="snappy", index=False)
    results_path.write_text(
        json.dumps(
            {
                "best_model": "xgboost",
                "feature_columns": ["feature_a", "feature_b"],
                "models": {"xgboost": {"auc_pr": 0.5}},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("src.models.shap_analyzer.load_best_model", lambda *_: object())
    monkeypatch.setattr(
        "src.models.shap_analyzer.compute_shap_values",
        lambda model, X_test, best_model: np.ones((len(X_test), X_test.shape[1])),
    )
    monkeypatch.setattr("src.models.shap_analyzer.save_shap_plots", lambda *_: None)

    payload = run_shap_analyzer(config_path)

    assert {"shap_values", "feature_names", "X_test"} <= set(payload)
    assert (tmp_path / "data/processed/shap_results.pkl").exists()
