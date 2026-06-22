from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from src.models.opponent_trainer import run_opponent_trainer


FEATURE_COLUMNS = [
    "bowler_type_encoded",
    "pitch_type_encoded",
    "match_phase_encoded",
    "over",
    "ball",
    "is_wide",
    "is_noball",
    "format_encoded",
    "temperature",
    "humidity",
    "cloud_cover",
    "wind_speed",
    "precipitation",
    "balls_faced_so_far",
    "runs_scored_so_far",
    "strike_rate_so_far",
    "wickets_fallen_before",
    "bowler_career_dismissals_of_kohli",
    "bowler_career_balls_to_kohli",
    "kohli_last5_avg",
    "kohli_last5_sr",
    "opposing_team_encoded",
    "is_home_match",
]


def opponent_features_fixture() -> pd.DataFrame:
    rows = []
    for i in range(96):
        row = {
            "match_id": f"m{i}",
            "date": "2020-01-01" if i < 64 else "2021-02-01",
            "is_dismissal": 1 if i % 11 in {0, 3} else 0,
        }
        for j, column in enumerate(FEATURE_COLUMNS):
            row[column] = float((i + j) % 9)
        row["format_encoded"] = i % 4
        row["is_home_match"] = i % 2
        row["opposing_team_encoded"] = i % 8
        rows.append(row)
    return pd.DataFrame(rows)


def opponent_deliveries_fixture() -> pd.DataFrame:
    teams = ["Australia", "England", "South Africa", "New Zealand", "Pakistan", "Sri Lanka"]
    rows = []
    for i in range(96):
        rows.append(
            {
                "opposing_team": teams[i % len(teams)],
                "is_dismissal": 1 if i % 11 in {0, 3} else 0,
                "wide": False,
                "noball": False,
            }
        )
    return pd.DataFrame(rows)


def test_opponent_trainer_writes_results_shap_and_rate_table(tmp_path: Path) -> None:
    config = {
        "project": {"random_state": 42},
        "paths": {
            "kohli_features_v2": "data/processed/kohli_features_v2.parquet",
            "kohli_deliveries_opponent": "data/processed/kohli_deliveries_opponent.parquet",
            "opponent_model_results": "data/processed/opponent_model_results.json",
            "xgboost_opponent_model": "models/xgboost_opponent.json",
        },
        "features": {"temporal_train_end_date": "2021-01-01"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    features_path = tmp_path / "data/processed/kohli_features_v2.parquet"
    features_path.parent.mkdir(parents=True)
    opponent_features_fixture().to_parquet(features_path, compression="snappy", index=False)
    deliveries_path = tmp_path / "data/processed/kohli_deliveries_opponent.parquet"
    opponent_deliveries_fixture().to_parquet(deliveries_path, compression="snappy", index=False)

    run_opponent_trainer(
        config_path,
        model_params={"n_estimators": 8, "early_stopping_rounds": 2, "max_depth": 2},
    )

    results_path = tmp_path / "data/processed/opponent_model_results.json"
    assert results_path.exists()
    results = json.loads(results_path.read_text(encoding="utf-8"))
    assert {"baseline", "opponent_model", "shap_top10", "dismissal_rate_by_team"} <= set(results)
    shap_features = {item["feature"] for item in results["shap_top10"]}
    assert "opposing_team_encoded" in shap_features or "opposing_team_encoded" in results["watched_feature_shap"]
    assert len(results["dismissal_rate_by_team"]) >= 5
