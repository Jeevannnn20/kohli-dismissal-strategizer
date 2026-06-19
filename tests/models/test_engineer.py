from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import yaml

from src.features.engineer import engineer_features_frame, run_engineer


def feature_fixture() -> pd.DataFrame:
    rows = []
    for i in range(10):
        innings = 1 if i < 5 else 2
        rows.append(
            {
                "match_id": "m1" if innings == 1 else "m2",
                "date": "2020-01-01" if innings == 1 else "2020-01-03",
                "format": "odis",
                "innings_number": innings,
                "delivery_index": i,
                "bowler_type": "RAPF" if i % 2 == 0 else "OB",
                "pitch_type": "flat",
                "match_phase": "powerplay",
                "over": i,
                "ball": (i % 6) + 1,
                "wide": i == 1,
                "noball": False,
                "batter": "V Kohli",
                "bowler": "Bowler A",
                "runs_off_bat": 1,
                "temperature": 28.0,
                "humidity": 60.0,
                "cloud_cover": 20.0,
                "wind_speed": 8.0,
                "precipitation": 0.0,
                "is_dismissal": 1 if i in {4, 9} else 0,
            }
        )
    return pd.DataFrame(rows)


def test_engineer_rolling_balls_and_no_nulls() -> None:
    features, _, _ = engineer_features_frame(
        feature_fixture(),
        {"tests": 0, "odis": 1, "t20is": 2, "ipl": 3},
        "V Kohli",
    )

    assert features["balls_faced_so_far"].tolist()[:5] == [0, 1, 1, 2, 3]
    assert features.isnull().sum().sum() == 0


def test_engineer_saves_label_encoders(tmp_path: Path) -> None:
    config = {
        "player": {"cricsheet_id": "V Kohli"},
        "paths": {
            "kohli_deliveries_enriched": "data/processed/kohli_deliveries_enriched.parquet",
            "kohli_features": "data/processed/kohli_features.parquet",
            "label_encoders": "data/external/label_encoders.pkl",
        },
        "features": {"format_encoding": {"tests": 0, "odis": 1, "t20is": 2, "ipl": 3}},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    input_path = tmp_path / "data/processed/kohli_deliveries_enriched.parquet"
    input_path.parent.mkdir(parents=True)
    feature_fixture().to_parquet(input_path, compression="snappy", index=False)

    run_engineer(config_path)

    encoders_path = tmp_path / "data/external/label_encoders.pkl"
    assert encoders_path.exists()
    encoders = joblib.load(encoders_path)
    assert set(encoders) == {"bowler_type", "pitch_type", "match_phase"}
