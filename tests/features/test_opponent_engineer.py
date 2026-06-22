from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import yaml

from src.features.opponent_engineer import add_opponent_context, run_opponent_engineer


def opponent_deliveries_fixture() -> pd.DataFrame:
    rows = []
    teams = [
        ["India", "Australia"],
        ["India", "England"],
        ["Royal Challengers Bangalore", "Mumbai Indians"],
        ["Royal Challengers Bengaluru", "Chennai Super Kings"],
    ]
    countries = ["India", "England", "India", "UAE"]
    formats = ["odis", "tests", "ipl", "ipl"]
    for i in range(12):
        rows.append(
            {
                "match_id": f"m{i // 3}",
                "date": "2020-01-01",
                "format": formats[i % 4],
                "innings_number": i // 6 + 1,
                "delivery_index": i,
                "teams": teams[i % 4],
                "country": countries[i % 4],
                "bowler_type": "RAPS" if i % 2 else "OB",
                "pitch_type": "flat",
                "match_phase": "powerplay",
                "over": i % 20,
                "ball": (i % 6) + 1,
                "wide": False,
                "noball": False,
                "batter": "V Kohli",
                "bowler": f"Bowler {i % 3}",
                "runs_off_bat": i % 4,
                "temperature": 28.0,
                "humidity": 60.0,
                "cloud_cover": 20.0,
                "wind_speed": 8.0,
                "precipitation": 0.0,
                "is_dismissal": 1 if i in {5, 11} else 0,
            }
        )
    return pd.DataFrame(rows)


def test_opponent_context_extracts_non_india_and_home_flag() -> None:
    enriched = add_opponent_context(opponent_deliveries_fixture())
    assert not enriched["opposing_team"].isin(
        ["India", "Royal Challengers Bangalore", "Royal Challengers Bengaluru"]
    ).any()
    assert enriched.loc[enriched["country"].eq("India"), "is_home_match"].eq(1).all()
    assert enriched.loc[~enriched["country"].eq("India"), "is_home_match"].eq(0).all()


def test_opponent_engineer_writes_outputs_without_nulls(tmp_path: Path) -> None:
    config = {
        "player": {"cricsheet_id": "V Kohli"},
        "paths": {
            "kohli_deliveries_enriched": "data/processed/kohli_deliveries_enriched.parquet",
            "kohli_deliveries_opponent": "data/processed/kohli_deliveries_opponent.parquet",
            "kohli_features_v2": "data/processed/kohli_features_v2.parquet",
            "label_encoders": "data/external/label_encoders.pkl",
        },
        "features": {"format_encoding": {"tests": 0, "odis": 1, "t20is": 2, "ipl": 3}},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    input_path = tmp_path / "data/processed/kohli_deliveries_enriched.parquet"
    input_path.parent.mkdir(parents=True)
    opponent_deliveries_fixture().to_parquet(input_path, compression="snappy", index=False)

    features = run_opponent_engineer(config_path)

    assert features.isnull().sum().sum() == 0
    assert "opposing_team_encoded" in features.columns
    assert "is_home_match" in features.columns
    encoders = joblib.load(tmp_path / "data/external/label_encoders.pkl")
    assert "opposing_team" in encoders
