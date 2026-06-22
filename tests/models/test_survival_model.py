from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import yaml

from src.models.survival_model import run_survival_model


def survival_deliveries_fixture() -> pd.DataFrame:
    rows = []
    for innings in range(1, 49):
        dismissed = innings % 3 != 0
        balls = 8 + (innings % 9)
        for ball in range(balls):
            rows.append(
                {
                    "match_id": f"m{innings}",
                    "date": "2020-01-01" if innings <= 32 else "2021-02-01",
                    "innings_number": innings,
                    "delivery_index": ball,
                    "wide": False,
                    "noball": False,
                    "runs_off_bat": ball % 4,
                    "is_dismissal": 1 if dismissed and ball == balls - 1 else 0,
                    "format": ["tests", "odis", "t20is", "ipl"][innings % 4],
                    "pitch_type": ["flat", "seam", "spin", "bounce"][innings % 4],
                    "match_phase": ["opening", "middle", "death"][innings % 3],
                    "bowler_type": ["LAPS", "RAPS", "OB", "LB"][innings % 4],
                    "cloud_cover": float(innings % 100),
                    "humidity": 50.0 + (innings % 20),
                }
            )
    return pd.DataFrame(rows)


def test_survival_model_outputs_innings_model_and_cindex(tmp_path: Path) -> None:
    config = {
        "paths": {
            "kohli_deliveries_enriched": "data/processed/kohli_deliveries_enriched.parquet",
            "kohli_innings": "data/processed/kohli_innings.parquet",
            "cox_model": "models/cox_ph_model.pkl",
            "plots_dir": "data/processed/plots",
        },
        "features": {
            "temporal_train_end_date": "2021-01-01",
            "format_encoding": {"tests": 0, "odis": 1, "t20is": 2, "ipl": 3},
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    deliveries_path = tmp_path / "data/processed/kohli_deliveries_enriched.parquet"
    deliveries_path.parent.mkdir(parents=True)
    survival_deliveries_fixture().to_parquet(deliveries_path, compression="snappy", index=False)

    results = run_survival_model(config_path)

    innings_path = tmp_path / "data/processed/kohli_innings.parquet"
    model_path = tmp_path / "models/cox_ph_model.pkl"
    assert innings_path.exists()
    assert model_path.exists()
    innings = pd.read_parquet(innings_path)
    assert {"duration", "event_observed", "format_encoded"} <= set(innings.columns)
    assert set(innings["event_observed"].unique()) <= {0, 1}
    assert (innings["duration"] > 0).all()
    model = joblib.load(model_path)
    assert hasattr(model, "predict_median")
    assert 0.5 <= results["test_c_index"] <= 1.0
