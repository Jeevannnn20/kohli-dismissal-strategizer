from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.ingestion.cleaner import clean_kohli_deliveries


def test_cleaner_drops_non_striker_kohli_dismissals(monkeypatch, tmp_path: Path) -> None:
    config = {
        "player": {"cricsheet_id": "V Kohli"},
        "paths": {
            "kohli_deliveries": "data/processed/kohli_deliveries.parquet",
            "kohli_dismissals": "data/processed/kohli_dismissals.parquet",
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    deliveries_path = tmp_path / "data/processed/kohli_deliveries.parquet"
    deliveries_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"batter": "V Kohli", "player_dismissed": None},
            {"batter": "RG Sharma", "player_dismissed": "V Kohli"},
        ]
    ).to_parquet(deliveries_path, compression="snappy", index=False)

    called = []
    monkeypatch.setattr(
        "src.ingestion.cleaner.extract_dismissals",
        lambda path: called.append(path),
    )

    cleaned = clean_kohli_deliveries(config_path)

    assert len(cleaned) == 1
    assert pd.read_parquet(deliveries_path).shape[0] == 1
    assert called == [config_path]
