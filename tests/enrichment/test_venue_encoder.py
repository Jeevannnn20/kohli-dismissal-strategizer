from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.enrichment.venue_encoder import enrich_venues, run_venue_encoder


def test_venue_join_assigns_unknown() -> None:
    deliveries = pd.DataFrame(
        {
            "venue": ["Eden Gardens", "Lord's", "Unknown Ground"],
            "is_dismissal": [1, 0, 1],
        }
    )
    venues = pd.DataFrame(
        {
            "venue": ["Eden Gardens", "Lord's"],
            "country": ["India", "England"],
            "pitch_type": ["spin", "seam"],
            "latitude": [22.5646, 51.5296],
            "longitude": [88.3433, -0.1728],
        }
    )

    enriched = enrich_venues(deliveries, venues)

    assert enriched["pitch_type"].tolist() == ["spin", "seam", "unknown"]
    assert enriched.loc[2, "country"] == "unknown"


def test_venue_encoder_writes_output_parquet(tmp_path: Path) -> None:
    config = {
        "paths": {
            "kohli_deliveries_v2": "data/processed/kohli_deliveries_v2.parquet",
            "venue_metadata": "data/external/venues.csv",
            "kohli_deliveries_v3": "data/processed/kohli_deliveries_v3.parquet",
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    deliveries_path = tmp_path / "data/processed/kohli_deliveries_v2.parquet"
    venues_path = tmp_path / "data/external/venues.csv"
    deliveries_path.parent.mkdir(parents=True)
    venues_path.parent.mkdir(parents=True)
    pd.DataFrame({"venue": ["Eden Gardens"], "is_dismissal": [1]}).to_parquet(
        deliveries_path, compression="snappy", index=False
    )
    pd.DataFrame(
        {
            "venue": ["Eden Gardens"],
            "country": ["India"],
            "pitch_type": ["spin"],
            "latitude": [22.5646],
            "longitude": [88.3433],
        }
    ).to_csv(venues_path, index=False)

    run_venue_encoder(config_path)

    output = pd.read_parquet(tmp_path / "data/processed/kohli_deliveries_v3.parquet")
    assert output.loc[0, "pitch_type"] == "spin"
