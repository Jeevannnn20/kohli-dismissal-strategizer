from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.enrichment.bowler_typer import enrich_bowler_types, run_bowler_typer


def test_bowler_type_join_assigns_unknown() -> None:
    deliveries = pd.DataFrame(
        {
            "bowler": ["K Rabada", "MM Ali", "A Zampa", "Mystery Bowler"],
            "is_dismissal": [1, 0, 1, 0],
        }
    )
    lookup = pd.DataFrame(
        {
            "bowler_name": ["K Rabada", "MM Ali", "A Zampa"],
            "bowler_type": ["RAPF", "OB", "LB"],
        }
    )

    enriched = enrich_bowler_types(deliveries, lookup)

    assert enriched["bowler_type"].tolist() == ["RAPF", "OB", "LB", "UNKNOWN"]


def test_bowler_typer_writes_output_parquet(tmp_path: Path) -> None:
    config = {
        "paths": {
            "kohli_deliveries": "data/processed/kohli_deliveries.parquet",
            "bowler_type_lookup": "data/external/bowler_types.csv",
            "kohli_deliveries_v2": "data/processed/kohli_deliveries_v2.parquet",
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    deliveries_path = tmp_path / "data/processed/kohli_deliveries.parquet"
    lookup_path = tmp_path / "data/external/bowler_types.csv"
    deliveries_path.parent.mkdir(parents=True)
    lookup_path.parent.mkdir(parents=True)
    pd.DataFrame({"bowler": ["K Rabada"], "is_dismissal": [1]}).to_parquet(
        deliveries_path, compression="snappy", index=False
    )
    pd.DataFrame(
        {"bowler_name": ["K Rabada"], "bowler_type": ["RAPF"]}
    ).to_csv(lookup_path, index=False)

    run_bowler_typer(config_path)

    output = pd.read_parquet(tmp_path / "data/processed/kohli_deliveries_v2.parquet")
    assert "bowler_type" in output.columns
    assert output.loc[0, "bowler_type"] == "RAPF"
