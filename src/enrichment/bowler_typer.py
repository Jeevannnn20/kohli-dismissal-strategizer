"""Enrich Kohli delivery rows with bowler type metadata."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_path: Path | None = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def project_root_from_config(config_path: Path | None = None) -> Path:
    return Path(config_path).resolve().parent if config_path else PROJECT_ROOT


def enrich_bowler_types(
    deliveries: pd.DataFrame, bowler_types: pd.DataFrame
) -> pd.DataFrame:
    lookup = bowler_types[["bowler_name", "bowler_type"]].drop_duplicates(
        "bowler_name"
    )
    enriched = deliveries.merge(
        lookup, how="left", left_on="bowler", right_on="bowler_name"
    )
    enriched["bowler_type"] = enriched["bowler_type"].fillna("UNKNOWN")
    return enriched.drop(columns=["bowler_name"])


def known_percentage(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float((series != "UNKNOWN").mean() * 100)


def run_bowler_typer(config_path: Path | None = None) -> pd.DataFrame:
    config = load_config(config_path)
    project_root = project_root_from_config(config_path)
    deliveries_path = project_root / Path(config["paths"]["kohli_deliveries"])
    bowler_types_path = project_root / Path(config["paths"]["bowler_type_lookup"])
    output_path = project_root / Path(config["paths"]["kohli_deliveries_v2"])

    deliveries = pd.read_parquet(deliveries_path)
    bowler_types = pd.read_csv(bowler_types_path)
    enriched = enrich_bowler_types(deliveries, bowler_types)

    dismissal_rows = enriched[enriched["is_dismissal"] == 1]
    print(
        "[bowler_typer] known bowler_type on dismissal balls: "
        f"{known_percentage(dismissal_rows['bowler_type']):.2f}%"
    )
    print(
        "[bowler_typer] known bowler_type on all deliveries: "
        f"{known_percentage(enriched['bowler_type']):.2f}%"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(output_path, compression="snappy", index=False)
    print(f"[bowler_typer] wrote {output_path}")
    return enriched


def main() -> None:
    run_bowler_typer()


if __name__ == "__main__":
    main()
