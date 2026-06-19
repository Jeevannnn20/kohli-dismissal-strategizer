"""Enrich Kohli delivery rows with venue metadata."""

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


def enrich_venues(deliveries: pd.DataFrame, venues: pd.DataFrame) -> pd.DataFrame:
    lookup = venues[
        ["venue", "country", "pitch_type", "latitude", "longitude"]
    ].drop_duplicates("venue")
    enriched = deliveries.merge(lookup, how="left", on="venue")
    enriched["pitch_type"] = enriched["pitch_type"].fillna("unknown")
    enriched["country"] = enriched["country"].fillna("unknown")
    enriched["latitude"] = enriched["latitude"].where(enriched["latitude"].notna(), None)
    enriched["longitude"] = enriched["longitude"].where(
        enriched["longitude"].notna(), None
    )
    return enriched


def run_venue_encoder(config_path: Path | None = None) -> pd.DataFrame:
    config = load_config(config_path)
    project_root = project_root_from_config(config_path)
    deliveries_path = project_root / Path(config["paths"]["kohli_deliveries_v2"])
    venues_path = project_root / Path(config["paths"]["venue_metadata"])
    output_path = project_root / Path(config["paths"]["kohli_deliveries_v3"])

    deliveries = pd.read_parquet(deliveries_path)
    venues = pd.read_csv(venues_path)
    enriched = enrich_venues(deliveries, venues)

    dismissals = enriched[enriched["is_dismissal"] == 1]
    matched_pct = 0.0
    if not dismissals.empty:
        matched_pct = float((dismissals["pitch_type"] != "unknown").mean() * 100)
    print(f"[venue_encoder] matched dismissal venues: {matched_pct:.2f}%")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(output_path, compression="snappy", index=False)
    print(f"[venue_encoder] wrote {output_path}")
    return enriched


def main() -> None:
    run_venue_encoder()


if __name__ == "__main__":
    main()
