"""Run the full Phase 2 feature enrichment pipeline."""

from __future__ import annotations

from pathlib import Path

from src.enrichment.bowler_typer import run_bowler_typer
from src.enrichment.venue_encoder import run_venue_encoder
from src.enrichment.weather_fetcher import run_weather_fetcher
from src.ingestion.cleaner import clean_kohli_deliveries


def run_pipeline(config_path: Path | None = None) -> None:
    print("[enrichment] Step 1/4: cleaning invalid non-striker dismissals")
    clean_kohli_deliveries(config_path)

    print("[enrichment] Step 2/4: adding bowler types")
    run_bowler_typer(config_path)

    print("[enrichment] Step 3/4: adding venue metadata")
    run_venue_encoder(config_path)

    print("[enrichment] Step 4/4: adding historical weather")
    run_weather_fetcher(config_path)

    print("[enrichment] Phase 2 enrichment complete")


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
