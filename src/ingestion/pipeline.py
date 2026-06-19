"""Run the full Phase 1 data ingestion pipeline."""

from __future__ import annotations

from pathlib import Path

from src.ingestion.downloader import run_downloader
from src.ingestion.extractor import extract_dismissals
from src.ingestion.parser import parse_all_matches


def run_pipeline(config_path: Path | None = None) -> None:
    print("[pipeline] Step 1/3: downloading and extracting Cricsheet data")
    run_downloader(config_path)

    print("[pipeline] Step 2/3: parsing Kohli deliveries")
    parse_all_matches(config_path)

    print("[pipeline] Step 3/3: extracting Kohli dismissals")
    extract_dismissals(config_path)

    print("[pipeline] Phase 1 ingestion complete")


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
