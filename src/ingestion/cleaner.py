"""Clean invalid Kohli delivery rows before enrichment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.ingestion.extractor import extract_dismissals


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_path: Path | None = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def project_root_from_config(config_path: Path | None = None) -> Path:
    return Path(config_path).resolve().parent if config_path else PROJECT_ROOT


def clean_kohli_deliveries(config_path: Path | None = None) -> pd.DataFrame:
    config = load_config(config_path)
    project_root = project_root_from_config(config_path)
    kohli_id = config["player"]["cricsheet_id"]
    deliveries_path = project_root / Path(config["paths"]["kohli_deliveries"])

    deliveries = pd.read_parquet(deliveries_path)
    invalid_mask = (deliveries["player_dismissed"] == kohli_id) & (
        deliveries["batter"] != kohli_id
    )
    dropped_rows = int(invalid_mask.sum())
    cleaned = deliveries.loc[~invalid_mask].copy()
    cleaned.to_parquet(deliveries_path, compression="snappy", index=False)

    print(f"[clean] dropped {dropped_rows} non-striker Kohli dismissal rows")
    print("[clean] regenerating dismissals from cleaned deliveries")
    extract_dismissals(config_path)
    return cleaned


def main() -> None:
    clean_kohli_deliveries()


if __name__ == "__main__":
    main()
