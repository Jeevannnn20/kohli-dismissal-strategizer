"""Download and extract Cricsheet bulk JSON archives."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import requests
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_path: Path | None = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def project_root_from_config(config_path: Path | None = None) -> Path:
    return Path(config_path).resolve().parent if config_path else PROJECT_ROOT


def run_downloader(config_path: Path | None = None) -> dict[str, int]:
    config = load_config(config_path)
    project_root = project_root_from_config(config_path)
    raw_data_dir = project_root / Path(config["paths"]["raw_data_dir"])
    base_url = config["cricsheet"]["base_download_url"].rstrip("/")
    extracted_counts: dict[str, int] = {}

    for format_name, format_config in config["cricsheet"]["formats"].items():
        zip_file_name = format_config["zip_file"]
        format_dir = raw_data_dir / format_name
        matches_dir = format_dir / "matches"
        zip_path = format_dir / zip_file_name

        format_dir.mkdir(parents=True, exist_ok=True)
        matches_dir.mkdir(parents=True, exist_ok=True)

        existing_json_files = list(matches_dir.glob("*.json"))
        if zip_path.exists():
            print(f"[download] {format_name}: using existing archive {zip_path}")
        else:
            url = f"{base_url}/{zip_file_name}"
            print(f"[download] {format_name}: downloading {url}")
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            zip_path.write_bytes(response.content)

        if existing_json_files:
            extracted_counts[format_name] = len(existing_json_files)
            print(
                f"[extract] {format_name}: {len(existing_json_files)} files already extracted"
            )
            continue

        with ZipFile(zip_path) as archive:
            json_members = [
                member for member in archive.namelist() if member.endswith(".json")
            ]
            archive.extractall(matches_dir, members=json_members)

        extracted_counts[format_name] = len(json_members)
        print(f"[extract] {format_name}: extracted {len(json_members)} JSON files")

    return extracted_counts


def main() -> None:
    run_downloader()


if __name__ == "__main__":
    main()
