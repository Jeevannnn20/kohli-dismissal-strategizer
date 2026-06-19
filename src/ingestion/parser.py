"""Parse Cricsheet JSON files into Kohli delivery-level parquet data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_path: Path | None = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def project_root_from_config(config_path: Path | None = None) -> Path:
    return Path(config_path).resolve().parent if config_path else PROJECT_ROOT


def match_phase(over: int, format_name: str, match_type: str | None = None) -> str:
    normalized_format = format_name.lower()
    normalized_match_type = (match_type or "").lower()

    if normalized_format == "tests" or normalized_match_type == "test":
        if over <= 14:
            return "opening"
        if over <= 59:
            return "middle"
        return "tail"

    if normalized_format == "odis":
        if over <= 5:
            return "powerplay"
        if over <= 39:
            return "middle"
        return "death"

    if over <= 5:
        return "powerplay"
    if over <= 14:
        return "middle"
    return "death"


def first_date(info: dict[str, Any]) -> str | None:
    dates = info.get("dates") or []
    return str(dates[0]) if dates else None


def first_dismissal(delivery: dict[str, Any]) -> dict[str, Any] | None:
    dismissals = delivery.get("wickets") or delivery.get("dismissals") or []
    return dismissals[0] if dismissals else None


def first_fielder(dismissal: dict[str, Any] | None) -> str | None:
    if not dismissal:
        return None
    fielders = dismissal.get("fielders") or []
    if not fielders:
        return None
    first = fielders[0]
    if isinstance(first, dict):
        return first.get("name")
    return str(first)


def dismissed_player(dismissal: dict[str, Any] | None) -> str | None:
    if not dismissal:
        return None
    return dismissal.get("player_out") or dismissal.get("player")


def parse_match_file(match_file: Path, format_name: str, kohli_id: str) -> list[dict]:
    with match_file.open("r", encoding="utf-8") as file:
        match = json.load(file)

    info = match.get("info", {})
    match_type = info.get("match_type")
    rows: list[dict] = []
    delivery_index = 0

    for innings_number, innings in enumerate(match.get("innings", []), start=1):
        if innings.get("super_over"):
            continue

        for over_block in innings.get("overs", []):
            over_number = int(over_block["over"])
            for ball_index, delivery in enumerate(
                over_block.get("deliveries", []), start=1
            ):
                dismissal = first_dismissal(delivery)
                player_dismissed = dismissed_player(dismissal)
                batter = delivery.get("batter")

                if batter != kohli_id and player_dismissed != kohli_id:
                    delivery_index += 1
                    continue

                extras_detail = delivery.get("extras") or {}
                runs = delivery.get("runs") or {}
                rows.append(
                    {
                        "match_id": match_file.stem,
                        "date": first_date(info),
                        "venue": info.get("venue"),
                        "city": info.get("city"),
                        "format": format_name,
                        "match_type": match_type,
                        "teams": info.get("teams"),
                        "season": str(info.get("season"))
                        if info.get("season") is not None
                        else None,
                        "innings_number": innings_number,
                        "over": over_number,
                        "ball": ball_index,
                        "delivery_index": delivery_index,
                        "batter": batter,
                        "bowler": delivery.get("bowler"),
                        "runs_off_bat": int(runs.get("batter", 0)),
                        "extras": int(runs.get("extras", 0)),
                        "total_runs": int(runs.get("total", 0)),
                        "wide": "wides" in extras_detail,
                        "noball": "noballs" in extras_detail,
                        "dismissal_kind": dismissal.get("kind") if dismissal else None,
                        "player_dismissed": player_dismissed,
                        "fielder": first_fielder(dismissal),
                        "is_dismissal": 1 if player_dismissed == kohli_id else 0,
                        "match_phase": match_phase(
                            over_number, format_name, str(match_type)
                        ),
                    }
                )
                delivery_index += 1

    return rows


def parse_all_matches(config_path: Path | None = None) -> pd.DataFrame:
    config = load_config(config_path)
    project_root = project_root_from_config(config_path)
    raw_data_dir = project_root / Path(config["paths"]["raw_data_dir"])
    output_path = project_root / Path(config["paths"]["kohli_deliveries"])
    kohli_id = config["player"]["cricsheet_id"]
    all_rows: list[dict] = []

    match_files: list[tuple[str, Path]] = []
    for format_name in config["cricsheet"]["formats"]:
        matches_dir = raw_data_dir / format_name / "matches"
        match_files.extend((format_name, path) for path in sorted(matches_dir.glob("*.json")))

    for format_name, match_file in tqdm(match_files, desc="Parsing Cricsheet matches"):
        all_rows.extend(parse_match_file(match_file, format_name, kohli_id))

    deliveries = pd.DataFrame(all_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    deliveries.to_parquet(output_path, compression="snappy", index=False)

    print(f"[parse] wrote {output_path}")
    print(f"[parse] shape: {deliveries.shape}")
    print("[parse] sample:")
    print(deliveries.head(5))
    return deliveries


def main() -> None:
    parse_all_matches()


if __name__ == "__main__":
    main()
