"""Extract Kohli dismissal records from parsed delivery-level data."""

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


def add_pre_dismissal_context(deliveries: pd.DataFrame, kohli_id: str) -> pd.DataFrame:
    deliveries = deliveries.sort_values(
        ["match_id", "innings_number", "delivery_index"]
    ).copy()
    dismissal_rows: list[pd.Series] = []

    for _, innings in deliveries.groupby(["match_id", "innings_number"], sort=False):
        prior_balls_faced = 0
        prior_runs_scored = 0

        for _, row in innings.iterrows():
            if row["is_dismissal"] == 1:
                dismissal = row.copy()
                dismissal["balls_faced_before_dismissal"] = prior_balls_faced
                dismissal["runs_scored_before_dismissal"] = prior_runs_scored
                dismissal_rows.append(dismissal)

            if (
                row["batter"] == kohli_id
                and not bool(row.get("wide", False))
                and not bool(row.get("noball", False))
            ):
                prior_balls_faced += 1
                prior_runs_scored += int(row["runs_off_bat"])

    if not dismissal_rows:
        return deliveries.iloc[0:0].assign(
            balls_faced_before_dismissal=pd.Series(dtype="int64"),
            runs_scored_before_dismissal=pd.Series(dtype="int64"),
        )

    dismissals = pd.DataFrame(dismissal_rows)
    dismissals["balls_faced_before_dismissal"] = dismissals[
        "balls_faced_before_dismissal"
    ].astype(int)
    dismissals["runs_scored_before_dismissal"] = dismissals[
        "runs_scored_before_dismissal"
    ].astype(int)
    return dismissals


def extract_dismissals(config_path: Path | None = None) -> pd.DataFrame:
    config = load_config(config_path)
    project_root = project_root_from_config(config_path)
    kohli_id = config["player"]["cricsheet_id"]
    deliveries_path = project_root / Path(config["paths"]["kohli_deliveries"])
    output_path = project_root / Path(config["paths"]["kohli_dismissals"])

    deliveries = pd.read_parquet(deliveries_path)
    dismissals = add_pre_dismissal_context(deliveries, kohli_id)
    dismissals.to_parquet(output_path, compression="snappy", index=False)

    print(f"[extract] wrote {output_path}")
    print(f"[extract] shape: {dismissals.shape}")
    print("[extract] sample:")
    print(dismissals.head(5))
    print("[extract] dismissals per format:")
    print(dismissals.groupby("format").size().sort_values(ascending=False))
    print("[extract] most common dismissal kinds:")
    print(dismissals["dismissal_kind"].value_counts().head(10))
    print("[extract] top 10 bowlers:")
    print(dismissals["bowler"].value_counts().head(10))
    return dismissals


def main() -> None:
    extract_dismissals()


if __name__ == "__main__":
    main()
