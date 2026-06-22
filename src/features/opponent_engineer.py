"""Add opponent context features and rebuild the feature matrix."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder

from src.features.engineer import (
    TARGET_COLUMN,
    add_bowler_history,
    add_innings_context,
    add_kohli_form,
    fill_numeric_nulls,
    fit_label_encoders,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
RCB_ALIASES = {"Royal Challengers Bangalore", "Royal Challengers Bengaluru"}
INDIA_TEAM = "India"


def load_config(config_path: Path | None = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def project_root_from_config(config_path: Path | None = None) -> Path:
    return Path(config_path).resolve().parent if config_path else PROJECT_ROOT


def parse_teams(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(team) for team in value]
    if not isinstance(value, str) and isinstance(value, Sequence):
        return [str(team) for team in value]
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, (list, tuple)):
            return [str(team) for team in converted]
    if pd.isna(value):
        return []
    text = str(value)
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return [part.strip() for part in text.split(",") if part.strip()]
    if isinstance(parsed, (list, tuple)):
        return [str(team) for team in parsed]
    return [str(parsed)]


def extract_opposing_team(teams: Any, format_name: str) -> str:
    team_list = parse_teams(teams)
    if not team_list:
        return "unknown"

    if str(format_name).lower() == "ipl":
        for team in team_list:
            if team not in RCB_ALIASES:
                return team
        return team_list[-1]

    for team in team_list:
        if team != INDIA_TEAM:
            return team
    return team_list[-1]


def add_opponent_context(deliveries: pd.DataFrame) -> pd.DataFrame:
    enriched = deliveries.copy()
    enriched["opposing_team"] = [
        extract_opposing_team(teams, format_name)
        for teams, format_name in zip(enriched["teams"], enriched["format"], strict=False)
    ]
    enriched["is_home_match"] = enriched["country"].fillna("").eq("India").astype(int)
    return enriched


def engineer_opponent_features_frame(
    deliveries: pd.DataFrame,
    format_encoding: dict[str, int],
    player_id: str = "V Kohli",
) -> tuple[pd.DataFrame, dict[str, LabelEncoder], list[str]]:
    df = deliveries.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "match_id", "innings_number", "delivery_index"]).reset_index(drop=True)
    df["ball"] = (pd.to_numeric(df["ball"], errors="coerce").fillna(1) - 1).clip(0, 5).astype(int)
    df, encoders = fit_label_encoders(df, ["bowler_type", "pitch_type", "match_phase"])

    opponent_encoder = LabelEncoder()
    df["opposing_team"] = df["opposing_team"].fillna("unknown").astype(str)
    df["opposing_team_encoded"] = opponent_encoder.fit_transform(df["opposing_team"])
    encoders["opposing_team"] = opponent_encoder

    df["format_encoded"] = df["format"].map(format_encoding).fillna(-1).astype(int)
    df["is_home_match"] = pd.to_numeric(df["is_home_match"], errors="coerce").fillna(0).astype(int)
    df = add_innings_context(df, player_id)
    df = add_bowler_history(df)
    df = add_kohli_form(df)

    feature_columns = [
        "bowler_type_encoded",
        "pitch_type_encoded",
        "match_phase_encoded",
        "over",
        "ball",
        "is_wide",
        "is_noball",
        "format_encoded",
        "temperature",
        "humidity",
        "cloud_cover",
        "wind_speed",
        "precipitation",
        "balls_faced_so_far",
        "runs_scored_so_far",
        "strike_rate_so_far",
        "wickets_fallen_before",
        "bowler_career_dismissals_of_kohli",
        "bowler_career_balls_to_kohli",
        "kohli_last5_avg",
        "kohli_last5_sr",
        "opposing_team_encoded",
        "is_home_match",
    ]
    keep_columns = ["match_id", "date", *feature_columns, TARGET_COLUMN]
    features = df[keep_columns].copy()
    features = fill_numeric_nulls(features, feature_columns + [TARGET_COLUMN])
    if features.isnull().sum().sum() != 0:
        raise ValueError("Opponent feature engineering produced null values")
    return features, encoders, feature_columns


def run_opponent_engineer(config_path: Path | None = None) -> pd.DataFrame:
    config = load_config(config_path)
    project_root = project_root_from_config(config_path)
    input_path = project_root / Path(config["paths"]["kohli_deliveries_enriched"])
    deliveries_output = project_root / Path(config["paths"]["kohli_deliveries_opponent"])
    features_output = project_root / Path(config["paths"]["kohli_features_v2"])
    encoders_path = project_root / Path(config["paths"]["label_encoders"])

    deliveries = pd.read_parquet(input_path)
    deliveries = add_opponent_context(deliveries)
    deliveries_output.parent.mkdir(parents=True, exist_ok=True)
    deliveries.to_parquet(deliveries_output, compression="snappy", index=False)
    print(f"[opponent_engineer] opponent deliveries shape: {deliveries.shape}")
    print(deliveries[["match_id", "format", "teams", "opposing_team", "is_home_match"]].head(5))

    features, encoders, feature_columns = engineer_opponent_features_frame(
        deliveries,
        config["features"]["format_encoding"],
        config["player"]["cricsheet_id"],
    )
    features_output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(features_output, compression="snappy", index=False)
    encoders_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoders, encoders_path)

    print(f"[opponent_engineer] feature matrix shape: {features.shape}")
    print(f"[opponent_engineer] feature list: {feature_columns}")
    print("[opponent_engineer] null check:")
    print(features.isnull().sum())
    print(features.head(5))
    return features


def main() -> None:
    run_opponent_engineer()


if __name__ == "__main__":
    main()
