"""Engineer model-ready features from enriched Kohli delivery data."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
TARGET_COLUMN = "is_dismissal"


def load_config(config_path: Path | None = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def project_root_from_config(config_path: Path | None = None) -> Path:
    return Path(config_path).resolve().parent if config_path else PROJECT_ROOT


def fit_label_encoders(
    df: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    encoded = df.copy()
    encoders: dict[str, LabelEncoder] = {}
    for column in columns:
        encoder = LabelEncoder()
        values = encoded[column].fillna("unknown").astype(str)
        encoded[f"{column}_encoded"] = encoder.fit_transform(values)
        encoders[column] = encoder
    return encoded, encoders


def add_innings_context(df: pd.DataFrame, player_id: str) -> pd.DataFrame:
    enriched = df.sort_values(["date", "match_id", "innings_number", "delivery_index"]).copy()
    enriched["is_wide"] = enriched["wide"].astype(int)
    enriched["is_noball"] = enriched["noball"].astype(int)
    enriched["legal_ball"] = (
        (enriched["batter"].eq(player_id))
        & ~enriched["wide"].astype(bool)
        & ~enriched["noball"].astype(bool)
    ).astype(int)
    enriched["kohli_runs_on_ball"] = np.where(
        enriched["batter"].eq(player_id), enriched["runs_off_bat"], 0
    )

    group_cols = ["match_id", "innings_number"]
    grouped = enriched.groupby(group_cols, sort=False)
    enriched["balls_faced_so_far"] = grouped["legal_ball"].cumsum() - enriched["legal_ball"]
    enriched["runs_scored_so_far"] = (
        grouped["kohli_runs_on_ball"].cumsum() - enriched["kohli_runs_on_ball"]
    )
    enriched["strike_rate_so_far"] = np.where(
        enriched["balls_faced_so_far"] > 0,
        enriched["runs_scored_so_far"] / enriched["balls_faced_so_far"] * 100,
        0,
    )
    enriched["wickets_fallen_before"] = grouped[TARGET_COLUMN].cumsum() - enriched[TARGET_COLUMN]
    return enriched


def add_bowler_history(df: pd.DataFrame) -> pd.DataFrame:
    match_bowler = (
        df.groupby(["bowler", "match_id", "date"], dropna=False)
        .agg(
            match_dismissals=(TARGET_COLUMN, "sum"),
            match_balls=("legal_ball", "sum"),
        )
        .reset_index()
        .sort_values(["bowler", "date", "match_id"])
    )
    grouped = match_bowler.groupby("bowler", dropna=False)
    match_bowler["bowler_career_dismissals_of_kohli"] = (
        grouped["match_dismissals"].cumsum() - match_bowler["match_dismissals"]
    )
    match_bowler["bowler_career_balls_to_kohli"] = (
        grouped["match_balls"].cumsum() - match_bowler["match_balls"]
    )
    return df.merge(
        match_bowler[
            [
                "bowler",
                "match_id",
                "bowler_career_dismissals_of_kohli",
                "bowler_career_balls_to_kohli",
            ]
        ],
        on=["bowler", "match_id"],
        how="left",
    )


def add_kohli_form(df: pd.DataFrame) -> pd.DataFrame:
    innings = (
        df.groupby(["match_id", "date", "innings_number"], sort=False)
        .agg(
            innings_runs=("kohli_runs_on_ball", "sum"),
            innings_balls=("legal_ball", "sum"),
            innings_dismissed=(TARGET_COLUMN, "max"),
        )
        .reset_index()
        .sort_values(["date", "match_id", "innings_number"])
    )
    innings["innings_sr"] = np.where(
        innings["innings_balls"] > 0,
        innings["innings_runs"] / innings["innings_balls"] * 100,
        0,
    )

    match_order = innings[["match_id", "date"]].drop_duplicates().sort_values(["date", "match_id"])
    form_rows: list[dict] = []
    prior_innings = innings.iloc[0:0].copy()
    for row in match_order.itertuples(index=False):
        last5 = prior_innings.tail(5)
        if last5.empty:
            avg = 0.0
            sr = 0.0
        else:
            dismissals = int(last5["innings_dismissed"].sum())
            avg = float(last5["innings_runs"].sum() / dismissals) if dismissals else float(last5["innings_runs"].mean())
            sr = float(last5["innings_sr"].mean())
        form_rows.append({"match_id": row.match_id, "kohli_last5_avg": avg, "kohli_last5_sr": sr})
        prior_innings = pd.concat(
            [prior_innings, innings[innings["match_id"].eq(row.match_id)]],
            ignore_index=True,
        )

    return df.merge(pd.DataFrame(form_rows), on="match_id", how="left")


def fill_numeric_nulls(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    filled = df.copy()
    for column in columns:
        filled[column] = pd.to_numeric(filled[column], errors="coerce")
        median = filled[column].median()
        if pd.isna(median):
            median = 0
        filled[column] = filled[column].fillna(median)
    return filled


def engineer_features_frame(
    deliveries: pd.DataFrame,
    format_encoding: dict[str, int],
    player_id: str = "V Kohli",
) -> tuple[pd.DataFrame, dict[str, LabelEncoder], list[str]]:
    df = deliveries.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "match_id", "innings_number", "delivery_index"]).reset_index(drop=True)
    df["ball"] = (pd.to_numeric(df["ball"], errors="coerce").fillna(1) - 1).clip(0, 5).astype(int)
    df, encoders = fit_label_encoders(df, ["bowler_type", "pitch_type", "match_phase"])
    df["format_encoded"] = df["format"].map(format_encoding).fillna(-1).astype(int)
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
    ]
    keep_columns = ["match_id", "date", *feature_columns, TARGET_COLUMN]
    features = df[keep_columns].copy()
    features = fill_numeric_nulls(features, feature_columns + [TARGET_COLUMN])
    if features.isnull().sum().sum() != 0:
        raise ValueError("Feature engineering produced null values")
    return features, encoders, feature_columns


def run_engineer(config_path: Path | None = None) -> pd.DataFrame:
    config = load_config(config_path)
    project_root = project_root_from_config(config_path)
    input_path = project_root / Path(config["paths"]["kohli_deliveries_enriched"])
    output_path = project_root / Path(config["paths"]["kohli_features"])
    encoders_path = project_root / Path(config["paths"]["label_encoders"])
    format_encoding = config["features"]["format_encoding"]

    deliveries = pd.read_parquet(input_path)
    features, encoders, feature_columns = engineer_features_frame(
        deliveries, format_encoding, config["player"]["cricsheet_id"]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoders_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, compression="snappy", index=False)
    joblib.dump(encoders, encoders_path)

    positives = int(features[TARGET_COLUMN].sum())
    negatives = int(len(features) - positives)
    print(f"[engineer] final feature matrix shape: {features.shape}")
    print(f"[engineer] feature names: {feature_columns}")
    print("[engineer] null count per column:")
    print(features.isnull().sum())
    print(f"[engineer] class balance: positives={positives}, negatives={negatives}")
    return features


def main() -> None:
    run_engineer()


if __name__ == "__main__":
    main()
