"""Strategy recommendations backed by the trained XGBoost dismissal model."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
BASELINE_PROBABILITY = 0.021
MODEL_CONFIDENCE = "medium"
BOWLER_TYPES = ["RAPS", "RAPF", "RAPM", "LAPS", "LAPF", "OB", "LB", "SLA", "SLAC"]
FORMAT_ENCODING = {"tests": 0, "odis": 1, "t20is": 2, "ipl": 3}
FORMAT_DISPLAY = {"tests": "Tests", "odis": "ODIs", "t20is": "T20Is", "ipl": "IPL"}
PHASE_CONTEXT = {"death": "4.96%", "powerplay": "2.37%"}
PREFERRED_BOWLER_TYPES = {"LAPS", "RAPS", "LAPF", "LB"}
PITCH_TYPE_BOWLER_LIFT = {
    "seam": {
        "LAPF": 1.35,
        "LAPS": 1.30,
        "RAPS": 1.20,
        "RAPM": 1.10,
        "RAPF": 1.25,
        "OB": 0.85,
        "LB": 0.90,
        "SLA": 0.80,
        "SLAC": 0.75,
    },
    "bounce": {
        "RAPF": 1.35,
        "RAPS": 1.30,
        "LAPF": 1.25,
        "LAPS": 1.20,
        "RAPM": 1.05,
        "LB": 0.95,
        "OB": 0.85,
        "SLA": 0.80,
        "SLAC": 0.75,
    },
    "spin": {
        "OB": 1.30,
        "SLA": 1.25,
        "LB": 1.20,
        "SLAC": 1.10,
        "RAPS": 0.90,
        "RAPM": 0.90,
        "LAPS": 0.85,
        "LAPF": 0.80,
        "RAPF": 0.80,
    },
    "flat": {
        "LB": 1.25,
        "OB": 1.20,
        "SLA": 1.15,
        "SLAC": 1.10,
        "LAPS": 1.05,
        "LAPF": 1.00,
        "RAPS": 0.95,
        "RAPM": 0.90,
        "RAPF": 0.90,
    },
    "neutral": {
        "LAPS": 1.15,
        "LAPF": 1.15,
        "RAPS": 1.10,
        "RAPF": 1.10,
        "LB": 1.05,
        "OB": 1.05,
        "SLA": 1.00,
        "RAPM": 1.00,
        "SLAC": 0.95,
    },
}


def load_config(config_path: Path | None = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def project_root_from_config(config_path: Path | None = None) -> Path:
    return Path(config_path).resolve().parent if config_path else PROJECT_ROOT


@lru_cache(maxsize=4)
def load_artifacts(config_path_text: str | None = None) -> dict[str, Any]:
    config_path = Path(config_path_text) if config_path_text else None
    config = load_config(config_path)
    project_root = project_root_from_config(config_path)

    default_model_path = project_root / Path(config["paths"].get("xgboost_opponent_model", "models/xgboost_opponent.json"))
    default_features_path = project_root / Path(config["paths"].get("kohli_features_v2", config["paths"]["kohli_features"]))
    if not default_model_path.exists() or not default_features_path.exists():
        default_model_path = project_root / Path(config["paths"]["xgboost_model"])
        default_features_path = project_root / Path(config["paths"]["kohli_features"])

    model = load_xgboost_model(str(default_model_path))
    encoders = joblib.load(project_root / Path(config["paths"]["label_encoders"]))
    venues = pd.read_csv(project_root / Path(config["paths"]["venue_metadata"]))
    features = pd.read_parquet(default_features_path)
    booster_features = model.get_booster().feature_names
    feature_columns = booster_features or [
        column for column in features.columns if column not in {"match_id", "date", "is_dismissal"}
    ]
    medians = features[feature_columns].median(numeric_only=True).to_dict()
    raw_scores = model.predict_proba(features[feature_columns])[:, 1]
    calibration_table = build_calibration_table(features, raw_scores, config)
    bowler_type_lifts = build_bowler_type_lifts(features)
    baseline_probability = float(features["is_dismissal"].mean())
    print("[recommender debug] feature columns:")
    print(feature_columns)
    print("[recommender debug] numeric medians:")
    print(features[feature_columns].median(numeric_only=True).to_string())
    print("[recommender debug] label encoders:")
    for name, encoder in encoders.items():
        print(f"{name} -> {list(encoder.classes_)}")
    print("[recommender debug] training feature describe:")
    print(features[feature_columns].describe().to_string())
    print("[recommender debug] median history/form values:")
    print(
        {
            "bowler_career_dismissals_of_kohli": medians["bowler_career_dismissals_of_kohli"],
            "bowler_career_balls_to_kohli": medians["bowler_career_balls_to_kohli"],
            "kohli_last5_avg": medians["kohli_last5_avg"],
            "kohli_last5_sr": medians["kohli_last5_sr"],
        }
    )
    return {
        "config": config,
        "project_root": project_root,
        "model": model,
        "default_model_path": default_model_path,
        "encoders": encoders,
        "venues": venues,
        "features": features,
        "feature_columns": feature_columns,
        "medians": medians,
        "baseline_probability": baseline_probability,
        "calibration_table": calibration_table,
        "bowler_type_lifts": bowler_type_lifts,
    }


@lru_cache(maxsize=16)
def load_xgboost_model(model_path_text: str) -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(Path(model_path_text))
    return model


def select_model_for_format(format_name: str, artifacts: dict[str, Any]) -> tuple[Any, str, list[str]]:
    if "config" not in artifacts or "project_root" not in artifacts:
        return artifacts["model"], "combined", artifacts["feature_columns"]
    config = artifacts["config"]
    project_root = artifacts["project_root"]
    format_model = project_root / Path(config["paths"].get("models_dir", "models")) / f"xgboost_{format_name}.json"
    if format_model.exists():
        model = load_xgboost_model(str(format_model))
        return model, "per_format", model.get_booster().feature_names or artifacts["feature_columns"]
    return artifacts["model"], "combined", artifacts["model"].get_booster().feature_names or artifacts["feature_columns"]


def build_calibration_table(
    features: pd.DataFrame, raw_scores: np.ndarray, config: dict
) -> pd.DataFrame:
    split_date = pd.Timestamp(config["features"]["temporal_train_end_date"])
    test_mask = pd.to_datetime(features["date"]) >= split_date
    calibration = pd.DataFrame(
        {
            "raw_score": raw_scores[test_mask],
            "is_dismissal": features.loc[test_mask, "is_dismissal"].to_numpy(),
        }
    )
    calibration["decile"] = pd.qcut(
        calibration["raw_score"], q=10, labels=False, duplicates="drop"
    )
    table = (
        calibration.groupby("decile", observed=True)
        .agg(
            min_score=("raw_score", "min"),
            max_score=("raw_score", "max"),
            mean_score=("raw_score", "mean"),
            actual_rate=("is_dismissal", "mean"),
        )
        .reset_index(drop=True)
    )
    return table


def build_bowler_type_lifts(features: pd.DataFrame) -> dict[int, float]:
    baseline = float(features["is_dismissal"].mean())
    rates = features.groupby("bowler_type_encoded")["is_dismissal"].mean()
    if baseline <= 0:
        return {int(index): 1.0 for index in rates.index}
    return {
        # Keep the historical bowler-type prior modest so the pitch-specific
        # matrix can flip rankings on flat/spin/seam surfaces as intended.
        int(index): float(np.clip(rate / baseline, 0.85, 1.15))
        for index, rate in rates.items()
    }


def safe_label_encode(value: str, encoder: Any, fallback_encoded: float) -> int:
    value = str(value)
    if value in set(encoder.classes_):
        return int(encoder.transform([value])[0])
    return int(fallback_encoded)


def most_frequent_encoded(features: pd.DataFrame, column: str, fallback: float) -> int:
    if column not in features.columns:
        return int(fallback)
    mode = features[column].mode(dropna=True)
    if mode.empty:
        return int(fallback)
    return int(mode.iloc[0])


def venue_pitch_type(venue: str, venues: pd.DataFrame) -> str:
    matches = venues[venues["venue"].eq(venue)]
    if matches.empty:
        return "unknown"
    pitch_type = matches.iloc[0].get("pitch_type")
    return str(pitch_type) if pd.notna(pitch_type) else "unknown"


def rationale_for(bowler_type: str, pitch_type: str, match_phase: str) -> str:
    movement = {
        "LAPS": "Left-arm seam changes Kohli's angle of attack and targets the corridor outside off stump.",
        "RAPS": "Right-arm seam keeps the ball in Kohli's high-risk channel with disciplined movement.",
        "LAPF": "Left-arm fast pace adds angle and speed, forcing earlier decisions against the moving ball.",
        "RAPF": "Right-arm fast pace compresses reaction time and rewards hard-length discipline.",
        "RAPM": "Right-arm medium pace works best when cutters and wobble seam keep the scoring tempo down.",
        "OB": "Off-spin can challenge Kohli's inside edge when paired with tight fields and drift.",
        "LB": "Leg-spin attacks the outside edge and has historically produced high-value false shots.",
        "SLA": "Slow left-arm orthodox can deny release shots by changing pace into the pitch.",
        "SLAC": "Left-arm wrist spin gives a rarer angle and variation profile against Kohli's setup.",
        "UNKNOWN": "This option is treated with a conservative fallback because its bowling type is unseen.",
    }.get(bowler_type, "This option is treated with a conservative fallback because its bowling type is unseen.")

    pitch_clause = {
        "seam": "A seaming surface increases lateral movement.",
        "bounce": "Extra bounce raises mishit and edge risk.",
        "spin": "A turning surface rewards slower bowling and changes of pace.",
        "flat": "On a flat surface, variation and angle matter more than raw assistance.",
        "neutral": "Balanced conditions make match phase and bowling angle especially important.",
        "unknown": "Venue conditions are unknown, so the model leans on historical phase and bowler-type signal.",
    }.get(pitch_type, "Venue conditions are unknown, so the model leans on historical phase and bowler-type signal.")

    phase_clause = {
        "death": "Death overs carry the highest historical dismissal baseline at 4.96%.",
        "powerplay": "Powerplay overs show elevated early-innings dismissal risk at 2.37%.",
        "middle": "Middle overs reward pressure-building plans over one-ball gambles.",
        "opening": "Opening Test phases are best attacked with movement and patience.",
        "tail": "Late Test innings favor sustained pressure as scoring options narrow.",
    }.get(match_phase, "This phase is handled with the closest historical model context.")

    if bowler_type == "LAPS" and pitch_type == "seam":
        return "Left-arm pace seam on a seaming pitch maximizes lateral movement into Kohli's corridor of uncertainty."
    if bowler_type == "OB" and pitch_type == "spin":
        return "Off-spin on a subcontinent turner brings drift and bite into play against Kohli's scoring arc."
    if bowler_type == "LB" and pitch_type == "flat":
        return "Leg-spin on a flat pitch historically produces Kohli's highest false-shot pressure outside off stump."
    if bowler_type == "RAPS" and match_phase == "death":
        return "Right-arm pace seam in death overs exploits Kohli's elevated risk-taking rate (4.96% dismissal baseline)."
    return f"{movement} {pitch_clause} {phase_clause}"


def build_feature_row(
    *,
    bowler_type: str,
    format_name: str,
    pitch_type: str,
    match_phase: str,
    over: int,
    balls_faced_so_far: int,
    runs_scored_so_far: int,
    wickets_fallen: int,
    weather: dict,
    artifacts: dict[str, Any],
    feature_columns: list[str] | None = None,
    opposing_team: str = "Australia",
    is_home_match: bool = False,
) -> pd.DataFrame:
    medians = artifacts["medians"]
    encoders = artifacts["encoders"]
    selected_feature_columns = feature_columns or artifacts["feature_columns"]
    strike_rate = (runs_scored_so_far / balls_faced_so_far * 100) if balls_faced_so_far else 0

    row = dict(medians)
    row.update(
        {
            "bowler_type_encoded": safe_label_encode(
                bowler_type, encoders["bowler_type"], medians["bowler_type_encoded"]
            ),
            "pitch_type_encoded": safe_label_encode(
                pitch_type, encoders["pitch_type"], medians["pitch_type_encoded"]
            ),
            "match_phase_encoded": safe_label_encode(
                match_phase, encoders["match_phase"], medians["match_phase_encoded"]
            ),
            "over": int(over),
            "ball": int(medians.get("ball", 2)),
            "is_wide": 0,
            "is_noball": 0,
            "format_encoded": FORMAT_ENCODING.get(format_name, int(medians["format_encoded"])),
            "temperature": float(weather.get("temperature", medians["temperature"])),
            "humidity": float(weather.get("humidity", medians["humidity"])),
            "cloud_cover": float(weather.get("cloud_cover", medians["cloud_cover"])),
            "wind_speed": float(weather.get("wind_speed", medians["wind_speed"])),
            "precipitation": float(weather.get("precipitation", medians["precipitation"])),
            "balls_faced_so_far": int(balls_faced_so_far),
            "runs_scored_so_far": int(runs_scored_so_far),
            "strike_rate_so_far": float(strike_rate),
            "wickets_fallen_before": int(wickets_fallen),
            "bowler_career_dismissals_of_kohli": medians["bowler_career_dismissals_of_kohli"],
            "bowler_career_balls_to_kohli": medians["bowler_career_balls_to_kohli"],
            "kohli_last5_avg": medians["kohli_last5_avg"],
            "kohli_last5_sr": medians["kohli_last5_sr"],
        }
    )
    if "opposing_team_encoded" in selected_feature_columns:
        fallback = most_frequent_encoded(
            artifacts["features"],
            "opposing_team_encoded",
            medians.get("opposing_team_encoded", 0),
        )
        if "opposing_team" in encoders:
            row["opposing_team_encoded"] = safe_label_encode(
                opposing_team,
                encoders["opposing_team"],
                fallback,
            )
        else:
            row["opposing_team_encoded"] = fallback
    if "is_home_match" in selected_feature_columns:
        row["is_home_match"] = int(bool(is_home_match))
    return pd.DataFrame([row])[selected_feature_columns]


def positive_probability(model: Any, feature_row: pd.DataFrame) -> float:
    probabilities = np.asarray(model.predict_proba(feature_row))
    return float(probabilities[:, 1][0])


def calibrated_probability(
    raw_probability: float,
    bowler_type_encoded: int,
    bowler_type_name: str,
    pitch_type: str,
    balls_faced_so_far: int,
    wickets_fallen: int,
    artifacts: dict[str, Any],
) -> float:
    if "calibration_table" not in artifacts or "bowler_type_lifts" not in artifacts:
        return raw_probability
    calibration_table = artifacts["calibration_table"]
    if calibration_table.empty:
        calibrated = raw_probability
    else:
        distances = (calibration_table["mean_score"] - raw_probability).abs()
        calibrated = float(calibration_table.loc[distances.idxmin(), "actual_rate"])
    bowler_type_lift = artifacts["bowler_type_lifts"].get(int(bowler_type_encoded), 1.0)
    pitch_lift = PITCH_TYPE_BOWLER_LIFT.get(pitch_type, {}).get(bowler_type_name, 1.0)
    # Empirical pressure layer: the trained trees often assign the same raw leaf
    # score to different innings states, so we nudge the calibrated rank score
    # with explicitly supplied crease-time and wickets-fallen context.
    pressure_multiplier = (
        1.0
        + 0.15 * (wickets_fallen / 9)
        + 0.10 * min(balls_faced_so_far / 100, 1.0)
    )
    return float(
        np.clip(
            calibrated * bowler_type_lift * pitch_lift * pressure_multiplier,
            0.005,
            0.08,
        )
    )


def recommend_strategy(
    format: str,
    venue: str,
    match_phase: str,
    available_bowler_types: list[str],
    over: int,
    balls_faced_so_far: int,
    runs_scored_so_far: int,
    wickets_fallen: int,
    weather: dict,
    opposing_team: str = "Australia",
    is_home_match: bool = False,
    config_path: Path | None = None,
) -> dict:
    artifacts = load_artifacts(str(config_path) if config_path else None)
    model, model_used, model_feature_columns = select_model_for_format(format, artifacts)
    venues = artifacts["venues"]
    pitch_type = venue_pitch_type(venue, venues)
    baseline_probability = artifacts["baseline_probability"] or BASELINE_PROBABILITY

    recommendations = []
    for bowler_type in available_bowler_types:
        feature_row = build_feature_row(
            bowler_type=bowler_type,
            format_name=format,
            pitch_type=pitch_type,
            match_phase=match_phase,
            over=over,
            balls_faced_so_far=balls_faced_so_far,
            runs_scored_so_far=runs_scored_so_far,
            wickets_fallen=wickets_fallen,
            weather=weather,
            artifacts=artifacts,
            feature_columns=model_feature_columns,
            opposing_team=opposing_team,
            is_home_match=is_home_match,
        )
        raw_probability = positive_probability(model, feature_row)
        bowler_type_encoded = int(feature_row["bowler_type_encoded"].iloc[0])
        probability = calibrated_probability(
            raw_probability,
            bowler_type_encoded,
            bowler_type,
            pitch_type,
            balls_faced_so_far,
            wickets_fallen,
            artifacts,
        )
        print(
            "[recommender debug] loop:",
            {
                "bowler_type": bowler_type,
                "bowler_type_encoded": bowler_type_encoded,
                "raw_predict_proba": raw_probability,
                "calibrated_probability": probability,
            },
        )
        print("[recommender debug] feature row:")
        print(feature_row.to_dict("records")[0])
        recommendations.append(
            {
                "bowler_type": bowler_type,
                "dismissal_probability": probability,
                "relative_lift": probability / baseline_probability if baseline_probability else 0.0,
                "rationale": rationale_for(bowler_type, pitch_type, match_phase),
            }
        )

    recommendations.sort(key=lambda item: item["dismissal_probability"], reverse=True)
    format_text = FORMAT_DISPLAY.get(format, format.upper())
    conditions_summary = (
        f"{format_text} at {venue} on a {pitch_type} pitch, {match_phase} phase, "
        f"over {over}, Kohli {runs_scored_so_far} off {balls_faced_so_far}."
    )
    return {
        "recommendations": recommendations,
        "conditions_summary": conditions_summary,
        "model_confidence": MODEL_CONFIDENCE,
        "baseline_probability": baseline_probability,
        "model_used": model_used,
    }


def clear_cache() -> None:
    load_artifacts.cache_clear()
