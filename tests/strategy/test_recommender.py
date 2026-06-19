from __future__ import annotations

import pandas as pd

from src.strategy import recommender
from src.strategy.recommender import recommend_strategy


class FakeEncoder:
    def __init__(self, classes: list[str]) -> None:
        self.classes_ = classes

    def transform(self, values: list[str]) -> list[int]:
        return [self.classes_.index(value) for value in values]


class FakeModel:
    def predict_proba(self, rows: pd.DataFrame):
        probabilities = []
        for _, row in rows.iterrows():
            probabilities.append(min(0.01 + float(row["bowler_type_encoded"]) * 0.01, 0.99))
        return [[1 - probability, probability] for probability in probabilities]


def fake_artifacts() -> dict:
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
    medians = {column: 1.0 for column in feature_columns}
    medians.update({"bowler_type_encoded": 1.0, "pitch_type_encoded": 1.0, "match_phase_encoded": 1.0})
    return {
        "model": FakeModel(),
        "encoders": {
            "bowler_type": FakeEncoder(["LAPS", "RAPS", "LB"]),
            "pitch_type": FakeEncoder(["flat", "seam", "unknown"]),
            "match_phase": FakeEncoder(["powerplay", "middle", "death"]),
        },
        "venues": pd.DataFrame(
            {
                "venue": ["Known Venue"],
                "pitch_type": ["seam"],
                "latitude": [1.0],
                "longitude": [2.0],
            }
        ),
        "feature_columns": feature_columns,
        "medians": medians,
        "baseline_probability": 0.02,
    }


def patch_artifacts(monkeypatch) -> None:
    monkeypatch.setattr(recommender, "load_artifacts", lambda *_: fake_artifacts())


def test_recommend_strategy_returns_expected_structure(monkeypatch) -> None:
    patch_artifacts(monkeypatch)

    result = recommend_strategy(
        format="ipl",
        venue="Known Venue",
        match_phase="death",
        available_bowler_types=["LAPS", "RAPS"],
        over=18,
        balls_faced_so_far=30,
        runs_scored_so_far=40,
        wickets_fallen=4,
        weather={},
    )

    assert {"recommendations", "conditions_summary", "model_confidence", "baseline_probability"} <= set(result)
    assert len(result["recommendations"]) == 2
    assert {"bowler_type", "dismissal_probability", "relative_lift", "rationale"} <= set(
        result["recommendations"][0]
    )


def test_recommendations_are_sorted_descending(monkeypatch) -> None:
    patch_artifacts(monkeypatch)

    result = recommend_strategy(
        "odis",
        "Known Venue",
        "middle",
        ["LAPS", "LB", "RAPS"],
        20,
        20,
        30,
        2,
        {},
    )

    probabilities = [item["dismissal_probability"] for item in result["recommendations"]]
    assert probabilities == sorted(probabilities, reverse=True)


def test_unknown_venue_falls_back_gracefully(monkeypatch) -> None:
    patch_artifacts(monkeypatch)

    result = recommend_strategy("tests", "Unknown Ground", "opening", ["LAPS"], 5, 10, 8, 1, {})

    assert result["recommendations"][0]["bowler_type"] == "LAPS"
    assert "unknown pitch" in result["conditions_summary"] or "unknown" in result["conditions_summary"]


def test_unknown_bowler_type_falls_back_gracefully(monkeypatch) -> None:
    patch_artifacts(monkeypatch)

    result = recommend_strategy("ipl", "Known Venue", "death", ["MYSTERY"], 18, 12, 20, 3, {})

    assert result["recommendations"][0]["bowler_type"] == "MYSTERY"
    assert result["recommendations"][0]["dismissal_probability"] > 0


def test_relative_lift_calculation(monkeypatch) -> None:
    patch_artifacts(monkeypatch)

    result = recommend_strategy("ipl", "Known Venue", "death", ["LAPS"], 18, 12, 20, 3, {})
    recommendation = result["recommendations"][0]

    assert recommendation["relative_lift"] == recommendation["dismissal_probability"] / result["baseline_probability"]
