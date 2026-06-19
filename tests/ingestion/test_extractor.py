from __future__ import annotations

import pandas as pd

from src.ingestion.extractor import add_pre_dismissal_context


def test_pre_dismissal_context_is_computed_per_innings() -> None:
    deliveries = pd.DataFrame(
        [
            {
                "match_id": "1",
                "innings_number": 1,
                "delivery_index": 0,
                "batter": "V Kohli",
                "runs_off_bat": 4,
                "wide": False,
                "noball": False,
                "is_dismissal": 0,
                "format": "odis",
                "dismissal_kind": None,
                "bowler": "Bowler A",
            },
            {
                "match_id": "1",
                "innings_number": 1,
                "delivery_index": 1,
                "batter": "V Kohli",
                "runs_off_bat": 0,
                "wide": True,
                "noball": False,
                "is_dismissal": 0,
                "format": "odis",
                "dismissal_kind": None,
                "bowler": "Bowler A",
            },
            {
                "match_id": "1",
                "innings_number": 1,
                "delivery_index": 2,
                "batter": "V Kohli",
                "runs_off_bat": 2,
                "wide": False,
                "noball": False,
                "is_dismissal": 0,
                "format": "odis",
                "dismissal_kind": None,
                "bowler": "Bowler B",
            },
            {
                "match_id": "1",
                "innings_number": 1,
                "delivery_index": 3,
                "batter": "V Kohli",
                "runs_off_bat": 0,
                "wide": False,
                "noball": False,
                "is_dismissal": 1,
                "format": "odis",
                "dismissal_kind": "bowled",
                "bowler": "Bowler C",
            },
            {
                "match_id": "1",
                "innings_number": 2,
                "delivery_index": 4,
                "batter": "V Kohli",
                "runs_off_bat": 1,
                "wide": False,
                "noball": False,
                "is_dismissal": 1,
                "format": "odis",
                "dismissal_kind": "run out",
                "bowler": "Bowler D",
            },
        ]
    )

    dismissals = add_pre_dismissal_context(deliveries, "V Kohli")

    assert len(dismissals) == 2
    first = dismissals.iloc[0]
    second = dismissals.iloc[1]
    assert first["balls_faced_before_dismissal"] == 2
    assert first["runs_scored_before_dismissal"] == 6
    assert second["balls_faced_before_dismissal"] == 0
    assert second["runs_scored_before_dismissal"] == 0
