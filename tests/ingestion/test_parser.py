from __future__ import annotations

import json
from pathlib import Path

from src.ingestion.parser import match_phase, parse_match_file


def write_match(path: Path) -> None:
    match = {
        "info": {
            "dates": ["2020-01-01", "2020-01-02"],
            "venue": "M Chinnaswamy Stadium",
            "match_type": "T20",
            "teams": ["India", "Australia"],
            "season": "2019/20",
        },
        "innings": [
            {
                "team": "India",
                "overs": [
                    {
                        "over": 0,
                        "deliveries": [
                            {
                                "batter": "V Kohli",
                                "bowler": "Bowler A",
                                "runs": {"batter": 1, "extras": 0, "total": 1},
                            },
                            {
                                "batter": "RG Sharma",
                                "bowler": "Bowler A",
                                "runs": {"batter": 0, "extras": 1, "total": 1},
                                "extras": {"wides": 1},
                            },
                        ],
                    },
                    {
                        "over": 15,
                        "deliveries": [
                            {
                                "batter": "V Kohli",
                                "bowler": "Bowler B",
                                "runs": {"batter": 0, "extras": 0, "total": 0},
                                "wickets": [
                                    {
                                        "kind": "caught",
                                        "player_out": "V Kohli",
                                        "fielders": [{"name": "Fielder A"}],
                                    }
                                ],
                            }
                        ],
                    },
                ],
            },
            {
                "team": "India",
                "super_over": True,
                "overs": [
                    {
                        "over": 0,
                        "deliveries": [
                            {
                                "batter": "V Kohli",
                                "bowler": "Bowler C",
                                "runs": {"batter": 6, "extras": 0, "total": 6},
                            }
                        ],
                    }
                ],
            },
        ],
    }
    path.write_text(json.dumps(match), encoding="utf-8")


def test_parser_extracts_kohli_rows_and_dismissal_metadata(tmp_path: Path) -> None:
    match_path = tmp_path / "fixture_match.json"
    write_match(match_path)

    rows = parse_match_file(match_path, "t20is", "V Kohli")

    assert len(rows) == 2
    assert rows[0]["match_id"] == "fixture_match"
    assert rows[0]["date"] == "2020-01-01"
    assert rows[0]["city"] is None
    assert rows[0]["is_dismissal"] == 0
    assert rows[0]["match_phase"] == "powerplay"
    assert rows[1]["is_dismissal"] == 1
    assert rows[1]["dismissal_kind"] == "caught"
    assert rows[1]["player_dismissed"] == "V Kohli"
    assert rows[1]["fielder"] == "Fielder A"
    assert rows[1]["match_phase"] == "death"


def test_match_phase_labels_by_format() -> None:
    assert match_phase(0, "tests", "Test") == "opening"
    assert match_phase(20, "tests", "Test") == "middle"
    assert match_phase(60, "tests", "Test") == "tail"
    assert match_phase(0, "odis", "ODI") == "powerplay"
    assert match_phase(20, "odis", "ODI") == "middle"
    assert match_phase(45, "odis", "ODI") == "death"
    assert match_phase(0, "ipl", "T20") == "powerplay"
    assert match_phase(10, "ipl", "T20") == "middle"
    assert match_phase(18, "ipl", "T20") == "death"
