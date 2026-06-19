from __future__ import annotations

from pathlib import Path

from src.ingestion.pipeline import run_pipeline


def test_pipeline_runs_steps_in_order(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, Path]] = []

    monkeypatch.setattr(
        "src.ingestion.pipeline.run_downloader",
        lambda config_path: calls.append(("download", config_path)),
    )
    monkeypatch.setattr(
        "src.ingestion.pipeline.parse_all_matches",
        lambda config_path: calls.append(("parse", config_path)),
    )
    monkeypatch.setattr(
        "src.ingestion.pipeline.extract_dismissals",
        lambda config_path: calls.append(("extract", config_path)),
    )

    config_path = tmp_path / "config.yaml"
    run_pipeline(config_path)

    assert calls == [
        ("download", config_path),
        ("parse", config_path),
        ("extract", config_path),
    ]
