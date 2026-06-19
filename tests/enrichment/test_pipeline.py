from __future__ import annotations

from pathlib import Path

from src.enrichment.pipeline import run_pipeline


def test_enrichment_pipeline_runs_steps_in_order(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, Path]] = []

    monkeypatch.setattr(
        "src.enrichment.pipeline.clean_kohli_deliveries",
        lambda config_path: calls.append(("clean", config_path)),
    )
    monkeypatch.setattr(
        "src.enrichment.pipeline.run_bowler_typer",
        lambda config_path: calls.append(("bowler", config_path)),
    )
    monkeypatch.setattr(
        "src.enrichment.pipeline.run_venue_encoder",
        lambda config_path: calls.append(("venue", config_path)),
    )
    monkeypatch.setattr(
        "src.enrichment.pipeline.run_weather_fetcher",
        lambda config_path: calls.append(("weather", config_path)),
    )

    config_path = tmp_path / "config.yaml"
    run_pipeline(config_path)

    assert calls == [
        ("clean", config_path),
        ("bowler", config_path),
        ("venue", config_path),
        ("weather", config_path),
    ]
