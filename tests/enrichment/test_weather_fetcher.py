from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.enrichment.weather_fetcher import build_weather_lookup, merge_weather


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        hours = [f"2020-01-01T{hour:02d}:00" for hour in range(24)]
        return {
            "hourly": {
                "time": hours,
                "temperature_2m": list(range(24)),
                "relativehumidity_2m": [60] * 24,
                "cloudcover": [40] * 24,
                "windspeed_10m": [12] * 24,
                "precipitation": [0.2] * 24,
            }
        }


class FakeSession:
    def __init__(self) -> None:
        self.calls = 0

    def get(self, url: str, params: dict, timeout: int) -> FakeResponse:
        self.calls += 1
        assert params["start_date"] == "2020-01-01"
        return FakeResponse()


def weather_config() -> dict:
    return {
        "weather": {
            "historical_api_url": "https://archive-api.open-meteo.com/v1/archive",
            "timezone": "auto",
            "target_hour": 14,
            "request_sleep_seconds": 0,
        }
    }


def test_weather_fetch_uses_cache_and_median_fill(tmp_path: Path) -> None:
    deliveries = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-01", "2020-01-02"],
            "latitude": [22.0, 22.0, None],
            "longitude": [88.0, 88.0, None],
        }
    )
    cache_path = tmp_path / "weather_cache.parquet"
    session = FakeSession()

    lookup, fetched, cached = build_weather_lookup(
        deliveries,
        weather_config(),
        cache_path,
        session=session,
        sleep_fn=lambda _: None,
    )
    assert fetched == 1
    assert cached == 0
    assert session.calls == 1
    assert lookup.loc[0, "temperature"] == 14

    _, fetched_again, cached_again = build_weather_lookup(
        deliveries,
        weather_config(),
        cache_path,
        session=session,
        sleep_fn=lambda _: None,
    )
    assert fetched_again == 0
    assert cached_again == 1
    assert session.calls == 1

    enriched = merge_weather(deliveries, lookup)
    missing_location_row = enriched[enriched["latitude"].isna()].iloc[0]
    assert missing_location_row["temperature"] == 14
    assert missing_location_row["humidity"] == 60
