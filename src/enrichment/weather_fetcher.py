"""Fetch and merge historical weather data for enriched Kohli deliveries."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
WEATHER_COLUMNS = [
    "temperature",
    "humidity",
    "cloud_cover",
    "wind_speed",
    "precipitation",
]
CACHE_KEY_COLUMNS = ["date", "latitude", "longitude"]


def load_config(config_path: Path | None = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def project_root_from_config(config_path: Path | None = None) -> Path:
    return Path(config_path).resolve().parent if config_path else PROJECT_ROOT


def empty_weather_cache() -> pd.DataFrame:
    return pd.DataFrame(columns=CACHE_KEY_COLUMNS + WEATHER_COLUMNS)


def cache_key_frame(weather_cache: pd.DataFrame) -> set[tuple[str, float, float]]:
    keys: set[tuple[str, float, float]] = set()
    if weather_cache.empty:
        return keys
    for row in weather_cache[CACHE_KEY_COLUMNS].itertuples(index=False):
        keys.add((str(row.date), round(float(row.latitude), 6), round(float(row.longitude), 6)))
    return keys


def hourly_value(hourly: dict[str, Any], candidate_names: list[str], index: int) -> Any:
    for name in candidate_names:
        values = hourly.get(name)
        if values is not None and len(values) > index:
            return values[index]
    return None


def hour_index(hourly_times: list[str], target_hour: int) -> int:
    target_suffix = f"T{target_hour:02d}:00"
    for index, value in enumerate(hourly_times):
        if str(value).endswith(target_suffix):
            return index
    return min(target_hour, max(len(hourly_times) - 1, 0))


def fetch_weather(
    date: str,
    latitude: float,
    longitude: float,
    config: dict,
    session: Any = requests,
) -> dict[str, Any]:
    weather_config = config["weather"]
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": date,
        "end_date": date,
        "hourly": "temperature_2m,relativehumidity_2m,cloudcover,windspeed_10m,precipitation",
        "timezone": weather_config.get("timezone", "auto"),
    }
    response = session.get(weather_config["historical_api_url"], params=params, timeout=60)
    response.raise_for_status()
    hourly = response.json().get("hourly", {})
    times = hourly.get("time", [])
    index = hour_index(times, int(weather_config.get("target_hour", 14)))
    return {
        "date": date,
        "latitude": latitude,
        "longitude": longitude,
        "temperature": hourly_value(hourly, ["temperature_2m"], index),
        "humidity": hourly_value(hourly, ["relativehumidity_2m", "relative_humidity_2m"], index),
        "cloud_cover": hourly_value(hourly, ["cloudcover", "cloud_cover"], index),
        "wind_speed": hourly_value(hourly, ["windspeed_10m", "wind_speed_10m"], index),
        "precipitation": hourly_value(hourly, ["precipitation"], index),
    }


def build_weather_lookup(
    deliveries: pd.DataFrame,
    config: dict,
    cache_path: Path,
    session: Any = requests,
    sleep_fn=time.sleep,
) -> tuple[pd.DataFrame, int, int]:
    if cache_path.exists():
        weather_cache = pd.read_parquet(cache_path)
    else:
        weather_cache = empty_weather_cache()

    combos = (
        deliveries.dropna(subset=["date", "latitude", "longitude"])[
            CACHE_KEY_COLUMNS
        ]
        .drop_duplicates()
        .copy()
    )
    combos["date"] = combos["date"].astype(str)
    existing_keys = cache_key_frame(weather_cache)
    fetched_rows: list[dict[str, Any]] = []
    served_from_cache = 0
    sleep_seconds = float(config["weather"].get("request_sleep_seconds", 0.5))

    for row in tqdm(
        combos.itertuples(index=False),
        total=len(combos),
        desc="Fetching weather",
    ):
        key = (str(row.date), round(float(row.latitude), 6), round(float(row.longitude), 6))
        if key in existing_keys:
            served_from_cache += 1
            continue

        fetched_rows.append(
            fetch_weather(str(row.date), float(row.latitude), float(row.longitude), config, session)
        )
        existing_keys.add(key)
        sleep_fn(sleep_seconds)

    if fetched_rows:
        fetched_frame = pd.DataFrame(fetched_rows)
        if weather_cache.empty:
            weather_cache = fetched_frame
        else:
            weather_cache = pd.concat(
                [weather_cache, fetched_frame], ignore_index=True
            )
        weather_cache = weather_cache.drop_duplicates(CACHE_KEY_COLUMNS, keep="last")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    weather_cache.to_parquet(cache_path, compression="snappy", index=False)
    return weather_cache, len(fetched_rows), served_from_cache


def merge_weather(deliveries: pd.DataFrame, weather_lookup: pd.DataFrame) -> pd.DataFrame:
    enriched = deliveries.merge(weather_lookup, how="left", on=CACHE_KEY_COLUMNS)
    for column in WEATHER_COLUMNS:
        enriched[column] = pd.to_numeric(enriched[column], errors="coerce")
        median = enriched[column].median()
        if pd.isna(median):
            median = 0
        enriched[column] = enriched[column].fillna(median)
    return enriched


def run_weather_fetcher(
    config_path: Path | None = None,
    session: Any = requests,
    sleep_fn=time.sleep,
) -> pd.DataFrame:
    config = load_config(config_path)
    project_root = project_root_from_config(config_path)
    deliveries_path = project_root / Path(config["paths"]["kohli_deliveries_v3"])
    cache_path = project_root / Path(config["paths"]["weather_cache_file"])
    output_path = project_root / Path(config["paths"]["kohli_deliveries_enriched"])

    deliveries = pd.read_parquet(deliveries_path)
    weather_lookup, fetched, cached = build_weather_lookup(
        deliveries, config, cache_path, session=session, sleep_fn=sleep_fn
    )
    enriched = merge_weather(deliveries, weather_lookup)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(output_path, compression="snappy", index=False)

    print(f"[weather_fetcher] fetched combos: {fetched}")
    print(f"[weather_fetcher] served from cache: {cached}")
    print(f"[weather_fetcher] wrote {output_path}")
    print(f"[weather_fetcher] final shape: {enriched.shape}")
    print(f"[weather_fetcher] columns: {enriched.columns.tolist()}")
    return enriched


def main() -> None:
    run_weather_fetcher()


if __name__ == "__main__":
    main()
