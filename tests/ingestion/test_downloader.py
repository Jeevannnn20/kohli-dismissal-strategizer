from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import yaml

from src.ingestion.downloader import run_downloader


class FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


def write_config(tmp_path: Path) -> Path:
    config = {
        "paths": {"raw_data_dir": "data/raw"},
        "cricsheet": {
            "base_download_url": "https://cricsheet.org/downloads/",
            "formats": {"tests": {"zip_file": "tests_json.zip"}},
        },
        "player": {"cricsheet_id": "V Kohli"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def zip_bytes() -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("12345.json", "{}")
        archive.writestr("README.txt", "ignored")
    return buffer.getvalue()


def test_downloader_saves_and_extracts_zip(monkeypatch, tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    calls: list[str] = []

    def fake_get(url: str, timeout: int) -> FakeResponse:
        calls.append(url)
        assert timeout == 120
        return FakeResponse(zip_bytes())

    monkeypatch.setattr("src.ingestion.downloader.requests.get", fake_get)

    counts = run_downloader(config_path)

    assert calls == ["https://cricsheet.org/downloads/tests_json.zip"]
    assert counts == {"tests": 1}
    assert (tmp_path / "data/raw/tests/tests_json.zip").exists()
    assert (tmp_path / "data/raw/tests/matches/12345.json").exists()
    assert not (tmp_path / "data/raw/tests/matches/README.txt").exists()


def test_downloader_is_idempotent(monkeypatch, tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    archive_path = tmp_path / "data/raw/tests/tests_json.zip"
    matches_dir = tmp_path / "data/raw/tests/matches"
    archive_path.parent.mkdir(parents=True)
    matches_dir.mkdir(parents=True)
    archive_path.write_bytes(zip_bytes())
    (matches_dir / "12345.json").write_text("{}", encoding="utf-8")

    def fail_get(url: str, timeout: int) -> FakeResponse:
        raise AssertionError("download should be skipped")

    monkeypatch.setattr("src.ingestion.downloader.requests.get", fail_get)

    counts = run_downloader(config_path)

    assert counts == {"tests": 1}
    assert (matches_dir / "12345.json").exists()
