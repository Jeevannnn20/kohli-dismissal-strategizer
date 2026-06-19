"""Tests for Cricsheet bulk download scaffolding."""

from src.ingestion import download_cricsheet


def test_download_cricsheet_module_imports() -> None:
    assert download_cricsheet is not None
