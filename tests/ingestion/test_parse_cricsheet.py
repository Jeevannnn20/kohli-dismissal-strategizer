"""Tests for Cricsheet parser scaffolding."""

from src.ingestion import parse_cricsheet


def test_parse_cricsheet_module_imports() -> None:
    assert parse_cricsheet is not None
