"""Tests for feature construction scaffolding."""

from src.features import build_features


def test_build_features_module_imports() -> None:
    assert build_features is not None
