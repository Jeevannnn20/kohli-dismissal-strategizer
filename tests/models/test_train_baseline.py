"""Tests for baseline training scaffolding."""

from src.models import train_baseline


def test_train_baseline_module_imports() -> None:
    assert train_baseline is not None
