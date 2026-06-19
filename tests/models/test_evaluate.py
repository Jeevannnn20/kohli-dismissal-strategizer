"""Tests for model evaluation scaffolding."""

from src.models import evaluate


def test_evaluate_module_imports() -> None:
    assert evaluate is not None
