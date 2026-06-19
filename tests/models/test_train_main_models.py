"""Tests for main model training scaffolding."""

from src.models import train_main_models


def test_train_main_models_module_imports() -> None:
    assert train_main_models is not None
