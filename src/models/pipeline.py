"""Run Phase 3 feature engineering, training, and SHAP analysis."""

from __future__ import annotations

from pathlib import Path

from src.features.engineer import run_engineer
from src.models.shap_analyzer import run_shap_analyzer
from src.models.trainer import run_trainer


def run_pipeline(config_path: Path | None = None) -> None:
    print("[models] Step 1/3: engineering features")
    run_engineer(config_path)

    print("[models] Step 2/3: training models")
    run_trainer(config_path)

    print("[models] Step 3/3: computing SHAP analysis")
    run_shap_analyzer(config_path)

    print("[models] Phase 3 complete")
    print("[models] produced: kohli_features.parquet, model_results.json, model files, shap_results.pkl, SHAP plots")


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
