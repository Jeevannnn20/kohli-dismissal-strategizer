"""Run Tier 1 model improvements end to end."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.features.opponent_engineer import run_opponent_engineer
from src.models.format_trainer import run_format_trainer
from src.models.opponent_trainer import run_opponent_trainer
from src.models.survival_model import run_survival_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_path: Path | None = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def project_root_from_config(config_path: Path | None = None) -> Path:
    return Path(config_path).resolve().parent if config_path else PROJECT_ROOT


def run_tier1_pipeline(config_path: Path | None = None) -> dict:
    config = load_config(config_path)
    project_root = project_root_from_config(config_path)

    print("[tier1] 1/4 training per-format XGBoost models")
    format_results = run_format_trainer(config_path)

    print("[tier1] 2/4 fitting Cox proportional hazards survival model")
    survival_results = run_survival_model(config_path)

    print("[tier1] 3/4 engineering opponent context features")
    run_opponent_engineer(config_path)

    print("[tier1] 4/4 training opponent-context XGBoost model")
    opponent_results = run_opponent_trainer(config_path)

    model_dir = project_root / Path(config["paths"].get("models_dir", "models"))
    created_models = sorted(path.name for path in model_dir.glob("*.json")) + sorted(
        path.name for path in model_dir.glob("*.pkl")
    )
    summary = {
        "format_auc_pr_vs_combined": {
            fmt: result.get("vs_combined")
            for fmt, result in format_results.items()
        },
        "survival_c_index": survival_results.get("test_c_index"),
        "opponent_auc_pr_vs_baseline": opponent_results["opponent_model"].get("auc_pr_delta"),
        "opposing_team_in_shap_top10": opponent_results.get("opposing_team_in_top10"),
        "is_home_match_in_shap_top10": opponent_results.get("is_home_match_in_top10"),
        "new_model_files_created": created_models,
    }
    print("[tier1] final summary:")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    run_tier1_pipeline()


if __name__ == "__main__":
    main()
