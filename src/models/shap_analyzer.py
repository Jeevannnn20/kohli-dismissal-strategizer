"""Compute SHAP explanations and plots for the best Kohli dismissal model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml
from xgboost import XGBClassifier

from src.models.trainer import TARGET_COLUMN, feature_columns, temporal_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_path: Path | None = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def project_root_from_config(config_path: Path | None = None) -> Path:
    return Path(config_path).resolve().parent if config_path else PROJECT_ROOT


def load_best_model(best_model: str, project_root: Path, config: dict) -> Any:
    if best_model == "xgboost":
        model = XGBClassifier()
        model.load_model(project_root / Path(config["paths"]["xgboost_model"]))
        return model
    if best_model == "lightgbm":
        return lgb.Booster(model_file=str(project_root / Path(config["paths"]["lightgbm_model"])))
    if best_model == "logistic_regression":
        return joblib.load(project_root / Path(config["paths"]["logistic_model"]))
    raise ValueError(f"Unsupported best model: {best_model}")


def normalize_shap_values(values: Any) -> np.ndarray:
    if isinstance(values, list):
        values = values[-1]
    if hasattr(values, "values"):
        values = values.values
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    return values


def compute_shap_values(model: Any, X_test: pd.DataFrame, best_model: str) -> np.ndarray:
    if best_model == "logistic_regression":
        estimator = model.named_steps["model"] if hasattr(model, "named_steps") else model
        transformed = (
            model.named_steps["scaler"].transform(X_test)
            if hasattr(model, "named_steps") and "scaler" in model.named_steps
            else X_test
        )
        explainer = shap.LinearExplainer(estimator, transformed)
        values = explainer.shap_values(transformed)
    else:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X_test)
    return normalize_shap_values(values)


def save_shap_payload(
    shap_values: np.ndarray,
    feature_names: list[str],
    X_test: pd.DataFrame,
    output_path: Path,
) -> dict[str, Any]:
    payload = {
        "shap_values": shap_values,
        "feature_names": feature_names,
        "X_test": X_test,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output_path)
    return payload


def save_shap_plots(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    plots_dir: Path,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(plots_dir / "shap_summary_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(plots_dir / "shap_summary_dot.png", dpi=150, bbox_inches="tight")
    plt.close()

    for feature, filename in [
        ("bowler_type_encoded", "shap_dependence_bowler_type.png"),
        ("over", "shap_dependence_over.png"),
        ("pitch_type_encoded", "shap_dependence_pitch_type.png"),
    ]:
        if feature in X_test.columns:
            plt.figure()
            shap.dependence_plot(feature, shap_values, X_test, show=False)
            plt.tight_layout()
            plt.savefig(plots_dir / filename, dpi=150, bbox_inches="tight")
            plt.close()


def print_top_features(shap_values: np.ndarray, feature_names: list[str]) -> None:
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked = sorted(zip(feature_names, mean_abs), key=lambda item: item[1], reverse=True)
    print("[shap] top 10 features by mean absolute SHAP value:")
    for name, score in ranked[:10]:
        print(f"{name}: {score:.6f}")


def run_shap_analyzer(config_path: Path | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    project_root = project_root_from_config(config_path)
    features_path = project_root / Path(config["paths"]["kohli_features"])
    results_path = project_root / Path(config["paths"]["model_results"])
    shap_path = project_root / Path(config["paths"]["shap_results"])
    plots_dir = project_root / Path(config["paths"]["plots_dir"])
    split_date = config["features"]["temporal_train_end_date"]

    results = json.loads(results_path.read_text(encoding="utf-8"))
    best_model = results["best_model"]
    features = pd.read_parquet(features_path)
    _, test = temporal_split(features, split_date)
    columns = results.get("feature_columns") or feature_columns(features)
    X_test = test[columns]

    model = load_best_model(best_model, project_root, config)
    shap_values = compute_shap_values(model, X_test, best_model)
    payload = save_shap_payload(shap_values, columns, X_test, shap_path)
    save_shap_plots(shap_values, X_test, plots_dir)
    print_top_features(shap_values, columns)
    print(f"[shap] wrote {shap_path}")
    print(f"[shap] plots dir: {plots_dir}")
    print(f"[shap] test matrix shape: {X_test.shape}")
    return payload


def main() -> None:
    run_shap_analyzer()


if __name__ == "__main__":
    main()
