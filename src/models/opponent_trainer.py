"""Train and evaluate the XGBoost model with opponent context features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from src.models.trainer import TARGET_COLUMN, temporal_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
BASELINE_AUC_ROC = 0.6981
BASELINE_AUC_PR = 0.0433


def load_config(config_path: Path | None = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def project_root_from_config(config_path: Path | None = None) -> Path:
    return Path(config_path).resolve().parent if config_path else PROJECT_ROOT


def feature_columns(features: pd.DataFrame) -> list[str]:
    return [column for column in features.columns if column not in {"match_id", "date", TARGET_COLUMN}]


def safe_auc_roc(y_true: pd.Series, probabilities: np.ndarray) -> float:
    if y_true.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, probabilities))


def safe_auc_pr(y_true: pd.Series, probabilities: np.ndarray) -> float:
    if y_true.nunique() < 2:
        return float("nan")
    return float(average_precision_score(y_true, probabilities))


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int,
    params: dict[str, Any] | None = None,
) -> XGBClassifier:
    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())
    scale_pos_weight = negatives / positives if positives else 1.0
    model_params: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "early_stopping_rounds": 30,
        "scale_pos_weight": scale_pos_weight,
        "random_state": random_state,
        "n_jobs": -1,
    }
    if params:
        model_params.update(params)
    model = XGBClassifier(**model_params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return model


def shap_summary(model: XGBClassifier, X_test: pd.DataFrame) -> tuple[list[dict[str, float]], dict[str, float]]:
    explainer = shap.TreeExplainer(model)
    shap_values = np.asarray(explainer.shap_values(X_test))
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked = (
        pd.DataFrame({"feature": X_test.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    top10 = ranked.head(10).to_dict("records")
    watched = {
        "opposing_team_encoded": float(
            ranked.loc[ranked["feature"].eq("opposing_team_encoded"), "mean_abs_shap"].iloc[0]
        )
        if "opposing_team_encoded" in ranked["feature"].values
        else 0.0,
        "is_home_match": float(
            ranked.loc[ranked["feature"].eq("is_home_match"), "mean_abs_shap"].iloc[0]
        )
        if "is_home_match" in ranked["feature"].values
        else 0.0,
    }
    return top10, watched


def dismissal_rate_by_team(deliveries: pd.DataFrame) -> pd.DataFrame:
    table = (
        deliveries.groupby("opposing_team", dropna=False)
        .agg(
            dismissals=(TARGET_COLUMN, "sum"),
            balls_faced=("legal_ball_proxy", "sum"),
        )
        .reset_index()
    )
    table["dismissal_rate_pct"] = np.where(
        table["balls_faced"] > 0,
        table["dismissals"] / table["balls_faced"] * 100,
        0.0,
    )
    return table.sort_values(["dismissals", "dismissal_rate_pct"], ascending=False).head(10)


def run_opponent_trainer(
    config_path: Path | None = None,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    project_root = project_root_from_config(config_path)
    features_path = project_root / Path(config["paths"]["kohli_features_v2"])
    deliveries_path = project_root / Path(config["paths"]["kohli_deliveries_opponent"])
    results_path = project_root / Path(config["paths"]["opponent_model_results"])
    model_path = project_root / Path(config["paths"]["xgboost_opponent_model"])

    features = pd.read_parquet(features_path)
    columns = feature_columns(features)
    train, test = temporal_split(features, config["features"]["temporal_train_end_date"])
    X_train, y_train = train[columns], train[TARGET_COLUMN].astype(int)
    X_test, y_test = test[columns], test[TARGET_COLUMN].astype(int)

    model = train_xgboost(
        X_train,
        y_train,
        X_test,
        y_test,
        int(config["project"]["random_state"]),
        model_params,
    )
    probabilities = model.predict_proba(X_test)[:, 1]
    auc_roc = safe_auc_roc(y_test, probabilities)
    auc_pr = safe_auc_pr(y_test, probabilities)
    improvement = auc_pr - BASELINE_AUC_PR if not np.isnan(auc_pr) else float("nan")
    top10, watched_shap = shap_summary(model, X_test)
    top10_features = [item["feature"] for item in top10]
    saved_model = bool(not np.isnan(auc_pr) and improvement > 0.002)
    if saved_model:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(model_path)

    deliveries = pd.read_parquet(deliveries_path)
    deliveries = deliveries.copy()
    deliveries["legal_ball_proxy"] = (
        ~deliveries["wide"].astype(bool) & ~deliveries["noball"].astype(bool)
    ).astype(int)
    rate_table = dismissal_rate_by_team(deliveries)

    results: dict[str, Any] = {
        "baseline": {"auc_roc": BASELINE_AUC_ROC, "auc_pr": BASELINE_AUC_PR},
        "opponent_model": {
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "auc_pr_delta": improvement,
            "saved_as_default_fallback": saved_model,
            "train_size": int(len(train)),
            "test_size": int(len(test)),
            "test_positives": int(y_test.sum()),
        },
        "shap_top10": top10,
        "opposing_team_in_top10": "opposing_team_encoded" in top10_features,
        "is_home_match_in_top10": "is_home_match" in top10_features,
        "watched_feature_shap": watched_shap,
        "dismissal_rate_by_team": rate_table.to_dict("records"),
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("[opponent_trainer] v1 combined: AUC-ROC=0.6981, AUC-PR=0.0433")
    print(f"[opponent_trainer] v2 opponent: AUC-ROC={auc_roc:.4f}, AUC-PR={auc_pr:.4f}")
    print(f"[opponent_trainer] AUC-PR delta: {improvement:.4f}")
    print("[opponent_trainer] SHAP top 10:")
    for item in top10:
        print(f"  {item['feature']}: {item['mean_abs_shap']:.6f}")
    print(
        "[opponent_trainer] watched features:",
        {
            "opposing_team_encoded_in_top10": results["opposing_team_in_top10"],
            "is_home_match_in_top10": results["is_home_match_in_top10"],
            **watched_shap,
        },
    )
    print("[opponent_trainer] dismissal rate by top opposing teams:")
    print(rate_table.to_string(index=False))
    print(f"[opponent_trainer] model saved as fallback: {saved_model}")
    return results


def main() -> None:
    run_opponent_trainer()


if __name__ == "__main__":
    main()
