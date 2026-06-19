"""Phase 3 diagnostics for trained dismissal models and engineered features."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data/processed/kohli_features.parquet"
ENRICHED_PATH = PROJECT_ROOT / "data/processed/kohli_deliveries_enriched.parquet"
MODEL_RESULTS_PATH = PROJECT_ROOT / "data/processed/model_results.json"
SHAP_RESULTS_PATH = PROJECT_ROOT / "data/processed/shap_results.pkl"
XGBOOST_MODEL_PATH = PROJECT_ROOT / "models/xgboost_kohli.json"
LIGHTGBM_MODEL_PATH = PROJECT_ROOT / "models/lightgbm_kohli.txt"
PLOTS_DIR = PROJECT_ROOT / "data/processed/plots"
TARGET_COLUMN = "is_dismissal"


def print_section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def load_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    features = pd.read_parquet(FEATURES_PATH)
    enriched = pd.read_parquet(ENRICHED_PATH)
    model_results = json.loads(MODEL_RESULTS_PATH.read_text(encoding="utf-8"))
    shap_results = joblib.load(SHAP_RESULTS_PATH)
    features["date"] = pd.to_datetime(features["date"])
    enriched["date"] = pd.to_datetime(enriched["date"])
    return features, enriched, model_results, shap_results


def temporal_test_set(features: pd.DataFrame, split_date: str) -> pd.DataFrame:
    return features[features["date"] >= pd.Timestamp(split_date)].copy()


def shap_top_10(shap_results: dict) -> list[str]:
    print_section("1. SHAP TOP 10 FEATURES")
    shap_values = np.asarray(shap_results["shap_values"])
    feature_names = list(shap_results["feature_names"])
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked = sorted(zip(feature_names, mean_abs), key=lambda item: item[1], reverse=True)
    for rank, (feature, score) in enumerate(ranked[:10], start=1):
        print(f"{rank:>2}. {feature:<40} {score:.6f}")
    return [feature for feature, _ in ranked[:5]]


def lightgbm_collapse_check(X_test: pd.DataFrame) -> None:
    print_section("2. LIGHTGBM COLLAPSE INVESTIGATION")
    model = lgb.Booster(model_file=str(LIGHTGBM_MODEL_PATH))
    probabilities = model.predict(X_test)
    unique_values = np.unique(np.round(probabilities, 10))
    print(f"Unique predicted probability values: {len(unique_values)}")
    print(unique_values[:30])
    if len(unique_values) > 30:
        print(f"... truncated, last value: {unique_values[-1]}")
    print(f"min probability:  {probabilities.min():.10f}")
    print(f"max probability:  {probabilities.max():.10f}")
    print(f"mean probability: {probabilities.mean():.10f}")
    print(f"predicted positives at threshold 0.5: {int((probabilities >= 0.5).sum())}")


def xgboost_probabilities(X_test: pd.DataFrame) -> np.ndarray:
    model = XGBClassifier()
    model.load_model(XGBOOST_MODEL_PATH)
    return model.predict_proba(X_test)[:, 1]


def calibration_check(X_test: pd.DataFrame, y_test: pd.Series) -> np.ndarray:
    print_section("3. CALIBRATION CHECK: XGBOOST DECILES")
    probabilities = xgboost_probabilities(X_test)
    calibration = pd.DataFrame(
        {"predicted_probability": probabilities, TARGET_COLUMN: y_test.to_numpy()}
    )
    calibration["decile"] = pd.qcut(
        calibration["predicted_probability"], q=10, labels=False, duplicates="drop"
    )
    summary = (
        calibration.groupby("decile", observed=True)
        .agg(
            rows=("predicted_probability", "size"),
            mean_predicted_probability=("predicted_probability", "mean"),
            actual_dismissal_rate=(TARGET_COLUMN, "mean"),
        )
        .reset_index()
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    return probabilities


def feature_distribution_check(features: pd.DataFrame, top_features: list[str]) -> None:
    print_section("4. FEATURE DISTRIBUTION CHECK: DISMISSALS VS NON-DISMISSALS")
    rows = []
    for feature in top_features:
        dismissal_mean = features.loc[features[TARGET_COLUMN] == 1, feature].mean()
        non_dismissal_mean = features.loc[features[TARGET_COLUMN] == 0, feature].mean()
        rows.append(
            {
                "feature": feature,
                "dismissal_mean": dismissal_mean,
                "non_dismissal_mean": non_dismissal_mean,
                "difference": dismissal_mean - non_dismissal_mean,
            }
        )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:.6f}"))


def bowler_type_breakdown(enriched: pd.DataFrame) -> None:
    print_section("5. BOWLER TYPE BREAKDOWN")
    summary = (
        enriched.groupby("bowler_type")
        .agg(
            balls_faced=("is_dismissal", "size"),
            dismissals=("is_dismissal", "sum"),
        )
        .assign(dismissal_rate=lambda df: df["dismissals"] / df["balls_faced"])
        .sort_values(["dismissals", "dismissal_rate"], ascending=False)
    )
    print(summary.to_string(float_format=lambda value: f"{value:.6f}"))


def phase_and_format_breakdown(enriched: pd.DataFrame) -> None:
    print_section("6. PHASE AND FORMAT BREAKDOWN")
    phase = (
        enriched.groupby("match_phase")
        .agg(rows=("is_dismissal", "size"), dismissals=("is_dismissal", "sum"))
        .assign(dismissal_rate=lambda df: df["dismissals"] / df["rows"])
        .sort_values("dismissal_rate", ascending=False)
    )
    print("\nDismissal rate by match_phase:")
    print(phase.to_string(float_format=lambda value: f"{value:.6f}"))

    fmt = (
        enriched.groupby("format")
        .agg(rows=("is_dismissal", "size"), dismissals=("is_dismissal", "sum"))
        .assign(dismissal_rate=lambda df: df["dismissals"] / df["rows"])
        .sort_values("dismissal_rate", ascending=False)
    )
    print("\nDismissal rate by format:")
    print(fmt.to_string(float_format=lambda value: f"{value:.6f}"))

    cross_tab = pd.pivot_table(
        enriched,
        index="format",
        columns="match_phase",
        values="is_dismissal",
        aggfunc="mean",
        observed=True,
    )
    print("\nFormat x match_phase dismissal rates:")
    print(cross_tab.to_string(float_format=lambda value: f"{value:.6f}"))


def over_by_over_dismissal_rate(enriched: pd.DataFrame) -> None:
    print_section("7. OVER-BY-OVER DISMISSAL RATE")
    limited = enriched[enriched["over"].between(0, 49)].copy()
    limited["over_bucket"] = (limited["over"] // 5 * 5).astype(int)
    limited["bucket_label"] = limited["over_bucket"].map(
        lambda start: f"{start:02d}-{start + 4:02d}"
    )
    bucketed = (
        limited.groupby(["over_bucket", "bucket_label"])
        .agg(rows=("is_dismissal", "size"), dismissals=("is_dismissal", "sum"))
        .assign(dismissal_rate=lambda df: df["dismissals"] / df["rows"])
        .reset_index()
        .sort_values("over_bucket")
    )
    print(bucketed[["bucket_label", "rows", "dismissals", "dismissal_rate"]].to_string(
        index=False, float_format=lambda value: f"{value:.6f}"
    ))

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(bucketed["bucket_label"], bucketed["dismissal_rate"], marker="o")
    plt.xlabel("Over bucket")
    plt.ylabel("Dismissal rate")
    plt.title("Kohli dismissal rate by over bucket")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    output_path = PLOTS_DIR / "dismissal_rate_by_over.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved plot: {output_path}")


def main() -> None:
    features, enriched, model_results, shap_results = load_artifacts()
    split_date = model_results["split_date"]
    feature_names = model_results["feature_columns"]
    test = temporal_test_set(features, split_date)
    X_test = test[feature_names]
    y_test = test[TARGET_COLUMN]

    top_features = shap_top_10(shap_results)
    lightgbm_collapse_check(X_test)
    calibration_check(X_test, y_test)
    feature_distribution_check(features, top_features)
    bowler_type_breakdown(enriched)
    phase_and_format_breakdown(enriched)
    over_by_over_dismissal_rate(enriched)


if __name__ == "__main__":
    main()
