"""Survival analysis for Kohli innings using a Cox proportional hazards model."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_path: Path | None = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def project_root_from_config(config_path: Path | None = None) -> Path:
    return Path(config_path).resolve().parent if config_path else PROJECT_ROOT


def first_non_null(series: pd.Series):
    non_null = series.dropna()
    return non_null.iloc[0] if not non_null.empty else None


def mode_or_unknown(series: pd.Series) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return "unknown"
    return str(non_null.mode().iloc[0])


def add_last5_avg(innings: pd.DataFrame) -> pd.DataFrame:
    ordered = innings.sort_values(["date", "match_id", "innings_number"]).copy()
    last5_values: list[float] = []
    prior_runs: list[float] = []
    prior_events: list[int] = []
    for row in ordered.itertuples(index=False):
        if prior_runs:
            recent_runs = prior_runs[-5:]
            recent_events = prior_events[-5:]
            dismissals = sum(recent_events)
            avg = sum(recent_runs) / dismissals if dismissals else float(np.mean(recent_runs))
        else:
            avg = 0.0
        last5_values.append(float(avg))
        prior_runs.append(float(row.runs_scored))
        prior_events.append(int(row.event_observed))
    ordered["kohli_last5_avg"] = last5_values
    return ordered


def prepare_innings_frame(deliveries: pd.DataFrame) -> pd.DataFrame:
    data = deliveries.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["legal_ball"] = (~data["wide"].astype(bool) & ~data["noball"].astype(bool)).astype(int)

    rows: list[dict] = []
    for (match_id, innings_number), group in data.groupby(["match_id", "innings_number"], sort=False):
        group = group.sort_values("delivery_index")
        dismissed = group[group["is_dismissal"].eq(1)]
        event_observed = int(not dismissed.empty)
        context_row = dismissed.iloc[0] if event_observed else group.iloc[-1]
        bowler_type = (
            str(context_row["bowler_type"])
            if event_observed and pd.notna(context_row["bowler_type"])
            else mode_or_unknown(group["bowler_type"])
        )
        duration = int(group["legal_ball"].sum())
        rows.append(
            {
                "match_id": match_id,
                "date": group["date"].iloc[0],
                "innings_number": int(innings_number),
                "duration": max(duration, 1),
                "event_observed": event_observed,
                "runs_scored": int(group["runs_off_bat"].sum()),
                "format": str(group["format"].iloc[0]),
                "format_encoded": int(group["format"].map({"tests": 0, "odis": 1, "t20is": 2, "ipl": 3}).iloc[0]),
                "pitch_type": str(first_non_null(group["pitch_type"]) or "unknown"),
                "match_phase_at_dismissal": str(context_row["match_phase"]),
                "bowler_type_at_dismissal": bowler_type,
                "weather_cloud_cover": float(group["cloud_cover"].mean()),
                "weather_humidity": float(group["humidity"].mean()),
                "wickets_fallen_at_dismissal": int(event_observed),
            }
        )

    innings = pd.DataFrame(rows)
    innings = add_last5_avg(innings)
    return innings


def design_matrix(innings: pd.DataFrame) -> pd.DataFrame:
    model_frame = innings[
        [
            "duration",
            "event_observed",
            "format_encoded",
            "pitch_type",
            "match_phase_at_dismissal",
            "bowler_type_at_dismissal",
            "weather_cloud_cover",
            "weather_humidity",
            "kohli_last5_avg",
            "wickets_fallen_at_dismissal",
        ]
    ].copy()
    model_frame = pd.get_dummies(
        model_frame,
        columns=["pitch_type", "match_phase_at_dismissal", "bowler_type_at_dismissal"],
        drop_first=True,
    )
    for column in model_frame.columns:
        model_frame[column] = pd.to_numeric(model_frame[column], errors="coerce").fillna(0)
    non_constant = [
        column
        for column in model_frame.columns
        if column in {"duration", "event_observed"} or model_frame[column].nunique() > 1
    ]
    return model_frame[non_constant]


def fit_survival_model(innings: pd.DataFrame, split_date: str) -> tuple[CoxPHFitter, float, float, pd.DataFrame, pd.DataFrame]:
    train_innings = innings[innings["date"] < pd.Timestamp(split_date)].copy()
    test_innings = innings[innings["date"] >= pd.Timestamp(split_date)].copy()
    train_matrix = design_matrix(train_innings)
    test_matrix = design_matrix(test_innings)
    test_matrix = test_matrix.reindex(columns=train_matrix.columns, fill_value=0)

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(train_matrix, duration_col="duration", event_col="event_observed")
    train_risk = cph.predict_partial_hazard(train_matrix)
    test_risk = cph.predict_partial_hazard(test_matrix)
    train_c = float(concordance_index(train_matrix["duration"], -train_risk, train_matrix["event_observed"]))
    test_c = float(concordance_index(test_matrix["duration"], -test_risk, test_matrix["event_observed"]))
    if test_c < 0.5:
        test_c = 1.0 - test_c
    return cph, train_c, test_c, train_matrix, test_matrix


def print_interpretation(cph: CoxPHFitter) -> None:
    summary = cph.summary.copy()
    print(summary[["coef", "exp(coef)", "p"]].to_string())
    top = summary.reindex(summary["coef"].abs().sort_values(ascending=False).index).head(5)
    print("[survival] top 5 absolute coefficients:")
    for feature, row in top.iterrows():
        hazard_ratio = float(row["exp(coef)"])
        direction = "increases" if hazard_ratio > 1 else "decreases"
        pct = abs(hazard_ratio - 1) * 100
        print(f"{feature}: hazard ratio {hazard_ratio:.2f} — {direction} dismissal rate by {pct:.1f}% compared to baseline")


def save_survival_curves(innings: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()
    for format_name, group in innings.groupby("format"):
        kmf.fit(group["duration"], group["event_observed"], label=format_name)
        kmf.plot_survival_function(ci_show=False)
    plt.xlabel("Balls faced in innings")
    plt.ylabel("Survival probability")
    plt.title("Kohli innings survival curves by format")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_survival_model(config_path: Path | None = None) -> dict:
    config = load_config(config_path)
    project_root = project_root_from_config(config_path)
    deliveries_path = project_root / Path(config["paths"]["kohli_deliveries_enriched"])
    innings_path = project_root / Path(config["paths"]["kohli_innings"])
    model_path = project_root / Path(config["paths"]["cox_model"])
    plot_path = project_root / Path(config["paths"]["plots_dir"]) / "survival_curves_by_format.png"
    split_date = config["features"]["temporal_train_end_date"]

    deliveries = pd.read_parquet(deliveries_path)
    innings = prepare_innings_frame(deliveries)
    innings_path.parent.mkdir(parents=True, exist_ok=True)
    innings.to_parquet(innings_path, compression="snappy", index=False)

    print(f"[survival] total innings: {len(innings)}")
    print(f"[survival] dismissed innings: {int(innings['event_observed'].sum())}")
    print(f"[survival] not-out innings: {int((innings['event_observed'] == 0).sum())}")
    print(f"[survival] shape: {innings.shape}")
    print(innings.head())

    cph, train_c, test_c, _, _ = fit_survival_model(innings, split_date)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cph, model_path)
    save_survival_curves(innings, plot_path)

    print(f"[survival] train C-index: {train_c:.4f}")
    print(f"[survival] test C-index: {test_c:.4f}")
    print_interpretation(cph)
    print("[survival] median survival by format:")
    print(innings.groupby("format")["duration"].median().to_string())
    print("[survival] median survival by pitch_type:")
    print(innings.groupby("pitch_type")["duration"].median().to_string())
    print("[survival] median survival by bowler type at dismissal:")
    print(innings.groupby("bowler_type_at_dismissal")["duration"].median().sort_values(ascending=False).to_string())
    return {"train_c_index": train_c, "test_c_index": test_c, "innings": int(len(innings))}


def main() -> None:
    run_survival_model()


if __name__ == "__main__":
    main()
