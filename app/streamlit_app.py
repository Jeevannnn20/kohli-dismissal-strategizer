"""Streamlit UI for the Kohli Dismissal Strategizer."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.strategy.recommender import BOWLER_TYPES, recommend_strategy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENUES_PATH = PROJECT_ROOT / "data/external/venues.csv"
SHAP_RESULTS_PATH = PROJECT_ROOT / "data/processed/shap_results.pkl"
LABEL_ENCODERS_PATH = PROJECT_ROOT / "data/external/label_encoders.pkl"


FORMAT_OPTIONS = {
    "Tests": "tests",
    "ODIs": "odis",
    "T20Is": "t20is",
    "IPL": "ipl",
}
PHASE_OPTIONS = {
    "tests": {"Opening": "opening", "Middle": "middle", "Tail": "tail"},
    "odis": {"Powerplay": "powerplay", "Middle": "middle", "Death": "death"},
    "t20is": {"Powerplay": "powerplay", "Middle": "middle", "Death": "death"},
    "ipl": {"Powerplay": "powerplay", "Middle": "middle", "Death": "death"},
}
OVER_RANGES = {
    "tests": (0, 90),
    "odis": (0, 49),
    "t20is": (0, 19),
    "ipl": (0, 19),
}


@st.cache_data
def load_venues() -> pd.DataFrame:
    return pd.read_csv(VENUES_PATH)


@st.cache_data
def load_top_shap_features() -> pd.DataFrame:
    payload = joblib.load(SHAP_RESULTS_PATH)
    shap_values = np.asarray(payload["shap_values"])
    feature_names = list(payload["feature_names"])
    mean_abs = np.abs(shap_values).mean(axis=0)
    top = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .head(5)
    )
    return top


@st.cache_data
def load_opposing_teams() -> list[str]:
    if not LABEL_ENCODERS_PATH.exists():
        return ["Australia", "England", "South Africa", "New Zealand", "Pakistan"]
    encoders = joblib.load(LABEL_ENCODERS_PATH)
    if "opposing_team" not in encoders:
        return ["Australia", "England", "South Africa", "New Zealand", "Pakistan"]
    return sorted(str(team) for team in encoders["opposing_team"].classes_)


def probability_bar(probability: float) -> None:
    st.progress(min(max(probability / 0.10, 0), 1))


def main() -> None:
    st.set_page_config(page_title="Kohli Dismissal Strategizer", layout="wide")
    st.title("Kohli Dismissal Strategizer")

    st.sidebar.header("About")
    st.sidebar.write(
        "A strategy assistant for ranking bowling options against Virat Kohli from historical ball-by-ball data. "
        "It combines Cricsheet dismissals with venue, weather, model, and SHAP-derived context."
    )
    st.sidebar.write("Model: XGBoost, AUC-ROC 0.698, AUC-PR 0.0433.")
    st.sidebar.write("Data source: cricsheet.org bulk public match files.")
    st.sidebar.warning(
        "Use probabilities as relative rankings, not absolute truth. Cricket context still matters."
    )

    venues = load_venues()
    venue_names = sorted(venues["venue"].dropna().unique().tolist())
    opposing_teams = load_opposing_teams()

    left, right = st.columns([0.95, 1.25], gap="large")

    with left:
        st.subheader("Inputs")
        selected_format_label = st.selectbox("Format", list(FORMAT_OPTIONS))
        selected_format = FORMAT_OPTIONS[selected_format_label]
        selected_venue = st.selectbox("Venue", venue_names)
        selected_opposing_team = st.selectbox("Opposing team", opposing_teams)
        is_home_match = st.checkbox("Playing in India?", value=False)
        phase_labels = list(PHASE_OPTIONS[selected_format])
        selected_phase_label = st.selectbox("Match phase", phase_labels)
        selected_phase = PHASE_OPTIONS[selected_format][selected_phase_label]
        min_over, max_over = OVER_RANGES[selected_format]
        over = st.slider("Over number", min_over, max_over, min_over)
        available_bowler_types = st.multiselect(
            "Available bowler types",
            BOWLER_TYPES,
            default=["LAPS", "RAPS", "LAPF", "LB"],
        )
        balls_faced = st.number_input(
            "Balls Kohli has faced this innings", min_value=0, max_value=300, value=24
        )
        runs_scored = st.number_input(
            "Runs Kohli has scored this innings", min_value=0, max_value=300, value=20
        )
        wickets_fallen = st.number_input("Wickets fallen", min_value=0, max_value=9, value=2)

        with st.expander("Weather inputs"):
            temperature = st.slider("Temperature °C", 15, 45, 29)
            humidity = st.slider("Humidity %", 20, 100, 50)
            cloud_cover = st.slider("Cloud cover %", 0, 100, 40)
            wind_speed = st.slider("Wind speed km/h", 0, 60, 13)
            precipitation = st.slider("Precipitation mm", 0.0, 20.0, 0.0, step=0.1)

        submitted = st.button("Get Strategy", type="primary")

    with right:
        st.subheader("Results")
        if not submitted:
            st.info("Choose match conditions and press Get Strategy.")
        elif not available_bowler_types:
            st.warning("Select at least one available bowler type.")
        else:
            result = recommend_strategy(
                format=selected_format,
                venue=selected_venue,
                match_phase=selected_phase,
                available_bowler_types=available_bowler_types,
                over=over,
                balls_faced_so_far=balls_faced,
                runs_scored_so_far=runs_scored,
                wickets_fallen=wickets_fallen,
                weather={
                    "temperature": temperature,
                    "humidity": humidity,
                    "cloud_cover": cloud_cover,
                    "wind_speed": wind_speed,
                    "precipitation": precipitation,
                },
                opposing_team=selected_opposing_team,
                is_home_match=is_home_match,
            )

            st.write(result["conditions_summary"])
            st.caption(
                f"Model confidence: {result['model_confidence']} | baseline probability: "
                f"{result['baseline_probability']:.2%} | model used: {result.get('model_used', 'combined')}"
            )

            for recommendation in result["recommendations"]:
                with st.container(border=True):
                    st.markdown(f"#### {recommendation['bowler_type']}")
                    st.metric(
                        "Dismissal probability",
                        f"{recommendation['dismissal_probability']:.2%}",
                        f"{recommendation['relative_lift']:.1f}x baseline",
                    )
                    probability_bar(recommendation["dismissal_probability"])
                    st.write(recommendation["rationale"])

            st.warning(
                "This model is trained on historical ball-by-ball data from cricsheet.org and enriched with venue and weather data. "
                "AUC-PR: 0.043 on post-2021 data. Probabilities indicate relative ranking of delivery types, not absolute predictions. "
                "Use as one input among many, not as sole decision basis."
            )

            st.markdown("#### Top SHAP Features")
            shap_top = load_top_shap_features()
            st.bar_chart(shap_top.set_index("feature")["mean_abs_shap"])


if __name__ == "__main__":
    main()
