# Kohli Dismissal Strategizer

Kohli Dismissal Strategizer is a machine learning project for analyzing Virat Kohli's dismissal patterns across Tests, ODIs, T20Is, and IPL matches. Given match format, venue, pitch type, weather conditions, and available bowler types, the finished system will estimate dismissal probability for delivery profiles and recommend actionable bowling strategies.

The project uses only free and public data sources. The primary cricket data source is Cricsheet bulk downloads, with historical weather enrichment from the Open-Meteo Historical Weather API.

## Demo

```bash
streamlit run app/streamlit_app.py
```

The app loads the trained XGBoost model, saved feature encoders, venue metadata, and SHAP results. It ranks available bowler types for the selected match conditions and explains the tactical rationale.

## Constraints

- Kohli's Cricsheet player identifier is `V Kohli`.
- The dismissal ball is identified by `player_dismissed == "V Kohli"`.
- Train/test splitting must be temporal: pre-2021 data for training, 2021 onwards for testing.
- Evaluation must use AUC-ROC and precision-recall AUC. Do not use accuracy for model assessment.
- Every script in `src/` must have a corresponding test in `tests/`.
- Use `pathlib.Path` for file paths.
- Keep constants in `config.yaml`.

## Setup

```bash
cd kohli-strategizer
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Project Structure

```text
kohli-strategizer/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── src/
│   ├── ingestion/
│   ├── enrichment/
│   ├── features/
│   ├── models/
│   └── strategy/
├── notebooks/
├── app/
├── tests/
├── requirements.txt
├── config.yaml
├── pytest.ini
├── .env.example
└── README.md
```

## Phase 1: Data Ingestion

Download Cricsheet bulk archives for Tests, ODIs, T20Is, and IPL. Parse match files into delivery-level records and isolate deliveries faced by `V Kohli`.

## Phase 2: Data Enrichment

Add venue metadata from `data/external/venues.csv`, bowler type lookups from `data/external/bowler_types.csv`, and historical weather features from Open-Meteo.

## Phase 3: Feature Engineering

Build reproducible feature pipelines for match context, innings state, delivery profile, venue conditions, pitch type, weather, and bowler attributes.

## Phase 4: Modeling

Train baseline scikit-learn models and main XGBoost/LightGBM models using a temporal split. Evaluate with AUC-ROC and precision-recall AUC.

## Phase 5: Interpretability

Use SHAP to explain the strongest drivers of dismissal risk by format, venue, bowler type, and delivery profile.

## Phase 6: Strategy Engine

Convert model predictions into ranked bowling recommendations for the selected match conditions and available bowler types.

## Phase 7: Streamlit UI

Build an interactive Streamlit app for entering match conditions and viewing recommended delivery strategies, predicted risk bands, and model explanations.

## Run The Full Pipeline

```bash
python -m src.ingestion.pipeline
python -m src.enrichment.pipeline
python -m src.models.pipeline
```

The first enrichment run fetches historical weather data from Open-Meteo and caches it in `data/external/weather_cache.parquet`.

## Launch The App

```bash
streamlit run app/streamlit_app.py
```

The app loads the pre-trained XGBoost model and saved feature artifacts. It does not retrain models.

## Results

- Best model: XGBoost
- XGBoost AUC-ROC: 0.698
- XGBoost AUC-PR: 0.043
- Top dismissal bowler types: LAPS, RAPS, LAPF, LB
- Highest risk phases: death overs (4.96%), powerplay (2.37%)
- Most common dismissal: caught (68.9%)
- Top dismissing bowlers: Rabada (13), Morkel (12), Southee (12)

The model is most useful for ranking strategy options, not for treating a probability as a literal forecast.

See [docs/MODEL_OBSERVATIONS.md](docs/MODEL_OBSERVATIONS.md) for the validation narrative, SHAP observations, calibration notes, and model-interpretation caveats.

## Tier 1 Improvements

### Survival Analysis (Cox Proportional Hazards)

**Model performance:**
- C-index: 0.7848 on post-2021 test data (78% of innings correctly
  ranked by predicted longevity — strongest model in the project)
- Training innings: 753 dismissed, 140 not-out (censored)

**Hazard ratios (features that increase dismissal rate per ball):**

| Feature | Hazard Ratio | Interpretation |
|---|---|---|
| Opening phase | 13.25 | Dismissal hazard is front-loaded — survive early or get out fast |
| Wickets fallen | 6.97 | Each additional wicket fallen increases dismissal rate 7x |
| Powerplay | 6.70 | Aggressive intent in powerplay creates elevated risk |
| Middle overs | 1.88 | Moderate elevation vs tail baseline |
| Seam pitch | 1.16 | 16% higher dismissal rate on seaming surfaces |
| LAPS bowler | 1.16 | Left-arm pace seam independently validates empirical lift layer |
| Tail phase | 0.32 (negative coef) | Once set in 60th+ over of a Test, extremely hard to dismiss |

**Median balls faced before dismissal:**

| Format | Median balls |
|---|---|
| Tests | 55.5 |
| ODIs | 46.5 |
| T20Is | 23.5 |
| IPL | 23.0 |

| Pitch type | Median balls |
|---|---|
| Bounce (Australia) | 40.0 |
| Seam (England/NZ/SA) | 34.0 |
| Spin (subcontinent) | 30.0 |
| Flat (subcontinent) | 29.5 |
| Neutral | 25.0 |

Kohli lasts longest on bounce pitches (40 balls) — Australian
conditions suit him despite Australia having the most dismissals
in absolute terms (113). He is most vulnerable on flat and spin
subcontinental pitches in short formats.

**Opponent dismissal rates:**

| Opponent | Dismissals | Balls faced | Rate |
|---|---|---|---|
| Mumbai Indians | 30 | 753 | 3.98% |
| Rajasthan Royals | 26 | 739 | 3.52% |
| Chennai Super Kings | 31 | 911 | 3.40% |
| New Zealand | 67 | 3,959 | 1.69% |
| England | 98 | 5,878 | 1.67% |
| Australia | 113 | 7,542 | 1.50% |
| Bangladesh | 27 | 1,860 | 1.45% |
| West Indies | 65 | 4,602 | 1.41% |
| Sri Lanka | 65 | 4,641 | 1.40% |
| South Africa | 59 | 4,742 | 1.24% |

IPL franchise rates (3-4%) reflect format dismissal rate, not
opponent-specific vulnerability. Among international opponents,
New Zealand and England dismiss Kohli most efficiently per ball.
South Africa has the lowest rate despite producing his most
frequent individual dismisser (Rabada, 13 dismissals).

## Limitations

- No ball-tracking data is available from the free public source, so line, length, swing, seam, and release-point detail are absent.
- The target is heavily imbalanced at roughly a 1.76% positive rate, which limits precision.
- Probabilities should be interpreted as relative rankings, not calibrated absolutes.
- Weather enrichment uses a 2pm local proxy, not exact match-session weather.
