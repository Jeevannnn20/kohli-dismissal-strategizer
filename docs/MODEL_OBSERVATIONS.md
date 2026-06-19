# Model Observations

This project should be presented as a ranking and strategy-support tool, not a literal wicket predictor. The dismissal class is rare: after cleaning, the dataset contains 753 dismissals across 42,741 Kohli delivery rows, a positive rate of roughly 1.76%.

## Validation Snapshot

- Temporal split only: training data before 2021, test data from 2021 onward.
- Best model: XGBoost.
- Test AUC-ROC: 0.698.
- Test AUC-PR: 0.0433.
- Test positive rate: 2.07%.
- Baseline model: logistic regression AUC-PR 0.0340.
- LightGBM AUC-PR: 0.0372.

The XGBoost model improves ranking quality over the baseline under the correct imbalanced-data metrics. Accuracy is intentionally not used.

## Calibration And Ranking

Raw XGBoost probabilities are high because training used imbalance weighting. Diagnostics showed the raw scores are useful for ranking, while empirical post-2021 deciles are better for display:

- Bottom decile actual dismissal rate: 0.00%.
- Top decile actual dismissal rate: 4.94%.
- The app therefore presents calibrated rank probabilities, not raw model probabilities.

## Strongest SHAP Signals

Top features by mean absolute SHAP value:

1. bowler_type_encoded
2. format_encoded
3. bowler_career_balls_to_kohli
4. bowler_career_dismissals_of_kohli
5. match_phase_encoded
6. over
7. balls_faced_so_far
8. ball
9. runs_scored_so_far
10. wind_speed

The model primarily learns bowling type, format, bowler history, innings phase, and over-context effects.

## Cricket Insights

- Highest dismissal-rate phases: death overs at 4.96%, powerplay at 2.37%.
- Format dismissal rates: IPL 3.21%, T20Is 2.70%, ODIs 1.55%, Tests 1.18%.
- Top dismissal bowler types by observed rate include LAPS, RAPS, LAPF, and LB.
- Most common dismissal mode: caught, about 68.9%.
- Top dismissing bowlers in the cleaned data: Kagiso Rabada 13, Morkel 12, Southee 12.

## Strategy Interpretation

The recommender combines:

- trained XGBoost ranking signal,
- empirical calibration from temporal test deciles,
- bowler-type historical lift,
- pitch-type tactical lift,
- innings pressure adjustment.

This makes the output more suitable for cricket strategy presentation: it answers "which option is better under these conditions?" rather than "what is the exact probability Kohli gets out?"
