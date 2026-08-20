# NVDA Data Pipeline, ML & Dashboard

End-to-end project on NVIDIA's 1999-present dataset: ETL -> feature
engineering -> walk-forward ML classification -> interactive dashboard.

## Setup

```bash
python -m venv venv && source venv/bin/activate      # optional but recommended
pip install -r requirements.txt
```

## Run the pipeline (in order)

```bash
python pipeline/etl.py              # -> data/processed/master_daily.parquet
python features/build_features.py   # -> data/processed/features_*.parquet
python models/train.py              # -> data/processed/predictions.parquet, model_metrics.json
streamlit run app/dashboard.py      # opens the dashboard in your browser
```

## Project structure

```
nvda-project/
├── data/
│   ├── raw/            # original 16 CSVs
│   └── processed/      # ETL + feature engineering + model outputs (generated)
├── pipeline/etl.py            # cleans & merges all raw sources into one daily table
├── features/build_features.py # technical indicators, targets, two feature sets
├── models/
│   ├── train.py         # walk-forward training + evaluation of 3 models + 2 baselines
│   └── artifacts/        # saved model files (generated)
└── app/dashboard.py      # Streamlit dashboard (4 tabs)
```

## Design decisions worth knowing about

**Two feature sets, on purpose.** The raw data has very different coverage
per source: daily prices go back to 1999, analyst upgrades/downgrades to
2016, earnings dates to 2020, and fundamentals only to ~2022 (annual) or
2025 (quarterly). Rather than truncate everything to the shortest series,
`build_features.py` produces:
- `features_long_history.parquet` — price/technical features only, full
  1999-2026 span, ~6,800 rows.
- `features_fundamentals_enriched.parquet` — adds annual fundamentals,
  earnings proximity, and analyst rolling activity, ~2022-2026, ~850 rows.

This is a deliberate bias-variance tradeoff (more data vs. richer
features) and is worth a paragraph in your writeup.

**Walk-forward validation, not k-fold.** `models/train.py` uses expanding-
window splits so every test fold is strictly after its training data.
Randomly shuffled k-fold CV on time series data leaks the future into
training and inflates accuracy — a common mistake this project explicitly
avoids.

**Lookahead-safe fundamentals.** In `pipeline/etl.py`, fundamentals are
joined with `merge_asof(..., direction="backward")` against the *report
date*, not the fiscal period end date. A quarter's numbers only become
"visible" to the model starting the day they were actually reported.

## Results (long-history feature set, next-day direction)

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Naive (repeat yesterday) | ~0.49 | 0.53 | n/a |
| Majority class | ~0.52 | 0.56 | n/a |
| Logistic Regression | ~0.53 | 0.62 | ~0.49 |
| Random Forest | ~0.53 | 0.64 | ~0.50 |
| XGBoost | ~0.51 | 0.57 | ~0.50 |

All models land close to the majority-class baseline, and ROC-AUC sits
right around 0.5 (coin-flip) across the board. **This is the expected,
honest result** — next-day direction for a single liquid stock is close
to market-efficient, and a model that can't beat a naive baseline here is
a *correct* finding, not a failed project. The interesting analysis is in
*why* (see dashboard's Model tab for rolling accuracy and confusion
matrix), not in chasing a higher number by overfitting.

## Possible extensions

- Try the `features_fundamentals_enriched` set — smaller sample, richer
  signal, good contrast to discuss in a report.
- Predict `target_up_5d` (5-day horizon) instead of 1-day — often less
  noisy.
- Add SHAP values for the Random Forest / XGBoost models to explain which
  features actually drive predictions.
- Extend the backtest to include transaction costs and position sizing.
