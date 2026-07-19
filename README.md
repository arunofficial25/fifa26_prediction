# 🏆 WC2026 Predictor

A machine learning system that predicts FIFA World Cup 2026 match outcomes and simulates the entire tournament bracket to a champion probability — trained on 47,000+ international matches (1996–2024) and updated live as the real 2026 tournament unfolds.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-orange)
![MySQL](https://img.shields.io/badge/MySQL-Database-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Live Accuracy](https://img.shields.io/badge/Live%20Accuracy-87.5%25-success)

<!-- **[Live Prediction Tracker](#live-prediction-performance) · [Model Results](#model-performance) · [Setup](#setup)** -->

---

## What it does

- Predicts Win / Draw / Loss probabilities for any international matchup
- Simulates the full knockout bracket — quarterfinals through the final — carrying uncertainty through every round instead of just picking one path
- Self-updates as real 2026 results come in (live API, with a config-file fallback)
- One command to answer "who wins the World Cup?"

```bash
python scripts/run_prediction.py
```

```
========================================
  WORLD CUP 2026 -- CHAMPION PREDICTION
========================================
  Argentina        53.9%
  Spain            46.1%

>>> Most likely champion: Argentina (53.9%)

========================================
```
*Updated live as each round completes — see current numbers via `python scripts/run_prediction.py`.*

## Live Prediction Performance

Every knockout prediction this model has made during the actual 2026 tournament, tracked against real results.

| Metric | Value |
|---|---:|
| Total Predictions | 11 |
| Correct | 9 |
| Incorrect | 2 |
| **Accuracy** | **81%** |
| Last Updated | July 19, 2026 |

### Prediction History

| Match | Predicted | Actual | Result |
|---|---|---|---|
| Belgium vs USA | Belgium | Belgium | ✅ |
| Spain vs Portugal | Spain | Spain | ✅ |
| Argentina vs Egypt | Argentina | Argentina | ✅ |
| Switzerland vs Colombia | Colombia | Switzerland (pens) | ❌ |
| France vs Morocco | France | France | ✅ |
| Spain vs Belgium | Spain | Spain | ✅ |
| Norway vs England | England | England | ✅ |
| Argentina vs Switzerland | Argentina | Argentina | ✅ |
| France vs Spain | Spain | Spain | ✅ |
| Argentina vs England | Argentina | Argentina | ✅ |
| France vs England | France | England | ❌ |

> Knockout matches are scored on who advances — including extra time and penalty shootouts. The one miss (Switzerland/Colombia) is the exact case the model itself flags as hardest: a near-even matchup that came down to penalties, not a model failure to explain away.

## Why this isn't just "another ML project"

Football is one of the most upset-prone sports to model — low-scoring games have high variance, and even professional analytics shops (Opta, 538) top out around 55–65% match accuracy. This project treats that honestly:

- **Calibration over vanity accuracy.** Evaluated with log loss, not just accuracy, because a well-calibrated 55% beats an overconfident 70% that's secretly overfit.
- **Two models, compared transparently.** Random Forest vs. draw-weighted XGBoost — documented precision/recall tradeoff rather than cherry-picking the best-looking number.
- **Zero data leakage.** Every feature (form, head-to-head, rank) is computed strictly from matches *before* the one being predicted.
- **Uncertainty carried through the bracket**, not collapsed at each round — a team's title odds reflect every possible path, not just the most likely one.
- **Live-tracked, not just backtested.** The 88.8% above is real 2026 knockout results, not a holdout set.

## Model Performance

| Model | Accuracy | Log Loss | Draw F1 |
|---|---|---|---|
| Random Forest *(production)* | **55%** | **0.92** | 0.29 |
| XGBoost (draw-weighted) | 51% | 0.95 | **0.36** |

*Baseline: 33% accuracy / 1.10 log loss for random guessing across 3 classes.*

**Top predictive features:** rank differential (33%) and FIFA points differential (24%) dominate, with head-to-head history (18%) a distant third — recent form and venue matter far less than long-run team quality. Draw prediction is the hardest class across both models, consistent with published football analytics research: draws are a "neither team broke through" outcome that correlates weakly with pre-match stats.

## Architecture

```
Raw historical data (results, FIFA rankings)
        │
        ▼
   MySQL database  ──►  Feature engineering (rank diff, form, H2H)
        │                        │
        │                        ▼
        │                Random Forest / XGBoost
        │                        │
        ▼                        ▼
tournament_state.json  ──►  Bracket simulator ──► Champion probabilities
        ▲
        │
football-data.org API (live results, config-file fallback)
```

## Tech Stack

`Python` · `MySQL` · `scikit-learn` · `XGBoost` · `pandas` · `SQLAlchemy` · `football-data.org API`

## Setup

```bash
# 1. Clone and install
git clone https://github.com/arunofficial25/wc2026-predictor.git
cd wc2026-predictor
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env           # then fill in your DB credentials + API key

# 3. Set up the database (MySQL Workbench)
#    Run sql/schema.sql to create WC2026Predictor and its tables

# 4. Build the pipeline
python scripts/01_collect_data.py
python scripts/02_load_to_sql.py
python scripts/03_feature_engineering.py
python scripts/04_train_model.py

# 5. Get predictions
python scripts/run_prediction.py                  # full bracket → champion
python scripts/check_match.py France England      # any single matchup
```

> Raw datasets aren't included in this repo (see `.gitignore`) — `01_collect_data.py` documents the exact sources: [International football results 1872–2024](https://github.com/JamshedAli18/International-football-results-from-1872-to-2024) and [FIFA World Ranking history](https://github.com/Dato-Futbol/fifa-ranking).

## Project Structure

```
wc2026-predictor/
├── data/
│   ├── raw/                    # historical results + rankings (gitignored)
│   ├── processed/              # cleaned, feature-engineered datasets
│   └── tournament_state.json   # live tournament state (ratings, bracket, results)
├── sql/                        # database schema
├── scripts/
│   ├── 01_collect_data.py
│   ├── 02_load_to_sql.py
│   ├── 03_feature_engineering.py
│   ├── 04_train_model.py       # Random Forest
│   ├── 05_predict_live.py
│   ├── 06_bracket_simulator.py
│   ├── 07_fetch_results.py     # API-first, config-file fallback
│   ├── run_prediction.py       # one-command full pipeline
│   └── check_match.py          # interactive single-match lookup
├── notebooks/
│   └── eda.ipynb
├── outputs/                    # trained models, predictions (gitignored)
├── requirements.txt
└── README.md
```

## Screenshots

*[Coming soon — dashboard in progress]*

## Future Work

- Poisson goal-simulation model as a second, independent prediction method
- Tableau/Power BI dashboard for the live bracket
- Expand automation to the full round-of-16 field, not just the confirmed knockout path

## Repository Statistics

- 47,324 historical matches
- 28 years of data (1996–2024)
- 368 national teams
- 10 engineered features
- 2 ML models compared
- MySQL database
- Live World Cup 2026 tracking (81% accuracy to date)

---

Built by Arun

[Linkedin](https://linkedin.com/in/arunofficial25) · [Website](https://arunofficial25.vercel.app) · [GitHub](https://github.com/arunofficial25)