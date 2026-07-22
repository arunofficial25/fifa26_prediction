# 🏆 WC2026 Predictor

A machine learning system that predicts FIFA World Cup 2026 match outcomes and simulates the entire tournament bracket to a champion probability — trained on 47,000+ international matches (1996–2024) and updated live as the real 2026 tournament unfolded.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-orange)
![MySQL](https://img.shields.io/badge/MySQL-Database-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Live Accuracy](https://img.shields.io/badge/Live%20Accuracy-83%25-success)

---

## 🇪🇸 Champions: Spain

> 🎉🏆 **Spain are the 2026 FIFA World Cup Champions** 🏆🎉
>
> Beat Argentina 1–0 after extra time in the final at MetLife Stadium — Spain's second World Cup title, first since 2010. The model's champion pick tracked Spain through the entire knockout stage, correctly calling every one of their matches from Round of 16 to the Final.
>
> 🥇 Spain · 🥈 Argentina · 🥉 England
>
> 🇪🇸 💐 ⚽ 🏆 🎊

---

## What it does

- Predicts Win / Draw / Loss probabilities for any international matchup
- Simulates the full knockout bracket — quarterfinals through the final — carrying uncertainty through every round instead of just picking one path
- Self-updated as real 2026 results came in (live API, with a config-file fallback)
- One command answered "who wins the World Cup?" — and got it right

```bash
python scripts/run_prediction.py
```

```
========================================
  WORLD CUP 2026 -- CHAMPION
========================================
1.  Spain             🥇
2.  Argentina         🥈
3.  England           🥉

>>>>> champion: Spain


========================================
Argentina vs Spain (Final)
========================================
Argentina        30.78%
Draw             37.97%
Spain            31.25%
----------------------------------------
Most likely: Draw (37.97%)

Match went to extra time, Spain won 1-0
========================================
```

## Live Prediction Performance

Every knockout prediction this model made during the actual 2026 tournament, tracked against real results — start to finish.

| Metric | Value |
|---|---:|
| Total Predictions | 12 |
| Correct | 10 |
| Incorrect | 2 |
| **Final Accuracy** | **83%** |
| Tournament Complete | July 20, 2026 |

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
| France vs Spain (SF) | Spain | Spain | ✅ |
| Argentina vs England (SF) | Argentina | Argentina | ✅ |
| France vs England (3rd Place) | France | England (6-4) | ❌ |
| **Argentina vs Spain (Final)** | **Spain** | **Spain (1-0, AET)** | ✅ |

> Knockout matches are scored on who advances — including extra time and penalty shootouts. Both misses were the model's own flagged weak spots: Switzerland/Colombia came down to penalties in a near-even matchup, and the third-place playoff (a low-stakes consolation game with rotated lineups) is inherently the least predictable fixture in any tournament — team ratings assume full-strength sides.

## Why this isn't just "another ML project"

Football is one of the most upset-prone sports to model — low-scoring games have high variance, and even professional analytics shops (Opta, 538) top out around 55–65% match accuracy. This project treats that honestly:

- **Calibration over vanity accuracy.** Evaluated with log loss, not just accuracy, because a well-calibrated 55% beats an overconfident 70% that's secretly overfit.
- **Two models, compared transparently.** Random Forest vs. draw-weighted XGBoost — documented precision/recall tradeoff rather than cherry-picking the best-looking number.
- **Zero data leakage.** Every feature (form, head-to-head, rank) is computed strictly from matches *before* the one being predicted.
- **Uncertainty carried through the bracket**, not collapsed at each round — a team's title odds reflected every possible path, not just the most likely one.
- **Live-tracked, not just backtested.** The 83% above is real 2026 knockout results, start to finish, not a holdout set picked after the fact.

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
git clone https://github.com/arunofficial25/fifa26_prediction.git
cd fifa26_prediction
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env           # then fill in your DB credentials + API key

# 3. Set up the database (MySQL Workbench)
#    Run sql/database.sql to create WC2026Predictor and its tables

# 4. Build the pipeline
python scripts/01_collect_data.py
python scripts/02_load_to_sql.py
python scripts/03_feature_engineering.py
python scripts/04_train_model.py

# 5. Get predictions
python scripts/run_prediction.py                  # full bracket → champion
python scripts/check_match.py Spain Argentina     # any single matchup
```

> Raw datasets aren't included in this repo (see `.gitignore`) — `01_collect_data.py` documents the exact sources: [International football results 1872–2024](https://github.com/JamshedAli18/International-football-results-from-1872-to-2024) and [FIFA World Ranking history](https://github.com/Dato-Futbol/fifa-ranking).

## Project Structure

```
fifa26_prediction/
├── data/
│   ├── raw/                    # historical results + rankings (gitignored)
│   ├── processed/              # cleaned, feature-engineered datasets
│   ├── tournament_state.json   # tournament state (ratings, bracket, confirmed results)
│   ├── match_details.json      # scorelines for every confirmed match
│   └── prediction_history.json # full record of predicted vs. actual outcomes
├── sql/                         # database schema
├── scripts/
│   ├── 01_collect_data.py
│   ├── 02_load_to_sql.py
│   ├── 03_feature_engineering.py
│   ├── 04_train_model.py       # Random Forest
│   ├── 05_predict_live.py
│   ├── 06_bracket_simulator.py
│   ├── 07_fetch_results.py     # API-first, config-file fallback
│   ├── 08_fetch_match_details.py
│   ├── run_prediction.py       # one-command full pipeline
│   └── check_match.py          # interactive single-match lookup
├── notebooks/
│   └── eda.ipynb
├── outputs/                    # trained models, predictions (gitignored)
├── requirements.txt
└── README.md
```

## Dashboard

Live-updating bracket visualization, built in Next.js: [fifa26_dashboard](https://github.com/arunofficial25/fifa26_dashboard)

## Future Work

- Poisson goal-simulation model as a second, independent prediction method
- Extend the interactive match checker to the full 368-team historical database, not just the 12 teams tracked through the knockout stage

## Repository Statistics

- 47,324 historical matches
- 28 years of data (1996–2024)
- 368 national teams
- 10 engineered features
- 2 ML models compared
- MySQL database
- **Final live 2026 tournament tracking: 83% accuracy (10/12)**

---

Built by Arun

[LinkedIn](https://linkedin.com/in/arunofficial25) · [Website](https://arunofficial25.vercel.app) · [GitHub](https://github.com/arunofficial25)