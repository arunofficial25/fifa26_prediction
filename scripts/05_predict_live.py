import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

model = joblib.load(os.path.join(OUTPUTS_DIR, 'model_rf.pkl'))

FEATURES = ['RankDiff', 'PointsDiff', 'HomeForm_Pts', 'HomeForm_GF', 'HomeForm_GA',
            'AwayForm_Pts', 'AwayForm_GF', 'AwayForm_GA', 'H2H_HomeWinPct', 'IsNeutralVenue']

# --- Manually verified fixture data (source: FIFA rankings, ESPN/SI match reports, July 6 2026) ---
# NOTE: Form values are informed estimates based on verified W/D/L tournament form,
# not exact historical joins (2025-26 season isn't in our training DB). Documented here
# for transparency -- a real production system would need a live-updating rankings/results feed.

fixtures = [
    {
        'Home': 'Morocco', 'Away': 'France', 'Stage': 'Quarterfinal', 'Date': '2026-07-09',
        'RankDiff': -9,          # France (#3) ranked well ahead of Morocco (~#13, using FIFA points gap)
        'PointsDiff': -120,      # approx FIFA points gap, France higher
        'HomeForm_Pts': 2.2,     # Morocco: unbeaten but several draws -> steady, not maximal
        'HomeForm_GF': 1.4, 'HomeForm_GA': 0.6,
        'AwayForm_Pts': 3.0,     # France: 5 wins from 5 -> maximum points
        'AwayForm_GF': 2.8, 'AwayForm_GA': 0.4,
        'H2H_HomeWinPct': 0.0,   # France beat Morocco in 2022 WC semifinal, historically dominant in this pairing
        'IsNeutralVenue': 1
    },
    {
        'Home': 'Norway', 'Away': 'England', 'Stage': 'Quarterfinal', 'Date': '2026-07-11',
        'RankDiff': 15,          # Norway ranked lower than England historically
        'PointsDiff': 90,
        'HomeForm_Pts': 2.4,     # Norway: strong knockout form, beat Brazil
        'HomeForm_GF': 2.0, 'HomeForm_GA': 1.0,
        'AwayForm_Pts': 2.0,     # England: wins but shaky/narrow
        'AwayForm_GF': 1.6, 'AwayForm_GA': 1.4,
        'H2H_HomeWinPct': 0.35,  # Norway has an underdog H2H record vs England historically
        'IsNeutralVenue': 1
    },
]

df = pd.DataFrame(fixtures)
X = df[FEATURES]

probs = model.predict_proba(X)
df['HomeWinProb'] = probs[:, 0]
df['DrawProb'] = probs[:, 1]
df['AwayWinProb'] = probs[:, 2]
df['Prediction'] = probs.argmax(axis=1)
df['Prediction'] = df['Prediction'].map({0: 'Home Win', 1: 'Draw', 2: 'Away Win'})

print(df[['Home', 'Away', 'Stage', 'HomeWinProb', 'DrawProb', 'AwayWinProb', 'Prediction']].to_string(index=False))

df.to_csv(os.path.join(OUTPUTS_DIR, 'predictions_live.csv'), index=False)
print(f"\nSaved to outputs/predictions_live.csv")