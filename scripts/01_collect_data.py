import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# --- Load raw files ---
results = pd.read_csv(os.path.join(RAW_DIR, 'results.csv'))
rankings = pd.read_csv(os.path.join(RAW_DIR, 'rankings.csv'))

print("=== RESULTS ===")
print("Shape:", results.shape)
print("Columns:", results.columns.tolist())
print(results.head(3))

print("\n=== RANKINGS ===")
print("Shape:", rankings.shape)
print("Columns:", rankings.columns.tolist())
print(rankings.head(3))

# --- Clean results ---
results['date'] = pd.to_datetime(results['date'], format='mixed', dayfirst=False, errors='coerce')

bad_dates = results['date'].isna().sum()
if bad_dates > 0:
    print(f"\nWarning: {bad_dates} rows in results had unparseable dates and will be dropped")
    results = results.dropna(subset=['date'])

# Drop rows with missing scores (unplayed/placeholder fixtures) -- MUST happen before building results_clean
missing_scores = results['home_score'].isna().sum()
if missing_scores > 0:
    print(f"Warning: {missing_scores} rows have missing scores (unplayed fixtures) and will be dropped")
    results = results.dropna(subset=['home_score', 'away_score'])

results_clean = results.rename(columns={
    'home_team': 'HomeTeam',
    'away_team': 'AwayTeam',
    'home_score': 'HomeGoals',
    'away_score': 'AwayGoals',
    'tournament': 'Tournament',
    'neutral': 'IsNeutralVenue'
})[['date', 'HomeTeam', 'AwayTeam', 'HomeGoals', 'AwayGoals', 'Tournament', 'IsNeutralVenue']]

results_clean['IsWC2026'] = 0

# --- Clean rankings ---
rankings['date'] = pd.to_datetime(rankings['date'], format='mixed', dayfirst=False, errors='coerce')

bad_rank_dates = rankings['date'].isna().sum()
if bad_rank_dates > 0:
    print(f"Warning: {bad_rank_dates} rows in rankings had unparseable dates and will be dropped")
    rankings = rankings.dropna(subset=['date'])

missing_points = rankings['total_points'].isna().sum()
if missing_points > 0:
    print(f"Warning: {missing_points} rows in rankings had missing points and will be dropped")
    rankings = rankings.dropna(subset=['total_points'])

rankings_clean = rankings.rename(columns={
    'team': 'TeamName',
    'total_points': 'Points'
})[['TeamName', 'date', 'Points']]

# Single, safe rank calculation -- no duplicate, no unsafe int cast
rankings_clean['Rank'] = rankings_clean.groupby('date')['Points'] \
    .rank(method='min', ascending=False).astype('Int64')

# --- Save cleaned versions ---
os.makedirs(PROCESSED_DIR, exist_ok=True)
results_clean.to_csv(os.path.join(PROCESSED_DIR, 'matches_clean.csv'), index=False)
rankings_clean.to_csv(os.path.join(PROCESSED_DIR, 'rankings_clean.csv'), index=False)

print("\nSaved cleaned files to data/processed/")
print("matches_clean shape:", results_clean.shape)
print("rankings_clean shape:", rankings_clean.shape)