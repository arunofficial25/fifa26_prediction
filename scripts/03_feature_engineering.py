import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = int(os.getenv('DB_PORT'))
DB_NAME = os.getenv('DB_NAME')

encoded_password = quote_plus(DB_PASSWORD)
engine = create_engine(f"mysql+pymysql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# --- Pull matches with team names joined in ---
matches = pd.read_sql("""
    SELECT m.MatchID, m.MatchDate, t1.TeamName AS HomeTeam, t2.TeamName AS AwayTeam,
           m.HomeGoals, m.AwayGoals, m.Tournament, m.IsNeutralVenue
    FROM Matches m
    JOIN Teams t1 ON m.HomeTeamID = t1.TeamID
    JOIN Teams t2 ON m.AwayTeamID = t2.TeamID
    ORDER BY m.MatchDate
""", engine)

rankings = pd.read_sql("""
    SELECT t.TeamName, r.RankDate, r.`Rank`, r.Points
    FROM FIFARankings r
    JOIN Teams t ON r.TeamID = t.TeamID
    ORDER BY r.RankDate
""", engine)

matches['MatchDate'] = pd.to_datetime(matches['MatchDate'])
rankings['RankDate'] = pd.to_datetime(rankings['RankDate'])

print(f"Loaded {len(matches)} matches, {len(rankings)} ranking records")

# --- Result label: 0 = Home Win, 1 = Draw, 2 = Away Win ---
def get_result(row):
    if row['HomeGoals'] > row['AwayGoals']:
        return 0
    elif row['HomeGoals'] == row['AwayGoals']:
        return 1
    else:
        return 2

matches['Result'] = matches.apply(get_result, axis=1)

# --- Helper: get each team's most recent rank/points as of a given date ---
rankings_sorted = rankings.sort_values('RankDate')

def get_latest_rank(team, match_date):
    sub = rankings_sorted[(rankings_sorted['TeamName'] == team) & (rankings_sorted['RankDate'] < match_date)]
    if len(sub) == 0:
        return np.nan, np.nan
    last = sub.iloc[-1]
    return last['Rank'], last['Points']

# Build a lookup dict per team: sorted list of (date, rank, points) -- much faster than repeated filtering
team_rank_history = {
    team: grp[['RankDate', 'Rank', 'Points']].sort_values('RankDate').values
    for team, grp in rankings_sorted.groupby('TeamName')
}

def fast_latest_rank(team, match_date):
    hist = team_rank_history.get(team)
    if hist is None:
        return np.nan, np.nan
    mask = hist[:, 0] < np.datetime64(match_date)
    valid = hist[mask]
    if len(valid) == 0:
        return np.nan, np.nan
    return valid[-1, 1], valid[-1, 2]

print("Computing rank/points for each match (this may take a few minutes)...")
home_ranks, home_points, away_ranks, away_points = [], [], [], []

for _, row in matches.iterrows():
    hr, hp = fast_latest_rank(row['HomeTeam'], row['MatchDate'])
    ar, ap = fast_latest_rank(row['AwayTeam'], row['MatchDate'])
    home_ranks.append(hr); home_points.append(hp)
    away_ranks.append(ar); away_points.append(ap)

matches['HomeRank'] = home_ranks
matches['HomePoints'] = home_points
matches['AwayRank'] = away_ranks
matches['AwayPoints'] = away_points

matches['RankDiff'] = matches['AwayRank'] - matches['HomeRank']   # positive = home team ranked better
matches['PointsDiff'] = matches['HomePoints'] - matches['AwayPoints']

print("Rank features done.")
print(matches[['MatchDate','HomeTeam','AwayTeam','HomeRank','AwayRank','RankDiff']].tail(10))

# --- Save checkpoint before the heavier form/h2h computation ---
matches.to_csv(os.path.join(PROCESSED_DIR, 'matches_with_rank.csv'), index=False)
print("\nCheckpoint saved: matches_with_rank.csv")

# --- Recompute Result if not already present (needed for form calc) ---
# (already computed above as matches['Result'])

print("\nComputing form, head-to-head, and goal averages (chronological pass)...")

matches = matches.sort_values('MatchDate').reset_index(drop=True)

# Track each team's match history as we go: list of dicts {date, goals_for, goals_against, points}
team_history = {}   # team -> list of (date, points_earned, goals_for, goals_against)
h2h_history = {}    # frozenset({teamA, teamB}) -> list of (date, winner) where winner is team name or 'Draw'

def get_form(team, n=5):
    hist = team_history.get(team, [])
    recent = hist[-n:]
    if len(recent) == 0:
        return np.nan, np.nan, np.nan
    avg_points = np.mean([h[1] for h in recent])
    avg_gf = np.mean([h[2] for h in recent])
    avg_ga = np.mean([h[3] for h in recent])
    return avg_points, avg_gf, avg_ga

def get_h2h_win_pct(teamA, teamB):
    key = frozenset([teamA, teamB])
    hist = h2h_history.get(key, [])
    if len(hist) == 0:
        return np.nan
    a_wins = sum(1 for _, winner in hist if winner == teamA)
    return a_wins / len(hist)

home_form_pts, home_form_gf, home_form_ga = [], [], []
away_form_pts, away_form_gf, away_form_ga = [], [], []
h2h_pct = []

for _, row in matches.iterrows():
    home, away = row['HomeTeam'], row['AwayTeam']

    hp, hgf, hga = get_form(home)
    ap, agf, aga = get_form(away)
    home_form_pts.append(hp); home_form_gf.append(hgf); home_form_ga.append(hga)
    away_form_pts.append(ap); away_form_gf.append(agf); away_form_ga.append(aga)

    h2h_pct.append(get_h2h_win_pct(home, away))

    # --- update history AFTER computing features (avoid leakage) ---
    if row['Result'] == 0:
        home_pts, away_pts, winner = 3, 0, home
    elif row['Result'] == 1:
        home_pts, away_pts, winner = 1, 1, 'Draw'
    else:
        home_pts, away_pts, winner = 0, 3, away

    team_history.setdefault(home, []).append((row['MatchDate'], home_pts, row['HomeGoals'], row['AwayGoals']))
    team_history.setdefault(away, []).append((row['MatchDate'], away_pts, row['AwayGoals'], row['HomeGoals']))

    key = frozenset([home, away])
    h2h_history.setdefault(key, []).append((row['MatchDate'], winner))

matches['HomeForm_Pts'] = home_form_pts
matches['HomeForm_GF'] = home_form_gf
matches['HomeForm_GA'] = home_form_ga
matches['AwayForm_Pts'] = away_form_pts
matches['AwayForm_GF'] = away_form_gf
matches['AwayForm_GA'] = away_form_ga
matches['H2H_HomeWinPct'] = h2h_pct

print("Form and H2H features done.")
print(matches[['MatchDate','HomeTeam','AwayTeam','HomeForm_Pts','AwayForm_Pts','H2H_HomeWinPct']].tail(10))

# --- Save final feature-engineered dataset ---
final_path = os.path.join(PROCESSED_DIR, 'matches_features_final.csv')
matches.to_csv(final_path, index=False)
print(f"\nSaved final dataset: {final_path}")
print("Final shape:", matches.shape)