import pandas as pd
import os
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# --- CONNECTION ---
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = int(os.getenv('DB_PORT'))
DB_NAME = os.getenv('DB_NAME')

encoded_password = quote_plus(DB_PASSWORD)
engine = create_engine(f"mysql+pymysql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# --- Load cleaned data ---
matches = pd.read_csv(os.path.join(PROCESSED_DIR, 'matches_clean.csv'))
rankings = pd.read_csv(os.path.join(PROCESSED_DIR, 'rankings_clean.csv'))

# --- Build unique Teams list from both files ---
teams = pd.concat([
    matches['HomeTeam'], matches['AwayTeam'], rankings['TeamName']
]).dropna().unique()
teams_df = pd.DataFrame({'TeamName': sorted(teams)})

print(f"Unique teams found: {len(teams_df)}")

# --- Insert Teams first (so we can map TeamID) ---
teams_df.to_sql('Teams', engine, if_exists='append', index=False)

# --- Pull back TeamID mapping ---
team_map = pd.read_sql("SELECT TeamID, TeamName FROM Teams", engine)
team_lookup = dict(zip(team_map['TeamName'], team_map['TeamID']))

# --- Map Matches to TeamIDs ---
matches['HomeTeamID'] = matches['HomeTeam'].map(team_lookup)
matches['AwayTeamID'] = matches['AwayTeam'].map(team_lookup)

# --- Diagnose unmapped teams before inserting ---
unmapped_home = matches[matches['HomeTeamID'].isna()]['HomeTeam'].unique()
unmapped_away = matches[matches['AwayTeamID'].isna()]['AwayTeam'].unique()

print(f"\nUnmapped HomeTeam values ({len(unmapped_home)}):", unmapped_home[:20])
print(f"Unmapped AwayTeam values ({len(unmapped_away)}):", unmapped_away[:20])

rows_before = len(matches)
matches = matches.dropna(subset=['HomeTeamID', 'AwayTeamID'])
rows_after = len(matches)
print(f"\nDropped {rows_before - rows_after} rows with unmapped teams. Proceeding with {rows_after} rows.")

matches['HomeTeamID'] = matches['HomeTeamID'].astype(int)
matches['AwayTeamID'] = matches['AwayTeamID'].astype(int)

matches_sql = matches.rename(columns={'date': 'MatchDate'})[
    ['MatchDate', 'HomeTeamID', 'AwayTeamID', 'HomeGoals', 'AwayGoals',
     'Tournament', 'IsNeutralVenue', 'IsWC2026']
]

matches_sql.to_sql('Matches', engine, if_exists='append', index=False, chunksize=1000)
print(f"Inserted {len(matches_sql)} matches")

# --- Map Rankings to TeamIDs ---
rankings['TeamID'] = rankings['TeamName'].map(team_lookup)

unmapped_rank = rankings[rankings['TeamID'].isna()]['TeamName'].unique()
print(f"\nUnmapped ranking TeamNames ({len(unmapped_rank)}):", unmapped_rank[:20])
rankings = rankings.dropna(subset=['TeamID'])
rankings['TeamID'] = rankings['TeamID'].astype(int)

rankings_sql = rankings.rename(columns={'date': 'RankDate', 'Rank': 'Rank'})[
    ['TeamID', 'RankDate', 'Rank', 'Points']
]

rankings_sql.to_sql('FIFARankings', engine, if_exists='append', index=False, chunksize=1000)
print(f"Inserted {len(rankings_sql)} ranking records")

print("\nDone. Data loaded into MySQL.")