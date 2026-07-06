import joblib
import pandas as pd
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
STATE_PATH = os.path.join(BASE_DIR, 'data', 'tournament_state.json')

model = joblib.load(os.path.join(OUTPUTS_DIR, 'model_rf.pkl'))
FEATURES = ['RankDiff', 'PointsDiff', 'HomeForm_Pts', 'HomeForm_GF', 'HomeForm_GA',
            'AwayForm_Pts', 'AwayForm_GF', 'AwayForm_GA', 'H2H_HomeWinPct', 'IsNeutralVenue']

with open(STATE_PATH, 'r') as f:
    STATE = json.load(f)
TEAM_RATINGS = STATE['team_ratings']

def predict_match(teamA, teamB):
    a, b = TEAM_RATINGS[teamA], TEAM_RATINGS[teamB]
    row = pd.DataFrame([{
        'RankDiff': b['Rank'] - a['Rank'],
        'PointsDiff': a['Points'] - b['Points'],
        'HomeForm_Pts': a['Form_Pts'], 'HomeForm_GF': a['Form_GF'], 'HomeForm_GA': a['Form_GA'],
        'AwayForm_Pts': b['Form_Pts'], 'AwayForm_GF': b['Form_GF'], 'AwayForm_GA': b['Form_GA'],
        'H2H_HomeWinPct': 0.5,
        'IsNeutralVenue': 1
    }])[FEATURES]
    p = model.predict_proba(row)[0]
    return p[0], p[1], p[2]

def find_team(name):
    """Case-insensitive match against TEAM_RATINGS keys."""
    for team in TEAM_RATINGS:
        if team.lower() == name.strip().lower():
            return team
    return None

def run_check(team_a_input, team_b_input):
    team_a = find_team(team_a_input)
    team_b = find_team(team_b_input)

    if not team_a or not team_b:
        print(f"\nCouldn't find one or both teams in tournament_state.json.")
        print("Available teams:", ", ".join(sorted(TEAM_RATINGS.keys())))
        return

    hw, draw, aw = predict_match(team_a, team_b)
    print(f"\n{team_a} vs {team_b}")
    print(f"  {team_a} win: {hw*100:5.1f}%")
    print(f"  Draw:        {draw*100:5.1f}%")
    print(f"  {team_b} win: {aw*100:5.1f}%")

    favorite = max([(team_a, hw), ('Draw', draw), (team_b, aw)], key=lambda x: x[1])
    print(f"  -> Most likely: {favorite[0]} ({favorite[1]*100:.1f}%)")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        # Usage: python scripts\check_match.py Spain Portugal
        run_check(sys.argv[1], sys.argv[2])
    else:
        # Interactive mode: keeps asking until you type 'exit'
        print("World Cup 2026 Match Checker (type 'exit' to quit)")
        print("Available teams:", ", ".join(sorted(TEAM_RATINGS.keys())))
        while True:
            team_a_input = input("\nEnter Team A: ").strip()
            if team_a_input.lower() == 'exit':
                break
            team_b_input = input("Enter Team B: ").strip()
            if team_b_input.lower() == 'exit':
                break
            run_check(team_a_input, team_b_input)