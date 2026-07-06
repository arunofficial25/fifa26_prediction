import joblib
import pandas as pd
import json
import os
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
STATE_PATH = os.path.join(BASE_DIR, 'data', 'tournament_state.json')

model = joblib.load(os.path.join(OUTPUTS_DIR, 'model_rf.pkl'))
FEATURES = ['RankDiff', 'PointsDiff', 'HomeForm_Pts', 'HomeForm_GF', 'HomeForm_GA',
            'AwayForm_Pts', 'AwayForm_GF', 'AwayForm_GA', 'H2H_HomeWinPct', 'IsNeutralVenue']

with open(STATE_PATH, 'r') as f:
    STATE = json.load(f)

TEAM_RATINGS = STATE['team_ratings']
RESULTS = STATE['confirmed_results']
BRACKET = STATE['bracket']

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

def advance_match(dist_a, dist_b):
    result = defaultdict(float)
    for team_a, p_a in dist_a.items():
        for team_b, p_b in dist_b.items():
            hw, draw, aw = predict_match(team_a, team_b)
            result[team_a] += p_a * p_b * (hw + 0.5 * draw)
            result[team_b] += p_a * p_b * (aw + 0.5 * draw)
    return dict(result)

def get_leg_result(leg_key, result_key):
    """
    If the R16 leg is already decided (per confirmed_results), return a locked 100% distribution.
    Otherwise, simulate it as an open match between the two possible teams.
    """
    team_a, team_b = BRACKET[leg_key]
    winner = RESULTS.get(result_key)
    if winner:
        return {winner: 1.0}
    return advance_match({team_a: 1.0}, {team_b: 1.0})

def show(stage, dist):
    print(f"\n=== {stage} ===")
    for team, p in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {team:15s} {p*100:5.1f}%")

# --- QF1: Morocco vs France (already a direct QF, no R16 leg needed) ---
qf1 = advance_match({BRACKET['QF1'][0]: 1.0}, {BRACKET['QF1'][1]: 1.0})
show("Quarterfinal 1: Morocco vs France", qf1)

# --- QF2 side ---
r16_a = get_leg_result('QF2_leg_a', 'R16_Belgium_USA')
r16_b = get_leg_result('QF2_leg_b', 'R16_Spain_Portugal')
qf2 = advance_match(r16_a, r16_b)
show("Quarterfinal 2: (Belgium/USA) vs (Spain/Portugal)", qf2)

# --- QF3: Norway vs England ---
qf3 = advance_match({BRACKET['QF3'][0]: 1.0}, {BRACKET['QF3'][1]: 1.0})
show("Quarterfinal 3: Norway vs England", qf3)

# --- QF4 side ---
r16_c = get_leg_result('QF4_leg_a', 'R16_Argentina_Egypt')
r16_d = get_leg_result('QF4_leg_b', 'R16_Switzerland_Colombia')
qf4 = advance_match(r16_c, r16_d)
show("Quarterfinal 4: (Argentina/Egypt) vs (Switzerland/Colombia)", qf4)

# --- Semifinals + Final ---
sf1 = advance_match(qf1, qf2)
show("Semifinal 1", sf1)

sf2 = advance_match(qf3, qf4)
show("Semifinal 2", sf2)

final = advance_match(sf1, sf2)
show("CHAMPION PROBABILITY", final)

# --- Save ---
all_stages = {'QF1': qf1, 'QF2': qf2, 'QF3': qf3, 'QF4': qf4, 'SF1': sf1, 'SF2': sf2, 'Final': final}
rows = [{'Stage': s, 'Team': t, 'Probability': p} for s, d in all_stages.items() for t, p in d.items()]
pd.DataFrame(rows).to_csv(os.path.join(OUTPUTS_DIR, 'bracket_simulation.csv'), index=False)
print(f"\nSaved to outputs/bracket_simulation.csv (state last updated: {STATE['last_updated']})")