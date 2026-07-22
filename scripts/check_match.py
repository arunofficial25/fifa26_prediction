import joblib
import pandas as pd
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
STATE_PATH = os.path.join(BASE_DIR, "data", "tournament_state.json")

model = joblib.load(os.path.join(OUTPUTS_DIR, "model_rf.pkl"))

FEATURES = [
    "RankDiff",
    "PointsDiff",
    "HomeForm_Pts",
    "HomeForm_GF",
    "HomeForm_GA",
    "AwayForm_Pts",
    "AwayForm_GF",
    "AwayForm_GA",
    "H2H_HomeWinPct",
    "IsNeutralVenue",
]

with open(STATE_PATH, "r") as f:
    STATE = json.load(f)

TEAM_RATINGS = STATE["team_ratings"]

# Tournament is complete -- no more "active"/"eliminated" distinction needed.
# Every team stored in tournament_state.json can be checked against any other.
ALL_TEAMS = sorted(TEAM_RATINGS.keys())


def predict_match(teamA, teamB):
    a = TEAM_RATINGS[teamA]
    b = TEAM_RATINGS[teamB]

    row = pd.DataFrame([{
        "RankDiff": b["Rank"] - a["Rank"],
        "PointsDiff": a["Points"] - b["Points"],
        "HomeForm_Pts": a["Form_Pts"],
        "HomeForm_GF": a["Form_GF"],
        "HomeForm_GA": a["Form_GA"],
        "AwayForm_Pts": b["Form_Pts"],
        "AwayForm_GF": b["Form_GF"],
        "AwayForm_GA": b["Form_GA"],
        "H2H_HomeWinPct": 0.5,
        "IsNeutralVenue": 1
    }])[FEATURES]

    return model.predict_proba(row)[0]


def find_team(name):
    """Case-insensitive lookup among all stored teams."""
    name = name.strip().lower()

    for team in ALL_TEAMS:
        if team.lower() == name:
            return team

    return None


def run_check(team_a_input, team_b_input):

    team_a = find_team(team_a_input)
    team_b = find_team(team_b_input)

    if not team_a or not team_b:

        print("\n❌ Invalid team name.")
        print("All Teams:")
        print(", ".join(ALL_TEAMS))
        return

    home, draw, away = predict_match(team_a, team_b)

    print("\n" + "=" * 40)
    print(f"{team_a} vs {team_b}")
    print("=" * 40)

    print(f"{team_a:15} {home*100:6.2f}%")
    print(f"{'Draw':15} {draw*100:6.2f}%")
    print(f"{team_b:15} {away*100:6.2f}%")

    favorite = max(
        [(team_a, home), ("Draw", draw), (team_b, away)],
        key=lambda x: x[1]
    )

    print("-" * 40)
    print(f"Most likely: {favorite[0]} ({favorite[1]*100:.2f}%)")


if __name__ == "__main__":

    if len(sys.argv) == 3:
        run_check(sys.argv[1], sys.argv[2])

    else:
        print("\n🏆 FIFA World Cup 2026 Match Predictor")
        print("-" * 40)
        print("Type 'exit' anytime to quit.\n")
        print("All Teams:")
        print(", ".join(ALL_TEAMS))

        while True:

            team_a = input("\nTeam A: ").strip()

            if team_a.lower() == "exit":
                break

            team_b = input("Team B: ").strip()

            if team_b.lower() == "exit":
                break

            run_check(team_a, team_b)