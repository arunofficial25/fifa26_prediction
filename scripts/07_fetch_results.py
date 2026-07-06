import os
import json
import requests
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_PATH = os.path.join(BASE_DIR, 'data', 'tournament_state.json')

API_KEY = os.environ.get('FOOTBALL_DATA_API_KEY')

def load_state():
    with open(STATE_PATH, 'r') as f:
        return json.load(f)

def save_state(state):
    with open(STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2)

def try_fetch_from_api(state):
    """Attempt to pull real results from football-data.org. Returns True if it succeeded."""
    if not API_KEY:
        print("No FOOTBALL_DATA_API_KEY found in environment -- skipping API, using config file.")
        return False

    try:
        resp = requests.get(
            "https://api.football-data.org/v4/competitions/WC/matches",
            headers={"X-Auth-Token": API_KEY},
            params={"status": "FINISHED"},
            timeout=10
        )
        resp.raise_for_status()
        matches = resp.json().get('matches', [])
        print(f"API responded with {len(matches)} finished matches.")

        # Map bracket legs to the actual team pairs we care about
        watch_pairs = {
            frozenset(['Belgium', 'United States']): 'R16_Belgium_USA',
            frozenset(['Spain', 'Portugal']): 'R16_Spain_Portugal',
            frozenset(['Argentina', 'Egypt']): 'R16_Argentina_Egypt',
            frozenset(['Switzerland', 'Colombia']): 'R16_Switzerland_Colombia',
        }

        updated = False
        for m in matches:
            home = m['homeTeam']['name']
            away = m['awayTeam']['name']
            pair = frozenset([home, away])
            for watch_key, state_key in watch_pairs.items():
                if pair == watch_key or pair.issubset(watch_key) or watch_key.issubset(pair):
                    home_score = m['score']['fullTime']['home']
                    away_score = m['score']['fullTime']['away']
                    if home_score is None:
                        continue
                    winner = home if home_score > away_score else away
                    # penalty shootout override if regulation was a draw
                    pens = m['score'].get('penalties', {})
                    if home_score == away_score and pens.get('home') is not None:
                        winner = home if pens['home'] > pens['away'] else away
                    if state['confirmed_results'].get(state_key) != winner:
                        state['confirmed_results'][state_key] = winner
                        updated = True
                        print(f"  Updated {state_key}: {winner} wins")

        if updated:
            save_state(state)
        else:
            print("  No new confirmed results from API.")
        return True

    except requests.exceptions.RequestException as e:
        print(f"API call failed ({e}) -- falling back to manually-edited config file.")
        return False

def main():
    state = load_state()
    success = try_fetch_from_api(state)
    if not success:
        print("Using existing data/tournament_state.json as-is (edit it manually if you have new results).")

    print("\n=== Current confirmed results ===")
    for k, v in state['confirmed_results'].items():
        print(f"  {k}: {v if v else 'pending'}")

if __name__ == "__main__":
    main()