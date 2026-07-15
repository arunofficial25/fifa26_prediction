import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DETAILS_PATH = os.path.join(BASE_DIR, 'data', 'match_details.json')
API_KEY = os.environ.get('FOOTBALL_DATA_API_KEY')

# Maps API team-name variants to the exact names used in match_details.json
NAME_ALIASES = {
    'United States': 'USA',
}

def normalize(name):
    return NAME_ALIASES.get(name, name)

def load_details():
    with open(DETAILS_PATH, 'r') as f:
        return json.load(f)

def save_details(details):
    with open(DETAILS_PATH, 'w') as f:
        json.dump(details, f, indent=2)

def fetch_finished_matches():
    if not API_KEY:
        print("No FOOTBALL_DATA_API_KEY found -- skipping API, leaving match_details.json as-is.")
        return None
    try:
        resp = requests.get(
            "https://api.football-data.org/v4/competitions/WC/matches",
            headers={"X-Auth-Token": API_KEY},
            params={"status": "FINISHED"},
            timeout=10
        )
        resp.raise_for_status()
        return resp.json().get('matches', [])
    except requests.exceptions.RequestException as e:
        print(f"API call failed ({e}) -- leaving match_details.json as-is.")
        return None

def main():
    details = load_details()
    matches = fetch_finished_matches()

    if matches is None:
        print("\nNo API data. Edit data/match_details.json by hand for now.")
        return

    updated_count = 0

    for key, entry in details.items():
        if entry['homeGoals'] is not None:
            continue  # already filled in, don't overwrite

        expected_home = entry['home']
        expected_away = entry['away']
        expected_pair = {expected_home, expected_away}

        for m in matches:
            api_home = normalize(m['homeTeam']['name'])
            api_away = normalize(m['awayTeam']['name'])

            if {api_home, api_away} != expected_pair:
                continue

            full_time = m['score']['fullTime']
            home_goals, away_goals = full_time['home'], full_time['away']
            if home_goals is None:
                continue  # match found but no score yet somehow

            # Match API's home/away orientation back to our stored home/away
            if api_home != expected_home:
                home_goals, away_goals = away_goals, home_goals

            entry['homeGoals'] = home_goals
            entry['awayGoals'] = away_goals

            pens = m['score'].get('penalties', {})
            if pens.get('home') is not None:
                p_home, p_away = pens['home'], pens['away']
                if api_home != expected_home:
                    p_home, p_away = p_away, p_home
                entry['penalties'] = f"{p_home}-{p_away} pens"

            print(f"  Filled {key}: {expected_home} {home_goals}-{away_goals} {expected_away}"
                  + (f" ({entry['penalties']})" if entry['penalties'] else ""))
            updated_count += 1
            break

    if updated_count > 0:
        save_details(details)
        print(f"\nUpdated {updated_count} match(es) in match_details.json")
    else:
        print("\nNo new finished matches found to fill in.")

if __name__ == "__main__":
    main()