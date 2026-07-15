import subprocess
import sys
import os
import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

def run_step(script_name):
    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPTS_DIR, script_name)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"ERROR running {script_name}:\n{result.stderr}")
        sys.exit(1)
    return result.stdout

print("Fetching latest results...")
fetch_output = run_step('07_fetch_results.py')

print("Fetching match scorelines...")
details_output = run_step('08_fetch_match_details.py')

print("Running bracket simulation...")
sim_output = run_step('06_bracket_simulator.py')

# --- Pull the champion probabilities from the saved CSV for a clean final summary ---
df = pd.read_csv(os.path.join(OUTPUTS_DIR, 'bracket_simulation.csv'))
champ = df[df['Stage'] == 'Final'].sort_values('Probability', ascending=False)

print("\n" + "=" * 40)
print("  WORLD CUP 2026 -- CHAMPION PREDICTION")
print("=" * 40)
for _, row in champ.head(5).iterrows():
    print(f"  {row['Team']:15s} {row['Probability']*100:5.1f}%")

top = champ.iloc[0]
print(f"\n>>> Most likely champion: {top['Team']} ({top['Probability']*100:.1f}%)")
print("=" * 40)