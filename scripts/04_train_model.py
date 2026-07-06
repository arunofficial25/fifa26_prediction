import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, log_loss, brier_score_loss
from sklearn.preprocessing import label_binarize

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(PROCESSED_DIR, 'matches_features_final.csv'))
df['MatchDate'] = pd.to_datetime(df['MatchDate'])

FEATURES = [
    'RankDiff', 'PointsDiff',
    'HomeForm_Pts', 'HomeForm_GF', 'HomeForm_GA',
    'AwayForm_Pts', 'AwayForm_GF', 'AwayForm_GA',
    'H2H_HomeWinPct', 'IsNeutralVenue'
]

# --- Drop rows missing key features (early matches with no history yet) ---
before = len(df)
df_model = df.dropna(subset=FEATURES + ['Result']).copy()
print(f"Dropped {before - len(df_model)} rows lacking full feature history (expected for early/rare matchups)")
print(f"Training on {len(df_model)} rows")

X = df_model[FEATURES]
y = df_model['Result']   # 0=Home Win, 1=Draw, 2=Away Win

# --- Time-based split: train on older matches, test on most recent ---
# This is more honest than a random split -- mimics predicting the "future" from the "past"
df_model = df_model.sort_values('MatchDate')
split_idx = int(len(df_model) * 0.85)

X_train = df_model.iloc[:split_idx][FEATURES]
y_train = df_model.iloc[:split_idx]['Result']
X_test = df_model.iloc[split_idx:][FEATURES]
y_test = df_model.iloc[split_idx:]['Result']

print(f"\nTrain: {len(X_train)} matches (up to {df_model.iloc[split_idx-1]['MatchDate'].date()})")
print(f"Test:  {len(X_test)} matches (from {df_model.iloc[split_idx]['MatchDate'].date()} onward)")

# --- Train Random Forest ---
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=10,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['Home Win', 'Draw', 'Away Win']))

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

logloss = log_loss(y_test, y_proba)
print(f"\nLog Loss: {logloss:.4f}  (lower is better; 1.0986 = random guessing baseline for 3 classes)")

# --- Feature importance ---
importance = pd.DataFrame({
    'Feature': FEATURES,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print("\n=== Feature Importance ===")
print(importance)

# --- Save model ---
joblib.dump(model, os.path.join(OUTPUTS_DIR, 'model_rf.pkl'))
print(f"\nModel saved to outputs/model_rf.pkl")