import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, log_loss

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(PROCESSED_DIR, 'matches_features_final.csv'))
df['MatchDate'] = pd.to_datetime(df['MatchDate'])

# --- New feature: absolute rank gap (closeness signals draw likelihood) ---
df['AbsRankDiff'] = df['RankDiff'].abs()
df['AbsPointsDiff'] = df['PointsDiff'].abs()

FEATURES = [
    'RankDiff', 'PointsDiff', 'AbsRankDiff', 'AbsPointsDiff',
    'HomeForm_Pts', 'HomeForm_GF', 'HomeForm_GA',
    'AwayForm_Pts', 'AwayForm_GF', 'AwayForm_GA',
    'H2H_HomeWinPct', 'IsNeutralVenue'
]

before = len(df)
df_model = df.dropna(subset=FEATURES + ['Result']).copy()
print(f"Dropped {before - len(df_model)} rows lacking full feature history")
print(f"Training on {len(df_model)} rows")

df_model = df_model.sort_values('MatchDate')
split_idx = int(len(df_model) * 0.85)

X_train = df_model.iloc[:split_idx][FEATURES]
y_train = df_model.iloc[:split_idx]['Result']
X_test = df_model.iloc[split_idx:][FEATURES]
y_test = df_model.iloc[split_idx:]['Result']

print(f"\nTrain: {len(X_train)} (up to {df_model.iloc[split_idx-1]['MatchDate'].date()})")
print(f"Test:  {len(X_test)} (from {df_model.iloc[split_idx]['MatchDate'].date()} onward)")

# --- Class weights: give Draw a bit more relative weight since it's the hardest, rarest-signal class ---
class_counts = y_train.value_counts()
total = len(y_train)
sample_weight = y_train.map(lambda c: total / (3 * class_counts[c]))
# manually boost draw weight a bit further since it's the true weak point
sample_weight = sample_weight * y_train.map({0: 1.0, 1: 1.3, 2: 1.0})

model = XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    random_state=42
)
model.fit(X_train, y_train, sample_weight=sample_weight)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=['Home Win', 'Draw', 'Away Win']))

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

logloss = log_loss(y_test, y_proba)
print(f"\nLog Loss: {logloss:.4f}  (baseline for random 3-class guessing = 1.0986)")

importance = pd.DataFrame({
    'Feature': FEATURES,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print("\n=== Feature Importance ===")
print(importance)

joblib.dump(model, os.path.join(OUTPUTS_DIR, 'model_xgb.pkl'))
print(f"\nModel saved to outputs/model_xgb2.pkl")

# also save the feature list so predict_live.py knows the exact order/names
import json
with open(os.path.join(OUTPUTS_DIR, 'feature_list.json'), 'w') as f:
    json.dump(FEATURES, f)