import os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("data/labels.csv")

FEATURES = ["block_var","entropy","decay_rate","aspect",
            "mean_intensity","std_dev","energy_top5","max_block_var"]

X = df[FEATURES].values
y = df["optimal_rank"].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42
)

# Train
model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50
)

# Evaluate
preds = model.predict(X_test)
preds_rounded = np.round(preds).astype(int)

print(f"MAE:  {mean_absolute_error(y_test, preds_rounded):.2f}")
print(f"R²:   {r2_score(y_test, preds_rounded):.4f}")

# Save
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/rank_predictor.pkl")
print("Model saved to models/rank_predictor.pkl")
