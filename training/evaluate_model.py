import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
import xgboost as xgb

df = pd.read_csv("data/labels.csv")
FEATURES = ["block_var","entropy","decay_rate","aspect",
            "mean_intensity","std_dev","energy_top5","max_block_var"]

X = df[FEATURES].values
y = df["optimal_rank"].values
_, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

model = joblib.load("models/rank_predictor.pkl")
preds = np.round(model.predict(X_test)).astype(int)

os.makedirs("results", exist_ok=True)

# Plot 1 — Predicted vs Actual rank scatter
plt.figure(figsize=(6,6))
plt.scatter(y_test, preds, alpha=0.3, s=10)
plt.plot([0,250],[0,250], 'r--', label="Perfect prediction")
plt.xlabel("Optimal Rank (Ground Truth)")
plt.ylabel("Predicted Rank")
plt.title("Rank Prediction: Learned vs Optimal")
plt.legend()
plt.tight_layout()
plt.savefig("results/rank_scatter.png", dpi=150)
print("Saved scatter plot to results/rank_scatter.png")

# Plot 2 — Feature importance
fig, ax = plt.subplots(figsize=(7,4))
xgb.plot_importance(model, ax=ax, importance_type='gain')
plt.tight_layout()
plt.savefig("results/feature_importance.png", dpi=150)
print("Saved feature importance to results/feature_importance.png")
