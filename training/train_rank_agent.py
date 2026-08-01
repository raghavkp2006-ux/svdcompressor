import os
import sys
import glob
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from PIL import Image

# Add root to sys path to import lib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from features.extract_features import extract_features
from lib.prescreening import complexity_score, recommend_rank

def process_image(img_path):
    """Process an image and extract features/true_k for each of its R, G, B channels."""
    try:
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img, dtype=np.float64)
        
        samples = []
        # Process per R/G/B channel exactly as prescreening does
        for c in range(3):
            channel = img_array[:, :, c]
            features = extract_features(channel)
            score = complexity_score(channel)
            true_k = recommend_rank(score, energy_percent=95.0, max_dim=min(channel.shape))
            samples.append(features + [true_k])
            
        return samples
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return []

def main():
    valid_dir = r"C:\Users\ragha\OneDrive\Desktop\papaer\DIV2K_valid_HR\DIV2K_valid_HR"
    
    print(f"Searching for images in {valid_dir}...")
    all_images = glob.glob(os.path.join(valid_dir, "*.png"))[:100]
    
    if not all_images:
        print("No images found! Check dataset path.")
        return
        
    print(f"Found {len(all_images)} images. Processing per-channel (R, G, B)...")
    
    dataset = []
    for idx, img_path in enumerate(all_images):
        if idx % 10 == 0:
            print(f"Processing image {idx+1}/{len(all_images)}...")
        dataset.extend(process_image(img_path))
        
    feature_names = [
        "block_var", "entropy", "decay_rate", "aspect",
        "mean_intensity", "std_dev", "energy_top5", "max_block_var"
    ]
    columns = feature_names + ["true_k"]
    
    df = pd.DataFrame(dataset, columns=columns)
    df = df.dropna()
    print(f"\nGenerated dataset with {len(df)} channel samples.")
    
    X = df[feature_names]
    y = df["true_k"]
    
    # Summary of true_k
    k_min, k_max, k_mean = y.min(), y.max(), y.mean()
    print(f"\n--- Ground Truth 'k' Summary ---")
    print(f"Min:  {k_min}")
    print(f"Max:  {k_max}")
    print(f"Mean: {k_mean:.2f}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8))
    ])
    
    print("\nTraining RandomForestRegressor...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n--- Evaluation Metrics (Test Set) ---")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    
    # Save model
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "rank_agent.pkl")
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to {model_path}")
    
    # 4. Feature importance plot
    rf_model = pipeline.named_steps['rf']
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances for Rank Prediction")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    fi_path = os.path.join(os.path.dirname(__file__), "rank_feature_importance.png")
    plt.savefig(fi_path)
    plt.close()
    print(f"Feature importance plot saved to {fi_path}")
    
    # 5. Scatter plot of predicted vs true k
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
    
    # Reference line y = x
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y = x (Perfect Prediction)')
    
    plt.title("Predicted vs. True Rank (k)")
    plt.xlabel("True k")
    plt.ylabel("Predicted k")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    scatter_path = os.path.join(os.path.dirname(__file__), "rank_scatter.png")
    plt.savefig(scatter_path)
    plt.close()
    print(f"Scatter plot saved to {scatter_path}")

if __name__ == "__main__":
    main()
