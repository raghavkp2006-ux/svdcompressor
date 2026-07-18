import os
import sys
import glob
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from PIL import Image

# Add root to sys path to import lib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from features.extract_features import extract_features
from lib.rsvd import rsvd, reconstruct

def compute_mse(a, b):
    return float(np.mean((a - b) ** 2))

def compute_psnr(mse_val):
    if mse_val == 0:
        return float('inf')
    return float(10 * np.log10(255**2 / mse_val))

def find_optimal_k(channel, max_k=250, target_psnr=30.0, step=5):
    """
    Find the smallest k such that PSNR >= target_psnr.
    Optimization: we compute rSVD once for max_k, and then slice the factors.
    Since SVD components are orthogonal, slicing the rank-250 decomposition 
    is equivalent to computing the rank-k decomposition directly, but saves 
    us from running rSVD 50 times per channel.
    """
    h, w = channel.shape
    max_k = min(max_k, h, w)
    
    # Compute the factorization for max_k once
    U, S, Vt = rsvd(channel, max_k, oversample=10, power_iter=2)
    
    for k in range(5, max_k + 1, step):
        # Slice to rank k
        Uk = U[:, :k]
        Sk = S[:k]
        Vtk = Vt[:k, :]
        
        compressed = reconstruct(Uk, Sk, Vtk)
        np.clip(compressed, 0, 255, out=compressed)
        
        mse = compute_mse(channel, compressed)
        psnr = compute_psnr(mse)
        
        if psnr >= target_psnr:
            return k
            
    return max_k

def process_image(img_path):
    """Extract features and empirical true_k for each R, G, B channel."""
    try:
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img, dtype=np.float64)
        
        samples = []
        for c in range(3):
            channel = img_array[:, :, c]
            features = extract_features(channel)
            true_k = find_optimal_k(channel)
            samples.append(features + [true_k])
            
        return samples
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return []

def main():
    start_time = time.time()
    valid_dir = r"C:\Users\ragha\OneDrive\Desktop\papaer\DIV2K_valid_HR\DIV2K_valid_HR"
    
    print(f"Searching for images in {valid_dir}...")
    all_images = glob.glob(os.path.join(valid_dir, "*.png"))[:100]
    
    if not all_images:
        print("No images found! Check dataset path.")
        return
        
    print(f"Found {len(all_images)} images. Processing per-channel (R, G, B) using multi-processing...")
    
    # Process images in parallel to drastically speed up grid search
    from joblib import Parallel, delayed
    results = Parallel(n_jobs=-1, verbose=10)(delayed(process_image)(img_path) for img_path in all_images)
    
    dataset = []
    for res in results:
        dataset.extend(res)
        
    search_time = time.time() - start_time
    print(f"\n--- Total Grid-Search Compute Time: {search_time:.2f} seconds ---")
    
    feature_names = [
        "block_var", "entropy", "decay_rate", "aspect",
        "mean_intensity", "std_dev", "energy_top5", "max_block_var"
    ]
    columns = feature_names + ["true_k"]
    
    df = pd.DataFrame(dataset, columns=columns)
    df = df.dropna()
    print(f"Generated dataset with {len(df)} channel samples.")
    
    X = df[feature_names]
    y = df["true_k"]
    
    # Summary of true_k
    k_min, k_max, k_mean = y.min(), y.max(), y.mean()
    print(f"\n--- Ground Truth 'k' Summary (Empirical PSNR >= 30) ---")
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
    
    print("\nTraining RandomForestRegressor (v2)...")
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
    
    # Subset metrics: capped vs non-capped
    capped_mask = (y_test == 250)
    capped_count = capped_mask.sum()
    non_capped_count = (~capped_mask).sum()
    print(f"\n--- Test Set Sub-population Analysis ---")
    print(f"Capped (k=250) samples in test set: {capped_count}")
    print(f"Non-capped samples in test set: {non_capped_count}")
    if capped_count > 0:
        capped_mae = mean_absolute_error(y_test[capped_mask], y_pred[capped_mask])
        print(f"MAE on capped samples:     {capped_mae:.4f}")
    if non_capped_count > 0:
        non_capped_mae = mean_absolute_error(y_test[~capped_mask], y_pred[~capped_mask])
        print(f"MAE on non-capped samples: {non_capped_mae:.4f}")
    
    # Save model
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "rank_agent_v2.pkl")
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to {model_path}")
    
    # 4. Feature importance plot
    rf_model = pipeline.named_steps['rf']
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances for Rank Prediction (Empirical)")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    fi_path = os.path.join(os.path.dirname(__file__), "rank_feature_importance_v2.png")
    plt.savefig(fi_path)
    plt.close()
    print(f"Feature importance plot saved to {fi_path}")
    
    # 5. Scatter plot of predicted vs true k
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
    
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y = x (Perfect Prediction)')
    
    plt.title("Predicted vs. True Rank (k) (Empirical Labels)")
    plt.xlabel("True k")
    plt.ylabel("Predicted k")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    scatter_path = os.path.join(os.path.dirname(__file__), "rank_scatter_v2.png")
    plt.savefig(scatter_path)
    plt.close()
    print(f"Scatter plot saved to {scatter_path}")

if __name__ == "__main__":
    main()
