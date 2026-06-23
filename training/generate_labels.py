import os
import sys
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from joblib import Parallel, delayed
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from features.extract_features import extract_features
from lib.rsvd import rsvd, reconstruct

IMAGE_DIR = "kodak_images"
RANKS_TO_SWEEP = [5,10,15,20,30,40,50,70,100,130,160,200]
OUTPUT_CSV = "data/labels.csv"

def best_rank_for_channel(channel: np.ndarray) -> int:
    """Find rank that maximizes SSIM for this channel."""
    h, w = channel.shape
    max_dim = min(h, w)
    best_rank, best_ssim = 5, -1.0

    for k in RANKS_TO_SWEEP:
        if k >= max_dim:
            break
        U, S, Vt = rsvd(channel, k, oversample=5, power_iter=1)
        recon = reconstruct(U, S, Vt)
        np.clip(recon, 0, 255, out=recon)
        score = ssim(channel, recon, data_range=255.0)
        if score > best_ssim:
            best_ssim = score
            best_rank = k
        if best_ssim > 0.97:  # early stop — good enough
            break

    return best_rank

def process_image(img_path: str) -> list:
    """Process one image → return rows (one per channel)."""
    rows = []
    try:
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float64)
        for c in range(3):
            channel = img[:, :, c]
            feats = extract_features(channel)
            label = best_rank_for_channel(channel)
            rows.append(feats + [label])
    except Exception as e:
        print(f"Skipping {img_path}: {e}")
    return rows

if __name__ == "__main__":
    # Collect all image paths
    all_paths = []
    if os.path.exists(IMAGE_DIR):
        for fname in os.listdir(IMAGE_DIR):
            if fname.endswith((".jpg", ".png")):
                all_paths.append(os.path.join(IMAGE_DIR, fname))

    print(f"Processing {len(all_paths)} images with joblib parallel...")

    # Parallel execution — uses all CPU cores
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_image)(p) for p in all_paths
    )

    # Flatten and save
    all_rows = [row for img_rows in results for row in img_rows]
    cols = ["block_var","entropy","decay_rate","aspect","mean_intensity",
            "std_dev","energy_top5","max_block_var","optimal_rank"]
    df = pd.DataFrame(all_rows, columns=cols)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_CSV}")
