import os
import sys
import glob
import time
import csv
import numpy as np
from PIL import Image

# Add root to sys path to import lib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.algorithms import compress_image_algo
from lib.compress import compress_image_basic
from features.extract_features import extract_features
from app import compute_mse, compute_psnr

def process_image(img_path, idx):
    print(f"Processing {img_path}...")
    try:
        img = Image.open(img_path).convert('RGB')
        
        # Resize to max 128x128 to speed up data generation
        max_dim = 128
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        img_array = np.array(img, dtype=np.float64)
        
        # Extract features (using grayscale for simplicity)
        gray_img = img.convert('L')
        gray_array = np.array(gray_img, dtype=np.float64)
        features = extract_features(gray_array)
        
        # Evaluate algorithms
        best_algo = None
        best_score = -float('inf')
        
        algorithms = ['svd', 'pca', 'nmf', 'dct']
        results = {}
        
        for algo in algorithms:
            current_k = 29 if algo == 'pca' else 30
            
            if algo == 'svd':
                comp = compress_image_basic(img_array, current_k)
            else:
                comp = compress_image_algo(img_array, current_k, algo)
            
            mse = compute_mse(img_array, comp)
            psnr = compute_psnr(mse)
            
            score = psnr 
            results[algo] = score
            
            if score > best_score:
                best_score = score
                best_algo = algo
                
        if idx < 10:
            print(f"  [Sample {idx+1}] Raw PSNRs: {results}")
        print(f"  Best: {best_algo}")
        return features, best_algo
        
    except Exception as e:
        print(f"  Error processing {img_path}: {e}")
        return None, None

def main():
    valid_dir = r"C:\Users\ragha\OneDrive\Desktop\papaer\DIV2K_valid_HR\DIV2K_valid_HR"
    
    # Grab images (limit to 100 total)
    all_images = glob.glob(os.path.join(valid_dir, "*.png"))[:100]
    
    if not all_images:
        print("No images found! Check dataset path.")
        return
        
    out_csv = os.path.join(os.path.dirname(__file__), "algorithm_dataset.csv")
    
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header matches extract_features output + label
        writer.writerow([
            "block_var", "entropy", "decay_rate", "aspect",
            "mean_intensity", "std_dev", "energy_top5", "max_block_var",
            "best_algorithm"
        ])
        
        for idx, img_path in enumerate(all_images):
            feats, label = process_image(img_path, idx)
            if feats and label:
                writer.writerow(feats + [label])
                
    print(f"Dataset saved to {out_csv}")

if __name__ == "__main__":
    main()
