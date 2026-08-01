import json, csv, os, time, sys
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
RESULTS = os.path.join(PROJECT_ROOT, 'benchmark_results.json')
OUT_DIR = os.path.join(PROJECT_ROOT, 'paper_data', 'corrected')

from lib.rsvd import rsvd, reconstruct
from lib.algorithms import pca_compress_channel, nmf_compress_channel, dct_compress_channel
from lib.prescreening import complexity_score, recommend_rank, DEFAULT_SKIP_THRESHOLD
from skimage.metrics import structural_similarity as ssim_skimage

def compute_mse(a, b):
    return float(np.mean((a - b) ** 2))

def compute_psnr(mse_val):
    if mse_val == 0: return float('inf')
    return float(10 * np.log10(255**2 / mse_val))

def compute_ssim(orig, comp):
    kwargs = {'data_range': 255.0}
    if orig.ndim == 3: kwargs['channel_axis'] = 2
    return float(ssim_skimage(orig, comp, **kwargs))

# PROBLEM 1: algo_comparison
print('=== Regenerating algo_comparison_corrected.csv ===')
test_img_path = os.path.join(PROJECT_ROOT, 'kodak_images', 'kodim23.png')
orig_pil = Image.open(test_img_path).convert('RGB')
orig_arr = np.array(orig_pil, dtype=np.float64)
h, w = orig_arr.shape[:2]
ch_count = orig_arr.shape[2]

k = 50
algorithms = ['rSVD', 'PCA', 'NMF', 'DCT']
rows_algo = []

for algo in algorithms:
    comp = np.zeros_like(orig_arr)
    t0 = time.time()
    
    for c in range(ch_count):
        channel = orig_arr[:,:,c]
        if algo == 'rSVD':
            U, S, Vt = rsvd(channel, k)
            recon = reconstruct(U, S, Vt)
            np.clip(recon, 0, 255, out=recon)
            comp[:,:,c] = recon
        elif algo == 'PCA':
            comp[:,:,c] = pca_compress_channel(channel, k)
        elif algo == 'NMF':
            comp[:,:,c] = nmf_compress_channel(channel, k)
        elif algo == 'DCT':
            comp[:,:,c] = dct_compress_channel(channel, k)
            
    latency = (time.time() - t0) * 1000
    mse = compute_mse(orig_arr, comp)
    psnr = compute_psnr(mse)
    ssim = compute_ssim(orig_arr, comp)
    
    orig_size = h * w * ch_count
    
    if algo in ['rSVD', 'PCA', 'NMF']:
        # Element count based on final truncated rank k
        comp_size = ch_count * (k * (h + w + 1))
        cr = float(orig_size) / comp_size
    elif algo == 'DCT':
        # DCT compresses via full-image 2D DCT truncation.
        # It retains the top-left k x k frequency block of coefficients per channel.
        # So compressed size is k^2 elements per channel.
        comp_size = ch_count * (k * k)
        cr = float(orig_size) / comp_size
        
    rows_algo.append({
        'algorithm': algo,
        'psnr': psnr,
        'ssim': ssim,
        'cr': cr,
        'latency_ms': latency
    })

algo_out = os.path.join(OUT_DIR, 'algo_comparison_corrected.csv')
with open(algo_out, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['algorithm','psnr','ssim','cr','latency_ms'])
    writer.writeheader()
    writer.writerows(rows_algo)
print('Saved', algo_out)


# PROBLEM 2: ablation_corrected.csv
print('\n=== Regenerating ablation_corrected.csv (on full Kodak24) ===')
# Run ablation on ALL 24 Kodak images to get identical CR to exp1
images = []
for fname in sorted([f for f in os.listdir(os.path.join(PROJECT_ROOT, 'kodak_images')) if f.endswith('.png')]):
    pil = Image.open(os.path.join(PROJECT_ROOT, 'kodak_images', fname)).convert('RGB')
    if max(pil.size) > 800:
        ratio = 800 / max(pil.size)
        pil = pil.resize((int(pil.size[0]*ratio), int(pil.size[1]*ratio)), Image.LANCZOS)
    images.append(np.array(pil, dtype=np.float64))

variants = [
    ('No pre-screener, q=0', False, 0),
    ('No pre-screener, q=2', False, 2),
    ('Pre-screener, q=0',    True,  0),
    ('Pre-screener, q=2 (proposed)', True, 2),
]
energy = 95.0
rows_ablation = []

for label, use_prescreen, q in variants:
    psnrs, ssims, crs, lats = [], [], [], []
    for img_arr in images:
        h, w = img_arr.shape[:2]
        ch_count = 3
        comp = np.zeros_like(img_arr)
        k_vals = []
        t0 = time.time()
        for c in range(ch_count):
            channel = img_arr[:,:,c]
            max_d = min(h, w)
            if use_prescreen:
                score = complexity_score(channel)
                if score < DEFAULT_SKIP_THRESHOLD:
                    comp[:,:,c] = channel
                    k_vals.append(0)
                    continue
                k = recommend_rank(score, energy, max_d)
            else:
                k = recommend_rank(4000.0, energy, max_d)
            
            k = min(k, max_d)
            U, S, Vt = rsvd(channel, k, oversample=10, power_iter=q)
            recon = reconstruct(U, S, Vt)
            np.clip(recon, 0, 255, out=recon)
            comp[:,:,c] = recon
            k_vals.append(k)
            
        latency = (time.time() - t0) * 1000
        mse = compute_mse(img_arr, comp)
        psnrs.append(compute_psnr(mse))
        ssims.append(compute_ssim(img_arr, comp))
        
        orig_size = h * w * ch_count
        comp_size = 0
        for k_val in k_vals:
            if k_val == 0:
                comp_size += h * w
            else:
                comp_size += k_val * (h + w + 1)
        crs.append(float(orig_size) / comp_size)
        lats.append(latency)
        
    rows_ablation.append({
        'config': label,
        'psnr_mean': round(np.mean(psnrs), 4),
        'ssim_mean': round(np.mean(ssims), 4),
        'cr_mean': round(np.mean(crs), 4),
        'latency_ms_mean': round(np.mean(lats), 1)
    })

ab_out = os.path.join(OUT_DIR, 'ablation_corrected.csv')
with open(ab_out, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['config','psnr_mean','ssim_mean','cr_mean','latency_ms_mean'])
    writer.writeheader()
    writer.writerows(rows_ablation)
print('Saved', ab_out)

print('\n=== CR CONSISTENCY CHECK (Proposed Config, rho=95, Kodak24) ===')
print(f"ablation_corrected.csv shows CR: {rows_ablation[-1]['cr_mean']}x")
