"""
Experiment 1 — Extended Ablation Sweep
=======================================
Sweeps tau, oversampling p, k_max, and rho independently on all 24 Kodak
images, holding other parameters at defaults (p=10, q=2, tau=80, k_min=5,
k_max=250, rho=95).

Output: paper_data/corrected/ablation_extended_corrected.csv
Columns: parameter, value, psnr_mean, ssim_mean, cr_mean, latency_ms_mean
"""

import os, sys, time, csv
import numpy as np
from PIL import Image

# ── Project root setup ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from lib.rsvd import rsvd, reconstruct
from lib.prescreening import complexity_score, recommend_rank
from scipy.ndimage import uniform_filter

# ── Paths ────────────────────────────────────────────────────────────────────
KODAK_DIR = os.path.join(PROJECT_ROOT, 'kodak_images')
OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'paper_data', 'ablation_extended.csv')

# ── Default parameters ──────────────────────────────────────────────────────
DEFAULTS = {
    'p': 10,        # oversampling
    'q': 2,         # power iterations
    'tau': 80,      # skip threshold
    'k_min': 5,     # minimum rank
    'k_max': 250,   # max rank for recommend_rank
    'rho': 95,      # energy percent
}

# ── Metric functions (from run_benchmarks.py) ────────────────────────────────
def compute_mse(a, b):
    return float(np.mean((a - b) ** 2))

def compute_psnr(mse_val):
    if mse_val == 0:
        return float('inf')
    return float(10 * np.log10(255**2 / mse_val))

from skimage.metrics import structural_similarity as ssim_skimage

def compute_ssim(orig, comp):
    kwargs = {'data_range': 255.0}
    if orig.ndim == 3:
        kwargs['channel_axis'] = 2
    return float(ssim_skimage(orig, comp, **kwargs))


def load_kodak_images(max_dim=800):
    """Load all Kodak images, resized to max_dim (consistent with existing benchmarks)."""
    images = []
    fnames = sorted([f for f in os.listdir(KODAK_DIR) if f.endswith('.png')])
    for fname in fnames:
        fpath = os.path.join(KODAK_DIR, fname)
        pil = Image.open(fpath).convert('RGB')
        if max(pil.size) > max_dim:
            ratio = max_dim / max(pil.size)
            new_size = (int(pil.size[0] * ratio), int(pil.size[1] * ratio))
            pil = pil.resize(new_size, Image.LANCZOS)
        images.append(np.array(pil, dtype=np.float64))
    return images


def compute_cr_perchannel(shape, k_vals):
    """Compute theoretical CR from per-channel rank values.
    
    Uses the formula from run_benchmarks.py: compressed_size = sum_c k_c * (m + n + 1)
    For skipped channels (k=0), the channel is stored as raw (m*n bytes).
    """
    if len(shape) == 2:
        m, n = shape; ch = 1
    else:
        m, n, ch = shape
    orig_size = m * n * ch
    comp_size = 0
    for k in k_vals:
        if k == 0:
            comp_size += m * n  # skipped channel stored raw
        else:
            comp_size += k * (m + n + 1)
    return float(orig_size / comp_size) if comp_size > 0 else 0.0


def run_single_config(images, tau, p, q, k_max, rho, k_min=5):
    """Run the adaptive rSVD pipeline on all images with given parameters.
    
    Returns (psnr_list, ssim_list, cr_list, latency_list).
    """
    psnr_list, ssim_list, cr_list, latency_list = [], [], [], []

    for img in images:
        h, w = img.shape[:2]
        ch_count = img.shape[2] if img.ndim == 3 else 1
        max_d = min(h, w)

        t0 = time.perf_counter()
        comp = np.zeros_like(img)
        k_vals = []

        for c in range(ch_count):
            channel = img[:, :, c] if img.ndim == 3 else img
            score = complexity_score(channel)

            # Prescreening with custom tau
            if score < tau:
                if img.ndim == 3:
                    comp[:, :, c] = channel.copy()
                else:
                    comp = channel.copy()
                k_vals.append(0)
                continue

            # Rank recommendation with custom k_max
            # Replicate recommend_rank logic but with custom max_rank
            max_rank = int(min(max_d, k_max) * (rho / 95.0))
            LOW, HIGH = 200.0, 8000.0
            t_score = min(1.0, max(0.0, (score - LOW) / (HIGH - LOW)))
            k = int(round(k_min + (max_rank - k_min) * (t_score ** 0.6)))
            k = min(k, max_d)
            k = max(k, 1)

            U, S, Vt = rsvd(channel, k, oversample=p, power_iter=q)
            recon = reconstruct(U, S, Vt)
            np.clip(recon, 0, 255, out=recon)

            if img.ndim == 3:
                comp[:, :, c] = recon
            else:
                comp = recon
            k_vals.append(k)

        latency = (time.perf_counter() - t0) * 1000

        mse = compute_mse(img, comp)
        psnr = compute_psnr(mse)
        ssim = compute_ssim(img, comp)
        cr = compute_cr_perchannel(img.shape, k_vals)

        psnr_list.append(psnr)
        ssim_list.append(ssim)
        cr_list.append(cr)
        latency_list.append(latency)

    return psnr_list, ssim_list, cr_list, latency_list


def main():
    print('=== Experiment 1: Extended Ablation Sweep ===')
    print(f'Loading Kodak images from {KODAK_DIR}...')
    images = load_kodak_images()
    print(f'  Loaded {len(images)} images')

    # Define sweep configurations
    sweeps = [
        ('tau',   [40, 60, 80, 100, 120]),
        ('p',     [5, 10, 15, 20]),
        ('k_max', [150, 200, 250, 300]),
        ('rho',   [80, 90, 95, 99]),
    ]

    rows = []

    for param_name, values in sweeps:
        print(f'\n  Sweeping {param_name}: {values}')
        for val in values:
            # Start from defaults, override the swept parameter
            cfg = dict(DEFAULTS)
            cfg[param_name] = val

            print(f'    {param_name}={val} ...', end=' ', flush=True)
            psnrs, ssims, crs, lats = run_single_config(
                images,
                tau=cfg['tau'],
                p=cfg['p'],
                q=cfg['q'],
                k_max=cfg['k_max'],
                rho=cfg['rho'],
                k_min=cfg['k_min'],
            )

            row = {
                'parameter': param_name,
                'value': val,
                'psnr_mean': round(np.mean(psnrs), 4),
                'ssim_mean': round(np.mean(ssims), 6),
                'cr_mean': round(np.mean(crs), 4),
                'latency_ms_mean': round(np.mean(lats), 2),
            }
            rows.append(row)
            print(f'PSNR={row["psnr_mean"]:.2f}  SSIM={row["ssim_mean"]:.4f}  '
                  f'CR={row["cr_mean"]:.2f}  Lat={row["latency_ms_mean"]:.1f}ms')

    # Write CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'parameter', 'value', 'psnr_mean', 'ssim_mean', 'cr_mean', 'latency_ms_mean'
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f'\n  Saved: {OUTPUT_CSV}')
    print(f'  Total rows: {len(rows)}')
    print('=== Experiment 1 Complete ===')


if __name__ == '__main__':
    main()
