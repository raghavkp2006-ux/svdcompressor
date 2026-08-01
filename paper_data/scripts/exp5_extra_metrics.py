"""
Experiment 5 — Additional Quality Metric (MS-SSIM)
===================================================
On Kodak24, run adaptive rSVD (rho=95) and compute PSNR, SSIM, and MS-SSIM
for each image.

MS-SSIM is computed using the standard 5-scale approach with Gaussian
downsampling (Wang et al., 2003).

Output: paper_data/corrected/extra_metrics_corrected.csv
Columns: image_name, psnr, ssim, ms_ssim
"""

import os, sys, csv
import numpy as np
from PIL import Image

# ── Project root setup ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from lib.compress import compress_image
from scipy.ndimage import uniform_filter, gaussian_filter

# ── Paths ────────────────────────────────────────────────────────────────────
KODAK_DIR = os.path.join(PROJECT_ROOT, 'kodak_images')
OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'paper_data', 'corrected', 'extra_metrics_corrected.csv')
# ── Metric functions ─────────────────────────────────────────────────────────
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


def _ssim_components(a, b, win=11):
    """Compute luminance (l), contrast (c), and structure (s) components of SSIM."""
    C1, C2, C3 = (0.01 * 255) ** 2, (0.03 * 255) ** 2, (0.03 * 255) ** 2 / 2.0
    mu1 = uniform_filter(a, win)
    mu2 = uniform_filter(b, win)
    mu1_sq, mu2_sq, mu1mu2 = mu1**2, mu2**2, mu1*mu2
    sigma1_sq = uniform_filter(a*a, win) - mu1_sq
    sigma2_sq = uniform_filter(b*b, win) - mu2_sq
    sigma12 = uniform_filter(a*b, win) - mu1mu2

    # Clamp variances to zero (numerical stability)
    sigma1_sq = np.maximum(sigma1_sq, 0)
    sigma2_sq = np.maximum(sigma2_sq, 0)

    l = (2 * mu1mu2 + C1) / (mu1_sq + mu2_sq + C1)
    c = (2 * np.sqrt(np.maximum(sigma1_sq * sigma2_sq, 0)) + C2) / (sigma1_sq + sigma2_sq + C2)
    s = (sigma12 + C3) / (np.sqrt(np.maximum(sigma1_sq * sigma2_sq, 0)) + C3)

    return float(np.mean(l)), float(np.mean(c)), float(np.mean(s))


def _downsample(img, factor=2):
    """Gaussian blur + downsample by factor."""
    blurred = gaussian_filter(img.astype(np.float64), sigma=1.0)
    return blurred[::factor, ::factor]


def compute_ms_ssim_channel(a, b, n_scales=5):
    """Multi-Scale SSIM for a single channel.
    
    Following Wang et al. (2003):
    - At each scale, compute contrast (c) and structure (s) components
    - At the finest scale, also include luminance (l)
    - Weights: [0.0448, 0.2856, 0.3001, 0.2363, 0.1333] (standard)
    """
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    n_scales = min(n_scales, len(weights))

    # Adjust number of scales based on image dimensions
    min_dim = min(a.shape[0], a.shape[1])
    max_scales = int(np.floor(np.log2(min_dim / 11)))  # need at least 11px for SSIM window
    n_scales = min(n_scales, max_scales)
    
    if n_scales < 1:
        return compute_ssim_channel(a, b)
    
    weights = weights[:n_scales]
    weights = weights / weights.sum()  # Re-normalize

    cs_vals = []
    for scale in range(n_scales):
        l, c, s = _ssim_components(a, b)
        cs_vals.append((l, c, s))

        if scale < n_scales - 1:
            a = _downsample(a)
            b = _downsample(b)

    # MS-SSIM = product of (c_j * s_j)^weight_j for j < M, times (l_M * c_M * s_M)^weight_M
    result = 1.0
    for j in range(n_scales - 1):
        l, c, s = cs_vals[j]
        # Use contrast * structure at intermediate scales
        cs = max(c * s, 1e-10)
        result *= cs ** weights[j]

    # At the coarsest scale, include luminance
    l, c, s = cs_vals[-1]
    lcs = max(l * c * s, 1e-10)
    result *= lcs ** weights[-1]

    return float(result)


def compute_ms_ssim(orig, comp):
    """Compute MS-SSIM for an RGB or grayscale image."""
    if orig.ndim == 2:
        return compute_ms_ssim_channel(orig, comp)
    return float(np.mean([
        compute_ms_ssim_channel(orig[:,:,c], comp[:,:,c])
        for c in range(orig.shape[2])
    ]))


def main():
    print('=== Experiment 5: Additional Quality Metric (MS-SSIM) ===')

    fnames = sorted([f for f in os.listdir(KODAK_DIR) if f.endswith('.png')])
    print(f'  Processing {len(fnames)} Kodak images...')

    rows = []

    for fname in fnames:
        fpath = os.path.join(KODAK_DIR, fname)
        pil = Image.open(fpath).convert('RGB')

        # Apply same resize as existing benchmarks
        max_dim = 800
        if max(pil.size) > max_dim:
            ratio = max_dim / max(pil.size)
            new_size = (int(pil.size[0] * ratio), int(pil.size[1] * ratio))
            pil = pil.resize(new_size, Image.LANCZOS)

        orig = np.array(pil, dtype=np.float64)

        # Run adaptive rSVD at rho=95
        comp, k_vals, scores, factors = compress_image(orig, rank=None, energy_percent=95.0)

        # Compute metrics
        mse = compute_mse(orig, comp)
        psnr = compute_psnr(mse)
        ssim = compute_ssim(orig, comp)
        ms_ssim = compute_ms_ssim(orig, comp)

        row = {
            'image_name': fname,
            'psnr': round(psnr, 4),
            'ssim': round(ssim, 6),
            'ms_ssim': round(ms_ssim, 6),
        }
        rows.append(row)
        print(f'  {fname:14s}  PSNR={psnr:.2f}  SSIM={ssim:.4f}  MS-SSIM={ms_ssim:.4f}')

    # Write CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_name', 'psnr', 'ssim', 'ms_ssim'])
        writer.writeheader()
        writer.writerows(rows)

    print(f'\n  Saved: {OUTPUT_CSV}')
    print(f'  Total rows: {len(rows)}')
    print('=== Experiment 5 Complete ===')


if __name__ == '__main__':
    main()
