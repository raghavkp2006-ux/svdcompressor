"""
Experiment 6 — High-Resolution Scalability
===========================================
Run the full adaptive pipeline on progressively larger images to measure
scalability. Since no native 4K images are available, we upsample kodim23
using bicubic interpolation (noted in output).

Resolutions tested: 768x512 (original), 1920x1080 (FHD), 3840x2160 (4K)

Output: paper_data/highres_scalability.csv
Columns: resolution, latency_ms, peak_memory_mb, psnr, ssim, cr
"""

import os, sys, time, csv, tracemalloc, gc
import numpy as np
from PIL import Image

# ── Project root setup ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from lib.compress import compress_image
from scipy.ndimage import uniform_filter

# ── Paths ────────────────────────────────────────────────────────────────────
KODAK_DIR = os.path.join(PROJECT_ROOT, 'kodak_images')
OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'paper_data', 'corrected', 'highres_scalability_corrected.csv')
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


# Test resolutions: (width, height, label, note)
RESOLUTIONS = [
    (768, 512,   '768x512',   'original Kodak resolution'),
    (1920, 1080, '1920x1080', 'upsampled via bicubic (FHD)'),
    (3840, 2160, '3840x2160', 'upsampled via bicubic (4K)'),
]


def compute_cr_perchannel(shape, k_vals):
    """Compute theoretical CR from per-channel rank values."""
    if len(shape) == 2:
        m, n = shape; ch = 1
    else:
        m, n, ch = shape
    orig_size = m * n * ch
    comp_size = 0
    for k in k_vals:
        if k == 0:
            comp_size += m * n
        else:
            comp_size += k * (m + n + 1)
    return float(orig_size / comp_size) if comp_size > 0 else 0.0


def main():
    print('=== Experiment 6: High-Resolution Scalability ===')

    source_path = os.path.join(KODAK_DIR, 'kodim23.png')
    source_pil = Image.open(source_path).convert('RGB')
    print(f'  Source: kodim23.png ({source_pil.size[0]}x{source_pil.size[1]})')

    rows = []

    for width, height, label, note in RESOLUTIONS:
        print(f'\n  {label} ({note}) ...', flush=True)

        # Resize to target resolution
        pil = source_pil.resize((width, height), Image.BICUBIC)
        img = np.array(pil, dtype=np.float64)

        # Force garbage collection before measurement
        gc.collect()

        # Measure with tracemalloc
        tracemalloc.start()
        t0 = time.perf_counter()

        comp_arr, k_vals, scores, factors = compress_image(
            img, rank=None, energy_percent=95.0
        )

        latency = (time.perf_counter() - t0) * 1000
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)

        # Quality metrics
        mse = compute_mse(img, comp_arr)
        psnr = compute_psnr(mse)
        ssim = compute_ssim(img, comp_arr)
        cr = compute_cr_perchannel(img.shape, k_vals)

        row = {
            'resolution': label,
            'latency_ms': round(latency, 2),
            'peak_memory_mb': round(peak_mb, 2),
            'psnr': round(psnr, 4),
            'ssim': round(ssim, 6),
            'cr': round(cr, 4),
        }
        rows.append(row)
        print(f'    Latency={latency:.1f}ms  Peak RAM={peak_mb:.1f}MB  '
              f'PSNR={psnr:.2f}  SSIM={ssim:.4f}  CR={cr:.2f}x  k={k_vals}')

    # Write CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'resolution', 'latency_ms', 'peak_memory_mb', 'psnr', 'ssim', 'cr'
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f'\n  Saved: {OUTPUT_CSV}')
    print(f'  NOTE: FHD and 4K resolutions were produced by upsampling kodim23.png')
    print(f'        via bicubic interpolation. This is noted for transparency.')
    print('=== Experiment 6 Complete ===')


if __name__ == '__main__':
    main()
