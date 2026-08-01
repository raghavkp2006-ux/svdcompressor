"""
Experiment 2 — Generalization to Other Datasets
================================================
Run the adaptive rSVD pipeline (rho=95, default params) on each available
dataset beyond Kodak24. Reports per-dataset aggregated statistics.

Looks for datasets at:
  - data/raw/BSR/BSDS500/data/images/test/  (BSD500 test set, 200 images)
  - data/genai_dataset/input/               (GenAI dataset if images present)
  - Kodak24 (always available, included as baseline)

Output: paper_data/corrected/generalization_results_corrected.csv
Columns: dataset, n_images, psnr_mean, psnr_std, ssim_mean, ssim_std,
         cr_mean, cr_std, latency_ms_mean, latency_ms_std
"""

import os, sys, time, csv, glob
import numpy as np
from PIL import Image

# ── Project root setup ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from lib.compress import compress_image
from scipy.ndimage import uniform_filter

# ── Paths ────────────────────────────────────────────────────────────────────
KODAK_DIR = os.path.join(PROJECT_ROOT, 'kodak_images')
OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'paper_data', 'generalization_results.csv')

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


def find_images(directory, extensions=('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
    """Find all image files in a directory (non-recursive)."""
    if not os.path.isdir(directory):
        return []
    files = []
    for f in sorted(os.listdir(directory)):
        if any(f.lower().endswith(ext) for ext in extensions):
            files.append(os.path.join(directory, f))
    return files


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


def run_on_dataset(image_paths, dataset_name, max_dim=800):
    """Run adaptive rSVD on all images and return per-image metrics."""
    psnrs, ssims, crs, lats = [], [], [], []

    for fpath in image_paths:
        try:
            pil = Image.open(fpath).convert('RGB')
        except Exception as e:
            print(f'    WARNING: Could not open {fpath}: {e}')
            continue

        # Apply same resize convention as existing benchmarks
        if max(pil.size) > max_dim:
            ratio = max_dim / max(pil.size)
            new_size = (int(pil.size[0] * ratio), int(pil.size[1] * ratio))
            pil = pil.resize(new_size, Image.LANCZOS)

        orig = np.array(pil, dtype=np.float64)

        t0 = time.perf_counter()
        comp, k_vals, scores, factors = compress_image(orig, rank=None, energy_percent=95.0)
        latency = (time.perf_counter() - t0) * 1000

        mse = compute_mse(orig, comp)
        psnr = compute_psnr(mse)
        # Cap at 100 dB for lossless-skip images (MSE=0 → inf PSNR)
        if psnr == float('inf'):
            psnr = 100.0
        ssim = compute_ssim(orig, comp)
        cr = compute_cr_perchannel(orig.shape, k_vals)

        psnrs.append(psnr)
        ssims.append(ssim)
        crs.append(cr)
        lats.append(latency)

    return psnrs, ssims, crs, lats


def main():
    print('=== Experiment 2: Generalization to Other Datasets ===')

    # Discover available datasets
    datasets = {}

    # 1. Kodak24 (always available)
    kodak_imgs = find_images(KODAK_DIR)
    if kodak_imgs:
        datasets['Kodak24'] = kodak_imgs

    # 2. BSD500 test set (if downloaded)
    bsd_paths = [
        os.path.join(PROJECT_ROOT, 'data', 'raw', 'BSR', 'BSDS500', 'data', 'images', 'test'),
        os.path.join(PROJECT_ROOT, 'data', 'raw', 'BSDS500', 'data', 'images', 'test'),
        os.path.join(PROJECT_ROOT, 'data', 'bsd500'),
        os.path.join(PROJECT_ROOT, 'data', 'BSD500'),
    ]
    for bp in bsd_paths:
        imgs = find_images(bp)
        if imgs:
            datasets['BSD500_test'] = imgs
            break

    # 3. GenAI dataset (input images)
    genai_path = os.path.join(PROJECT_ROOT, 'data', 'genai_dataset', 'input')
    genai_imgs = find_images(genai_path)
    if genai_imgs:
        datasets['GenAI'] = genai_imgs

    # 4. GenAI dataset (target images as alternative)
    genai_target_path = os.path.join(PROJECT_ROOT, 'data', 'genai_dataset', 'target')
    genai_target_imgs = find_images(genai_target_path)
    if genai_target_imgs and 'GenAI' not in datasets:
        datasets['GenAI_target'] = genai_target_imgs

    print(f'  Found {len(datasets)} dataset(s):')
    for name, imgs in datasets.items():
        print(f'    {name}: {len(imgs)} images')

    if len(datasets) < 2:
        print('\n  WARNING: Only Kodak24 is available. No additional datasets found.')
        print('  To add BSD500, run: python download_bsd500.py')
        print('  Proceeding with available datasets only.')

    rows = []

    for dataset_name, image_paths in datasets.items():
        print(f'\n  Running on {dataset_name} ({len(image_paths)} images)...', flush=True)
        psnrs, ssims, crs, lats = run_on_dataset(image_paths, dataset_name)

        if not psnrs:
            print(f'    No valid images processed for {dataset_name}')
            continue

        row = {
            'dataset': dataset_name,
            'n_images': len(psnrs),
            'psnr_mean': round(np.mean(psnrs), 4),
            'psnr_std': round(np.std(psnrs), 4),
            'ssim_mean': round(np.mean(ssims), 6),
            'ssim_std': round(np.std(ssims), 6),
            'cr_mean': round(np.mean(crs), 4),
            'cr_std': round(np.std(crs), 4),
            'latency_ms_mean': round(np.mean(lats), 2),
            'latency_ms_std': round(np.std(lats), 2),
        }
        rows.append(row)
        print(f'    PSNR={row["psnr_mean"]:.2f}±{row["psnr_std"]:.2f}  '
              f'SSIM={row["ssim_mean"]:.4f}±{row["ssim_std"]:.4f}  '
              f'CR={row["cr_mean"]:.2f}±{row["cr_std"]:.2f}  '
              f'Lat={row["latency_ms_mean"]:.1f}±{row["latency_ms_std"]:.1f}ms')

    # Write CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'dataset', 'n_images', 'psnr_mean', 'psnr_std', 'ssim_mean', 'ssim_std',
            'cr_mean', 'cr_std', 'latency_ms_mean', 'latency_ms_std'
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f'\n  Saved: {OUTPUT_CSV}')
    print(f'  Total datasets: {len(rows)}')
    print('=== Experiment 2 Complete ===')


if __name__ == '__main__':
    main()
