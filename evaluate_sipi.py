import os, sys, time, csv
import numpy as np
from PIL import Image

# ── Project root setup ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from lib.compress import compress_image
from skimage.metrics import structural_similarity as ssim_skimage

def compute_mse(a, b):
    return float(np.mean((a - b) ** 2))

def compute_psnr(mse_val):
    if mse_val == 0:
        return float('inf')
    return float(10 * np.log10(255**2 / mse_val))

def compute_ssim(orig, comp):
    kwargs = {'data_range': 255.0}
    if orig.ndim == 3:
        kwargs['channel_axis'] = 2
    return float(ssim_skimage(orig, comp, **kwargs))

def find_images(directory, extensions=('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
    if not os.path.isdir(directory):
        return []
    files = []
    for f in sorted(os.listdir(directory)):
        if any(f.lower().endswith(ext) for ext in extensions):
            files.append(os.path.join(directory, f))
    return files

def compute_cr_perchannel(shape, k_vals):
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
    psnrs, ssims, crs, lats = [], [], [], []
    for fpath in image_paths:
        try:
            pil = Image.open(fpath).convert('RGB')
        except Exception as e:
            print(f'    WARNING: Could not open {fpath}: {e}')
            continue

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
        
        # Detect outliers
        is_outlier = psnr > 60.0 or psnr == float('inf')
        if is_outlier:
            skipped_all = all(k == 0 for k in k_vals)
            print(f"  [OUTLIER] {os.path.basename(fpath)} (size: {orig.shape})")
            print(f"      PSNR before clamp: {psnr}")
            print(f"      MSE: {mse}")
            print(f"      All channels skipped (k_vals={k_vals}): {skipped_all}")
            
        if psnr == float('inf'):
            psnr = 100.0
            
        # The user's prompt suggested checking if it was applied incorrectly.
        # Actually, numpy can return np.inf if the division overflows, but compute_psnr uses float('inf')
        # However, just to be safe:
        if np.isinf(psnr) or psnr > 100.0: # Cap all at 100
             psnr = 100.0

        ssim = compute_ssim(orig, comp)
        cr = compute_cr_perchannel(orig.shape, k_vals)

        print(f"  {os.path.basename(fpath):20s}  PSNR: {psnr:.2f}  SSIM: {ssim:.4f}")

        psnrs.append(psnr)
        ssims.append(ssim)
        crs.append(cr)
        lats.append(latency)
    return psnrs, ssims, crs, lats

def main():
    sipi_dir = os.path.join(PROJECT_ROOT, 'data', 'usc_sipi', 'misc')
    if not os.path.exists(sipi_dir):
        sipi_dir = os.path.join(PROJECT_ROOT, 'data', 'usc_sipi') # check if it extracted flat
    
    image_paths = find_images(sipi_dir)
    print(f'Found {len(image_paths)} images in {sipi_dir}')
    
    psnrs, ssims, crs, lats = run_on_dataset(image_paths, 'USC-SIPI')
    
    row = {
        'dataset': 'USC-SIPI',
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
    
    print(row)
    
    output_csv = os.path.join(PROJECT_ROOT, 'usc_sipi_results.csv')
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'dataset', 'n_images', 'psnr_mean', 'psnr_std', 'ssim_mean', 'ssim_std',
            'cr_mean', 'cr_std', 'latency_ms_mean', 'latency_ms_std'
        ])
        writer.writeheader()
        writer.writerow(row)
    
    print(f'Saved to {output_csv}')

if __name__ == '__main__':
    main()
