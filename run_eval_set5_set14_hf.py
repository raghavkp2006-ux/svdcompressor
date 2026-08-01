import os, sys, time, csv
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim_skimage

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from lib.compress import compress_image

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

def run_on_dataset(image_paths, dataset_name):
    psnrs, ssims, crs, lats = [], [], [], []
    print(f"\n--- Evaluating on {dataset_name} ({len(image_paths)} images) ---")
    for fpath in image_paths:
        try:
            pil = Image.open(fpath).convert('RGB')
        except Exception as e:
            print(f'    WARNING: Could not open {fpath}: {e}')
            continue

        orig = np.array(pil, dtype=np.float64)

        t0 = time.perf_counter()
        comp, k_vals, scores, factors = compress_image(orig, rank=None, energy_percent=95.0)
        latency = (time.perf_counter() - t0) * 1000

        mse = compute_mse(orig, comp)
        psnr = compute_psnr(mse)
        
        is_outlier = psnr > 60.0 or psnr == float('inf')
        if is_outlier:
            skipped_all = all(k == 0 for k in k_vals)
            print(f"  [OUTLIER] {os.path.basename(fpath)} (size: {orig.shape})")
            print(f"      PSNR before clamp: {psnr}")
            print(f"      MSE: {mse}")
            print(f"      All channels skipped (k_vals={k_vals}): {skipped_all}")
            
        if np.isinf(psnr) or psnr > 100.0:
             psnr = 100.0

        ssim = compute_ssim(orig, comp)
        cr = compute_cr_perchannel(orig.shape, k_vals)

        psnrs.append(psnr)
        ssims.append(ssim)
        crs.append(cr)
        lats.append(latency)
        
    return psnrs, ssims, crs, lats

def main():
    t0_main = time.time()
    
    set5_dir = os.path.join(PROJECT_ROOT, "data", "Set5")
    set14_dir = os.path.join(PROJECT_ROOT, "data", "Set14")
    
    from datasets import load_dataset
    import shutil
    os.makedirs(set5_dir, exist_ok=True)
    os.makedirs(set14_dir, exist_ok=True)
    
    print("Loading Set5 from HF...")
    ds5 = load_dataset("eugenesiow/Set5", "bicubic_x2", split="validation")
    set5_paths = []
    for i, item in enumerate(ds5):
        img = item['hr']
        path = os.path.join(set5_dir, f"{i:03d}.png")
        img.save(path)
        set5_paths.append(path)
        
    print("Loading Set14 from HF...")
    ds14 = load_dataset("eugenesiow/Set14", "bicubic_x2", split="validation")
    set14_paths = []
    for i, item in enumerate(ds14):
        img = item['hr']
        path = os.path.join(set14_dir, f"{i:03d}.png")
        img.save(path)
        set14_paths.append(path)
    
    results = []
    for dname, paths in [('Set5', set5_paths), ('Set14', set14_paths)]:
        psnrs, ssims, crs, lats = run_on_dataset(paths, dname)
        if len(psnrs) > 0:
            row = {
                'dataset': dname,
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
            results.append(row)
            if row['psnr_std'] > row['psnr_mean']:
                print(f"\nWARNING: {dname} has psnr_std ({row['psnr_std']}) > psnr_mean ({row['psnr_mean']})")

    output_csv = os.path.join(PROJECT_ROOT, 'bsd_set5_set14_results.csv')
    if results:
        with open(output_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'dataset', 'n_images', 'psnr_mean', 'psnr_std', 'ssim_mean', 'ssim_std',
                'cr_mean', 'cr_std', 'latency_ms_mean', 'latency_ms_std'
            ])
            writer.writerows(results)
        print(f"\nResults appended to {output_csv}")
    
    print(f"Total runtime: {time.time() - t0_main:.2f} seconds")

if __name__ == '__main__':
    main()
