import os, sys, time, csv, io
import numpy as np
import requests, zipfile, tarfile
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

def find_images(directory, extensions=('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
    if not os.path.isdir(directory):
        return []
    files = []
    for f in sorted(os.listdir(directory)):
        if any(f.lower().endswith(ext) for ext in extensions):
            files.append(os.path.join(directory, f))
    return files

def download_with_retries(url, dest_path=None, max_retries=3, is_zip_mem=False):
    for attempt in range(1, max_retries + 1):
        try:
            print(f"    Attempt {attempt} for {url}")
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, stream=True, headers=headers, timeout=30)
            response.raise_for_status()
            
            if is_zip_mem:
                content = response.content
                print(f"      Download complete ({len(content)} bytes)")
                return io.BytesIO(content)
            else:
                with open(dest_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"      Download complete, saved to {dest_path}")
                return dest_path
        except Exception as e:
            print(f"      Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    return None

def download_bsd500():
    target_dir = os.path.join(PROJECT_ROOT, "data", "raw", "BSR", "BSDS500", "data", "images", "test")
    if os.path.exists(target_dir) and len(find_images(target_dir)) > 0:
        print("BSD500 already exists.")
        return target_dir, "Already local"

    os.makedirs(os.path.join(PROJECT_ROOT, "data", "raw"), exist_ok=True)
    
    # 1. Retry original URL
    url1 = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
    tgz_path = os.path.join(PROJECT_ROOT, "data", "raw", "BSR_bsds500.tgz")
    print(f"Trying BSD500 Fallback 1: {url1}")
    res = download_with_retries(url1, dest_path=tgz_path)
    if res:
        try:
            print("      Extracting BSD500 tgz...")
            with tarfile.open(tgz_path, "r:gz") as tar:
                tar.extractall(os.path.join(PROJECT_ROOT, "data", "raw"))
            if os.path.exists(target_dir):
                return target_dir, url1
        except Exception as e:
            print(f"      Extraction failed: {e}")

    # 2. Try Kaggle CLI
    print("Trying BSD500 Fallback 2: Kaggle CLI")
    try:
        import subprocess
        res = subprocess.run(["kaggle", "datasets", "download", "-d", "balraj98/berkeley-segmentation-dataset-500-bsds500", "-p", os.path.join(PROJECT_ROOT, "data", "raw")], capture_output=True, text=True)
        if res.returncode == 0:
            print("      Kaggle download successful, extracting...")
            k_zip = os.path.join(PROJECT_ROOT, "data", "raw", "berkeley-segmentation-dataset-500-bsds500.zip")
            with zipfile.ZipFile(k_zip, 'r') as z:
                z.extractall(os.path.join(PROJECT_ROOT, "data", "raw", "kaggle_bsd"))
            kaggle_test_dir = os.path.join(PROJECT_ROOT, "data", "raw", "kaggle_bsd", "images", "test")
            if os.path.exists(kaggle_test_dir):
                return kaggle_test_dir, "Kaggle CLI"
        else:
            print(f"      Kaggle CLI failed: {res.stderr}")
    except Exception as e:
        print(f"      Kaggle fallback failed: {e}")

    # 3. Try Github mirror
    url3 = "https://github.com/BIDS/BSDS500/archive/refs/heads/master.zip"
    print(f"Trying BSD500 Fallback 3: {url3}")
    zip_mem = download_with_retries(url3, is_zip_mem=True)
    if zip_mem:
        try:
            print("      Extracting BSD500 GitHub zip...")
            with zipfile.ZipFile(zip_mem) as z:
                z.extractall(os.path.join(PROJECT_ROOT, "data", "raw"))
            git_target = os.path.join(PROJECT_ROOT, "data", "raw", "BSDS500-master", "BSDS500", "data", "images", "test")
            if os.path.exists(git_target):
                return git_target, url3
        except Exception as e:
            print(f"      Extraction failed: {e}")
            
    print("All BSD500 downloads failed.")
    return None, None

def download_set5_set14():
    set5_dir = os.path.join(PROJECT_ROOT, "data", "Set5")
    set14_dir = os.path.join(PROJECT_ROOT, "data", "Set14")
    
    already_have = os.path.exists(set5_dir) and len(find_images(set5_dir)) > 0 and os.path.exists(set14_dir) and len(find_images(set14_dir)) > 0
    if already_have:
        print("Set5/Set14 already exist.")
        return set5_dir, set14_dir, "Already local"

    # 1. & 2. Try SelfExSR Zip
    url1 = "https://github.com/jbhuang0604/SelfExSR/archive/refs/heads/master.zip"
    print(f"Trying Set5/Set14 Fallback 1 & 2: {url1}")
    zip_mem = download_with_retries(url1, is_zip_mem=True)
    if zip_mem:
        try:
            import shutil
            print("      Extracting Set5 and Set14...")
            with zipfile.ZipFile(zip_mem) as z:
                for file_info in z.infolist():
                    if file_info.filename.startswith("SelfExSR-master/data/Set5/") or file_info.filename.startswith("SelfExSR-master/data/Set14/"):
                        if not file_info.filename.endswith('/'):
                            z.extract(file_info, os.path.join(PROJECT_ROOT, "data", "tmp_selfexsr"))
            
            tmp_set5 = os.path.join(PROJECT_ROOT, "data", "tmp_selfexsr", "SelfExSR-master", "data", "Set5")
            tmp_set14 = os.path.join(PROJECT_ROOT, "data", "tmp_selfexsr", "SelfExSR-master", "data", "Set14")
            
            if not os.path.exists(set5_dir) and os.path.exists(tmp_set5):
                shutil.copytree(tmp_set5, set5_dir)
            if not os.path.exists(set14_dir) and os.path.exists(tmp_set14):
                shutil.copytree(tmp_set14, set14_dir)
            
            if os.path.exists(os.path.join(PROJECT_ROOT, "data", "tmp_selfexsr")):
                shutil.rmtree(os.path.join(PROJECT_ROOT, "data", "tmp_selfexsr"))
            
            if os.path.exists(set5_dir) and os.path.exists(set14_dir):
                return set5_dir, set14_dir, url1
        except Exception as e:
            print(f"      Extraction failed: {e}")

    # 3. Huggingface Datasets
    print("Trying Set5/Set14 Fallback 3: Huggingface datasets")
    try:
        from datasets import load_dataset
        import shutil
        os.makedirs(set5_dir, exist_ok=True)
        os.makedirs(set14_dir, exist_ok=True)
        
        print("      Loading Set5 from HF...")
        ds5 = load_dataset("eugenesiow/Set5", "bicubic_x2", split="validation")
        for i, item in enumerate(ds5):
            img = item['hr']
            img.save(os.path.join(set5_dir, f"{i:03d}.png"))
            
        print("      Loading Set14 from HF...")
        ds14 = load_dataset("eugenesiow/Set14", "bicubic_x2", split="validation")
        for i, item in enumerate(ds14):
            img = item['hr']
            img.save(os.path.join(set14_dir, f"{i:03d}.png"))
            
        return set5_dir, set14_dir, "Huggingface datasets"
    except ImportError:
        print("      Huggingface 'datasets' package not installed.")
    except Exception as e:
        print(f"      HF fallback failed: {e}")
        
    print("All Set5/Set14 downloads failed.")
    return None, None, None


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
        
        # print(f"  {os.path.basename(fpath):20s}  PSNR: {psnr:.2f}  SSIM: {ssim:.4f}")
        
    return psnrs, ssims, crs, lats

def main():
    t0_main = time.time()
    
    print("--- Starting Downloads ---")
    bsd500_dir, b_src = download_bsd500()
    set5_dir, set14_dir, s_src = download_set5_set14()
    
    if b_src: print(f"BSD500 Source: {b_src}")
    if s_src: print(f"Set5/Set14 Source: {s_src}")
    
    datasets = []
    if bsd500_dir: datasets.append(('BSD500', bsd500_dir))
    if set5_dir: datasets.append(('Set5', set5_dir))
    if set14_dir: datasets.append(('Set14', set14_dir))
        
    if not datasets:
        print("\nNo datasets could be acquired. Exiting.")
        return
        
    results = []
    for dname, ddir in datasets:
        image_paths = find_images(ddir)
        if len(image_paths) == 0:
            print(f"No images found in {ddir} for {dname}.")
            continue
        psnrs, ssims, crs, lats = run_on_dataset(image_paths, dname)
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
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'dataset', 'n_images', 'psnr_mean', 'psnr_std', 'ssim_mean', 'ssim_std',
                'cr_mean', 'cr_std', 'latency_ms_mean', 'latency_ms_std'
            ])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {output_csv}")
    
    print(f"Total runtime: {time.time() - t0_main:.2f} seconds")

if __name__ == '__main__':
    main()
