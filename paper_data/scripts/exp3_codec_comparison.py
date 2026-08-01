"""
Experiment 3 — Modern Codec Baselines
======================================
On Kodak24, encode with WebP, JPEG2000, and AVIF (if available) at quality
levels roughly matching JPEG Q=50, 75, 95.

Output: paper_data/codec_comparison.csv
Columns: method, quality_level, psnr_mean, ssim_mean, file_size_kb_mean,
         cr_mean, latency_ms_mean
"""

import os, sys, io, time, csv
import numpy as np
from PIL import Image

# ── Project root setup ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from scipy.ndimage import uniform_filter

# ── Paths ────────────────────────────────────────────────────────────────────
KODAK_DIR = os.path.join(PROJECT_ROOT, 'kodak_images')
OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'paper_data', 'corrected', 'codec_comparison_corrected.csv')
# ── Check AVIF support ──────────────────────────────────────────────────────
HAS_AVIF = False
try:
    # Try pillow-avif-plugin
    import pillow_avif  # noqa: F401
    HAS_AVIF = True
except ImportError:
    pass

if not HAS_AVIF:
    try:
        # Try pillow-heif
        from pillow_heif import register_avif_opener  # noqa: F401
        register_avif_opener()
        HAS_AVIF = True
    except ImportError:
        pass

if not HAS_AVIF:
    print("NOTE: AVIF encoding not available (pillow-avif-plugin / pillow-heif not installed).")
    print("      AVIF will be skipped. Install with: pip install pillow-avif-plugin")

# ── Check JPEG XL support ───────────────────────────────────────────────────
HAS_JXL = False
try:
    import pillow_jxl  # noqa: F401
    HAS_JXL = True
except ImportError:
    pass

if not HAS_JXL:
    print("NOTE: JPEG XL encoding not available (pillow-jxl-plugin not installed).")
    print("      JPEG XL will be skipped. Install with: pip install pillow-jxl-plugin")

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
    """Load all Kodak images, resized to max_dim."""
    images = []
    pil_images = []
    fnames = sorted([f for f in os.listdir(KODAK_DIR) if f.endswith('.png')])
    for fname in fnames:
        fpath = os.path.join(KODAK_DIR, fname)
        pil = Image.open(fpath).convert('RGB')
        if max(pil.size) > max_dim:
            ratio = max_dim / max(pil.size)
            new_size = (int(pil.size[0] * ratio), int(pil.size[1] * ratio))
            pil = pil.resize(new_size, Image.LANCZOS)
        images.append(np.array(pil, dtype=np.float64))
        pil_images.append(pil)
    return images, pil_images


def encode_webp(pil_img, quality):
    """Encode with WebP and return (decoded_array, file_size_bytes, latency_ms)."""
    t0 = time.perf_counter()
    buf = io.BytesIO()
    pil_img.save(buf, format='WEBP', quality=quality)
    file_bytes = buf.tell()
    buf.seek(0)
    decoded = np.array(Image.open(buf).convert('RGB'), dtype=np.float64)
    latency = (time.perf_counter() - t0) * 1000
    return decoded, file_bytes, latency


def encode_jpeg2000(pil_img, orig_arr, rate):
    """Encode with JPEG2000 per-channel and return (decoded_array, file_size_bytes, latency_ms).
    
    rate controls quality_layers for Pillow JPEG2000 (lower rate = higher quality).
    Approximate mapping: rate=40 ~ Q50, rate=20 ~ Q75, rate=5 ~ Q95.
    """
    t0 = time.perf_counter()
    comp = np.zeros_like(orig_arr)
    total_bytes = 0

    for c in range(3):
        pil_c = Image.fromarray(orig_arr[:, :, c].astype(np.uint8), mode='L')
        buf = io.BytesIO()
        pil_c.save(buf, format='JPEG2000', quality_mode='rates', quality_layers=[rate])
        total_bytes += buf.tell()
        buf.seek(0)
        comp[:, :, c] = np.array(Image.open(buf).convert('L'), dtype=np.float64)

    latency = (time.perf_counter() - t0) * 1000
    return comp, total_bytes, latency


def encode_avif(pil_img, quality):
    """Encode with AVIF and return (decoded_array, file_size_bytes, latency_ms)."""
    t0 = time.perf_counter()
    buf = io.BytesIO()
    pil_img.save(buf, format='AVIF', quality=quality)
    file_bytes = buf.tell()
    buf.seek(0)
    decoded = np.array(Image.open(buf).convert('RGB'), dtype=np.float64)
    latency = (time.perf_counter() - t0) * 1000
    return decoded, file_bytes, latency


def encode_jpegxl(pil_img, quality):
    """Encode with JPEG XL and return (decoded_array, file_size_bytes, latency_ms).

    Uses pillow-jxl-plugin which registers format='JXL' with Pillow.
    quality parameter uses the same 0-100 scale as JPEG/WebP.
    """
    t0 = time.perf_counter()
    buf = io.BytesIO()
    pil_img.save(buf, format='JXL', quality=quality)
    file_bytes = buf.tell()
    buf.seek(0)
    decoded = np.array(Image.open(buf).convert('RGB'), dtype=np.float64)
    latency = (time.perf_counter() - t0) * 1000
    return decoded, file_bytes, latency


def main():
    print('=== Experiment 3: Modern Codec Baselines ===')
    images, pil_images = load_kodak_images()
    print(f'  Loaded {len(images)} Kodak images')

    # Define codec configurations
    # quality_level is our label (matching JPEG Q=50/75/95 intent)
    codecs = []

    # WebP
    for q in [50, 75, 95]:
        codecs.append(('WebP', q, q))

    # JPEG2000 — rate parameter (lower = higher quality)
    # rate=40 gives very lossy (~Q50), rate=20 medium (~Q75), rate=5 high quality (~Q95)
    jp2_rates = {50: 40, 75: 20, 95: 5}
    for q_label, rate in jp2_rates.items():
        codecs.append(('JPEG2000', q_label, rate))

    # AVIF (if available)
    if HAS_AVIF:
        for q in [50, 75, 95]:
            codecs.append(('AVIF', q, q))

    # JPEG XL (if available)
    if HAS_JXL:
        for q in [50, 75, 95]:
            codecs.append(('JPEG_XL', q, q))

    # Collect results per codec config
    results = []

    for method, quality_level, param in codecs:
        print(f'\n  {method} Q~{quality_level} ...', end=' ', flush=True)
        psnrs, ssims, sizes_kb, crs, lats = [], [], [], [], []

        for i, (img, pil_img) in enumerate(zip(images, pil_images)):
            orig_bytes = img.shape[0] * img.shape[1] * 3

            if method == 'WebP':
                decoded, file_bytes, latency = encode_webp(pil_img, param)
            elif method == 'JPEG2000':
                decoded, file_bytes, latency = encode_jpeg2000(pil_img, img, param)
            elif method == 'AVIF':
                decoded, file_bytes, latency = encode_avif(pil_img, param)
            elif method == 'JPEG_XL':
                decoded, file_bytes, latency = encode_jpegxl(pil_img, param)

            mse = compute_mse(img, decoded)
            psnr = compute_psnr(mse)
            ssim = compute_ssim(img, decoded)
            cr = orig_bytes / file_bytes if file_bytes > 0 else 0.0

            psnrs.append(psnr)
            ssims.append(ssim)
            sizes_kb.append(file_bytes / 1024)
            crs.append(cr)
            lats.append(latency)

        row = {
            'method': method,
            'quality_level': quality_level,
            'psnr_mean': round(np.mean(psnrs), 4),
            'ssim_mean': round(np.mean(ssims), 6),
            'file_size_kb_mean': round(np.mean(sizes_kb), 2),
            'cr_mean': round(np.mean(crs), 4),
            'latency_ms_mean': round(np.mean(lats), 2),
        }
        results.append(row)
        print(f'PSNR={row["psnr_mean"]:.2f}  SSIM={row["ssim_mean"]:.4f}  '
              f'Size={row["file_size_kb_mean"]:.1f}KB  CR={row["cr_mean"]:.1f}x  '
              f'Lat={row["latency_ms_mean"]:.1f}ms')

    # Write CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'method', 'quality_level', 'psnr_mean', 'ssim_mean',
            'file_size_kb_mean', 'cr_mean', 'latency_ms_mean'
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f'\n  Saved: {OUTPUT_CSV}')
    print(f'  Total rows: {len(results)}')
    if not HAS_AVIF:
        print('  NOTE: AVIF was skipped (not installed)')
    if not HAS_JXL:
        print('  NOTE: JPEG XL was skipped (not installed)')
    print('=== Experiment 3 Complete ===')


if __name__ == '__main__':
    main()
