"""
run_benchmarks.py
=================
Runs ALL high-priority benchmarks for the research paper:
  1. JPEG baseline comparison (Q=50, 75, 95) on Kodak23
  2. Full ablation study (4 system variants)
  3. Full Kodak 24-image benchmark (adaptive rSVD)
  4. Generates publication-ready figures

Imports lib/ modules directly -- no Flask server required.
"""

import os, sys, io, time, json
import numpy as np
import scipy.stats as stats
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Add project root to path so lib/ can be imported ─────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.rsvd import rsvd, reconstruct
from lib.prescreening import complexity_score, recommend_rank, DEFAULT_SKIP_THRESHOLD
from lib.compress import compress_image, compress_image_basic, true_compression_ratio, serialize_factors
from scipy.ndimage import uniform_filter

# ── Paths ────────────────────────────────────────────────────────────────────
KODAK_DIR  = os.path.join(os.path.dirname(__file__), 'kodak_images')
SCREENSHOT = os.path.join(os.path.dirname(__file__), 'screenshots')
RESULTS    = os.path.join(os.path.dirname(__file__), 'benchmark_results.json')
os.makedirs(KODAK_DIR, exist_ok=True)
os.makedirs(SCREENSHOT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'figure.dpi': 150, 'savefig.dpi': 200,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})

# ── Metrics (same as app.py) ─────────────────────────────────────────────────
def compute_mse(a, b):
    return float(np.mean((a - b) ** 2))

def compute_psnr(mse_val):
    if mse_val == 0:
        return float('inf')
    return float(10 * np.log10(255**2 / mse_val))

def compute_ssim_channel(a, b, win=11):
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1 = uniform_filter(a, win); mu2 = uniform_filter(b, win)
    mu1_sq, mu2_sq, mu1mu2 = mu1**2, mu2**2, mu1*mu2
    s1 = uniform_filter(a*a, win) - mu1_sq
    s2 = uniform_filter(b*b, win) - mu2_sq
    s12 = uniform_filter(a*b, win) - mu1mu2
    num = (2*mu1mu2+C1)*(2*s12+C2)
    den = (mu1_sq+mu2_sq+C1)*(s1+s2+C2)
    return float(np.mean(num/den))

def compute_ssim(orig, comp):
    if orig.ndim == 2:
        return compute_ssim_channel(orig, comp)
    return float(np.mean([
        compute_ssim_channel(orig[:,:,c], comp[:,:,c])
        for c in range(orig.shape[2])
    ]))

def compute_cr(shape, k):
    if len(shape) == 2:
        m, n = shape; ch = 1
    else:
        m, n, ch = shape
    orig_size = m * n * ch
    comp_size = ch * k * (m + n + 1)
    return float(orig_size / comp_size) if comp_size > 0 else 0.0

def savefig(fig, name):
    path = os.path.join(SCREENSHOT, name)
    fig.savefig(path); plt.close(fig)
    print('    saved -> %s' % name)


# ==============================================================================
#  STEP 0: Download all 24 Kodak images
# ==============================================================================
def download_kodak():
    import urllib.request
    print('=== Downloading Kodak images ===')
    for i in range(1, 25):
        fname = 'kodim%02d.png' % i
        fpath = os.path.join(KODAK_DIR, fname)
        if os.path.exists(fpath) and os.path.getsize(fpath) > 10000:
            continue
        url = 'https://r0k.us/graphics/kodak/kodak/%s' % fname
        print('  Downloading %s ...' % fname, end=' ')
        try:
            urllib.request.urlretrieve(url, fpath)
            print('OK (%d KB)' % (os.path.getsize(fpath) // 1024))
        except Exception as e:
            print('FAILED: %s' % str(e))
    count = len([f for f in os.listdir(KODAK_DIR) if f.endswith('.png')])
    print('  Kodak images available: %d/24\n' % count)
    return count


# ==============================================================================
#  STEP 1: JPEG baseline comparison
# ==============================================================================
def run_jpeg_baseline():
    print('=== JPEG Baseline Comparison ===')
    test_img_path = os.path.join(KODAK_DIR, 'kodim23.png')
    if not os.path.exists(test_img_path):
        test_img_path = os.path.join(os.path.dirname(__file__), 'test_image.png')

    orig_pil = Image.open(test_img_path).convert('RGB')
    # Resize to match our pipeline (max 800px)
    max_dim = 800
    if max(orig_pil.size) > max_dim:
        ratio = max_dim / max(orig_pil.size)
        new_size = (int(orig_pil.size[0] * ratio), int(orig_pil.size[1] * ratio))
        orig_pil = orig_pil.resize(new_size, Image.LANCZOS)

    orig_arr = np.array(orig_pil, dtype=np.float64)
    orig_bytes = orig_arr.shape[0] * orig_arr.shape[1] * 3  # raw RGB bytes

    results = []
    jpeg_images = {}

    for fmt, quality in [('JPEG',50), ('JPEG',75), ('JPEG',95), ('WEBP',50), ('WEBP',75), ('WEBP',95)]:
        t0 = time.time()
        buf = io.BytesIO()
        orig_pil.save(buf, format=fmt, quality=quality)
        jpeg_bytes = buf.tell()
        buf.seek(0)
        jpeg_pil = Image.open(buf).convert('RGB')
        latency = (time.time() - t0) * 1000

        jpeg_arr = np.array(jpeg_pil, dtype=np.float64)
        mse = compute_mse(orig_arr, jpeg_arr)
        psnr = compute_psnr(mse)
        ssim = compute_ssim(orig_arr, jpeg_arr)
        cr = orig_bytes / jpeg_bytes

        if fmt == 'JPEG':
            jpeg_images[quality] = jpeg_pil
            
        row = {
            'method': '%s Q=%d' % (fmt, quality),
            'quality': quality,
            'psnr': round(psnr, 2),
            'ssim': round(ssim, 4),
            'mse': round(mse, 2),
            'file_kb': round(jpeg_bytes / 1024, 1),
            'cr': round(cr, 1),
            'latency_ms': round(latency, 1),
        }
        results.append(row)
        print('  %s Q=%-3d  PSNR=%.2f  SSIM=%.4f  Size=%.1fKB  CR=%.1fx  Lat=%.1fms'
              % (fmt, quality, psnr, ssim, jpeg_bytes/1024, cr, latency))

    # Also run our rSVD adaptive at 95% energy for direct comparison
    t0 = time.time()
    comp_arr, k_vals, scores, factors = compress_image(orig_arr, rank=None, energy_percent=95.0)
    latency = (time.time() - t0) * 1000
    used_k = max(k_vals)
    mse = compute_mse(orig_arr, comp_arr)
    psnr = compute_psnr(mse)
    ssim = compute_ssim(orig_arr, comp_arr)
    U_list, S_list, Vt_list = zip(*factors)
    cr = true_compression_ratio(orig_arr.shape, U_list, S_list, Vt_list)

    row = {
        'method': 'rSVD Adaptive (proposed)',
        'psnr': round(psnr, 2),
        'ssim': round(ssim, 4),
        'mse': round(mse, 2),
        'cr': round(cr, 1),
        'latency_ms': round(latency, 1),
        'k_used': used_k,
        'k_per_channel': k_vals,
    }
    results.append(row)
    print('  rSVD Adpt   PSNR=%.2f  SSIM=%.4f  CR=%.1fx  Lat=%.1fms  k=%s'
          % (psnr, ssim, cr, latency, k_vals))

    # Generate comparison figure
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
    axes[0].imshow(orig_pil); axes[0].set_title('Original', fontweight='bold'); axes[0].axis('off')
    for idx, q in enumerate([50, 75, 95]):
        axes[idx+1].imshow(jpeg_images[q])
        axes[idx+1].set_title('JPEG Q=%d\nPSNR=%.1f  SSIM=%.4f' % (q, results[idx]['psnr'], results[idx]['ssim']),
                              fontweight='bold', fontsize=8)
        axes[idx+1].axis('off')
    # rSVD result
    comp_pil = Image.fromarray(np.clip(comp_arr, 0, 255).astype(np.uint8))
    axes[4].imshow(comp_pil)
    axes[4].set_title('rSVD Adaptive\nPSNR=%.1f  SSIM=%.4f' % (row['psnr'], row['ssim']),
                      fontweight='bold', fontsize=8, color='#4F46E5')
    axes[4].axis('off')
    fig.suptitle('JPEG vs. Proposed rSVD Adaptive Compression on Kodak23', fontweight='bold', y=1.02)
    plt.tight_layout()
    savefig(fig, 'fig10_jpeg_comparison.png')

    print()
    return results


# ==============================================================================
#  STEP 2: Ablation study
# ==============================================================================
def run_ablation():
    print('=== Ablation Study ===')
    test_img_path = os.path.join(KODAK_DIR, 'kodim23.png')
    if not os.path.exists(test_img_path):
        test_img_path = os.path.join(os.path.dirname(__file__), 'test_image.png')

    orig_pil = Image.open(test_img_path).convert('RGB')
    max_dim = 800
    if max(orig_pil.size) > max_dim:
        ratio = max_dim / max(orig_pil.size)
        new_size = (int(orig_pil.size[0] * ratio), int(orig_pil.size[1] * ratio))
        orig_pil = orig_pil.resize(new_size, Image.LANCZOS)

    orig_arr = np.array(orig_pil, dtype=np.float64)

    variants = [
        # (label, use_prescreener, power_iter)
        ('No pre-screener, q=0', False, 0),
        ('No pre-screener, q=2', False, 2),
        ('Pre-screener, q=0',    True,  0),
        ('Pre-screener, q=2 (proposed)', True, 2),
    ]

    results = []
    energy = 95.0

    for label, use_prescreen, q in variants:
        print('  %s ...' % label, end=' ')
        h, w = orig_arr.shape[:2]
        ch_count = orig_arr.shape[2] if orig_arr.ndim == 3 else 1

        t0 = time.time()
        comp = np.zeros_like(orig_arr)
        k_vals = []
        U_list = []; S_list = []; Vt_list = []

        for c in range(ch_count):
            channel = orig_arr[:,:,c] if orig_arr.ndim == 3 else orig_arr
            max_d = min(h, w)

            if use_prescreen:
                score = complexity_score(channel)
                if score < DEFAULT_SKIP_THRESHOLD:
                    comp[:,:,c] = channel.copy() if orig_arr.ndim == 3 else channel.copy()
                    k_vals.append(0)
                    U_list.append(None); S_list.append(None); Vt_list.append(None)
                    continue
                k = recommend_rank(score, energy, max_d)
            else:
                k = recommend_rank(4000.0, energy, max_d)  # mid-range default

            k = min(k, max_d)
            U, S, Vt = rsvd(channel, k, oversample=10, power_iter=q)
            recon = reconstruct(U, S, Vt)
            np.clip(recon, 0, 255, out=recon)

            if orig_arr.ndim == 3:
                comp[:,:,c] = recon
            else:
                comp = recon
            k_vals.append(k)
            U_list.append(U); S_list.append(S); Vt_list.append(Vt)

        latency = (time.time() - t0) * 1000
        used_k = max(k_vals) if k_vals else 0
        mse = compute_mse(orig_arr, comp)
        psnr = compute_psnr(mse)
        ssim = compute_ssim(orig_arr, comp)
        cr = true_compression_ratio(orig_arr.shape, U_list, S_list, Vt_list)

        row = {
            'variant': label,
            'use_prescreener': use_prescreen,
            'power_iter': q,
            'psnr': round(psnr, 2),
            'ssim': round(ssim, 4),
            'mse': round(mse, 2),
            'cr': round(cr, 1),
            'latency_ms': round(latency, 1),
            'k_per_channel': k_vals,
        }
        results.append(row)
        print('PSNR=%.2f  SSIM=%.4f  CR=%.1fx  Lat=%.1fms  k=%s'
              % (psnr, ssim, cr, latency, k_vals))

    # Generate ablation figure (bar chart comparing PSNR and SSIM)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    labels = [r['variant'].replace(' (proposed)', '\n(proposed)') for r in results]
    x = range(len(labels))
    colors = ['#94a3b8', '#64748b', '#818cf8', '#4F46E5']

    axes[0].bar(x, [r['psnr'] for r in results], color=colors, width=0.6, edgecolor='white', linewidth=1.2)
    for i, r in enumerate(results):
        axes[0].text(i, r['psnr'] + 0.15, '%.1f' % r['psnr'], ha='center', fontsize=8, fontweight='bold')
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=7)
    axes[0].set_ylabel('PSNR (dB)'); axes[0].set_title('PSNR', fontweight='bold')
    axes[0].grid(True, axis='y', alpha=0.3)

    axes[1].bar(x, [r['ssim'] for r in results], color=colors, width=0.6, edgecolor='white', linewidth=1.2)
    for i, r in enumerate(results):
        axes[1].text(i, r['ssim'] + 0.0003, '%.4f' % r['ssim'], ha='center', fontsize=8, fontweight='bold')
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=7)
    axes[1].set_ylabel('SSIM'); axes[1].set_title('SSIM', fontweight='bold')
    axes[1].grid(True, axis='y', alpha=0.3)

    axes[2].bar(x, [r['latency_ms'] for r in results], color=colors, width=0.6, edgecolor='white', linewidth=1.2)
    for i, r in enumerate(results):
        axes[2].text(i, r['latency_ms'] + 0.5, '%.0f ms' % r['latency_ms'], ha='center', fontsize=8, fontweight='bold')
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels, fontsize=7)
    axes[2].set_ylabel('Latency (ms)'); axes[2].set_title('Execution Time', fontweight='bold')
    axes[2].grid(True, axis='y', alpha=0.3)

    fig.suptitle('Ablation Study: Contribution of Pre-screener and Power Iterations', fontweight='bold')
    plt.tight_layout()
    savefig(fig, 'fig11_ablation_study.png')

    print()
    return results


# ==============================================================================
#  STEP 3: Full Kodak 24-image benchmark
# ==============================================================================
def run_kodak_benchmark():
    print('=== Full Kodak 24-Image Benchmark ===')
    images = sorted([f for f in os.listdir(KODAK_DIR) if f.endswith('.png')])
    if len(images) < 24:
        print('  WARNING: Only %d/24 Kodak images found.' % len(images))

    all_results = []

    for fname in images:
        fpath = os.path.join(KODAK_DIR, fname)
        orig_pil = Image.open(fpath).convert('RGB')

        # Resize same as production
        max_dim = 800
        if max(orig_pil.size) > max_dim:
            ratio = max_dim / max(orig_pil.size)
            new_size = (int(orig_pil.size[0] * ratio), int(orig_pil.size[1] * ratio))
            orig_pil = orig_pil.resize(new_size, Image.LANCZOS)

        orig_arr = np.array(orig_pil, dtype=np.float64)

        t0 = time.time()
        comp_arr, k_vals, scores, factors = compress_image(orig_arr, rank=None, energy_percent=95.0)
        latency = (time.time() - t0) * 1000

        used_k = max(k_vals)
        mse = compute_mse(orig_arr, comp_arr)
        psnr = compute_psnr(mse)
        ssim = compute_ssim(orig_arr, comp_arr)
        U_list, S_list, Vt_list = zip(*factors)
        cr = true_compression_ratio(orig_arr.shape, U_list, S_list, Vt_list)

        skipped = sum(1 for k in k_vals if k == 0)

        # Compute JPEG Q=50 baseline for paired t-test
        buf = io.BytesIO()
        orig_pil.save(buf, format='JPEG', quality=50)
        buf.seek(0)
        jpeg_pil = Image.open(buf).convert('RGB')
        jpeg_arr = np.array(jpeg_pil, dtype=np.float64)
        jpeg_mse = compute_mse(orig_arr, jpeg_arr)
        jpeg_psnr = compute_psnr(jpeg_mse)

        row = {
            'image': fname,
            'psnr': round(psnr, 2),
            'jpeg_q50_psnr': round(jpeg_psnr, 2),
            'ssim': round(ssim, 4),
            'mse': round(mse, 2),
            'cr': round(cr, 1),
            'latency_ms': round(latency, 1),
            'k_per_channel': k_vals,
            'scores': [round(s, 1) for s in scores],
            'channels_skipped': skipped,
        }
        all_results.append(row)
        print('  %-14s  PSNR=%.2f  (JPEG=%.2f)  SSIM=%.4f  CR=%5.1fx  Lat=%6.1fms  k=%s  skip=%d'
              % (fname, psnr, jpeg_psnr, ssim, cr, latency, k_vals, skipped))

    # Compute summary statistics
    psnrs  = [r['psnr'] for r in all_results]
    jpeg_psnrs = [r['jpeg_q50_psnr'] for r in all_results]
    ssims  = [r['ssim'] for r in all_results]
    crs    = [r['cr'] for r in all_results]
    lats   = [r['latency_ms'] for r in all_results]
    skips  = [r['channels_skipped'] for r in all_results]

    summary = {
        'n_images': len(all_results),
        'psnr_mean': round(np.mean(psnrs), 2),
        'psnr_std':  round(np.std(psnrs), 2),
        'ssim_mean': round(np.mean(ssims), 4),
        'ssim_std':  round(np.std(ssims), 4),
        'cr_mean':   round(np.mean(crs), 1),
        'cr_std':    round(np.std(crs), 1),
        'lat_mean':  round(np.mean(lats), 1),
        'lat_std':   round(np.std(lats), 1),
        'skip_mean': round(np.mean(skips), 2),
    }

    # Statistical significance testing
    diffs = np.array(psnrs) - np.array(jpeg_psnrs)
    t_stat, p_val = stats.ttest_rel(psnrs, jpeg_psnrs)
    mean_diff = np.mean(diffs)
    se = stats.sem(diffs)
    ci_margin = se * stats.t.ppf((1 + 0.95) / 2., len(diffs) - 1)
    ci_lower = mean_diff - ci_margin
    ci_upper = mean_diff + ci_margin

    summary['t_stat'] = round(float(t_stat), 4)
    summary['p_val'] = round(float(p_val), 6)
    summary['mean_diff_psnr'] = round(float(mean_diff), 4)
    summary['ci_95_lower'] = round(float(ci_lower), 4)
    summary['ci_95_upper'] = round(float(ci_upper), 4)

    print()
    print('  --- SUMMARY (%d images) ---' % summary['n_images'])
    print('  PSNR:    %.2f +/- %.2f dB' % (summary['psnr_mean'], summary['psnr_std']))
    print('  SSIM:    %.4f +/- %.4f'    % (summary['ssim_mean'], summary['ssim_std']))
    print('  CR:      %.1fx +/- %.1f'   % (summary['cr_mean'], summary['cr_std']))
    print('  Latency: %.1f +/- %.1f ms' % (summary['lat_mean'], summary['lat_std']))
    print('  Avg channels skipped: %.2f/3' % summary['skip_mean'])
    print()
    print('  --- STATISTICAL SIGNIFICANCE vs JPEG Q=50 ---')
    print('  Mean PSNR Difference: +%.4f dB (Adaptive rSVD vs JPEG Q=50)' % summary['mean_diff_psnr'])
    print('  Paired t-test:        t = %.4f, p = %.6f' % (summary['t_stat'], summary['p_val']))
    print('  95%% Conf. Interval:   [%.4f, %.4f] dB' % (summary['ci_95_lower'], summary['ci_95_upper']))

    # Generate summary bar chart
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))

    # PSNR distribution
    axes[0].bar(range(len(psnrs)), sorted(psnrs, reverse=True), color='#4F46E5', width=0.7, alpha=0.85)
    axes[0].axhline(summary['psnr_mean'], ls='--', color='red', lw=1.2,
                    label='Mean=%.1f dB' % summary['psnr_mean'])
    axes[0].set_xlabel('Image (sorted)'); axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_title('PSNR Distribution', fontweight='bold'); axes[0].legend(fontsize=7)
    axes[0].grid(True, axis='y', alpha=0.3)

    # SSIM distribution
    axes[1].bar(range(len(ssims)), sorted(ssims, reverse=True), color='#059669', width=0.7, alpha=0.85)
    axes[1].axhline(summary['ssim_mean'], ls='--', color='red', lw=1.2,
                    label='Mean=%.4f' % summary['ssim_mean'])
    axes[1].set_xlabel('Image (sorted)'); axes[1].set_ylabel('SSIM')
    axes[1].set_title('SSIM Distribution', fontweight='bold'); axes[1].legend(fontsize=7)
    axes[1].grid(True, axis='y', alpha=0.3)

    # CR distribution
    axes[2].bar(range(len(crs)), sorted(crs, reverse=True), color='#D97706', width=0.7, alpha=0.85)
    axes[2].axhline(summary['cr_mean'], ls='--', color='red', lw=1.2,
                    label='Mean=%.1fx' % summary['cr_mean'])
    axes[2].set_xlabel('Image (sorted)'); axes[2].set_ylabel('Compression Ratio')
    axes[2].set_title('CR Distribution', fontweight='bold'); axes[2].legend(fontsize=7)
    axes[2].grid(True, axis='y', alpha=0.3)

    # Latency distribution
    axes[3].bar(range(len(lats)), sorted(lats), color='#DC2626', width=0.7, alpha=0.85)
    axes[3].axhline(summary['lat_mean'], ls='--', color='blue', lw=1.2,
                    label='Mean=%.0f ms' % summary['lat_mean'])
    axes[3].set_xlabel('Image (sorted)'); axes[3].set_ylabel('Latency (ms)')
    axes[3].set_title('Latency Distribution', fontweight='bold'); axes[3].legend(fontsize=7)
    axes[3].grid(True, axis='y', alpha=0.3)

    fig.suptitle('Adaptive rSVD Performance Across Full Kodak 24-Image Benchmark (energy=95%)',
                 fontweight='bold')
    plt.tight_layout()
    savefig(fig, 'fig12_kodak_benchmark.png')

    print()
    return all_results, summary


def run_rd_curve():
    print('=== Rate-Distortion Curve ===')
    test_img_path = os.path.join(KODAK_DIR, 'kodim23.png')
    img = np.array(Image.open(test_img_path).convert('RGB'), dtype=np.float64)
    h, w = img.shape[:2]; total_pixels = h * w

    rsvd_pts = []
    for rho in [60,70,75,80,85,90,95,99]:
        comp, k_vals, scores, factors = compress_image(img, energy_percent=rho)
        total_bits = 0
        for (U, S, Vt) in factors:
            total_bits += len(serialize_factors(U,S,Vt)) * 8
        bpp = total_bits / total_pixels if total_pixels > 0 else 0
        psnr = compute_psnr(compute_mse(img, comp))
        ssim = compute_ssim(img, comp)
        rsvd_pts.append({'rho':rho,'bpp':bpp,'psnr':psnr,'ssim':ssim})
        print(f"  rSVD rho={rho} -> bpp={bpp:.3f}, PSNR={psnr:.2f}, SSIM={ssim:.4f}")

    jpeg_pts = []
    orig_pil = Image.fromarray(img.astype(np.uint8))
    for q in [10,20,30,40,50,60,75,95]:
        buf = io.BytesIO()
        orig_pil.save(buf, 'JPEG', quality=q)
        bits = buf.tell() * 8
        bpp = bits / total_pixels
        buf.seek(0)
        ja = np.array(Image.open(buf).convert('RGB'), dtype=np.float64)
        psnr = compute_psnr(compute_mse(img,ja))
        ssim = compute_ssim(img,ja)
        jpeg_pts.append({'q':q,'bpp':bpp, 'psnr':psnr, 'ssim':ssim})
        print(f"  JPEG q={q} -> bpp={bpp:.3f}, PSNR={psnr:.2f}, SSIM={ssim:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([p['bpp'] for p in rsvd_pts],
            [p['psnr'] for p in rsvd_pts],
            'o-', color='#4F46E5', lw=2, label='rSVD Adaptive (proposed)')
    ax.plot([p['bpp'] for p in jpeg_pts],
            [p['psnr'] for p in jpeg_pts],
            's--', color='#D97706', lw=2, label='JPEG')
    ax.set_xlabel('Bits per pixel (bpp)')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Rate–distortion curve on Kodak23', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    savefig(fig, 'fig_rd_curve.png')
    print()
    return rsvd_pts, jpeg_pts

def run_tau_sensitivity():
    print('=== Tau Sensitivity Analysis ===')
    test_img_path = os.path.join(KODAK_DIR, 'kodim23.png')
    img = np.array(Image.open(test_img_path).convert('RGB'), dtype=np.float64)
    rows = []
    for tau in [40, 60, 80, 100, 120, 150]:
        comp = np.zeros_like(img)
        k_vals = []
        U_list = []; S_list = []; Vt_list = []
        for c in range(3):
            ch = img[:,:,c]
            score = complexity_score(ch)
            if score < tau:
                comp[:,:,c] = ch; k_vals.append(0)
                U_list.append(None); S_list.append(None); Vt_list.append(None)
                continue
            k = recommend_rank(score, 95.0, min(img.shape[:2]))
            U, S, Vt = rsvd(ch, k)
            comp[:,:,c] = reconstruct(U, S, Vt)
            np.clip(comp[:,:,c], 0, 255, out=comp[:,:,c])
            k_vals.append(k)
            U_list.append(U); S_list.append(S); Vt_list.append(Vt)
        psnr = compute_psnr(compute_mse(img, comp))
        cr = true_compression_ratio(img.shape, U_list, S_list, Vt_list)
        rows.append({'tau':tau,'psnr':psnr,'cr':cr})
        print(f"  tau={tau} -> PSNR={psnr:.2f}, CR={cr:.2f}")

    fig, ax1 = plt.subplots(figsize=(7, 5))
    taus = [r['tau'] for r in rows]
    ax1.plot(taus, [r['psnr'] for r in rows], 'o-', color='#4F46E5', label='PSNR')
    ax1.set_xlabel(r'Threshold $\tau$')
    ax1.set_ylabel('PSNR (dB)', color='#4F46E5')
    
    ax2 = ax1.twinx()
    ax2.plot(taus, [r['cr'] for r in rows], 's--', color='#10B981', label='CR')
    ax2.set_ylabel('Compression Ratio', color='#10B981')
    
    plt.title(r'Sensitivity Analysis of $\tau$', fontweight='bold')
    savefig(fig, 'fig_tau_sensitivity.png')
    print()
    return rows

def run_pca_rsvd_timing_benchmark():
    print('=== PCA vs rSVD Timing Inconsistency Check ===')
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("  sklearn not installed, skipping PCA benchmark.")
        return None
        
    # Benchmark at FULL resolution (e.g. 512x768 Kodak image, not thumbnail)
    test_img_path = os.path.join(KODAK_DIR, 'kodim23.png')
    img = np.array(Image.open(test_img_path).convert('L'), dtype=np.float64) # single channel
    k = 50
    print(f"  Benchmarking on full resolution {img.shape} at k={k}...")
    
    # PCA
    t0 = time.time()
    pca = PCA(n_components=k)
    pca.fit_transform(img)
    pca_time = (time.time() - t0) * 1000
    print(f"  PCA: {pca_time:.1f} ms")
    
    # rSVD
    t0 = time.time()
    rsvd(img, k)
    rsvd_time = (time.time() - t0) * 1000
    print(f"  rSVD: {rsvd_time:.1f} ms")
    
    print("  NOTE: rSVD advantage over PCA grows with rank k relative to image dimensions;")
    print("  at k=50 on 512x768, rSVD might be slower because the random projection overhead")
    print("  dominates at moderate k. At k>=150 rSVD is faster due to O(mnk) vs O(mn*min(m,n)) scaling.")
    print()
    return {'pca_time': pca_time, 'rsvd_time': rsvd_time}

# ==============================================================================
#  MAIN
# ==============================================================================
if __name__ == '__main__':
    all_data = {}

    n = download_kodak()

    print('=' * 70)
    jpeg_results = run_jpeg_baseline()
    all_data['jpeg_baseline'] = jpeg_results

    print('=' * 70)
    ablation_results = run_ablation()
    all_data['ablation'] = ablation_results

    print('=' * 70)
    kodak_results, kodak_summary = run_kodak_benchmark()
    all_data['kodak_results'] = kodak_results
    all_data['kodak_summary'] = kodak_summary

    print('=' * 70)
    all_data['rd_curve'] = run_rd_curve()
    
    print('=' * 70)
    all_data['tau_sensitivity'] = run_tau_sensitivity()
    
    print('=' * 70)
    all_data['pca_vs_rsvd'] = run_pca_rsvd_timing_benchmark()

    # Save all results to JSON
    with open(RESULTS, 'w') as f:
        json.dump(all_data, f, indent=2, default=str)
    print('All results saved to: %s' % RESULTS)

    # Final listing
    print()
    print('=== Generated Screenshots ===')
    for fname in sorted(os.listdir(SCREENSHOT)):
        if fname.startswith('fig1'):
            sz = os.path.getsize(os.path.join(SCREENSHOT, fname))
            print('  %-45s  %d KB' % (fname, sz // 1024))

    print()
    print('=== ALL BENCHMARKS COMPLETE ===')
