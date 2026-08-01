"""
generate_fig13_codec_qualitative.py
====================================
Creates a qualitative comparison figure (fig13) showing WebP, JPEG 2000,
JPEG XL, and adaptive rSVD side by side on kodim23.png.

Layout: 2 rows x 5 columns
  Row 1 -- Full images:   Original | WebP Q75 | JPEG2000 rate=20 | JPEG XL Q75 | rSVD rho=95
  Row 2 -- Zoomed crops:  same order, 150x150 region highlighting fine texture

Each panel is labeled with PSNR and SSIM.
Uses the SAME larger font rcParams as other regenerated project figures.
"""

import os, sys, io
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ── Consistent rcParams (matching generate_report_photos.py & run_benchmarks.py) ──
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
})

# ── Project root setup ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from lib.compress import compress_image, true_compression_ratio
from skimage.metrics import structural_similarity as ssim_skimage

# ── Check JPEG XL support ───────────────────────────────────────────────────
HAS_JXL = False
try:
    import pillow_jxl  # noqa: F401
    HAS_JXL = True
except ImportError:
    print('WARNING: pillow-jxl-plugin not installed. JPEG XL panel will be skipped.')

# ── Paths ────────────────────────────────────────────────────────────────────
KODAK_DIR  = os.path.join(PROJECT_ROOT, 'kodak_images')
SCREENSHOT = os.path.join(PROJECT_ROOT, 'screenshots')
os.makedirs(SCREENSHOT, exist_ok=True)

# ── Metrics ──────────────────────────────────────────────────────────────────
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


# ── Codec helpers ────────────────────────────────────────────────────────────
def encode_webp(pil_img, quality=75):
    """Encode with WebP, return decoded float64 array and file size."""
    buf = io.BytesIO()
    pil_img.save(buf, format='WEBP', quality=quality)
    file_bytes = buf.tell()
    buf.seek(0)
    decoded = np.array(Image.open(buf).convert('RGB'), dtype=np.float64)
    return decoded, file_bytes


def encode_jpeg2000(orig_arr, rate=20):
    """Encode per-channel JPEG2000 at given rate. rate=20 ~ Q75."""
    comp = np.zeros_like(orig_arr)
    total_bytes = 0
    for c in range(3):
        pil_c = Image.fromarray(orig_arr[:, :, c].astype(np.uint8), mode='L')
        buf = io.BytesIO()
        pil_c.save(buf, format='JPEG2000', quality_mode='rates', quality_layers=[rate])
        total_bytes += buf.tell()
        buf.seek(0)
        comp[:, :, c] = np.array(Image.open(buf).convert('L'), dtype=np.float64)
    return comp, total_bytes


def encode_jpegxl(pil_img, quality=75):
    """Encode with JPEG XL via pillow-jxl-plugin, return decoded float64 array and file size."""
    buf = io.BytesIO()
    pil_img.save(buf, format='JXL', quality=quality)
    file_bytes = buf.tell()
    buf.seek(0)
    decoded = np.array(Image.open(buf).convert('RGB'), dtype=np.float64)
    return decoded, file_bytes


def main():
    print('=== Generating Fig 13: Codec Qualitative Comparison ===')

    # ── Load kodim23 ─────────────────────────────────────────────────────────
    img_path = os.path.join(KODAK_DIR, 'kodim23.png')
    if not os.path.exists(img_path):
        print(f'ERROR: {img_path} not found. Run download_kodak() first.')
        sys.exit(1)

    orig_pil = Image.open(img_path).convert('RGB')
    # Resize to 800px max (consistent with all other benchmarks)
    max_dim = 800
    if max(orig_pil.size) > max_dim:
        ratio = max_dim / max(orig_pil.size)
        new_size = (int(orig_pil.size[0] * ratio), int(orig_pil.size[1] * ratio))
        orig_pil = orig_pil.resize(new_size, Image.LANCZOS)

    orig_arr = np.array(orig_pil, dtype=np.float64)
    h, w = orig_arr.shape[:2]
    orig_bytes = h * w * 3
    print(f'  Image size: {w}x{h}')

    # ── 1. WebP Q=75 ─────────────────────────────────────────────────────────
    webp_arr, webp_bytes = encode_webp(orig_pil, quality=75)
    webp_psnr = compute_psnr(compute_mse(orig_arr, webp_arr))
    webp_ssim = compute_ssim(orig_arr, webp_arr)
    webp_cr = orig_bytes / webp_bytes
    print(f'  WebP Q75:   PSNR={webp_psnr:.2f} dB  SSIM={webp_ssim:.4f}  CR={webp_cr:.1f}x')

    # ── 2. JPEG2000 rate=20 (~ Q75 tier) ─────────────────────────────────────
    jp2_arr, jp2_bytes = encode_jpeg2000(orig_arr, rate=20)
    jp2_psnr = compute_psnr(compute_mse(orig_arr, jp2_arr))
    jp2_ssim = compute_ssim(orig_arr, jp2_arr)
    jp2_cr = orig_bytes / jp2_bytes
    print(f'  JP2 r=20:   PSNR={jp2_psnr:.2f} dB  SSIM={jp2_ssim:.4f}  CR={jp2_cr:.1f}x')

    # ── 3. JPEG XL Q=75 ─────────────────────────────────────────────────────
    jxl_arr = jxl_psnr = jxl_ssim = jxl_cr = None
    if HAS_JXL:
        jxl_arr, jxl_bytes = encode_jpegxl(orig_pil, quality=75)
        jxl_psnr = compute_psnr(compute_mse(orig_arr, jxl_arr))
        jxl_ssim = compute_ssim(orig_arr, jxl_arr)
        jxl_cr = orig_bytes / jxl_bytes
        print(f'  JXL Q75:    PSNR={jxl_psnr:.2f} dB  SSIM={jxl_ssim:.4f}  CR={jxl_cr:.1f}x')
    else:
        print('  JXL Q75:    SKIPPED (not installed)')

    # ── 4. Adaptive rSVD rho=95 ──────────────────────────────────────────────
    rsvd_arr, k_vals, scores, factors = compress_image(orig_arr, rank=None, energy_percent=95.0)
    rsvd_psnr = compute_psnr(compute_mse(orig_arr, rsvd_arr))
    rsvd_ssim = compute_ssim(orig_arr, rsvd_arr)
    U_list, S_list, Vt_list = zip(*factors)
    rsvd_cr = true_compression_ratio(orig_arr.shape, U_list, S_list, Vt_list)
    print(f'  rSVD rho=95: PSNR={rsvd_psnr:.2f} dB  SSIM={rsvd_ssim:.4f}  CR={rsvd_cr:.1f}x  k={k_vals}')

    # ── Pick crop region ─────────────────────────────────────────────────────
    # kodim23 is the macaw image (768x512 after resize).
    # Pick a 150x150 crop capturing the eye/feather detail area --
    # rich texture with fine barbs, ideal for artifact visibility.
    crop_size = 150
    # Target the macaw's head/eye area (upper-center region)
    crop_y = max(0, int(h * 0.30) - crop_size // 2)
    crop_x = max(0, int(w * 0.52) - crop_size // 2)
    # Clamp
    crop_y = min(crop_y, h - crop_size)
    crop_x = min(crop_x, w - crop_size)
    cy, cx = crop_y, crop_x
    cs = crop_size

    print(f'  Crop region: ({cx}, {cy}) to ({cx+cs}, {cy+cs})')

    # ── Prepare display arrays ───────────────────────────────────────────────
    def to_uint8(arr):
        return np.clip(arr, 0, 255).astype(np.uint8)

    orig_disp = to_uint8(orig_arr)
    webp_disp = to_uint8(webp_arr)
    jp2_disp  = to_uint8(jp2_arr)
    rsvd_disp = to_uint8(rsvd_arr)

    # Build column lists: Original, WebP, JP2, [JXL if available], rSVD
    panel_images = [orig_disp, webp_disp, jp2_disp]
    labels = [
        'Original',
        f'WebP Q=75\nPSNR={webp_psnr:.2f} dB  SSIM={webp_ssim:.4f}\nCR={webp_cr:.1f}x',
        f'JPEG 2000 (rate=20)\nPSNR={jp2_psnr:.2f} dB  SSIM={jp2_ssim:.4f}\nCR={jp2_cr:.1f}x',
    ]
    crop_labels = ['Original (crop)', 'WebP (crop)', 'JPEG 2000 (crop)']
    title_colors = ['black', '#2563EB', '#059669']

    if HAS_JXL and jxl_arr is not None:
        jxl_disp = to_uint8(jxl_arr)
        panel_images.append(jxl_disp)
        labels.append(f'JPEG XL Q=75\nPSNR={jxl_psnr:.2f} dB  SSIM={jxl_ssim:.4f}\nCR={jxl_cr:.1f}x')
        crop_labels.append('JPEG XL (crop)')
        title_colors.append('#DC2626')

    panel_images.append(rsvd_disp)
    labels.append(f'Adaptive rSVD (rho=95)\nPSNR={rsvd_psnr:.2f} dB  SSIM={rsvd_ssim:.4f}\nCR={rsvd_cr:.1f}x')
    crop_labels.append('rSVD (crop)')
    title_colors.append('#7C3AED')

    ncols = len(panel_images)
    crops = [img[cy:cy+cs, cx:cx+cs] for img in panel_images]

    # ── Build figure ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 10.5),
                              gridspec_kw={'height_ratios': [2.2, 1.6],
                                           'hspace': 0.15, 'wspace': 0.06})

    # Row 1: Full images
    for col in range(ncols):
        ax = axes[0, col]
        ax.imshow(panel_images[col])
        ax.set_title(labels[col], fontweight='bold', fontsize=11,
                     color=title_colors[col], pad=8)
        ax.axis('off')
        # Draw crop rectangle on full images
        rect = Rectangle((cx, cy), cs, cs,
                          linewidth=2.5, edgecolor='#EF4444', facecolor='none',
                          linestyle='--')
        ax.add_patch(rect)

    # Row 2: Zoomed crops
    for col in range(ncols):
        ax = axes[1, col]
        ax.imshow(crops[col], interpolation='nearest')
        ax.set_title(crop_labels[col], fontweight='bold', fontsize=11,
                     color=title_colors[col])
        ax.axis('off')
        # Red border around crop for emphasis
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#EF4444')
            spine.set_linewidth(2)

    codec_list = 'WebP vs JPEG 2000 vs JPEG XL vs Adaptive rSVD' if HAS_JXL else 'WebP vs JPEG 2000 vs Adaptive rSVD'
    fig.suptitle('Qualitative Codec Comparison on Kodak23\n' + codec_list,
                 fontweight='bold', fontsize=16, y=1.02)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = os.path.join(SCREENSHOT, 'fig13_codec_qualitative_comparison.png')
    fig.savefig(out_path)
    plt.close(fig)
    print(f'\n  Saved -> {out_path}')
    print('=== Done ===')


if __name__ == '__main__':
    main()
