import os, sys, io, time, json
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.compress import compress_image, true_compression_ratio
from run_benchmarks import KODAK_DIR, compute_mse, compute_psnr, compute_ssim, download_kodak

def jpeg2000_compress_channel(channel: np.ndarray, quality_layers=None) -> np.ndarray:
    """Compress via JPEG 2000 wavelet (Pillow backend). quality_layers=[40] ≈ k=50."""
    pil = Image.fromarray(channel.astype(np.uint8), mode='L')
    buf = io.BytesIO()
    pil.save(buf, format='JPEG2000',
             quality_mode='rates',
             quality_layers=quality_layers or [40])
    buf.seek(0)
    result = np.array(Image.open(buf).convert('L'), dtype=np.float64)
    return np.clip(result, 0, 255)

def run_webp_jpeg2000_benchmark():
    print("=== WebP & JPEG 2000 vs proposed rSVD Benchmark ===")
    images = sorted([f for f in os.listdir(KODAK_DIR) if f.endswith('.png')])
    if not images:
        download_kodak()
        images = sorted([f for f in os.listdir(KODAK_DIR) if f.endswith('.png')])

    results = []

    for fname in images:
        fpath = os.path.join(KODAK_DIR, fname)
        orig_pil = Image.open(fpath).convert('RGB')
        
        max_dim = 800
        if max(orig_pil.size) > max_dim:
            ratio = max_dim / max(orig_pil.size)
            new_size = (int(orig_pil.size[0] * ratio), int(orig_pil.size[1] * ratio))
            orig_pil = orig_pil.resize(new_size, Image.LANCZOS)

        orig_arr = np.array(orig_pil, dtype=np.float64)
        total_pixels = orig_arr.shape[0] * orig_arr.shape[1]
        orig_bytes = total_pixels * 3

        # rSVD Adaptive at 95%
        comp_arr, k_vals, scores, factors = compress_image(orig_arr, rank=None, energy_percent=95.0)
        psnr_rsvd = compute_psnr(compute_mse(orig_arr, comp_arr))
        U_list, S_list, Vt_list = zip(*factors)
        cr_rsvd = true_compression_ratio(orig_arr.shape, U_list, S_list, Vt_list)

        # WebP at Q=50, 75
        webp_res = {}
        for q in [50, 75]:
            buf = io.BytesIO()
            orig_pil.save(buf, format='WEBP', quality=q)
            buf.seek(0)
            webp_arr = np.array(Image.open(buf).convert('RGB'), dtype=np.float64)
            webp_res[q] = {
                'psnr': compute_psnr(compute_mse(orig_arr, webp_arr)),
                'cr': orig_bytes / buf.getbuffer().nbytes
            }

        # JPEG 2000 at q=[40], [60]
        jp2_res = {}
        for q in [40, 60]:
            comp_jp2 = np.zeros_like(orig_arr)
            jp2_bytes = 0
            for c in range(3):
                pil_c = Image.fromarray(orig_arr[:,:,c].astype(np.uint8), mode='L')
                buf = io.BytesIO()
                pil_c.save(buf, format='JPEG2000', quality_mode='rates', quality_layers=[q])
                jp2_bytes += buf.getbuffer().nbytes
                buf.seek(0)
                comp_jp2[:,:,c] = np.array(Image.open(buf).convert('L'), dtype=np.float64)
            jp2_res[q] = {
                'psnr': compute_psnr(compute_mse(orig_arr, comp_jp2)),
                'cr': orig_bytes / jp2_bytes
            }

        print(f"{fname:12} | rSVD: PSNR={psnr_rsvd:.2f} CR={cr_rsvd:.1f}x | WebP(Q75): PSNR={webp_res[75]['psnr']:.2f} CR={webp_res[75]['cr']:.1f}x | JP2(Q60): PSNR={jp2_res[60]['psnr']:.2f} CR={jp2_res[60]['cr']:.1f}x")
        
        results.append({
            'image': fname,
            'rsvd': {'psnr': psnr_rsvd, 'cr': cr_rsvd},
            'webp_q50': webp_res[50],
            'webp_q75': webp_res[75],
            'jp2_q40': jp2_res[40],
            'jp2_q60': jp2_res[60]
        })

    print("\n--- Summary ---")
    keys = ['rsvd', 'webp_q50', 'webp_q75', 'jp2_q40', 'jp2_q60']
    for k in keys:
        avg_psnr = np.mean([r[k]['psnr'] for r in results])
        avg_cr = np.mean([r[k]['cr'] for r in results])
        print(f"{k:10} | Avg PSNR: {avg_psnr:.2f} dB | Avg CR: {avg_cr:.1f}x")

if __name__ == '__main__':
    run_webp_jpeg2000_benchmark()
