import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})
import time
import os
import sys

# Add project root to sys.path
sys.path.append(r'c:\antigravity_projects\svds\svdcompressor')

from lib.compress import compress_image

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

def main():
    img_path = r'c:\antigravity_projects\svd compressor\kodak_images\kodim01.png'
    original_img = Image.open(img_path).convert('RGB')
    original_array = np.array(original_img, dtype=np.float64)

    start_time = time.time()
    compressed_array, k_values, scores, _ = compress_image(original_array, energy_percent=95.0)
    latency_ms = (time.time() - start_time) * 1000

    compressed_uint8 = np.clip(compressed_array, 0, 255).astype(np.uint8)
    original_uint8 = original_array.astype(np.uint8)

    psnr = calculate_psnr(original_array, compressed_array)

    h, w, c = original_array.shape
    original_size = h * w * c
    compressed_size = sum([k * (h + w + 1) for k in k_values])
    cr = original_size / compressed_size if compressed_size > 0 else 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.imshow(original_uint8)
    ax1.set_title("Original Image (Kodak 24: kodim01)")
    ax1.axis('off')

    ax2.imshow(compressed_uint8)
    ax2.set_title(f"SVD Compressed (Adaptive)\nPSNR: {psnr:.2f} dB, CR: {cr:.2f}x\nLatency: {latency_ms:.1f} ms, Ranks: {k_values}")
    ax2.axis('off')

    plt.tight_layout()
    out_dir = r'C:\Users\ragha\.gemini\antigravity-ide\brain\09577c5f-300c-4f5e-a027-64295fcca3d4\artifacts'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'kodim01_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved report to {out_path}")

if __name__ == '__main__':
    main()
