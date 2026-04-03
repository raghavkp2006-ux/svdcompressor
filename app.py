"""
SVD Image Compression — Flask Backend
Handles image upload, SVD compression (basic & adaptive), and metric computation.
"""

import os
import io
import base64
import json
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Removed UPLOAD_FOLDER creation because Vercel has a read-only filesystem
# The image is processed directly in memory via file.stream anyway.
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max


# ─── Utility Functions ───────────────────────────────────────────────

def image_to_array(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array (float64, 0-255)."""
    return np.array(image, dtype=np.float64)


def array_to_image(arr: np.ndarray) -> Image.Image:
    """Convert numpy array back to PIL Image."""
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def compute_mse(original: np.ndarray, compressed: np.ndarray) -> float:
    """Mean Squared Error between two images."""
    return float(np.mean((original - compressed) ** 2))


def compute_psnr(mse_val: float) -> float:
    """Peak Signal-to-Noise Ratio."""
    if mse_val == 0:
        return float('inf')
    return float(10 * np.log10((255 ** 2) / mse_val))


def compute_ssim_channel(orig: np.ndarray, comp: np.ndarray) -> float:
    """Simplified SSIM for a single channel."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu_x = np.mean(orig)
    mu_y = np.mean(comp)
    sigma_x2 = np.var(orig)
    sigma_y2 = np.var(comp)
    sigma_xy = np.mean((orig - mu_x) * (comp - mu_y))

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)

    return float(numerator / denominator)


def compute_ssim(original: np.ndarray, compressed: np.ndarray) -> float:
    """Compute SSIM across all channels."""
    if original.ndim == 2:
        return compute_ssim_channel(original, compressed)
    channels = original.shape[2]
    ssim_vals = []
    for c in range(channels):
        ssim_vals.append(compute_ssim_channel(original[:, :, c], compressed[:, :, c]))
    return float(np.mean(ssim_vals))


def compute_compression_ratio(original_shape, k: int) -> float:
    """Compute compression ratio."""
    if len(original_shape) == 2:
        m, n = original_shape
        channels = 1
    else:
        m, n, channels = original_shape
    original_size = m * n * channels
    compressed_size = channels * k * (m + n + 1)
    return float(original_size / compressed_size) if compressed_size > 0 else 0


def image_to_base64(image: Image.Image, fmt='PNG') -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


# ─── SVD Compression ─────────────────────────────────────────────────

def svd_compress_channel(channel: np.ndarray, k: int) -> np.ndarray:
    """Compress a single channel using SVD with k singular values."""
    U, S, Vt = np.linalg.svd(channel, full_matrices=False)
    k = min(k, len(S))
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]


def svd_compress_image(img_array: np.ndarray, k: int) -> np.ndarray:
    """Compress an image using basic SVD with k singular values."""
    if img_array.ndim == 2:
        return svd_compress_channel(img_array, k)

    channels = img_array.shape[2]
    result = np.zeros_like(img_array)
    for c in range(channels):
        result[:, :, c] = svd_compress_channel(img_array[:, :, c], k)
    return result


def adaptive_svd_compress(img_array: np.ndarray, energy_percent: float) -> tuple:
    """
    Adaptive SVD compression that selects k based on energy threshold.
    Returns (compressed_array, k_values_per_channel).
    """
    def compress_channel_adaptive(channel, energy_pct):
        U, S, Vt = np.linalg.svd(channel, full_matrices=False)
        total_energy = np.sum(S ** 2)
        cumulative_energy = np.cumsum(S ** 2)
        threshold = energy_pct / 100.0 * total_energy
        k = int(np.searchsorted(cumulative_energy, threshold) + 1)
        k = min(k, len(S))
        return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :], k

    if img_array.ndim == 2:
        compressed, k = compress_channel_adaptive(img_array, energy_percent)
        return compressed, [k]

    channels = img_array.shape[2]
    result = np.zeros_like(img_array)
    k_values = []
    for c in range(channels):
        compressed_ch, k = compress_channel_adaptive(img_array[:, :, c], energy_percent)
        result[:, :, c] = compressed_ch
        k_values.append(k)
    return result, k_values


def get_singular_values(img_array: np.ndarray) -> dict:
    """Get singular values for visualization."""
    if img_array.ndim == 2:
        _, S, _ = np.linalg.svd(img_array, full_matrices=False)
        return {'gray': S.tolist()[:100]}  # Cap at 100 for perf

    result = {}
    channel_names = ['red', 'green', 'blue']
    for c in range(min(img_array.shape[2], 3)):
        _, S, _ = np.linalg.svd(img_array[:, :, c], full_matrices=False)
        result[channel_names[c]] = S.tolist()[:100]
    return result


def get_energy_curve(img_array: np.ndarray) -> dict:
    """Get cumulative energy curve data for visualization."""
    if img_array.ndim == 2:
        _, S, _ = np.linalg.svd(img_array, full_matrices=False)
        total = np.sum(S ** 2)
        cumulative = np.cumsum(S ** 2) / total * 100
        return {'gray': cumulative.tolist()[:100]}

    result = {}
    channel_names = ['red', 'green', 'blue']
    for c in range(min(img_array.shape[2], 3)):
        _, S, _ = np.linalg.svd(img_array[:, :, c], full_matrices=False)
        total = np.sum(S ** 2)
        cumulative = np.cumsum(S ** 2) / total * 100
        result[channel_names[c]] = cumulative.tolist()[:100]
    return result


# ─── Routes ───────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/compress', methods=['POST'])
def compress():
    """Main compression endpoint."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    mode = request.form.get('mode', 'basic')  # 'basic' or 'adaptive'
    k = int(request.form.get('k', 50))
    energy = float(request.form.get('energy', 95.0))

    try:
        # Open and convert image
        img = Image.open(file.stream).convert('RGB')

        # Resize if too large (for performance)
        max_dim = 800
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        img_array = image_to_array(img)
        max_k = min(img_array.shape[0], img_array.shape[1])

        # Get singular values & energy curve for charts
        singular_values = get_singular_values(img_array)
        energy_curve = get_energy_curve(img_array)

        # Perform compression
        if mode == 'adaptive':
            compressed_array, k_values = adaptive_svd_compress(img_array, energy)
            used_k = max(k_values)
        else:
            k = min(k, max_k)
            compressed_array = svd_compress_image(img_array, k)
            k_values = [k] * img_array.shape[2] if img_array.ndim == 3 else [k]
            used_k = k

        # Compute metrics
        mse_val = compute_mse(img_array, compressed_array)
        psnr_val = compute_psnr(mse_val)
        ssim_val = compute_ssim(img_array, compressed_array)
        cr_val = compute_compression_ratio(img_array.shape, used_k)

        # Convert images to base64
        original_b64 = image_to_base64(img)
        compressed_img = array_to_image(compressed_array)
        compressed_b64 = image_to_base64(compressed_img)

        # Build multi-k comparison data (for the chart)
        comparison_ks = [5, 10, 20, 50, 100, min(150, max_k), min(200, max_k)]
        comparison_ks = sorted(set([kk for kk in comparison_ks if kk <= max_k]))
        comparison_data = []
        for ck in comparison_ks:
            c_arr = svd_compress_image(img_array, ck)
            c_mse = compute_mse(img_array, c_arr)
            c_psnr = compute_psnr(c_mse)
            c_ssim = compute_ssim(img_array, c_arr)
            c_cr = compute_compression_ratio(img_array.shape, ck)
            comparison_data.append({
                'k': ck,
                'psnr': round(c_psnr, 2),
                'ssim': round(c_ssim, 4),
                'mse': round(c_mse, 2),
                'cr': round(c_cr, 2)
            })

        return jsonify({
            'success': True,
            'original': f'data:image/png;base64,{original_b64}',
            'compressed': f'data:image/png;base64,{compressed_b64}',
            'metrics': {
                'mse': round(mse_val, 4),
                'psnr': round(psnr_val, 4),
                'ssim': round(ssim_val, 4),
                'cr': round(cr_val, 4),
                'k_used': used_k,
                'k_per_channel': k_values,
                'max_k': max_k,
                'image_size': list(img.size),
                'mode': mode
            },
            'singular_values': singular_values,
            'energy_curve': energy_curve,
            'comparison': comparison_data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download', methods=['POST'])
def download():
    """Download compressed image."""
    data = request.json
    if not data or 'image_data' not in data:
        return jsonify({'error': 'No image data'}), 400

    try:
        # Remove data URI prefix
        img_data = data['image_data'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        return send_file(
            io.BytesIO(img_bytes),
            mimetype='image/png',
            as_attachment=True,
            download_name='svd_compressed.png'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
