"""
SVD Image Compression — Flask Backend
Handles image upload, rSVD compression (basic & adaptive), and metric computation.

Architecture:
  app.py (routes + metrics) → lib/compress.py → lib/rsvd.py + lib/prescreening.py
"""

import os
import io
import base64
import json
import time
import joblib
import numpy as np
from scipy.ndimage import uniform_filter
from PIL import Image, ImageDraw
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.units import inch
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# ─── Import modular compression library ──────────────────────────────
from lib.compress import compress_image, compress_image_basic, compress_channel
from lib.prescreening import complexity_score, recommend_rank, DEFAULT_SKIP_THRESHOLD
from lib.rsvd import rsvd, reconstruct
from lib.algorithms import compress_image_algo
from features.extract_features import extract_features

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


from skimage.metrics import structural_similarity as ssim_skimage
import numpy as np

def compute_ssim(original: np.ndarray, compressed: np.ndarray) -> float:
    kwargs = {'data_range': 255.0}
    if original.ndim == 3:
        kwargs['channel_axis'] = 2
    return float(ssim_skimage(original, compressed, **kwargs))


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


# ─── Visualization Functions ─────────────────────────────────────────

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


def compute_error_map(original: np.ndarray, compressed: np.ndarray) -> Image.Image:
    """Generate error/difference visualization: bright = high error, dark = low error."""
    diff = np.abs(original.astype(float) - compressed.astype(float))
    
    # Handle RGB vs grayscale
    if diff.ndim == 3:
        diff = np.mean(diff, axis=2)  # Average across channels
    
    # Normalize to 0-255
    if diff.max() > 0:
        diff_normalized = (diff / diff.max() * 255).astype(np.uint8)
    else:
        diff_normalized = np.zeros_like(diff, dtype=np.uint8)
    
    # Convert to Image
    error_img = Image.fromarray(diff_normalized, mode='L')
    return error_img


def compute_block_heatmap(original: np.ndarray, compressed: np.ndarray, k_per_channel: list, mode: str) -> Image.Image:
    """
    Generate a heatmap showing complexity per block.
    Blue = simple (low k), Red = complex (high k).
    For adaptive mode, show the k values used per channel.
    """
    h, w = original.shape[:2]
    block_size = 32
    
    # Create heatmap array
    heatmap = np.zeros((h, w, 3), dtype=np.uint8)
    
    if mode == 'adaptive':
        # For adaptive mode, show which k was used per channel
        max_k = max([k for k in k_per_channel if k > 0]) if any(k > 0 for k in k_per_channel) else 1
        
        for by in range(0, h - block_size + 1, block_size):
            for bx in range(0, w - block_size + 1, block_size):
                # Calculate average k used for this region
                channel_ks = [k for k in k_per_channel]
                avg_k = np.mean([k for k in channel_ks if k > 0]) if any(k > 0 for k in channel_ks) else 0
                
                # Map k to color: blue (low) -> red (high)
                if max_k > 0:
                    k_normalized = avg_k / max_k  # 0 to 1
                else:
                    k_normalized = 0
                
                # Blue to Red gradient
                r = int(k_normalized * 255)
                b = int((1 - k_normalized) * 255)
                color = (r, 0, b)
                
                # Fill block
                x_end = min(bx + block_size, w)
                y_end = min(by + block_size, h)
                heatmap[by:y_end, bx:x_end] = color
    else:
        # For basic mode, show error magnitude per block
        error = np.abs(original.astype(float) - compressed.astype(float))
        if error.ndim == 3:
            error = np.mean(error, axis=2)
        
        max_error = error.max() if error.max() > 0 else 1
        
        for by in range(0, h - block_size + 1, block_size):
            for bx in range(0, w - block_size + 1, block_size):
                block_error = np.mean(error[by:by+block_size, bx:bx+block_size])
                error_normalized = block_error / max_error
                
                # Red gradient (more error = more red)
                r = int(error_normalized * 255)
                g = int((1 - error_normalized) * 128)
                b = int((1 - error_normalized) * 128)
                color = (r, g, b)
                
                # Fill block
                x_end = min(bx + block_size, w)
                y_end = min(by + block_size, h)
                heatmap[by:y_end, bx:x_end] = color
    
    return Image.fromarray(heatmap, mode='RGB')


def generate_pdf_report(metrics: dict, original_b64: str, compressed_b64: str, 
                       error_map_b64: str, heatmap_b64: str) -> bytes:
    """Generate a PDF report with compression results."""
    if not HAS_REPORTLAB:
        # Fallback if reportlab not installed
        return None
    
    # Create PDF in memory
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#7c3aed'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#6366f1'),
        spaceAfter=8,
        fontName='Helvetica-Bold'
    )
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6
    )
    
    # Title
    story.append(Paragraph("SVD Image Compression Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Header info
    info_data = [
        ['Mode:', metrics.get('mode', 'N/A').upper()],
        ['K Value Used:', str(metrics.get('k_used', 'N/A'))],
        ['Image Size:', f"{metrics['image_size'][0]} × {metrics['image_size'][1]} px"],
        ['Processing Time:', f"{metrics.get('compute_time', 0):.2f}s"],
        ['Algorithm:', 'Randomized SVD (Halko et al. 2011)'],
    ]
    info_table = Table(info_data, colWidths=[2*inch, 3*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f3e8ff')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Metrics
    story.append(Paragraph("Compression Metrics", heading_style))
    metrics_data = [
        ['Metric', 'Value', 'Description'],
        ['PSNR (dB)', f"{metrics.get('psnr', 0):.2f}", 'Peak Signal-to-Noise Ratio - Higher is better'],
        ['SSIM', f"{metrics.get('ssim', 0):.4f}", 'Structural Similarity - Closer to 1.0 is better'],
        ['MSE', f"{metrics.get('mse', 0):.2f}", 'Mean Squared Error - Lower is better'],
        ['Compression Ratio', f"{metrics.get('cr', 0):.2f}×", 'Original size / Compressed size'],
    ]
    metrics_table = Table(metrics_data, colWidths=[1.5*inch, 1*inch, 3*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6366f1')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Images
    story.append(Paragraph("Image Comparison", heading_style))
    try:
        # Decode base64 images
        orig_data = base64.b64decode(original_b64.split(',')[1])
        comp_data = base64.b64decode(compressed_b64.split(',')[1])
        error_data = base64.b64decode(error_map_b64.split(',')[1])
        hmap_data = base64.b64decode(heatmap_b64.split(',')[1])
        
        # Create image rows
        img_orig = RLImage(io.BytesIO(orig_data), width=2*inch, height=1.5*inch)
        img_comp = RLImage(io.BytesIO(comp_data), width=2*inch, height=1.5*inch)
        img_error = RLImage(io.BytesIO(error_data), width=2*inch, height=1.5*inch)
        img_hmap = RLImage(io.BytesIO(hmap_data), width=2*inch, height=1.5*inch)
        
        img_table = Table([
            [img_orig, img_comp],
            [img_error, img_hmap]
        ], colWidths=[2.2*inch, 2.2*inch])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(img_table)
    except Exception as e:
        story.append(Paragraph(f"<i>Could not embed images: {str(e)}</i>", normal_style))
    
    story.append(Spacer(1, 0.3*inch))
    
    # Footer
    story.append(Paragraph("Generated by SVD Compressor", normal_style))
    
    # Build PDF
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


# ─── Routes ───────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/compress', methods=['POST'])
def compress():
    """Main compression endpoint."""
    start_time = time.time()

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    mode = request.form.get('mode', 'basic')  # 'basic' or 'adaptive'
    k = int(request.form.get('k', 50))
    energy = float(request.form.get('energy', 95.0))
    algo = request.form.get('algo', 'svd')

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

        # Get singular values & energy curve for charts (only if SVD or PCA)
        if algo in ['svd', 'pca']:
            singular_values = get_singular_values(img_array)
            energy_curve = get_energy_curve(img_array)
        else:
            singular_values = {}
            energy_curve = {}

        # Perform compression using modular library
        if algo not in ['svd', 'pca']:
            k = min(k, max_k)
            compressed_array = compress_image_algo(img_array, k, algo)
            k_values = [k] * (img_array.shape[2] if img_array.ndim == 3 else 1)
            scores = []
            used_k = k
        elif mode == 'adaptive':
            if algo == 'svd':
                compressed_array, k_values, scores, _ = compress_image(
                    img_array, rank=None, energy_percent=energy
                )
                used_k = max(k_values)
            else: # pca
                channels = img_array.shape[2] if img_array.ndim == 3 else 1
                k_values = []
                scores = []
                for c in range(channels):
                    channel = img_array[:, :, c] if img_array.ndim == 3 else img_array
                    score = complexity_score(channel)
                    k_channel = recommend_rank(score, energy, max_k)
                    k_values.append(k_channel)
                    scores.append(score)
                used_k = max(k_values)
                compressed_array = compress_image_algo(img_array, used_k, 'pca')
        else:
            k = min(k, max_k)
            if algo == 'svd':
                compressed_array = compress_image_basic(img_array, k)
            else:
                compressed_array = compress_image_algo(img_array, k, 'pca')
            k_values = [k] * (img_array.shape[2] if img_array.ndim == 3 else 1)
            scores = []
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
        
        # Generate error map and heatmap
        error_map_img = compute_error_map(img_array, compressed_array)
        error_map_b64 = image_to_base64(error_map_img)
        
        heatmap_img = compute_block_heatmap(img_array, compressed_array, k_values, mode)
        heatmap_b64 = image_to_base64(heatmap_img)

        # Build multi-k comparison data (for the chart)
        comparison_data = []
        if algo in ['svd', 'dct']:
            comparison_ks = [5, 10, 20, 50, 100, min(150, max_k), min(200, max_k)]
            comparison_ks = sorted(set([kk for kk in comparison_ks if kk <= max_k]))
            for ck in comparison_ks:
                if algo == 'svd':
                    c_arr = compress_image_basic(img_array, ck)
                else:
                    c_arr = compress_image_algo(img_array, ck, algo)
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
        else:
            # For slow algorithms (NMF, PCA), only append the currently computed point
            comparison_data.append({
                'k': used_k,
                'psnr': round(psnr_val, 2),
                'ssim': round(ssim_val, 4),
                'mse': round(mse_val, 2),
                'cr': round(cr_val, 2)
            })

        # Compute total execution time
        compute_time = time.time() - start_time

        return jsonify({
            'success': True,
            'original': f'data:image/png;base64,{original_b64}',
            'compressed': f'data:image/png;base64,{compressed_b64}',
            'error_map': f'data:image/png;base64,{error_map_b64}',
            'heatmap': f'data:image/png;base64,{heatmap_b64}',
            'metrics': {
                'mse': round(mse_val, 4),
                'psnr': round(psnr_val, 4),
                'ssim': round(ssim_val, 4),
                'cr': round(cr_val, 4),
                'k_used': used_k,
                'k_per_channel': k_values,
                'max_k': max_k,
                'image_size': list(img.size),
                'mode': mode,
                'compute_time': compute_time,
                'algorithm': algo.upper() if algo != 'svd' else 'rSVD (Halko et al. 2011)',
                'power_iterations': 2 if algo == 'svd' else None,
                'complexity_scores': scores if scores else None,
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


@app.route('/api/benchmark', methods=['POST'])
def api_benchmark():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image provided'}), 400

    file = request.files['image']
    try:
        k = int(request.form.get('k', 50))
    except ValueError:
        k = 50

    try:
        img = Image.open(file.stream).convert('RGB')
        # Resize image for benchmark to prevent NMF from taking too long
        img.thumbnail((256, 256))
        img_array = image_to_array(img)
        
        times = {}
        algorithms = ['svd', 'pca', 'nmf', 'dct']
        
        for algo in algorithms:
            start_time = time.time()
            if algo == 'svd':
                compress_image_basic(img_array, k)
            else:
                compress_image_algo(img_array, k, algo)
            times[algo] = round(time.time() - start_time, 3)
            
        return jsonify({
            'success': True,
            'times': times,
            'image_size': img_array.shape[:2]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/report', methods=['POST'])
def report():
    """Generate and download PDF report."""
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    try:
        if not HAS_REPORTLAB:
            return jsonify({'error': 'PDF generation not available. Install reportlab.'}), 501
        
        metrics = data.get('metrics', {})
        original_b64 = data.get('original', '')
        compressed_b64 = data.get('compressed', '')
        error_map_b64 = data.get('error_map', '')
        heatmap_b64 = data.get('heatmap', '')
        
        pdf_bytes = generate_pdf_report(metrics, original_b64, compressed_b64, 
                                       error_map_b64, heatmap_b64)
        
        if pdf_bytes is None:
            return jsonify({'error': 'Failed to generate PDF'}), 500
        
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name='svd_compression_report.pdf'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/recommend_algorithm', methods=['POST'])
def recommend_algorithm():
    """Predict the best compression algorithm for the uploaded image."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    try:
        img = Image.open(file.stream).convert('RGB')
        
        # We need the image as grayscale to extract features
        gray_img = img.convert('L')
        gray_array = np.array(gray_img, dtype=np.float64)
        
        # Extract features
        feats = extract_features(gray_array)
        
        # Load model
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'algorithm_agent.pkl')
        if not os.path.exists(model_path):
            return jsonify({'error': 'Algorithm recommendation model not found.'}), 404
            
        model = joblib.load(model_path)
        
        # Predict best algorithm
        predicted_algo = model.predict([feats])[0]
        
        # Determine complexity for k/energy recommendation
        score = complexity_score(gray_array)
        
        # Decide mode based on complexity
        # If image is very simple (low score), adaptive is best.
        # If highly complex, basic is best to ensure sufficient k.
        recommended_mode = 'adaptive' if score < 1000 else 'basic'
        
        # Calculate a reasonable default k
        h, w = gray_array.shape
        max_dim = min(h, w)
        recommended_k = recommend_rank(score, 95.0, max_dim)
        
        return jsonify({
            'success': True,
            'recommended_algorithm': predicted_algo,
            'recommended_mode': recommended_mode,
            'recommended_k': recommended_k,
            'complexity_score': score
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
