"""
Alternative matrix decomposition and compression algorithms:
- PCA (Principal Component Analysis)
- NMF (Non-negative Matrix Factorization)
- DCT (Discrete Cosine Transform)
"""
import numpy as np
import warnings

try:
    from sklearn.decomposition import PCA, NMF
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from scipy.fftpack import dct, idct
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def pca_compress_channel(channel: np.ndarray, k: int) -> np.ndarray:
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required for PCA.")
    k = min(k, min(channel.shape))
    if k == 0:
        return np.zeros_like(channel)
    pca = PCA(n_components=k)
    transformed = pca.fit_transform(channel)
    reconstructed = pca.inverse_transform(transformed)
    return np.clip(reconstructed, 0, 255)

def nmf_compress_channel(channel: np.ndarray, k: int) -> np.ndarray:
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required for NMF.")
    k = min(k, min(channel.shape))
    if k == 0:
        return np.zeros_like(channel)
    # NMF requires non-negative inputs, image channels are 0-255
    nmf = NMF(n_components=k, init='random', random_state=42, max_iter=200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        W = nmf.fit_transform(channel)
        H = nmf.components_
    reconstructed = np.dot(W, H)
    return np.clip(reconstructed, 0, 255)

def dct_compress_channel(channel: np.ndarray, k: int) -> np.ndarray:
    if not HAS_SCIPY:
        raise ImportError("scipy is required for DCT.")
    h, w = channel.shape
    # Approximate k as the number of frequency components to keep (top-left block)
    max_k = min(h, w)
    cutoff = min(k, max_k)
    if cutoff == 0:
        return np.zeros_like(channel)
        
    # Perform 2D DCT
    dct_coeffs = dct(dct(channel.T, norm='ortho').T, norm='ortho')
    
    # Truncate high frequencies
    truncated_coeffs = np.zeros_like(dct_coeffs)
    truncated_coeffs[:cutoff, :cutoff] = dct_coeffs[:cutoff, :cutoff]
    
    # Perform 2D IDCT
    reconstructed = idct(idct(truncated_coeffs.T, norm='ortho').T, norm='ortho')
    return np.clip(reconstructed, 0, 255)

def compress_image_algo(img_array: np.ndarray, k: int, algo: str) -> np.ndarray:
    """
    Compress image using a specified algorithm.
    algo can be 'pca', 'nmf', or 'dct'.
    """
    if algo == 'pca':
        func = pca_compress_channel
    elif algo == 'nmf':
        func = nmf_compress_channel
    elif algo == 'dct':
        func = dct_compress_channel
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    if img_array.ndim == 2:
        return func(img_array, k)

    channels = img_array.shape[2]
    result = np.zeros_like(img_array)
    for c in range(channels):
        result[:, :, c] = func(img_array[:, :, c], k)
        
    return result
