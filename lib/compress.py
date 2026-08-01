"""
Top-level compression API — the function your Flask routes call.

Handles:
  - Per-channel pre-screening (skip flat channels entirely)
  - Rank recommendation based on complexity score
  - rSVD compression per channel
  - Full RGB image assembly
"""

import zlib, struct, io
import numpy as np
from .prescreening import complexity_score, recommend_rank, DEFAULT_SKIP_THRESHOLD
from .rsvd import rsvd, reconstruct

def serialize_factors(U, S, Vt) -> bytes:
    """Pack U, S, Vt into bytes then zlib-compress."""
    if U is None or S is None or Vt is None:
        return b''
    buf = io.BytesIO()
    for arr in (U, S, Vt):
        data = arr.astype(np.float32).tobytes()
        if arr.ndim == 1:
            buf.write(struct.pack('>II', arr.shape[0], 1))
        else:
            buf.write(struct.pack('>II', *arr.shape))
        buf.write(data)
    return zlib.compress(buf.getvalue(), level=6)

def compressed_size_bytes(U, S, Vt) -> int:
    return len(serialize_factors(U, S, Vt))

def true_compression_ratio(orig_shape, U_list, S_list, Vt_list) -> float:
    """Compute true CR given lists of factors per channel."""
    if len(orig_shape) == 2:
        m, n = orig_shape
        ch = 1
    else:
        m, n, ch = orig_shape
        
    orig_bytes = m * n * ch
    comp_bytes = 0
    for i in range(ch):
        comp_bytes += compressed_size_bytes(U_list[i], S_list[i], Vt_list[i])
        
    return orig_bytes / comp_bytes if comp_bytes > 0 else 0.0


def compress_channel(
    channel: np.ndarray,
    rank: int | None = None,
    energy_percent: float = 95.0,
) -> tuple[np.ndarray, int, float, bool, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Compress a single image channel using rSVD with pre-screening.
    
    Args:
        channel:        2D numpy array (height × width), float64 values 0–255.
        rank:           Explicit rank (None = auto-select via complexity score).
        energy_percent: Energy slider value for auto-rank calculation.
    
    Returns:
        (compressed_channel, rank_used, complexity_score, was_skipped)
    """
    h, w = channel.shape
    score = complexity_score(channel)

    # Pre-screening: skip SVD for flat/smooth channels
    if score < DEFAULT_SKIP_THRESHOLD:
        return channel.copy(), 0, score, True, None, None, None

    # Determine rank
    max_dim = min(h, w)
    
    # Use learned predictor if no explicit rank is given
    # Scale prediction by energy_percent to retain slider functionality
    if rank is not None:
        k = rank
    else:
        k = recommend_rank(score, energy_percent, max_dim)

    k = min(k, max_dim)  # Safety clamp

    # Core rSVD
    U, S, Vt = rsvd(channel, k)
    compressed = reconstruct(U, S, Vt)

    # Clamp to valid pixel range
    np.clip(compressed, 0, 255, out=compressed)

    return compressed, k, score, False, U, S, Vt


def compress_image(
    img_array: np.ndarray,
    rank: int | None = None,
    energy_percent: float = 95.0,
) -> tuple[np.ndarray, list[int], list[float], list[tuple]]:
    """
    Compress a full RGB (or grayscale) image.
    
    Each channel is independently pre-screened and compressed at its own
    optimal rank. This means a noisy red channel might use rank 80 while
    a smooth blue channel uses rank 20 — maximizing quality per byte.
    
    Args:
        img_array:      3D numpy array (H × W × C) or 2D for grayscale.
        rank:           Explicit rank for all channels (None = per-channel auto).
        energy_percent: Energy slider for auto-rank.
    
    Returns:
        (compressed_array, k_values_per_channel, scores_per_channel, factors_per_channel)
    """
    if img_array.ndim == 2:
        compressed, k, score, _, U, S, Vt = compress_channel(
            img_array, rank, energy_percent
        )
        return compressed, [k], [score], [(U, S, Vt)]

    channels = img_array.shape[2]
    result = np.zeros_like(img_array)
    k_values = []
    scores = []
    factors = []

    for c in range(channels):
        compressed, k, score, _, U, S, Vt = compress_channel(
            img_array[:, :, c], rank, energy_percent
        )
        result[:, :, c] = compressed
        k_values.append(k)
        scores.append(score)
        factors.append((U, S, Vt))

    return result, k_values, scores, factors


def compress_image_basic(img_array: np.ndarray, k: int) -> np.ndarray:
    """
    Basic mode: compress all channels at a fixed rank k.
    No pre-screening, no adaptive rank selection.
    
    Args:
        img_array: 3D numpy array (H × W × C) or 2D.
        k:         Fixed rank for all channels.
    
    Returns:
        Compressed image array.
    """
    if img_array.ndim == 2:
        U, S, Vt = rsvd(img_array, min(k, min(img_array.shape)))
        result = reconstruct(U, S, Vt)
        np.clip(result, 0, 255, out=result)
        return result

    channels = img_array.shape[2]
    result = np.zeros_like(img_array)
    max_dim = min(img_array.shape[0], img_array.shape[1])
    k_clamped = min(k, max_dim)

    for c in range(channels):
        U, S, Vt = rsvd(img_array[:, :, c], k_clamped)
        result[:, :, c] = reconstruct(U, S, Vt)

    np.clip(result, 0, 255, out=result)
    return result
