"""
Top-level compression API — the function your Flask routes call.

Handles:
  - Per-channel pre-screening (skip flat channels entirely)
  - Rank recommendation based on complexity score
  - rSVD compression per channel
  - Full RGB image assembly
"""

import numpy as np
from .prescreening import complexity_score, recommend_rank, SKIP_THRESHOLD
from .rsvd import rsvd, reconstruct


def compress_channel(
    channel: np.ndarray,
    rank: int | None = None,
    energy_percent: float = 95.0,
) -> tuple[np.ndarray, int, float, bool]:
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
    if score < SKIP_THRESHOLD:
        return channel.copy(), 0, score, True

    # Determine rank
    max_dim = min(h, w)
    k = rank if rank is not None else recommend_rank(score, energy_percent, max_dim)
    k = min(k, max_dim)  # Safety clamp

    # Core rSVD
    U, S, Vt = rsvd(channel, k)
    compressed = reconstruct(U, S, Vt)

    # Clamp to valid pixel range
    np.clip(compressed, 0, 255, out=compressed)

    return compressed, k, score, False


def compress_image(
    img_array: np.ndarray,
    rank: int | None = None,
    energy_percent: float = 95.0,
) -> tuple[np.ndarray, list[int], list[float]]:
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
        (compressed_array, k_values_per_channel, scores_per_channel)
    """
    if img_array.ndim == 2:
        compressed, k, score, _ = compress_channel(
            img_array, rank, energy_percent
        )
        return compressed, [k], [score]

    channels = img_array.shape[2]
    result = np.zeros_like(img_array)
    k_values = []
    scores = []

    for c in range(channels):
        compressed, k, score, _ = compress_channel(
            img_array[:, :, c], rank, energy_percent
        )
        result[:, :, c] = compressed
        k_values.append(k)
        scores.append(score)

    return result, k_values, scores


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
