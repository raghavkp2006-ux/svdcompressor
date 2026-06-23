import os
import sys
import numpy as np
from scipy.stats import entropy as scipy_entropy

# Add root to sys path to import lib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lib.rsvd import rsvd

def extract_features(channel: np.ndarray) -> list:
    """Extract 8 features from a single image channel."""
    h, w = channel.shape

    # 1. Block variance (top quartile)
    block_size = 16
    variances = []
    for by in range(0, h - block_size + 1, block_size):
        for bx in range(0, w - block_size + 1, block_size):
            block = channel[by:by+block_size, bx:bx+block_size]
            variances.append(np.var(block))
    
    # Handle small images where no 16x16 block fits
    if not variances:
        variances = [0.0]

    variances.sort(reverse=True)
    top_q = variances[:max(1, len(variances)//4)]
    block_var = float(np.mean(top_q))

    # 2. Global entropy
    hist, _ = np.histogram(channel.flatten(), bins=256, range=(0,255))
    hist = hist / hist.sum()  # normalize to probability
    ent = float(scipy_entropy(hist + 1e-10))  # +1e-10 avoids log(0)

    # 3. Singular value decay rate (fit exponential to top-20 SVs)
    rank_probe = min(20, min(h, w) - 1)
    if rank_probe > 0:
        _, S, _ = rsvd(channel, rank_probe, oversample=5, power_iter=1)
        S_norm = S / (S[0] + 1e-10)  # normalize so S[0]=1
        indices = np.arange(len(S_norm))
        # fit log(S) = a*i + b → decay rate = -a
        coeffs = np.polyfit(indices, np.log(S_norm + 1e-10), 1)
        decay_rate = float(-coeffs[0])  # positive = fast decay = simple image
    else:
        S = np.array([1.0])
        decay_rate = 0.0

    # 4. Aspect ratio
    aspect = float(h / w)

    # 5. Mean pixel intensity
    mean_intensity = float(np.mean(channel))

    # 6. Std deviation
    std_dev = float(np.std(channel))

    # 7. Energy in top-5 singular values
    energy_top5 = float(np.sum(S[:5]**2) / (np.sum(S**2) + 1e-10))

    # 8. Max block variance (captures sharpest edge)
    max_block_var = float(variances[0])

    return [
        block_var, ent, decay_rate, aspect,
        mean_intensity, std_dev, energy_top5, max_block_var
    ]
