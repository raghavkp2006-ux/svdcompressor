"""
Randomized SVD — Halko, Martinsson, Tropp (2011)

Core algorithm for fast low-rank matrix approximation.
Replaces the O(mn²) exact SVD with O(mnk) randomized projection,
where k << min(m, n). For a 1000×1000 channel at rank 50, this is
~50× faster than numpy's full SVD.

Algorithm flow:
  1. Random Gaussian projection Ω (n × r)     — creates sketch directions
  2. Sketch Y = A · Ω (m × r)                 — reduces column space
  3. QR decomposition Q, _ = qr(Y)             — orthonormal basis for column space
  4. Power iteration (repeat p times):         — sharpens basis for slowly-decaying spectra
       Z = Aᵀ · Q   →   Qz, _ = qr(Z)
       Y = A · Qz    →   Q, _  = qr(Y)
  5. Project into low-dim: B = Qᵀ · A (r × n) — small matrix
  6. Exact SVD on B: Ub, S, Vt = svd(B)       — cheap (r × n instead of m × n)
  7. Recover: U = Q · Ub                       — left singular vectors of A

Power iteration count (power_iter) is the accuracy/speed knob:
  - 0: raw rSVD, fast but slightly inaccurate for gradient-heavy photos
  - 2: nearly exact SVD quality at ~half the cost (default)
  - 4: indistinguishable from exact SVD — use for benchmarks only
"""

import numpy as np


def rsvd(
    channel: np.ndarray,
    rank: int,
    oversample: int = 10,
    power_iter: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute rank-k randomized SVD of a 2D matrix.
    
    Args:
        channel:    2D numpy array (m × n), typically one color channel.
        rank:       Target rank k (number of singular values to keep).
        oversample: Extra dimensions for the random sketch (improves accuracy).
        power_iter: Number of power iterations (accuracy/speed trade-off).
    
    Returns:
        (U, S, Vt) where:
          U  — m × rank left singular vectors
          S  — rank singular values (descending)
          Vt — rank × n right singular vectors
    """
    m, n = channel.shape
    r = min(rank + oversample, min(m, n))

    # Step 1 — Random Gaussian projection matrix Ω (n × r)
    omega = np.random.randn(n, r)

    # Step 2 — Initial sketch Y = A · Ω  (m × r)
    Y = channel @ omega

    # Step 3 — QR to get orthonormal basis
    Q, _ = np.linalg.qr(Y)

    # Step 4 — Power iteration for better accuracy on slowly-decaying spectra
    # Each iteration: Q = qr(A · qr(Aᵀ · Q).Q).Q
    # This concentrates the basis on the dominant singular subspace.
    for _ in range(power_iter):
        Z = channel.T @ Q           # n × r  (Aᵀ · Q)
        Qz, _ = np.linalg.qr(Z)    # n × r  (re-orthogonalize to avoid drift)
        Y = channel @ Qz            # m × r  (A · Qz)
        Q, _ = np.linalg.qr(Y)     # m × r  (final orthonormal basis)

    # Step 5 — Project A into low-dim subspace: B = Qᵀ · A  (r × n)
    B = Q.T @ channel

    # Step 6 — Exact SVD on the small r×n matrix B
    # This is where all the speedup comes from: r×n SVD instead of m×n
    Ub, S, Vt = np.linalg.svd(B, full_matrices=False)

    # Step 7 — Recover left singular vectors of A: U = Q · Ub
    U = Q @ Ub

    # Truncate to requested rank
    k = min(rank, len(S))
    return U[:, :k], S[:k], Vt[:k, :]


def reconstruct(
    U: np.ndarray,
    S: np.ndarray,
    Vt: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct the approximated matrix: Â = U · diag(S) · Vt
    
    Args:
        U:  m × k left singular vectors
        S:  k singular values
        Vt: k × n right singular vectors
    
    Returns:
        Reconstructed matrix (m × n), float64.
    """
    return U @ np.diag(S) @ Vt
