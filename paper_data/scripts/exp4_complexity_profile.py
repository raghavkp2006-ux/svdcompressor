"""
Experiment 4 — Complexity & Memory Profiling
=============================================
For rSVD, PCA, NMF, DCT at resolutions 256x256, 512x512, 1024x1024:
measure peak RAM (tracemalloc) and wall-clock latency.

Uses a resized Kodak image (kodim23) as test input for realistic content.

Output: paper_data/complexity_profile.csv
Columns: algorithm, resolution, peak_memory_mb, latency_ms
"""

import os, sys, time, csv, tracemalloc, gc
import numpy as np
from PIL import Image

# ── Project root setup ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from lib.rsvd import rsvd, reconstruct
from lib.algorithms import pca_compress_channel, nmf_compress_channel, dct_compress_channel

# ── Paths ────────────────────────────────────────────────────────────────────
KODAK_DIR = os.path.join(PROJECT_ROOT, 'kodak_images')
OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'paper_data', 'corrected', 'complexity_profile_corrected.csv')
# Fixed rank for fair comparison
RANK = 50
RESOLUTIONS = [256, 512, 1024]


def get_test_channel(res):
    """Load kodim23 and resize to res x res, return single channel (float64)."""
    fpath = os.path.join(KODAK_DIR, 'kodim23.png')
    pil = Image.open(fpath).convert('L')
    pil = pil.resize((res, res), Image.LANCZOS)
    return np.array(pil, dtype=np.float64)


def measure_algorithm(algo_name, func, channel, rank):
    """Run algorithm with tracemalloc and return (peak_memory_mb, latency_ms)."""
    gc.collect()
    tracemalloc.start()

    t0 = time.perf_counter()
    func(channel, rank)
    latency = (time.perf_counter() - t0) * 1000

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / (1024 * 1024)
    return peak_mb, latency


def rsvd_wrapper(channel, rank):
    """Wrapper matching the (channel, rank) signature."""
    U, S, Vt = rsvd(channel, rank, oversample=10, power_iter=2)
    return reconstruct(U, S, Vt)


def main():
    print('=== Experiment 4: Complexity & Memory Profiling ===')

    algorithms = {
        'rSVD': rsvd_wrapper,
        'PCA': pca_compress_channel,
        'NMF': nmf_compress_channel,
        'DCT': dct_compress_channel,
    }

    rows = []

    for res in RESOLUTIONS:
        print(f'\n  Resolution: {res}x{res}')
        channel = get_test_channel(res)

        for algo_name, func in algorithms.items():
            print(f'    {algo_name:6s} ...', end=' ', flush=True)

            try:
                # Warm-up run (excluded from measurement)
                func(channel, RANK)

                # Measured run
                peak_mb, latency = measure_algorithm(algo_name, func, channel, RANK)

                row = {
                    'algorithm': algo_name,
                    'resolution': f'{res}x{res}',
                    'peak_memory_mb': round(peak_mb, 2),
                    'latency_ms': round(latency, 2),
                }
                rows.append(row)
                print(f'Peak RAM={peak_mb:.2f} MB  Latency={latency:.1f} ms')

            except Exception as e:
                print(f'FAILED: {e}')
                rows.append({
                    'algorithm': algo_name,
                    'resolution': f'{res}x{res}',
                    'peak_memory_mb': None,
                    'latency_ms': None,
                })

    # Write CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'algorithm', 'resolution', 'peak_memory_mb', 'latency_ms'
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f'\n  Saved: {OUTPUT_CSV}')
    print(f'  Total rows: {len(rows)}')
    print('=== Experiment 4 Complete ===')


if __name__ == '__main__':
    main()
