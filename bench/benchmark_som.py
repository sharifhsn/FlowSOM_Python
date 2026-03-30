"""Benchmark SOM training at various dataset sizes.

Compares:
  - Online SOM (SOMEstimator) - Numba JIT, sequential
  - Batch SOM (BatchSOMEstimator) - Numba JIT, parallel batches

Tests at 10K, 50K, 100K, 500K events with 10x10 grid, 10 epochs.
Each config is run 3 times; reports mean +/- std.
"""

import time

import numpy as np

import flowsom as fs
from flowsom.models import SOMEstimator
from flowsom.models.batch.som_estimator import BatchSOMEstimator


def load_and_oversample(n_events):
    """Load FCS test data and oversample to n_events rows."""
    ff = fs.io.read_FCS("tests/data/ff.fcs")
    cols = [8, 11, 13, 14, 15, 16, 17]
    X = ff[:, cols].X.astype(np.float64)
    # Tile to reach desired size
    repeats = (n_events // X.shape[0]) + 1
    X_big = np.tile(X, (repeats, 1))[:n_events]
    return X_big


def benchmark_estimator(estimator_cls, X, n_runs=3, **kwargs):
    """Time estimator.fit(X) over n_runs, return (mean, std) in seconds."""
    times = []
    for _ in range(n_runs):
        est = estimator_cls(xdim=10, ydim=10, rlen=10, seed=42, **kwargs)
        t0 = time.perf_counter()
        est.fit(X)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
    return np.mean(times), np.std(times)


def main():
    sizes = [10_000, 50_000, 100_000, 500_000]
    n_runs = 3

    print("=" * 80)
    print("SOM TRAINING BENCHMARK")
    print(f"Grid: 10x10, Epochs: 10, Runs per config: {n_runs}")
    print("=" * 80)

    # --- Warmup: trigger Numba JIT ---
    print("\nWarming up Numba JIT (first run compiles)...")
    X_warmup = load_and_oversample(1000)
    est = SOMEstimator(xdim=5, ydim=5, rlen=1, seed=42)
    est.fit(X_warmup)
    est2 = BatchSOMEstimator(xdim=5, ydim=5, rlen=1, seed=42, num_batches=2)
    est2.fit(X_warmup)
    print("JIT warmup complete.\n")

    # --- Benchmark ---
    print(f"{'Events':>10}  {'Online SOM (s)':>18}  {'Batch SOM (s)':>18}  {'Ratio (batch/online)':>20}")
    print("-" * 72)

    for n in sizes:
        X = load_and_oversample(n)

        mean_online, std_online = benchmark_estimator(SOMEstimator, X, n_runs=n_runs)
        mean_batch, std_batch = benchmark_estimator(
            BatchSOMEstimator, X, n_runs=n_runs, num_batches=10
        )
        ratio = mean_batch / mean_online if mean_online > 0 else float("inf")

        print(
            f"{n:>10,}  "
            f"{mean_online:>8.3f} +/- {std_online:.3f}  "
            f"{mean_batch:>8.3f} +/- {std_batch:.3f}  "
            f"{ratio:>18.2f}x"
        )

    print("\nNote: Online SOM is inherently sequential (order-dependent).")
    print("Batch SOM uses parallel batches via Numba prange but is NOT the")
    print("true BL-FlowSOM algorithm. The Rust implementation will use the")
    print("batch-learning SOM from Otsuka et al. 2025.")


if __name__ == "__main__":
    main()
