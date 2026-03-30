"""Profile FlowSOM pipeline to identify where time is spent.

Breaks down the full FlowSOM() call into stages:
  1. read_input (I/O)
  2. run_model (SOM training + metaclustering)
  3. _update_derived_values (statistics + MST)

Also runs cProfile on the full pipeline for detailed function-level breakdown.
"""

import cProfile
import pstats
import time

import numpy as np

import flowsom as fs


def time_stages():
    """Time each stage of the FlowSOM pipeline independently."""
    print("=" * 70)
    print("STAGE-BY-STAGE TIMING")
    print("=" * 70)

    # Load data
    ff = fs.io.read_FCS("tests/data/ff.fcs")
    print(f"Data shape: {ff.X.shape}")

    cols_to_use = [8, 11, 13, 14, 15, 16, 17]
    n_clusters = 10

    # --- Warmup: trigger Numba JIT compilation ---
    print("\nWarming up Numba JIT...")
    t0 = time.perf_counter()
    _ = fs.FlowSOM(ff, cols_to_use=cols_to_use, n_clusters=n_clusters, seed=42)
    warmup_time = time.perf_counter() - t0
    print(f"Warmup (includes JIT compilation): {warmup_time:.3f}s")

    # --- Timed run: JIT already compiled ---
    print("\n--- Timed run (post-JIT) ---\n")

    # Build FlowSOM object without running __init__ pipeline
    fsom = fs.FlowSOM.__new__(fs.FlowSOM)
    fsom.cols_to_use = cols_to_use
    fsom.mad_allowed = 4
    fsom.xdim = 10
    fsom.ydim = 10
    fsom.rlen = 10
    fsom.mst = 1
    fsom.alpha = (0.05, 0.01)
    fsom.seed = 42
    fsom.n_clusters = n_clusters

    from mudata import MuData
    import anndata as ad

    fsom.mudata = MuData({"cell_data": ad.AnnData(), "cluster_data": ad.AnnData()})

    from flowsom.models.flowsom_estimator import FlowSOMEstimator

    fsom.model = FlowSOMEstimator(
        xdim=10, ydim=10, rlen=10, mst=1, alpha=(0.05, 0.01), seed=42, n_clusters=n_clusters
    )

    # Stage 1: read_input
    t0 = time.perf_counter()
    fsom.read_input(ff)
    t_read = time.perf_counter() - t0

    # Stage 2: run_model (SOM + metaclustering)
    t0 = time.perf_counter()
    fsom.run_model()
    t_model = time.perf_counter() - t0

    # Stage 3: _update_derived_values (stats + MST)
    t0 = time.perf_counter()
    fsom._update_derived_values()
    t_derived = time.perf_counter() - t0

    total = t_read + t_model + t_derived

    print(f"{'Stage':<35} {'Time (s)':>10} {'% Total':>10}")
    print("-" * 57)
    print(f"{'1. read_input (I/O)':<35} {t_read:>10.4f} {100*t_read/total:>9.1f}%")
    print(f"{'2. run_model (SOM + metacluster)':<35} {t_model:>10.4f} {100*t_model/total:>9.1f}%")
    print(f"{'3. _update_derived_values (MST+stats)':<35} {t_derived:>10.4f} {100*t_derived/total:>9.1f}%")
    print("-" * 57)
    print(f"{'TOTAL':<35} {total:>10.4f} {'100.0%':>10}")

    # --- Now time SOM training vs metaclustering within run_model ---
    print("\n\n--- Breaking down run_model ---\n")

    from flowsom.models import SOMEstimator, ConsensusCluster

    X = fsom.mudata["cell_data"][:, cols_to_use].X

    # SOM training only
    est = SOMEstimator(xdim=10, ydim=10, rlen=10, mst=1, alpha=(0.05, 0.01), seed=42)
    t0 = time.perf_counter()
    est.fit(X)
    t_som = time.perf_counter() - t0

    # Metaclustering only
    cc = ConsensusCluster(n_clusters=n_clusters)
    t0 = time.perf_counter()
    cc.fit_predict(est.codes)
    t_meta = time.perf_counter() - t0

    print(f"{'Sub-stage':<35} {'Time (s)':>10} {'% of run_model':>15}")
    print("-" * 62)
    print(f"{'SOM training (fit)':<35} {t_som:>10.4f} {100*t_som/(t_som+t_meta):>14.1f}%")
    print(f"{'Metaclustering (ConsensusCluster)':<35} {t_meta:>10.4f} {100*t_meta/(t_som+t_meta):>14.1f}%")
    print("-" * 62)
    print(f"{'Total':<35} {t_som+t_meta:>10.4f}")


def run_cprofile():
    """Run cProfile on the full FlowSOM pipeline."""
    print("\n\n" + "=" * 70)
    print("cProfile: FULL FlowSOM() PIPELINE (top 30 by cumulative time)")
    print("=" * 70 + "\n")

    ff = fs.io.read_FCS("tests/data/ff.fcs")

    # Warmup
    _ = fs.FlowSOM(ff, cols_to_use=[8, 11, 13, 14, 15, 16, 17], n_clusters=10, seed=42)

    # Profiled run
    profiler = cProfile.Profile()
    profiler.enable()
    _ = fs.FlowSOM(ff, cols_to_use=[8, 11, 13, 14, 15, 16, 17], n_clusters=10, seed=42)
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(30)


if __name__ == "__main__":
    time_stages()
    run_cprofile()
