# Plan: Rust-Accelerated Parallel SOM for FlowSOM_Python

## What You're Building

A PyO3 extension module called `flowsom-rs` that replaces FlowSOM_Python's Numba-compiled SOM training with a Rust implementation using `rayon` for multi-core parallelism. The module implements batch-learning SOM (from the BL-FlowSOM paper) which also eliminates the input-order-dependent nondeterminism in current FlowSOM.

## Why It Will Be Accepted

The FlowSOM_Python authors themselves documented two unsolved problems:

1. **Multi-threading doesn't work.** Their paper states: "The execution speed did not change significantly when running the algorithm on different number of threads using the NUMBA_NUM_THREADS environment variable." This is a known Numba limitation with array allocations and GIL contention.

2. **Batch SOM is listed as future work.** Their paper concludes: "Future work includes implementing the remainder of the visualizations as in R, further broadening the documentation and **implementing batch SOM to further speed up the algorithm**."

You'd be delivering their stated future work, with a solution to a problem they couldn't solve with Numba.

## The Algorithm

### Current FlowSOM: Online SOM (what they have now)

```
for epoch in 1..T:
    for each event x_i (in random order):      ← sequential, order-dependent
        find BMU: j* = argmin_j ||x_i - w_j||²
        update all nodes: w_j += α(t) · h(j,j*,t) · (x_i - w_j)
```

This is inherently sequential — each weight update depends on the previous one. Numba JITs it to native code, but you can't parallelize the inner loop because of the data dependency. This is why `NUMBA_NUM_THREADS` didn't help.

### BL-FlowSOM: Batch-Learning SOM (what you'll implement)

```
for epoch in 1..T:
    // Phase 1: Assign all events to BMUs (PARALLEL)
    for each event x_i in parallel:             ← rayon par_iter
        bmu[i] = argmin_j ||x_i - w_j||²

    // Phase 2: Update weights (accumulate then divide)
    for each node j:
        numerator_j = Σ_{i: bmu[i]=j} h(j,bmu[i],t) · x_i
        denominator_j = Σ_{i: bmu[i]=j} h(j,bmu[i],t)
        w_j = numerator_j / denominator_j
```

Phase 1 is embarrassingly parallel — each event's BMU search is independent. Phase 2 is a reduction. Both are trivially parallelizable with `rayon`. And because the weight update uses all events simultaneously rather than one-at-a-time, **the result is deterministic regardless of input order**.

Reference: Otsuka et al., "BL-FlowSOM: Consistent and Highly Accelerated FlowSOM Based on Parallelized Batch Learning," Cytometry Part A, 2025.

---

## Phase 0: Setup and Reconnaissance (Days 1–2)

### 0.1 Fork and understand FlowSOM_Python

```bash
git clone https://github.com/saeyslab/FlowSOM_Python.git
cd FlowSOM_Python
uv venv && uv pip install -e ".[dev,test]"
```

Find the SOM training code. Based on the paper, it's in `src/flowsom/` — look for Numba-decorated functions that do the Kohonen competitive learning. Identify:

- The exact function signature (inputs: data matrix, grid dimensions, epochs, learning rate, sigma)
- The exact output format (node weight matrix, mapping of events to nodes)
- How it integrates with the rest of the FlowSOM pipeline (what calls it, what consumes its output)
- The test suite — find the tests that exercise SOM training so you can validate your replacement

### 0.2 Run the existing benchmarks

Reproduce the paper's benchmark: use the demo FCS file, oversample to 500K / 1M / 3M events, time the full `fs.FlowSOM()` call. Also time just the SOM training step in isolation. This gives you your baseline.

### 0.3 Profile the SOM training

```python
import cProfile
cProfile.run('fs.FlowSOM(ff, cols_to_use=cols, xdim=10, ydim=10, n_clusters=10)')
```

Confirm what fraction of total runtime is SOM training vs. I/O vs. metaclustering vs. MST. This tells you the ceiling on your speedup — if SOM training is 60% of total time and you make it 10× faster, the end-to-end speedup is ~2.3×.

---

## Phase 1: Rust Core Library (Days 3–8)

### 1.1 Create the crate

```bash
cargo init flowsom-rs --lib
cd flowsom-rs
```

`Cargo.toml`:

```toml
[package]
name = "flowsom-rs"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]   # cdylib for PyO3, rlib for Rust tests

[dependencies]
ndarray = { version = "0.16", features = ["rayon"] }
rayon = "1.10"
rand = "0.8"
rand_chacha = "0.3"               # deterministic RNG

[dev-dependencies]
criterion = "0.5"
approx = "0.5"

[[bench]]
name = "som_benchmark"
harness = false
```

### 1.2 Implement the batch-learning SOM

The core module needs three functions:

**`initialize_pca`**: Initialize node weights via PCA of the input data (as BL-FlowSOM does), not random. This eliminates initialization-dependent randomness.

```rust
/// Initialize a `grid_x * grid_y` SOM grid with PCA-based initialization.
/// Returns node weights as Array2<f64> with shape (n_nodes, n_features).
pub fn initialize_pca(
    data: &ArrayView2<f64>,
    grid_x: usize,
    grid_y: usize,
) -> Array2<f64>
```

**`train_batch_som`**: The main training loop with parallel BMU search.

```rust
/// Train a batch-learning SOM.
///
/// # Arguments
/// * `data` - Event matrix, shape (n_events, n_features)
/// * `weights` - Initial node weights, shape (n_nodes, n_features) — mutated in place
/// * `grid_x`, `grid_y` - Grid dimensions
/// * `epochs` - Number of training epochs
/// * `sigma_start`, `sigma_end` - Neighborhood radius decay
///
/// # Returns
/// * `mapping` - Vec<usize> of length n_events, mapping each event to its BMU node index
pub fn train_batch_som(
    data: &ArrayView2<f64>,
    weights: &mut Array2<f64>,
    grid_x: usize,
    grid_y: usize,
    epochs: usize,
    sigma_start: f64,
    sigma_end: f64,
) -> Vec<usize>
```

Inside `train_batch_som`, the hot loop:

```rust
for epoch in 0..epochs {
    let sigma = /* decay from sigma_start to sigma_end */;

    // Phase 1: Parallel BMU search
    let bmus: Vec<usize> = data.axis_iter(Axis(0))
        .into_par_iter()
        .map(|event| {
            let mut best_dist = f64::MAX;
            let mut best_node = 0;
            for j in 0..n_nodes {
                let node = weights.row(j);
                let dist = squared_euclidean(&event, &node);
                if dist < best_dist {
                    best_dist = dist;
                    best_node = j;
                }
            }
            best_node
        })
        .collect();

    // Phase 2: Accumulate weighted sums per node (can also be parallelized)
    let mut numerators = Array2::<f64>::zeros((n_nodes, n_features));
    let mut denominators = Array1::<f64>::zeros(n_nodes);

    for (i, bmu) in bmus.iter().enumerate() {
        for j in 0..n_nodes {
            let h = neighborhood(grid_x, grid_y, *bmu, j, sigma);
            if h > 1e-6 {  // skip negligible contributions
                let event = data.row(i);
                numerators.row_mut(j).scaled_add(h, &event);
                denominators[j] += h;
            }
        }
    }

    // Phase 3: Update weights
    for j in 0..n_nodes {
        if denominators[j] > 0.0 {
            weights.row_mut(j).assign(&(&numerators.row(j) / denominators[j]));
        }
    }
}
```

**`neighborhood`**: Gaussian neighborhood function on the 2D grid.

```rust
fn neighborhood(
    grid_x: usize, grid_y: usize,
    bmu: usize, node: usize,
    sigma: f64,
) -> f64 {
    let (bmu_x, bmu_y) = (bmu % grid_x, bmu / grid_x);
    let (node_x, node_y) = (node % grid_x, node / grid_x);
    let dx = bmu_x as f64 - node_x as f64;
    let dy = bmu_y as f64 - node_y as f64;
    let dist_sq = dx * dx + dy * dy;
    (-dist_sq / (2.0 * sigma * sigma)).exp()
}
```

### 1.3 Write Rust tests

Test against known inputs/outputs. Use the Samusik_01 or Levine_13dim benchmark datasets (available from FlowRepository). The key correctness criterion: given the same data, your batch SOM should produce metaclusters with F1-scores comparable to FlowSOM_Python's (within the variance they report across 10 runs).

### 1.4 Benchmark in pure Rust

Use `criterion` to benchmark `train_batch_som` on synthetic data (1M events × 7 features, 10×10 grid, 10 epochs) at 1, 2, 4, 8 threads (controlled via `rayon::ThreadPoolBuilder`). This gives you the pure Rust scaling numbers before Python overhead.

---

## Phase 2: PyO3 Bindings (Days 9–12)

### 2.1 Add PyO3

Add to `Cargo.toml`:

```toml
[dependencies]
pyo3 = { version = "0.23", features = ["extension-module"] }
numpy = "0.23"  # pyo3 numpy interop
```

Create `pyproject.toml` for `maturin`:

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "flowsom-rs"
requires-python = ">=3.9"
```

### 2.2 Write the Python-facing API

The binding should match the interface FlowSOM_Python's SOM training expects. Based on the codebase, you need a function that takes a numpy array and returns node weights + event-to-node mapping.

```rust
#[pyfunction]
#[pyo3(signature = (data, xdim=10, ydim=10, epochs=10, sigma_start=None, sigma_end=None, seed=None))]
fn train_som(
    py: Python<'_>,
    data: PyReadonlyArray2<f64>,
    xdim: usize,
    ydim: usize,
    epochs: usize,
    sigma_start: Option<f64>,
    sigma_end: Option<f64>,
    seed: Option<u64>,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray1<usize>>)> {
    let data = data.as_array();

    let sigma_s = sigma_start.unwrap_or_else(|| (xdim.max(ydim) as f64) / 2.0);
    let sigma_e = sigma_end.unwrap_or(1.0);

    // PCA initialization (deterministic)
    let mut weights = initialize_pca(&data, xdim, ydim);

    // Train
    let mapping = train_batch_som(&data, &mut weights, xdim, ydim, epochs, sigma_s, sigma_e);

    // Convert back to numpy
    let weights_py = weights.into_pyarray(py).to_owned();
    let mapping_py = Array1::from(mapping).into_pyarray(py).to_owned();

    Ok((weights_py, mapping_py))
}

#[pymodule]
fn flowsom_rs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train_som, m)?)?;
    Ok(())
}
```

### 2.3 Build and test from Python

```bash
maturin develop --release
```

```python
import numpy as np
import flowsom_rs

data = np.random.randn(1_000_000, 7)
weights, mapping = flowsom_rs.train_som(data, xdim=10, ydim=10, epochs=10)
print(weights.shape)   # (100, 7)
print(mapping.shape)   # (1000000,)
```

---

## Phase 3: Integration Benchmark (Days 13–16)

### 3.1 Drop-in replacement test

Write a script that runs FlowSOM_Python's full pipeline but swaps the SOM step for your Rust version. Verify that:

- Metaclustering F1-scores are comparable to the Numba version (using the same benchmark datasets from the paper: Levine_13dim, Levine_32dim, Mosmann_rare, Nilsson_rare, Samusik_01, Samusik_all)
- Results are deterministic across runs (same seed → bit-identical output)
- Results are deterministic across input orders (shuffle the events → same output)

### 3.2 Benchmark comparison

Create a comprehensive benchmark script:

```python
# /// script
# requires-python = ">=3.9"
# dependencies = ["flowsom", "flowsom-rs", "numpy", "time"]
# ///

import flowsom as fs
import flowsom_rs
import numpy as np
import time

# Load test data
ff = fs.io.read_FCS("tests/data/ff.fcs")

for n_events in [500_000, 1_000_000, 2_000_000, 3_000_000]:
    data = np.tile(ff.X, (n_events // ff.X.shape[0] + 1, 1))[:n_events]

    # Numba version (FlowSOM_Python)
    start = time.perf_counter()
    # ... call their internal SOM function ...
    numba_time = time.perf_counter() - start

    # Rust version
    start = time.perf_counter()
    weights, mapping = flowsom_rs.train_som(data, xdim=10, ydim=10, epochs=10)
    rust_time = time.perf_counter() - start

    print(f"{n_events:>10}: Numba={numba_time:.2f}s  Rust={rust_time:.2f}s  Speedup={numba_time/rust_time:.1f}x")
```

Run on 1, 2, 4, 8 cores (control via `RAYON_NUM_THREADS`) to show scaling.

### 3.3 Consistency test

Reproduce the BL-FlowSOM paper's consistency experiment: take the Samusik_all dataset, concatenate its 10 source samples in 10 different orders, run your SOM on each, and show that all 10 produce identical cluster assignments. Then run FlowSOM_Python's Numba SOM on the same 10 orderings and show they produce different results.

---

## Phase 4: Prepare the Contribution (Days 17–20)

### 4.1 Open a discussion issue first

Before submitting code, open an issue on `saeyslab/FlowSOM_Python` titled something like:

> **[RFC] Rust-accelerated batch-learning SOM with multi-core parallelism**
>
> Hi — I've implemented a batch-learning SOM in Rust (via PyO3) that addresses two items from your paper:
>
> 1. Multi-core parallelism for SOM training. Your paper noted that `NUMBA_NUM_THREADS` didn't help — this is a known Numba limitation with array allocations. The Rust implementation uses `rayon` and shows N× scaling on N cores.
>
> 2. Batch-learning SOM (per Otsuka et al. 2025, BL-FlowSOM) which produces deterministic results regardless of input order, unlike the current online SOM.
>
> Benchmark results: [table of speedups]. F1-scores on the six benchmark datasets: [table showing comparable accuracy].
>
> Would you be open to a PR that adds this as an optional backend? The API would be `fs.FlowSOM(..., som_backend="rust")` with the Numba version remaining the default for zero-dependency installations.
>
> Happy to adapt the approach based on your feedback.

This respects the maintainers' ownership and lets them weigh in before you invest in PR polish.

### 4.2 Structure the PR

If they're receptive, prepare a PR with:

**New files:**
- `rust/` directory with the Rust crate and `pyproject.toml` (or a separate `flowsom-rs` package on PyPI that `flowsom` lists as an optional dependency)
- Integration shim in `src/flowsom/` that detects whether `flowsom_rs` is installed and uses it when requested

**Modified files:**
- The SOM training call site, adding a `som_backend` parameter
- Documentation noting the Rust backend as optional

**Unchanged files:**
- All existing tests should still pass with the Numba backend
- New tests exercising the Rust backend

**The optional dependency pattern:**

```toml
# In FlowSOM_Python's pyproject.toml
[project.optional-dependencies]
rust = ["flowsom-rs>=0.1.0"]
```

Users install with `pip install flowsom[rust]` to get the accelerated backend, or plain `pip install flowsom` to keep the pure-Python/Numba version.

### 4.3 Write a benchmark report

Include a markdown document or Jupyter notebook in the PR showing:
- Speedup curves (events vs. runtime, Numba vs. Rust at 1/4/8 cores)
- F1-score comparison table across the six benchmark datasets
- Consistency comparison (10 input orderings)
- Memory usage comparison

---

## Phase 5: If They Don't Want a PR (Contingency)

If the Saeyslab team prefers not to merge Rust into their codebase (totally reasonable — it adds a build dependency), publish `flowsom-rs` as a standalone package on PyPI and crates.io. Write a wrapper that monkey-patches FlowSOM_Python:

```python
import flowsom_rs

# Drop-in acceleration for FlowSOM_Python
def patch_flowsom():
    """Replace FlowSOM_Python's SOM training with Rust backend."""
    import flowsom.models
    flowsom.models._original_som = flowsom.models.som_training_function
    flowsom.models.som_training_function = flowsom_rs.train_som

# Usage:
# import flowsom_rs; flowsom_rs.patch_flowsom()
# Then use FlowSOM_Python as normal — it's now using the Rust SOM
```

This way the ecosystem benefits even without upstream adoption.

---

## Timeline Summary

| Phase | Days | Deliverable |
|-------|------|-------------|
| 0: Recon | 1–2 | Profile data, baseline benchmarks |
| 1: Rust core | 3–8 | Working batch SOM with rayon, Criterion benchmarks |
| 2: PyO3 | 9–12 | `pip install`-able Python module |
| 3: Integration | 13–16 | Head-to-head benchmarks, F1 validation, consistency proof |
| 4: Contribute | 17–20 | Issue, PR, or standalone package |

Total: ~3 weeks of focused work.

## Dependencies

### Rust side
- `ndarray` (0.16) + `ndarray-linalg` (for PCA init)
- `rayon` (1.10)
- `rand` + `rand_chacha` (deterministic RNG)
- `pyo3` (0.23) + `numpy` (0.23)

### Python side
- `maturin` (build tool)
- `numpy` (runtime)
- `flowsom` (for integration testing)
- `pytest` + `pytest-benchmark` (testing)

### Build
- Rust stable toolchain
- `maturin` (`pip install maturin`)
- That's it — no C compiler, no Fortran, no LAPACK

## License

FlowSOM_Python is GPL-3.0. Your Rust SOM is implemented from the algorithm description in academic papers (Kohonen 1982 for SOM, Monti 2003 for consensus clustering, Otsuka 2025 for batch learning), not ported from their GPL code. You can license the Rust crate however you want — MIT or Apache-2.0 are fine for a standalone crate. If it gets merged into FlowSOM_Python, the integration code would be GPL-3.0 to match.

## Key Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Batch SOM produces different clusters than online SOM | This is expected — they're different algorithms. Show that F1-scores are comparable (BL-FlowSOM paper already proves this). Frame as "comparable accuracy + determinism + parallelism." |
| Maintainers don't want Rust in their build | Publish standalone `flowsom-rs` on PyPI with monkey-patch approach |
| Multi-core speedup is less than expected | Even 3× on 8 cores + determinism is a publishable contribution. The consistency story may matter more than raw speed to clinical users. |
| PyO3 numpy interop adds overhead | Benchmark the Python↔Rust boundary cost. For 1M+ events the SOM compute should dominate the FFI overhead. |
| FlowSOM_Python changes their SOM API before you finish | Pin to their current release (v0.2.2) for development, adapt to any changes at PR time. |
