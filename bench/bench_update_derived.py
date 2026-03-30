"""Benchmark _update_derived_values for hyperfine.

Usage:
    hyperfine --warmup 2 \
        '.venv/bin/python bench/bench_update_derived.py' \
        --export-markdown bench/hyperfine_results.md
"""

import flowsom as fs

ff = fs.io.read_FCS("tests/data/ff.fcs")
fsom = fs.FlowSOM(ff, cols_to_use=[8, 11, 13, 14, 15, 16, 17], n_clusters=10, seed=42)

# Run _update_derived_values 10 times to amplify the signal
for _ in range(10):
    fsom._update_derived_values()
