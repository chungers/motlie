# SIFT Benchmark Experiments: SearchConfig Validation

## Overview

This document describes experiments to validate the `SearchConfig` API and measure
performance across different scales and distance metrics.

## Experiment Goals

1. **Validate SearchConfig Auto-Selection**
   - Cosine distance → RaBitQ (Hamming approximation)
   - L2 distance → Exact (no approximation)

2. **Measure Performance at Scale**
   - 1K, 10K, 100K vector scales
   - Index build throughput (vectors/sec)
   - Search QPS and latency percentiles

3. **Compare Distance Metrics**
   - L2 (Euclidean) - standard for SIFT
   - Cosine (Angular) - normalized vectors

4. **Validate Recall**
   - Target: Recall@10 > 95%
   - Measure impact of rerank_factor

## Experiment Matrix

| Scale | Distance | Strategy (auto) | ef | rerank | Expected Recall |
|-------|----------|-----------------|-----|--------|-----------------|
| 1K    | L2       | Exact           | 100 | -      | ~100%           |
| 1K    | Cosine   | RaBitQ(cached)  | 100 | 4      | >95%            |
| 10K   | L2       | Exact           | 100 | -      | >98%            |
| 10K   | Cosine   | RaBitQ(cached)  | 100 | 4      | >95%            |
| 100K  | L2       | Exact           | 100 | -      | >95%            |
| 100K  | Cosine   | RaBitQ(cached)  | 100 | 4      | >90%            |

## Data Preparation

SIFT vectors are normalized for Cosine distance tests:
```
v_norm = v / ||v||
```

This ensures fair comparison since:
- L2 distance works on raw SIFT vectors
- Cosine distance requires unit vectors for optimal performance

## Metrics Collected

| Metric | Description |
|--------|-------------|
| Build Time | Time to construct HNSW index |
| Build Throughput | Vectors indexed per second |
| QPS | Queries per second |
| Avg Latency | Mean search latency (ms) |
| P50 Latency | Median search latency (ms) |
| P99 Latency | 99th percentile latency (ms) |
| Recall@10 | Fraction of true top-10 found |
| Disk Usage | RocksDB storage size |

## Configuration

### HNSW Parameters
- M: 16 (connections per layer)
- M_max: 32
- ef_construction: 200
- ef_search: 100 (varied for tuning)

### RaBitQ Parameters
- bits_per_dim: 1 (32x compression)
- rerank_factor: 4 (fetch 4x candidates)
- use_cache: true (BinaryCodeCache)

### Hardware
- Block cache: 256 MB
- Temp directory for each run

## Running Experiments

### Quick Validation (1K scale)
```bash
./run_experiments.sh 1000
```

### Full Benchmark Suite
```bash
./run_experiments.sh 1000 10000 100000
```

### Individual Runs
```bash
# L2 (exact) - standard
cargo run --release --example vector2 -- \
  --dataset sift10k --num-vectors 1000 --num-queries 100 \
  --k 10 --ef 100

# Cosine (RaBitQ) - with BinaryCodeCache
cargo run --release --example vector2 -- \
  --dataset sift10k --num-vectors 1000 --num-queries 100 \
  --k 10 --ef 100 --rabitq-cached --rerank-factor 4 --cosine
```

## Expected Results

### Build Throughput
| Scale | Expected (vec/s) |
|-------|------------------|
| 1K    | ~3000-5000       |
| 10K   | ~1000-2000       |
| 100K  | ~150-250         |

### Search QPS
| Scale | L2 (Exact) | Cosine (RaBitQ) |
|-------|------------|-----------------|
| 1K    | ~5000      | ~4000           |
| 10K   | ~1000      | ~1500           |
| 100K  | ~200       | ~300-400        |

Note: RaBitQ should outperform exact search at larger scales due to
Hamming distance filtering reducing the candidate set.

## Analysis

Results are stored in `results/` with naming convention:
```
searchconfig_{scale}_{distance}_{timestamp}.txt
```

Summary table generated in:
```
results/searchconfig_summary.md
```
