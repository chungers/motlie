# Vector2: Phase 2 HNSW Benchmark

This example benchmarks the Phase 2 HNSW implementation in `motlie_db::vector` using
RocksDB-backed storage with RoaringBitmap edges.

## Quick Start

```bash
# Run with random data
cargo run --release --example vector2 -- --num-vectors 10000 --num-queries 100 --k 10

# Run with SIFT10K dataset
cargo run --release --example vector2 -- --dataset sift10k --num-vectors 10000 --k 10

# Run with SIFT1M dataset (larger scale)
cargo run --release --example vector2 -- --dataset sift1m --num-vectors 100000 --k 10
```

## Benchmark Results

### Configuration

All benchmarks run with:
- HNSW parameters: M=16, M_max=32, ef_construction=200
- Search parameters: ef=100, k=10
- Vector dimension: 128 (SIFT)
- Platform: Linux aarch64 (NEON SIMD)

### Results Summary

| Dataset | Vectors | Build Time | Build Rate | Recall@10 | QPS | P50 Lat | P99 Lat | Disk |
|---------|---------|------------|------------|-----------|-----|---------|---------|------|
| SIFT10K | 1,000 | 0.86s | 1,161 vec/s | 100.0% | 1,483 | 0.67ms | 0.87ms | 2 MB |
| SIFT10K | 10,000 | 20.54s | 487 vec/s | 100.0% | 522 | 1.97ms | 2.82ms | 22 MB |
| SIFT1M | 100,000 | 453.40s | 221 vec/s | 88.4% | 282 | 3.44ms | 5.75ms | 241 MB |
| SIFT1M | 1,000,000 | 16,133s | 62 vec/s | **99.7%** | 47 | 21.05ms | 37.30ms | 2.85 GB |

### Comparison with Baseline (Old HNSW)

| Scale | Old Build Rate | New Build Rate | Old Recall | New Recall | Improvement |
|-------|----------------|----------------|------------|------------|-------------|
| 1M | 39.9 vec/s | 62.0 vec/s | 95.3% | **99.7%** | **1.55x faster, +4.4% recall** |

### Analysis

1. **Recall**: The implementation achieves excellent recall on smaller datasets (100% at 1K-10K vectors).
   At 100K scale, recall drops to 88.4% which is reasonable for HNSW with ef=100 search parameter.
   Increasing ef would improve recall at the cost of latency.

2. **Build Throughput**: Indexing speed decreases as dataset size increases, which is expected
   for HNSW due to the increasing graph connectivity requirements.

3. **Query Latency**: Sub-5ms P99 latency at 100K vectors demonstrates good search performance
   with RocksDB storage.

4. **Disk Usage**: ~2.4 KB/vector at 100K scale, which includes vectors (512 bytes) + HNSW edges
   + metadata.

## Detailed Results

Raw benchmark outputs are stored in the `results/` directory with phase prefixes
for tracking performance evolution across ROADMAP phases:

- `results/phase2_sift10k_1k_vectors.txt` - Phase 2: SIFT10K with 1,000 vectors
- `results/phase2_sift10k_10k_vectors.txt` - Phase 2: SIFT10K with 10,000 vectors
- `results/phase2_sift1m_100k_vectors.txt` - Phase 2: SIFT1M with 100,000 vectors
- `results/phase2_sift1m_1m_vectors.txt` - Phase 2: SIFT1M with 1,000,000 vectors

## Command Line Options

```
USAGE:
    vector2 [OPTIONS]

OPTIONS:
    --db-path <PATH>           Database path (uses temp dir if not specified)
    --dataset <NAME>           Dataset: sift1m, sift10k, random [default: random]
    --num-vectors <N>          Number of vectors to index [default: 10000]
    --num-queries <N>          Number of queries to run [default: 100]
    --k <N>                    Number of nearest neighbors [default: 10]
    --ef <N>                   Search beam width [default: 100]
    --m <N>                    HNSW M parameter [default: 16]
    --ef-construction <N>      ef_construction parameter [default: 200]
    --compare                  Compare with old implementation
    -v, --verbose              Verbose output
```

## Architecture

This benchmark exercises the following components:

- `motlie_db::vector::HnswIndex` - HNSW insert and search operations
- `motlie_db::vector::NavigationCache` - Layer traversal optimization
- `motlie_db::vector::Storage` - RocksDB-backed vector storage
- `motlie_db::vector::merge::EdgeOp` - RocksDB merge operators for edges

The implementation stores:
- Vectors in the `vector/vectors` column family
- HNSW edges as RoaringBitmaps in the `vector/edges` column family
- Graph metadata in the `vector/graph_meta` column family
- Vector metadata (layer assignments) in the `vector/vec_meta` column family
