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
    --cache-size-mb <N>        Block cache size in MB [default: 256]
    --compare                  Compare with old implementation
    -v, --verbose              Verbose output
```

## Future Improvements

The following optimizations are planned to improve search QPS. Items are prioritized based on
impact and alignment with the ROADMAP phases.

### High Priority (Phase 3: Batch Operations)

These are the next planned improvements from `libs/db/src/vector/ROADMAP.md`:

| Task | Description | Expected Impact | Effort |
|------|-------------|-----------------|--------|
| **3.1 O(1) Degree Queries** | Use `RoaringBitmap.len()` for degree checks | 95% prune overhead reduction | 0.5 day |
| **3.2 Batch Neighbor Fetch** | MultiGet for beam search candidates | 3-5x search speedup | 1 day |
| **3.3 Batch Vector Retrieval** | MultiGet for re-ranking vectors | 2x re-ranking speedup | 0.5 day |
| **3.4 Batch ULID Resolution** | MultiGet for search result IDs | Faster result mapping | 0.5 day |

### Medium Priority (Phase 2 Deferred + Cache Tuning)

| Improvement | Description | Expected Impact |
|-------------|-------------|-----------------|
| **Upper Layer Edge Caching** | Cache edges for layers 2+ in memory | Reduce I/O for upper traversal |
| **LRU Vector Cache** | In-memory cache for hot layer-0 vectors | 2-3x QPS for repeated queries |
| **Larger Block Cache** | Increase from 256MB to 2-4GB | 2-5x QPS at 1M+ scale (see below) |
| **Pin Hot Column Families** | Pin L0 blocks and bloom filters | 20-50% latency reduction |

### Cache Size Analysis

**Key insight:** Cache size impacts **search QPS**, not build speed. Build is write-bound (~62 vec/s
at 1M scale = ~4.5 hours). Search is read-bound and benefits from larger cache when data exceeds
cache size.

| Scale | Disk Size | 256MB Cache | 2GB+ Cache | Speedup |
|-------|-----------|-------------|------------|---------|
| 100K | 241 MB | 267 QPS | 279 QPS | ~5% (data fits in either) |
| 1M | 2.85 GB | 47 QPS | TBD | Expected 2-5x |

**Why 1M cache comparison is deferred:** Building a 1M index takes ~4.5 hours. We will re-run
this comparison after Phase 3 batch operations are implemented, which will:
1. Speed up build time via MultiGet during greedy search (target: 2-3x faster)
2. Speed up search via batch neighbor/vector fetches (target: 3-5x faster)

This allows us to measure cache impact on search with reasonable build times.

Use `--cache-size-mb` to configure:
```bash
# Large cache for production workloads
cargo run --release --example vector2 -- --dataset sift1m --num-vectors 1000000 --cache-size-mb 4096
```

### Lower Priority (Future Phases)

| Phase | Improvement | Description |
|-------|-------------|-------------|
| Phase 4 | **Product Quantization (RaBitQ)** | 128D → 32 bytes, 16x smaller vectors |
| Phase 4 | **Two-Stage Search** | Fast PQ scan + exact re-ranking |
| Future | **SIMD Distance (AVX-512)** | 4-8x faster distance computation |
| Future | **Memory-Mapped Vectors** | OS page cache for large datasets |

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
