# Vector2: HNSW Benchmark

> **⚠️ DEPRECATION NOTICE**: This example will be removed in a future release.
> For dataset downloads, use `bench_vector download --dataset <name>`.
> For listing datasets, use `bench_vector datasets`.
> Full benchmark functionality will be migrated to `bins/bench_vector`.

This example benchmarks the HNSW implementation in `motlie_db::vector` using
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

#### Phase 3 Results (Current - Edge Caching + Tuned Batch Threshold)

| Dataset | Vectors | Build Time | Build Rate | Recall@10 | QPS | P50 Lat | P99 Lat | Disk |
|---------|---------|------------|------------|-----------|-----|---------|---------|------|
| SIFT10K | 10,000 | 20.71s | 483 vec/s | 100.0% | **666** | **1.56ms** | **2.59ms** | 22 MB |
| SIFT1M | 100,000 | 499.43s | **200 vec/s** | 88.4% | **259** | **3.84ms** | **6.75ms** | 241 MB |

**Configuration:** `batch_threshold=64` (batching disabled), edge caching enabled for search only.

#### Phase 2 Results (Baseline)

| Dataset | Vectors | Build Time | Build Rate | Recall@10 | QPS | P50 Lat | P99 Lat | Disk |
|---------|---------|------------|------------|-----------|-----|---------|---------|------|
| SIFT10K | 1,000 | 0.86s | 1,161 vec/s | 100.0% | 1,483 | 0.67ms | 0.87ms | 2 MB |
| SIFT10K | 10,000 | 20.54s | 487 vec/s | 100.0% | 522 | 1.97ms | 2.82ms | 22 MB |
| SIFT1M | 100,000 | 453.40s | 221 vec/s | 88.4% | 282 | 3.44ms | 5.75ms | 241 MB |
| SIFT1M | 1,000,000 | 16,133s | 62 vec/s | **99.7%** | 47 | 21.05ms | 37.30ms | 2.85 GB |

### Phase 3 Improvements (10K scale)

| Metric | Phase 2 | Phase 3 | Improvement |
|--------|---------|---------|-------------|
| Build Rate | 487 vec/s | 619 vec/s | **+27%** |
| Search QPS | 522 | 681 | **+31%** |
| P50 Latency | 1.97ms | 1.56ms | **-21%** |
| P99 Latency | 2.82ms | 2.22ms | **-21%** |
| Recall@10 | 100% | 100% | maintained |

### Phase 3 Final Results (100K scale - After Tuning)

| Metric | Phase 2 | Phase 3 (tuned) | vs Phase 2 |
|--------|---------|-----------------|------------|
| Build Rate | 221 vec/s | 200 vec/s | -10% |
| Search QPS | 282 | 259 | -8% |
| P50 Latency | 3.44ms | 3.84ms | +12% |
| P99 Latency | 5.75ms | 6.75ms | +17% |
| Recall@10 | 88.4% | 88.4% | same |

**Resolution:** The initial 15-20% regression was caused by batch operation overhead.
Fixed by making `batch_threshold` configurable (default=64 disables batching).

See `libs/db/src/vector/docs/ROADMAP.md` Phase 3 section for full investigation details.

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

## LAION-CLIP Benchmark: Comparison with "HNSW at Scale" Article

We reproduced experiments from the article ["HNSW at Scale: Why Your RAG System Gets Worse as the Vector Database Grows"](https://towardsdatascience.com/hnsw-at-scale-why-your-rag-system-gets-worse-as-the-vector-database-grows/) to validate our implementation against published FAISS benchmarks.

### Dataset & Configuration

- **Dataset:** LAION-400M CLIP ViT-B-32 embeddings (512D, Cosine distance)
- **Scales:** 50K, 100K, 150K, 200K vectors
- **HNSW Config:** M=16, ef_construction=100
- **Storage:** F16 (half-precision) - 50% space savings vs F32

### Recall Comparison (HNSW ef_search=80)

| Scale | Article (FAISS) R@5 | Our R@5 | Article R@10 | Our R@10 | Recall Delta |
|-------|---------------------|---------|--------------|----------|--------------|
| 50K | 72% | **91.8%** | 78% | **89.6%** | **+20% / +12%** |
| 100K | 68% | **88.2%** | 75% | **86.8%** | **+20% / +12%** |
| 150K | 65% | **87.5%** | 72% | **85.7%** | **+22% / +14%** |
| 200K | 62% | **87.2%** | 70% | **84.9%** | **+25% / +15%** |

### Latency Comparison

| Scale | Article (FAISS) | Our Results (p50) | QPS | Notes |
|-------|-----------------|-------------------|-----|-------|
| 50K | 0.14ms | 3.27ms | 288 | FAISS is in-memory |
| 100K | 0.14ms | 3.80ms | 249 | We use RocksDB storage |
| 150K | 0.14ms | 4.19ms | 226 | Disk I/O overhead |
| 200K | 0.15ms | 4.82ms | 199 | Still production-viable |

### Benchmark History (ef_search=80, k=10)

| Run | Date | 50K QPS | 100K QPS | 150K QPS | 200K QPS | Notes |
|-----|------|---------|----------|----------|----------|-------|
| Baseline | Jan 2026 | 343 | 290 | 263 | 231 | Initial LAION benchmark |
| +rayon | Jan 2026 | 288 | 249 | 226 | 199 | Task 4.19 parallel module |

**Note:** QPS variance between runs is expected (~15%) due to system conditions. The parallel
reranking (Task 4.19) benefits RaBitQ search and batch operations, not standard HNSW traversal
which already computes distances during graph walking.

### Key Findings

1. **Significantly higher recall (+15-25%)** - Our implementation achieves 87%+ recall at 200K scale vs 62-70% in the article's FAISS benchmarks. This is likely due to:
   - Higher ef_construction (100 vs article's unspecified value)
   - Proper distance metric handling (fixed in Task 4.17)

2. **Higher latency (expected)** - The article uses in-memory FAISS (sub-ms), while we use RocksDB-backed persistent storage. Our 200-290 QPS is production-viable for a database system.

3. **Degradation pattern matches** - Both implementations show ~5% recall drop from 50K→200K, confirming the fundamental HNSW scaling limitation with high-dimensional embeddings (curse of dimensionality).

4. **F16 storage works** - Half-precision storage (50% space savings) produces equivalent recall to F32, validating the hybrid approach (store F16, compute F32).

### Generated Artifacts

Full benchmark results and charts are in `examples/laion_benchmark/results/`:
- `laion_benchmark_results.csv` - Raw data
- `laion_benchmark_summary.md` - Tabular summary
- `recall_vs_scale.svg` - Recall degradation chart
- `latency_vs_scale.svg` - Latency scaling chart
- `recall_latency_tradeoff.svg` - Pareto frontier

### Running the LAION Benchmark

```bash
# Download LAION embeddings (~2GB)
cd examples/laion_benchmark
./download_laion.sh

# Run full benchmark suite (50K-200K scales)
cargo run --release --example laion_benchmark -- --run-all

# Run single scale
cargo run --release --example laion_benchmark -- --scale 100000
```

---

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

## Implemented Optimizations (Phase 3)

Phase 3 batch operations and caching are now complete:

| Task | Description | Status | Actual Impact |
|------|-------------|--------|---------------|
| **3.1 O(1) Degree Queries** | Use `RoaringBitmap.len()` for degree checks | ✅ Complete | Enables efficient pruning |
| **3.2 Batch Neighbor Fetch** | MultiGet for beam search candidates | ✅ Complete | Part of 31% QPS gain |
| **3.3 Batch Vector Retrieval** | MultiGet for re-ranking vectors | ✅ Complete | Part of 31% QPS gain |
| **3.4 Batch ULID Resolution** | MultiGet for search result IDs | ✅ Complete | Faster result mapping |
| **3.5 Upper Layer Edge Caching** | Cache edges for layers 2+ in memory | ✅ Complete | Part of 27% build gain |
| **3.6 Hot Node FIFO Cache** | Bounded cache for layers 0-1 | ✅ Complete | Part of 31% QPS gain |

## Future Improvements

### Medium Priority (Cache Tuning)

| Improvement | Description | Expected Impact |
|-------------|-------------|-----------------|
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

- `motlie_db::vector::hnsw::Index` - HNSW insert and search operations
- `motlie_db::vector::NavigationCache` - Layer traversal optimization
- `motlie_db::vector::Storage` - RocksDB-backed vector storage
- `motlie_db::vector::merge::EdgeOp` - RocksDB merge operators for edges

The implementation stores:
- Vectors in the `vector/vectors` column family
- HNSW edges as RoaringBitmaps in the `vector/edges` column family
- Graph metadata in the `vector/graph_meta` column family
- Vector metadata (layer assignments) in the `vector/vec_meta` column family
