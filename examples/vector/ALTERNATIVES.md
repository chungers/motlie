# Alternative Architectures for Vector Search

This document explores alternative approaches to vector search that could complement or replace the current phased design ([POC](./POC.md) → [HNSW2](./HNSW2.md) → [IVFPQ](./IVFPQ.md) → [HYBRID](./HYBRID.md)).

**Last Updated**: 2025-12-24

---

## Current Approach Summary

The current phased design is **graph-centric**:

| Phase | Approach | Focus |
|-------|----------|-------|
| POC | HNSW/Vamana on RocksDB | Proof of concept |
| HNSW2 | Roaring bitmap HNSW | Build throughput |
| IVFPQ | GPU-accelerated | Search QPS (optional) |
| HYBRID | HNSW + PQ | Billion-scale |

**Core assumption**: Graph-based navigation (HNSW) is the optimal structure for high-recall vector search.

This document challenges that assumption with two alternatives from recent research.

---

## Critical Constraint: No Pre-Training Data (DATA-1)

**From [REQUIREMENTS.md](./REQUIREMENTS.md) Section 5.4:**

> motlie_db is a general-purpose graph database. Vector search must work with any embedding type without prior knowledge of the data distribution.

| Requirement | Description |
|-------------|-------------|
| **DATA-1** | No representative training data available |
| **DATA-2** | Unknown data distribution (clustered vs uniform) |
| **DATA-3** | All algorithms must work incrementally (online) |

### Impact on Alternatives

| Approach | Training Required | DATA-1 Compliant | Verdict |
|----------|-------------------|------------------|---------|
| **Current (HNSW/Vamana)** | No | **Yes** | Viable |
| **HNSW2 (roaring bitmaps)** | No | **Yes** | Viable |
| **HYBRID (PQ)** | Yes (codebooks) | **No** | Needs modification |
| **Alt A: SPFresh/SPANN** | Yes (centroids) | **No** | Not viable without changes |
| **Alt B: ScaNN** | Yes (anisotropic) | **No** | Not viable |
| **Alt B: RaBitQ** | No (random rotation) | **Yes** | **Viable** |

### Why RaBitQ Works Without Training

Unlike PQ which requires k-means clustering, RaBitQ uses a **random orthonormal rotation matrix**:

```
Traditional PQ (training required):
  1. Collect representative vectors
  2. Run k-means on each subspace → codebooks
  3. Encode new vectors using learned codebooks
  Problem: Step 1-2 require training data

RaBitQ (training-free):
  1. Generate random orthonormal matrix R (dimension D×D)
  2. For each vector v: rotated = R × v
  3. Binary code = sign(rotated)  (1 if positive, 0 if negative)
  No training data required!
```

**Mathematical Guarantee** (from [RaBitQ paper](https://arxiv.org/abs/2405.12497)):
> "The estimation error is O(1/√D), which is asymptotically optimal."

For 128D vectors: error ≈ 8.8%. For 1024D: error ≈ 3.1%.

The error bound holds for **any** data distribution because it depends only on the dimension, not the data.

### Revised Alternative Viability

Given DATA-1 constraint:

| Alternative | Original Verdict | DATA-1 Verdict | Action |
|-------------|------------------|----------------|--------|
| Alt A: SPFresh | Promising | **Not viable** | Requires training centroids |
| Alt B: ScaNN | Not viable | Not viable | Requires learned quantization |
| Alt B: RaBitQ | **Viable** | **Viable** | Training-free, use this |
| Current: [HYBRID](./HYBRID.md) | Promising | **Needs modification** | Replace PQ with RaBitQ |

**Conclusion**: The **[HNSW2](./HNSW2.md) + RaBitQ** combination is the clear winner for DATA-1 compliance.

---

## Alternative A: SPFresh-Inspired Cluster-Centric Architecture

> **DATA-1 Warning**: This approach requires training centroids via k-means, violating DATA-1.
> Included for completeness, but **not recommended** for motlie_db.

### Research Background

[SPFresh: Incremental In-Place Update for Billion-Scale Vector Search](https://arxiv.org/abs/2410.14452) (SOSP 2023) introduces a fundamentally different approach:

- **Cluster-centric** rather than graph-centric
- **LIRE protocol** (Lightweight Incremental REbalancing) for streaming updates
- Achieves stable P99.9 latency during continuous updates

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                   SPFresh/SPANN Architecture                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              In-Memory Centroid Index (SPTAG)                │    │
│  │  • Graph-based index over cluster centroids only             │    │
│  │  • Typically 1M centroids for 1B vectors                     │    │
│  │  • ~50-100 MB RAM                                            │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼ Find top-k centroids                  │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              Disk-Based Posting Lists (RocksDB)              │    │
│  │  • Each centroid → list of vectors in that cluster           │    │
│  │  • Vectors stored with PQ codes for fast scanning            │    │
│  │  • ~1000 vectors per posting (average)                       │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼ LIRE Protocol                         │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              Incremental Update Modules                       │    │
│  │  • In-place Updater: Append vectors to postings               │    │
│  │  • Local Rebuilder: Split oversized postings                  │    │
│  │  • Block Controller: Manage disk layout                       │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Innovation: LIRE Protocol

Unlike graph-based approaches that must update edges on every insert:

1. **Insert**: Append to nearest centroid's posting list (O(1) amortized)
2. **Rebalance**: When posting exceeds threshold, split and reassign boundary vectors
3. **Key insight**: Only ~1% of vectors need reassignment during splits

From the SPFresh paper:
> "LIRE achieves low-overhead vector updates by only reassigning vectors at the boundary between partitions, where in a high-quality vector index the amount of such vectors is deemed small."

### Comparison with Current Design

| Aspect | Current (HNSW-centric) | SPFresh (Cluster-centric) |
|--------|------------------------|---------------------------|
| **Insert Model** | Per-insert graph update | Append to posting + async rebalance |
| **Memory** | Graph edges (~50 bytes/vec) | Centroids only (~50-100 MB total) |
| **Consistency** | Immediate (with flush) | Eventually consistent (LIRE async) |
| **Recall** | 95%+ via navigation | 95%+ via multi-probe |
| **P99 Stability** | Varies with graph size | Stable during updates |
| **Deletion** | Complex orphan repair | Simple posting removal |

### Requirements Alignment

| Requirement | Target | SPFresh Potential |
|-------------|--------|-------------------|
| **THR-1** (5K inserts/s) | 5,000/s | **10,000+/s** (append-only) |
| **THR-3** (500 QPS) | 500 | 500-1,000 (posting scan) |
| **REC-1** (95% recall) | > 95% | 95%+ (with nprobe tuning) |
| **SCALE-1** (1B vectors) | 1B | Designed for 1B+ |
| **LAT-4** (10ms P99 insert) | < 10ms | < 1ms (async) |

### Advantages Over Current Design

1. **Simpler streaming updates**: No graph edge maintenance
2. **Predictable P99**: No graph traversal variability
3. **Natural batching**: Postings are already batched by cluster
4. **Filtering**: Cluster-level filtering is efficient (vs. HNSW fragmentation)

### Disadvantages

1. **Initial training required**: Must train centroids on representative data
2. **Distribution shift**: Requires periodic rebalancing if data distribution changes
3. **Lower peak recall**: Graph-based may achieve higher recall with same compute
4. **Less mature for RocksDB**: SPANN designed for custom storage

### Engineering Effort

| Component | Effort | Risk |
|-----------|--------|------|
| SPTAG centroid index | 1-2 weeks | Medium (C++ interop) |
| Posting list storage | 1 week | Low (RocksDB CFs) |
| LIRE split/merge | 2-3 weeks | Medium |
| Block controller | 1 week | Low |
| **Total** | **5-7 weeks** | Medium |

---

## Alternative B: ScaNN-Style Learned Quantization with RaBitQ

### Research Background

Two key innovations from Google and NTU:

1. [ScaNN: Accelerating Large-Scale Inference with Anisotropic Vector Quantization](https://arxiv.org/abs/1908.10396) (ICML 2020)
   - Learned quantization that penalizes parallel error more than orthogonal
   - 2x speedup over prior quantization methods

2. [RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound](https://arxiv.org/abs/2405.12497) (SIGMOD 2024)
   - 1-bit quantization with theoretical guarantees
   - O(1/√D) error bound (asymptotically optimal)
   - Achieves >95% recall with reranking

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│              ScaNN + RaBitQ Hybrid Architecture                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │         Training Phase (Offline, One-Time)                    │    │
│  │  • Learn anisotropic codebooks from representative data       │    │
│  │  • Compute RaBitQ rotation matrices                           │    │
│  │  • Build IVF centroids (optional coarse partitioning)        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              Three-Tier Storage                               │    │
│  │                                                               │    │
│  │  Tier 1: RaBitQ Binary Codes (1 bit/dim)                      │    │
│  │  ├── 128D → 16 bytes per vector                               │    │
│  │  ├── 1B vectors → 16 GB                                       │    │
│  │  └── SIMD popcount for distance approximation                 │    │
│  │                                                               │    │
│  │  Tier 2: ScaNN PQ Codes (Optional, 8 bytes/vector)            │    │
│  │  ├── Anisotropic quantization for better approximation        │    │
│  │  └── Used for mid-quality reranking                           │    │
│  │                                                               │    │
│  │  Tier 3: Full Vectors (On Disk, 512 bytes/vector)             │    │
│  │  ├── Loaded for final reranking                               │    │
│  │  └── Only top-100 candidates from Tier 1/2                    │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Innovations

**Anisotropic Loss (ScaNN)**:

Traditional PQ minimizes reconstruction error uniformly. ScaNN observes:
> "For a given query, the database points that have the largest inner products are more relevant."

Therefore, error parallel to the vector direction matters more than orthogonal error. This yields better ranking with same compression.

**RaBitQ Theoretical Bound**:

RaBitQ proves that for D-dimensional vectors:
> "The estimation error is O(1/√D), which is asymptotically optimal."

For 128D vectors: error ≈ 8.8%. For 1024D: error ≈ 3.1%.

This means higher-dimensional embeddings get better compression "for free."

### Comparison with Current Design

| Aspect | Current ([HYBRID.md](./HYBRID.md) PQ) | ScaNN + RaBitQ |
|--------|------------------------|----------------|
| **Compression** | 8 bytes/vector (PQ) | 16 bytes (RaBitQ) or 8 bytes (PQ) |
| **Theoretical guarantee** | None | O(1/√D) error bound |
| **Training** | Standard PQ | Anisotropic + rotation |
| **High-dim advantage** | Same compression ratio | Better with higher D |
| **Memory at 1B (128D)** | 8 GB | 16 GB (RaBitQ) or 8 GB (hybrid) |
| **Recall mechanism** | HNSW navigation | Binary scan + rerank |

### Requirements Alignment

| Requirement | Target | ScaNN+RaBitQ Potential |
|-------------|--------|------------------------|
| **THR-1** (5K inserts/s) | 5,000/s | **50,000+/s** (append + encode) |
| **THR-3** (500 QPS) | 500 | **10,000+** (SIMD popcount) |
| **REC-1** (95% recall) | > 95% | 95%+ (with 100 candidates) |
| **SCALE-1** (1B vectors) | 1B | 1B+ (16 GB for binary) |
| **STOR-4** (PQ compression) | Implemented | **32x better** (1-bit vs 8-byte) |
| **STOR-5** (SIMD) | Not implemented | Native (popcount) |

### Advantages Over Current Design

1. **Extreme compression**: 1 bit/dimension vs 8 bytes/vector
2. **Theoretical guarantees**: Provable error bounds
3. **Insert simplicity**: Encode and append (no graph updates)
4. **SIMD-native**: Binary operations are fastest possible
5. **Industry adoption**: Milvus 2.6, Elasticsearch 8.16, LanceDB

From [Milvus blog](https://milvus.io/blog/bring-vector-compression-to-the-extreme-how-milvus-serves-3%C3%97-more-queries-with-rabitq.md):
> "Milvus serves 3× the QPS while maintaining a comparable level of accuracy."

### Disadvantages

1. **Training dependency**: Must train on representative data
2. **Reranking required**: Pure binary gives ~70% recall
3. **Update overhead**: New vectors need encoding
4. **Navigation penalty**: No graph → must scan more candidates

### Engineering Effort

| Component | Effort | Risk |
|-----------|--------|------|
| RaBitQ encoding/rotation | 1-2 weeks | Low (algorithm is simple) |
| Binary distance SIMD | 1 week | Low (popcount intrinsics) |
| Reranking pipeline | 1 week | Low |
| Training infrastructure | 2 weeks | Medium |
| Integration with RocksDB | 1 week | Low |
| **Total** | **6-8 weeks** | Low-Medium |

---

## Comparative Analysis

### Architecture Comparison

| Dimension | Current (HYBRID) | Alt A (SPFresh) | Alt B (ScaNN+RaBitQ) |
|-----------|------------------|-----------------|----------------------|
| **Index Type** | Graph (HNSW) | Cluster (IVF) | Flat + Quantization |
| **Navigation** | Multi-layer graph | Centroid lookup | Linear scan + prune |
| **Compression** | PQ (8 bytes) | PQ (8 bytes) | RaBitQ (16 bytes) or 1-bit |
| **Memory Model** | Hybrid | Mostly disk | Mostly memory |
| **Update Model** | Async graph | LIRE rebalance | Append + encode |
| **Filtering** | Post-filter | Cluster pre-filter | Bitmap intersection |

### Performance Projections

| Metric | HYBRID | SPFresh | ScaNN+RaBitQ |
|--------|--------|---------|--------------|
| **Insert throughput** | 5-10K/s | 10-20K/s | 50K+/s |
| **Search QPS (1B)** | 100-200 | 200-500 | 5,000-10,000 |
| **Memory (1B, 128D)** | 58 GB | 10 GB | 16-24 GB |
| **Recall@10** | > 95% | > 95% | > 95% |
| **P99 Insert** | < 10ms | < 1ms | < 0.1ms |
| **P99 Search** | < 100ms | < 50ms | < 10ms |

### Trade-off Analysis

```
                        Memory Efficiency
                              ▲
                              │
              SPFresh ●       │       ● ScaNN+RaBitQ
                              │
                              │
        ────────────────────────────────────────► Update Speed
                              │
                              │
                    HYBRID ●  │
                              │
                              │
```

```
                         Recall Quality
                              ▲
                              │
                   HYBRID ●   │
                              │
              SPFresh ●       │       ● ScaNN+RaBitQ (with rerank)
                              │
        ────────────────────────────────────────► Query Throughput
                              │
                              │
                              │
```

---

## Recommendation Matrix

### By Primary Use Case

| Use Case | Recommended | Rationale |
|----------|-------------|-----------|
| **RAG (UC-4)** | HYBRID | Highest recall, moderate QPS sufficient |
| **Image Search (UC-2)** | ScaNN+RaBitQ | High-dim embeddings benefit from RaBitQ |
| **Real-time Recs (UC-3)** | SPFresh | Best insert latency, stable P99 |
| **Semantic Search (UC-1)** | HYBRID or SPFresh | Both achieve >95% recall |

### By Constraint Priority

| Priority | Recommended | Rationale |
|----------|-------------|-----------|
| **Memory-constrained** | SPFresh | Smallest footprint (centroids only in RAM) |
| **Latency-constrained** | ScaNN+RaBitQ | SIMD binary is fastest |
| **Recall-constrained** | HYBRID | Graph navigation is most reliable |
| **Update-constrained** | ScaNN+RaBitQ | Append-only, no rebalancing |

### Engineering Pragmatism

| Factor | HYBRID | SPFresh | ScaNN+RaBitQ |
|--------|--------|---------|--------------|
| **Builds on POC** | Yes | Partially | Yes |
| **RocksDB native** | Yes | Requires adaptation | Yes |
| **Rust ecosystem** | roaring-rs | SPTAG (C++) | Native Rust possible |
| **Production examples** | hnswlib, pgvector | Azure Cosmos DB | Milvus, Elasticsearch |
| **Risk** | Low | Medium | Low |

---

## Proposed Hybrid Strategy

Rather than choosing one approach, consider a **staged hybrid**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Recommended Hybrid Evolution                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Phase 1: Current POC                     ← COMPLETED                │
│  └── HNSW/Vamana on RocksDB, Flush/Transaction API                   │
│                                                                      │
│  Phase 2: HNSW2 + RaBitQ (NEW)            ← RECOMMENDED              │
│  ├── Roaring bitmap edges (from HNSW2.md)                            │
│  ├── RaBitQ binary codes for candidate filtering                     │
│  └── Benefits: 3x QPS improvement, minimal risk                      │
│                                                                      │
│  Phase 3: SPFresh Hybrid (NEW)            ← OPTIONAL                 │
│  ├── SPANN clusters for coarse navigation                            │
│  ├── HNSW within-cluster for fine navigation                         │
│  └── Benefits: Best of both worlds                                   │
│                                                                      │
│  Phase 4: Full Production (HYBRID.md)     ← UNCHANGED                │
│  └── Add async updater, PQ reranking                                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Document Links**: [POC.md](./POC.md) | [HNSW2.md](./HNSW2.md) | [HYBRID.md](./HYBRID.md)

### Phase 2 Modification: [HNSW2](./HNSW2.md) + RaBitQ

Add RaBitQ binary codes alongside HNSW2:

```rust
// New column family
CF: binary_codes
Key:   node_id (u32, 4 bytes)
Value: RaBitQ binary code (D/8 bytes, e.g., 16 bytes for 128D)

// Search modification
fn search(query: &[f32], k: usize) -> Vec<(f32, u32)> {
    // 1. Encode query to binary
    let query_bits = rabitq_encode(query);

    // 2. Scan binary codes for top-1000 candidates (SIMD popcount)
    let candidates = binary_scan(&query_bits, 1000);

    // 3. Navigate HNSW from candidates (existing logic)
    let results = hnsw_search_from(candidates, query, k * 2);

    // 4. Rerank with exact distances
    rerank_exact(results, k)
}
```

**Benefit**: 3x QPS with minimal code changes.

---

## References

### Papers

1. [Accelerating Large-Scale Inference with Anisotropic Vector Quantization](https://arxiv.org/abs/1908.10396) - ScaNN, ICML 2020
2. [SPFresh: Incremental In-Place Update for Billion-Scale Vector Search](https://arxiv.org/abs/2410.14452) - SOSP 2023
3. [RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound](https://arxiv.org/abs/2405.12497) - SIGMOD 2024
4. [FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search](https://arxiv.org/pdf/2105.09613) - Microsoft Research
5. [Extended-RaBitQ: Practical and Asymptotically Optimal Quantization](https://arxiv.org/html/2409.09913v1) - SIGMOD 2025

### Industry Implementations

- [Microsoft DiskANN](https://github.com/microsoft/DiskANN) - Graph-based, streaming support
- [Google ScaNN](https://github.com/google-research/google-research/tree/master/scann) - Learned quantization
- [Milvus RaBitQ Support](https://milvus.io/blog/bring-vector-compression-to-the-extreme-how-milvus-serves-3%C3%97-more-queries-with-rabitq.md) - Binary quantization
- [Elasticsearch BBQ](https://www.elastic.co/search-labs/blog/better-binary-quantization-lucene-elasticsearch) - RaBitQ derivative
- [Azure Cosmos DB DiskANN](https://devblogs.microsoft.com/cosmosdb/azure-cosmos-db-with-diskann-part-4-stable-vector-search-recall-with-streaming-data/) - Production streaming

### Comparison Resources

- [Milvus: IVF vs HNSW](https://milvus.io/blog/understanding-ivf-vector-index-how-It-works-and-when-to-choose-it-over-hnsw.md)
- [AWS: k-NN Algorithm Choice for Billion Scale](https://aws.amazon.com/blogs/big-data/choose-the-k-nn-algorithm-for-your-billion-scale-use-case-with-opensearch/)
- [LanceDB RaBitQ](https://lancedb.com/blog/feature-rabitq-quantization/)

---

*Generated: 2025-12-24*
*Purpose: Research exploration - not for implementation*
