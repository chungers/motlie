# IVF-PQ and GPU-Accelerated Vector Search

**Phase 3 - GPU-Accelerated Search (Optional)**

This document explores Inverted File with Product Quantization (IVF-PQ) and GPU-accelerated alternatives to our current graph-based HNSW/Vamana implementations, with a focus on scaling to 1 billion vectors.

**Last Updated**: 2025-12-24

---

## Document Hierarchy

```
REQUIREMENTS.md     ← Ground truth for all design decisions
    ↓
POC.md              ← Phase 1: Current implementation
    ↓
HNSW2.md            ← Phase 2: Optimized HNSW with roaring bitmaps
    ↓
IVFPQ.md (this)     ← Phase 3: GPU-accelerated search (you are here)
    ↓
HYBRID.md           ← Phase 4: Billion-scale production architecture
```

## Target Requirements

This phase focuses on achieving:

| Requirement | Target | Current (POC.md) | IVFPQ Target |
|-------------|--------|------------------|--------------|
| **THR-3** | > 500 QPS | 47 QPS | 10,000+ QPS (GPU) |
| **STOR-5** | SIMD distance | Not implemented | CAGRA GPU kernels |
| **STOR-4** | PQ compression | Not implemented | 512x compression |
| **UC-2** | Image similarity | 100M vectors | 100M-1B vectors |

See [REQUIREMENTS.md](./REQUIREMENTS.md) for full requirement definitions.

**Note**: This phase is optional and requires GPU infrastructure (NVIDIA CUDA). For CPU-only deployments, skip to [HYBRID.md](./HYBRID.md).

---

## Table of Contents

1. [Algorithm Overview](#algorithm-overview)
2. [Comparison with Graph-Based Methods](#comparison-with-graph-based-methods)
3. [Library Options](#library-options)
4. [Integration Architecture](#integration-architecture)
5. [Project Plan](#project-plan)
6. [Performance Projections](#performance-projections)
7. [IVF-PQ vs HNSW2: RocksDB Integration Analysis](#ivf-pq-vs-hnsw2-rocksdb-integration-analysis)
8. [IVF-PQ Performance: CPU vs GPU vs RocksDB](#ivf-pq-performance-cpu-vs-gpu-vs-rocksdb)
9. [Recommendations](#recommendations)

---

## Algorithm Overview

### IVF-Flat (Inverted File Index)

IVF-Flat partitions the vector space into clusters using k-means, then performs exhaustive search only within the most relevant clusters.

```
┌─────────────────────────────────────────────────────────────────┐
│                    IVF-Flat Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Training Phase:                                                │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  All Vectors ──► k-means ──► Cluster Centroids (nlist)  │   │
│   │                              C₁, C₂, C₃, ..., Cₙ        │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Index Structure:                                               │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│   │ Cluster 1   │  │ Cluster 2   │  │ Cluster n   │            │
│   │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │            │
│   │ │ vec₁    │ │  │ │ vec₅    │ │  │ │ vec₉    │ │            │
│   │ │ vec₂    │ │  │ │ vec₆    │ │  │ │ vec₁₀   │ │            │
│   │ │ vec₃    │ │  │ │ vec₇    │ │  │ │ vec₁₁   │ │            │
│   │ │ vec₄    │ │  │ │ vec₈    │ │  │ │ ...     │ │            │
│   │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │            │
│   └─────────────┘  └─────────────┘  └─────────────┘            │
│                                                                  │
│   Search Phase:                                                  │
│   Query ──► Find nprobe nearest centroids ──► Exhaustive search │
│             within those clusters                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key Parameters:**
- `nlist`: Number of clusters (typically √n to 16√n)
- `nprobe`: Number of clusters to search (higher = better recall, slower)

**Characteristics:**
- Exact distances within searched clusters
- Memory: O(n × d) - stores full vectors
- Build: O(n × d × iterations) for k-means
- Search: O(nprobe × n/nlist × d)

### IVF-PQ (Inverted File + Product Quantization)

IVF-PQ adds compression on top of IVF-Flat by encoding vectors using Product Quantization.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Product Quantization                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Original Vector (1024 dimensions):                             │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ [0.1, 0.3, 0.5, ..., 0.2, 0.4, 0.6, ..., 0.8, 0.9, ...] │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│   Split into M subvectors (M=8 for 1024-dim):                   │
│   ┌────────┐ ┌────────┐ ┌────────┐     ┌────────┐              │
│   │ Sub 1  │ │ Sub 2  │ │ Sub 3  │ ... │ Sub 8  │              │
│   │ 128-dim│ │ 128-dim│ │ 128-dim│     │ 128-dim│              │
│   └────────┘ └────────┘ └────────┘     └────────┘              │
│       │          │          │              │                    │
│       ▼          ▼          ▼              ▼                    │
│   ┌────────┐ ┌────────┐ ┌────────┐     ┌────────┐              │
│   │Code: 42│ │Code: 17│ │Code: 89│ ... │Code: 63│              │
│   │ 8 bits │ │ 8 bits │ │ 8 bits │     │ 8 bits │              │
│   └────────┘ └────────┘ └────────┘     └────────┘              │
│                                                                  │
│   Compressed: 1024 × 4 bytes = 4096 bytes ──► 8 bytes (512× !) │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Compression Example (1024-dim, M=8, nbits=8):**

| Format | Size per Vector | 1M Vectors | 1B Vectors |
|--------|-----------------|------------|------------|
| Full f32 | 4,096 bytes | 4 GB | 4 TB |
| PQ (M=8, 8-bit) | 8 bytes | 8 MB | 8 GB |
| Compression ratio | **512×** | - | - |

**Trade-offs:**
- Lossy compression reduces recall (mitigated by refinement)
- Fast approximate distance via lookup tables
- Memory-efficient: fits billion-scale in GPU memory

### CAGRA (CUDA ANNS Graph-based)

CAGRA is NVIDIA's GPU-optimized graph-based algorithm, designed from scratch for parallel execution.

```
┌─────────────────────────────────────────────────────────────────┐
│                    CAGRA vs HNSW                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   HNSW (CPU):                     CAGRA (GPU):                  │
│   ┌─────────────────────┐         ┌─────────────────────┐       │
│   │ Layer 2: sparse     │         │ Single optimized    │       │
│   │    ○─────────○      │         │ k-NN graph built    │       │
│   │                     │         │ via NN-Descent      │       │
│   │ Layer 1: medium     │         │                     │       │
│   │   ○───○───○───○     │         │ ┌─────────────────┐ │       │
│   │                     │         │ │ Parallel graph  │ │       │
│   │ Layer 0: dense      │         │ │ construction    │ │       │
│   │  ○─○─○─○─○─○─○─○    │         │ │ with GPU-aware  │ │       │
│   │                     │         │ │ pruning         │ │       │
│   │ Sequential insert   │         │ └─────────────────┘ │       │
│   └─────────────────────┘         └─────────────────────┘       │
│                                                                  │
│   Build: O(n log n)               Build: Highly parallel        │
│   Insert: One at a time           Insert: Batch optimized       │
│   Search: ~10-50 hops             Search: Parallel BFS          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key Innovations:**
1. **NN-Descent for construction**: Parallel graph building vs sequential HNSW inserts
2. **GPU-aware pruning**: Optimized for parallel execution
3. **Interoperability**: Can export to HNSW format for CPU search

**Performance (vs HNSW):**
- Build time: 2.2-27× faster
- Large batch (10K queries): 33-77× faster
- Single query: 3.4-53× faster at 95% recall

---

## Comparison with Graph-Based Methods

### Algorithm Characteristics

| Aspect | IVF-Flat | IVF-PQ | HNSW | CAGRA | Vamana |
|--------|----------|--------|------|-------|--------|
| **Type** | Cluster-based | Cluster + compression | Graph | Graph (GPU) | Graph |
| **Memory** | O(n×d) | O(n×M) | O(n×M×log(n)) | O(n×degree) | O(n×R) |
| **Build Parallel** | Yes | Yes | No (sequential) | Yes | Partial |
| **GPU Optimized** | Yes | Yes | No (CPU) | Yes | No |
| **Online Insert** | Yes | Yes | Yes | Limited | No |
| **Recall** | Tunable | Tunable (lower) | High | High | Medium |

### Performance at Scale

| Scale | HNSW (CPU) | IVF-PQ (GPU) | CAGRA (GPU) | Notes |
|-------|------------|--------------|-------------|-------|
| **Build 1M** | ~7 hours | ~1-2 min | ~30 sec | GPU 200-400× faster |
| **Build 100M** | ~30 days | ~30-60 min | ~15-30 min | GPU essential |
| **Build 1B** | ~300 days | ~5-10 hours | ~3-5 hours | Only GPU practical |
| **Search QPS (batch)** | 50-100 | 50,000-100,000 | 100,000-500,000 | GPU 1000× faster |
| **Search QPS (single)** | 1,000-5,000 | 5,000-20,000 | 10,000-50,000 | Less dramatic |
| **Memory (1B)** | 4+ TB | 8-50 GB | 100-500 GB | IVF-PQ most efficient |

### Recall vs Throughput Trade-offs

```
                        Recall@10 vs QPS (log scale)
    100% ┤
         │                                    ★ CAGRA (high batch)
     95% ┤                    ◆ HNSW         ★ CAGRA
         │        ○ IVF-Flat (high nprobe)  ★
     90% ┤                    ◆             ○ IVF-Flat
         │    ○                ◆ HNSW       □ IVF-PQ + refine
     85% ┤   ○ IVF-Flat                     □
         │  ○ (low nprobe)                  □ IVF-PQ + refine
     80% ┤ □ IVF-PQ
         │ □
     75% ┤□ IVF-PQ (no refine)
         │
     70% ┼────────┬────────┬────────┬────────┬────────►
              100      1K      10K     100K      1M   QPS

Legend: ○ IVF-Flat  □ IVF-PQ  ◆ HNSW  ★ CAGRA
```

### When to Use Each Algorithm

| Use Case | Recommended | Reason |
|----------|-------------|--------|
| **< 1M, high recall** | HNSW | Best recall, acceptable speed |
| **< 1M, real-time, GPU available** | CAGRA | Fastest at all recall levels |
| **1M-100M, batch queries** | IVF-PQ (GPU) | Memory efficient, fast batches |
| **100M-1B, memory constrained** | IVF-PQ (GPU) | Only fits in GPU memory |
| **1B+, disk-based** | Vamana/DiskANN | Designed for disk |
| **Online updates critical** | HNSW or IVF-Flat | Support incremental |
| **Filtered search** | IVF + bitmap | Natural partition filtering |

---

## Library Options

### NVIDIA cuVS (Recommended for GPU)

[cuVS](https://developer.nvidia.com/cuvs) is NVIDIA's production vector search library, part of RAPIDS.

**Algorithms Supported:**
- IVF-Flat, IVF-PQ
- CAGRA (state-of-the-art graph)
- Brute-force
- HNSW (for CPU fallback)

**Language Bindings:**
- C/C++ (primary)
- Python
- **Rust** (official support)
- Go, Java

**Key Features:**
- Multi-GPU support
- Index serialization
- Filtering support
- CPU-GPU interoperability (build on GPU, search on CPU)

**Performance Claims:**
- Build indexes 12× faster at 95% recall
- Search latency 8× lower at 95% recall
- IVF-PQ: 4-5× compression with competitive recall

**Rust Integration:**

```rust
use cuvs::cagra::{Index, IndexParams, SearchParams};
use cuvs::ivf_pq::{Index as IvfPqIndex, IndexParams as IvfPqParams};

// CAGRA example
let params = IndexParams::new(/* graph_degree */ 64);
let index = Index::build(&dataset, &params)?;

let search_params = SearchParams::new(/* itopk_size */ 64);
let (distances, indices) = index.search(&queries, k, &search_params)?;

// IVF-PQ example
let params = IvfPqParams::new(/* n_lists */ 1024, /* pq_dim */ 64);
let index = IvfPqIndex::build(&dataset, &params)?;
```

**Requirements:**
- NVIDIA GPU (compute capability 7.0+)
- CUDA 12 or 13
- ~500MB-2GB GPU memory for library

### Faiss (Meta)

[Faiss](https://github.com/facebookresearch/faiss) is the industry-standard vector search library.

**Rust Bindings:** [faiss-rs](https://github.com/Enet4/faiss-rs)

```rust
use faiss::{Index, index_factory, MetricType};

// Create IVF-PQ index
let index = index_factory(128, "IVF1024,PQ8", MetricType::L2)?;

// Train on sample data
index.train(&training_data)?;

// Add vectors
index.add(&vectors)?;

// Search
let (distances, labels) = index.search(&query, k)?;
```

**Comparison with cuVS:**

| Aspect | Faiss | cuVS |
|--------|-------|------|
| **Maturity** | 7+ years | 2+ years |
| **CPU Support** | Full | Limited |
| **GPU Support** | Good | Excellent |
| **Rust Bindings** | Community | Official |
| **CAGRA** | No | Yes |
| **Integration** | Standalone | RAPIDS ecosystem |

### Other Options

| Library | Language | GPU | Notes |
|---------|----------|-----|-------|
| **USearch** | C++/Rust/Python | Partial | Fast, lightweight |
| **Qdrant** | Rust | No | Full vector DB |
| **LanceDB** | Rust | No | Columnar storage |
| **Milvus** | Go/C++ | Yes (cuVS) | Distributed |

---

## Integration Architecture

### Option A: cuVS as Separate Index (Recommended)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Hybrid Architecture                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                     motlie_db                            │   │
│   │  ┌─────────────────┐  ┌─────────────────┐               │   │
│   │  │ Graph Storage   │  │ Vector Metadata │               │   │
│   │  │ (Nodes, Edges)  │  │ (IDs, timestamps)│              │   │
│   │  └─────────────────┘  └─────────────────┘               │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              │ ID mapping                        │
│                              ▼                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                  cuVS IVF-PQ/CAGRA Index                 │   │
│   │  ┌─────────────────┐  ┌─────────────────┐               │   │
│   │  │ Compressed      │  │ GPU-optimized   │               │   │
│   │  │ Vectors (PQ)    │  │ Search          │               │   │
│   │  └─────────────────┘  └─────────────────┘               │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    GPU Memory                            │   │
│   │  Index: 8-50 GB for 1B vectors (IVF-PQ)                 │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Flow:
1. Insert: motlie_db stores metadata ──► cuVS adds to index
2. Search: cuVS returns IDs ──► motlie_db fetches full data
3. Delete: motlie_db marks deleted ──► periodic cuVS rebuild
```

**Advantages:**
- Leverage motlie_db for metadata, relationships, temporal data
- Use cuVS for pure vector search (what it's optimized for)
- Clear separation of concerns
- GPU acceleration where it matters most

### Option B: cuVS-backed Storage Layer

```
┌─────────────────────────────────────────────────────────────────┐
│                    motlie_db with cuVS Backend                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                Storage Abstraction Layer                 │   │
│   │                                                          │   │
│   │  impl VectorIndex for CuvsIndex {                        │   │
│   │      fn insert(&self, id: u32, vec: &[f32]) -> Result    │   │
│   │      fn search(&self, query: &[f32], k: usize) -> Vec    │   │
│   │      fn delete(&self, id: u32) -> Result                 │   │
│   │  }                                                        │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│          ┌───────────────────┼───────────────────┐              │
│          ▼                   ▼                   ▼              │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐      │
│   │ HNSW        │     │ IVF-PQ      │     │ CAGRA       │      │
│   │ (current)   │     │ (cuVS)      │     │ (cuVS)      │      │
│   │ CPU/RocksDB │     │ GPU         │     │ GPU         │      │
│   └─────────────┘     └─────────────┘     └─────────────┘      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Advantages:**
- Unified API across algorithms
- Easier to switch between CPU/GPU
- Retains motlie_db primitives for non-vector operations

### ID Mapping Strategy

cuVS uses u32 IDs internally. Map to motlie_db UUIDs:

```rust
struct IdMapper {
    // UUID -> u32 (for insert/search)
    uuid_to_u32: HashMap<Uuid, u32>,
    // u32 -> UUID (for search results)
    u32_to_uuid: Vec<Uuid>,
    next_id: AtomicU32,
}

impl IdMapper {
    fn get_or_create(&mut self, uuid: Uuid) -> u32 {
        *self.uuid_to_u32.entry(uuid).or_insert_with(|| {
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
            self.u32_to_uuid.push(uuid);
            id
        })
    }

    fn to_uuid(&self, id: u32) -> Option<Uuid> {
        self.u32_to_uuid.get(id as usize).copied()
    }
}
```

---

## Project Plan

### Phase 1: Research & Prototyping (2 weeks)

| Task | Effort | Deliverable |
|------|--------|-------------|
| Set up cuVS development environment | 2 days | Working Rust + cuVS build |
| Prototype IVF-PQ index with cuVS Rust bindings | 3 days | Basic index/search working |
| Prototype CAGRA index | 2 days | Compare with IVF-PQ |
| Benchmark on SIFT1M | 2 days | Performance baseline |
| Document findings | 1 day | Updated IVFPQ.md |

**Milestone**: Working cuVS prototype with benchmark results

### Phase 2: Integration Design (1 week)

| Task | Effort | Deliverable |
|------|--------|-------------|
| Design ID mapping layer | 1 day | IdMapper implementation |
| Design storage abstraction | 2 days | VectorIndex trait |
| Design motlie_db integration | 2 days | Architecture doc |

**Milestone**: Approved integration architecture

### Phase 3: IVF-PQ Implementation (2 weeks)

| Task | Effort | Deliverable |
|------|--------|-------------|
| Implement IvfPqIndex wrapper | 3 days | Rust wrapper for cuVS |
| Implement batch insert | 2 days | Efficient bulk loading |
| Implement search with ID mapping | 2 days | UUID results |
| Implement index persistence | 2 days | Save/load to disk |
| Add motlie_db metadata integration | 1 day | Store vector metadata in RocksDB |

**Milestone**: Working IVF-PQ with motlie_db integration

### Phase 4: CAGRA Implementation (1 week)

| Task | Effort | Deliverable |
|------|--------|-------------|
| Implement CagraIndex wrapper | 2 days | Rust wrapper |
| Add GPU-CPU interop (CAGRA → HNSW) | 2 days | CPU fallback path |
| Benchmark vs IVF-PQ | 1 day | Performance comparison |

**Milestone**: CAGRA as alternative algorithm

### Phase 5: Production Hardening (2 weeks)

| Task | Effort | Deliverable |
|------|--------|-------------|
| Error handling & recovery | 2 days | Graceful failures |
| GPU memory management | 2 days | OOM handling, multi-GPU |
| Index update strategies | 3 days | Incremental vs rebuild |
| Comprehensive benchmarks | 2 days | 1M, 10M, 100M, 1B |
| Documentation & examples | 1 day | Usage guide |

**Milestone**: Production-ready vector search

### Phase 6: Scale Testing (1 week)

| Task | Effort | Deliverable |
|------|--------|-------------|
| 100M vector benchmark | 2 days | Build time, recall, QPS |
| 1B vector benchmark | 3 days | Verify scalability claims |

**Milestone**: Validated at billion scale

### Total Timeline: 9-10 weeks

```
Week 1-2:   [Phase 1: Research & Prototyping]
Week 3:     [Phase 2: Integration Design]
Week 4-5:   [Phase 3: IVF-PQ Implementation]
Week 6:     [Phase 4: CAGRA Implementation]
Week 7-8:   [Phase 5: Production Hardening]
Week 9-10:  [Phase 6: Scale Testing]
```

---

## Performance Projections

### Build Time Comparison

| Scale | Current HNSW | IVF-PQ (GPU) | CAGRA (GPU) |
|-------|--------------|--------------|-------------|
| 1K | 9.65s | **<1s** | **<1s** |
| 10K | 2.4 min | **<5s** | **<5s** |
| 100K | 33 min | **<30s** | **<30s** |
| 1M | 7 hours | **1-2 min** | **30s-1 min** |
| 10M | 3 days | **10-20 min** | **5-10 min** |
| 100M | 38 days | **1-2 hours** | **30-60 min** |
| 1B | 460 days | **5-10 hours** | **3-5 hours** |

### Memory Requirements

| Scale | HNSW (RAM) | IVF-PQ (GPU) | CAGRA (GPU) |
|-------|------------|--------------|-------------|
| 1M | 4 GB | 8-50 MB | 500 MB |
| 10M | 40 GB | 80-500 MB | 5 GB |
| 100M | 400 GB | 800 MB - 5 GB | 50 GB |
| 1B | 4 TB | **8-50 GB** | 500 GB |

### Search Performance (Batch of 10K queries)

| Scale | HNSW QPS | IVF-PQ QPS | CAGRA QPS |
|-------|----------|------------|-----------|
| 1M | 500-1K | 50,000-100,000 | 100,000-500,000 |
| 100M | 50-100 | 10,000-50,000 | 50,000-200,000 |
| 1B | 5-10 | 5,000-20,000 | 20,000-100,000 |

### Expected Recall

| Algorithm | Recall@10 (tuned) | Notes |
|-----------|-------------------|-------|
| HNSW | 95-99% | Highest recall |
| CAGRA | 93-97% | Close to HNSW |
| IVF-PQ + refine | 90-95% | Refinement crucial |
| IVF-PQ (no refine) | 80-90% | Lossy compression |

---

## IVF-PQ vs HNSW2: RocksDB Integration Analysis

This section compares IVF-PQ (cuVS) with the proposed [HNSW2 design](./HNSW2.md) specifically in terms of RocksDB integration complexity and operational characteristics.

### Architecture Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                    HNSW2 (RocksDB-Native)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    RocksDB                               │   │
│   │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │   │
│   │  │ vectors  │ │  edges   │ │ node_meta│ │graph_meta│   │   │
│   │  │ (binary) │ │ (bitmap) │ │          │ │          │   │   │
│   │  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │   │
│   │                                                          │   │
│   │  - Everything in RocksDB                                 │   │
│   │  - Merge operators for atomic updates                    │   │
│   │  - Block cache for hot data                              │   │
│   │  - Single storage layer                                  │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    IVF-PQ (cuVS + RocksDB)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────────────┐    ┌──────────────────────────────┐ │
│   │      RocksDB         │    │         cuVS (GPU)           │ │
│   │  ┌────────────────┐  │    │  ┌────────────────────────┐  │ │
│   │  │ Metadata       │  │◄───│  │ Compressed vectors     │  │ │
│   │  │ (UUID→u32 map) │  │    │  │ (PQ codes in GPU mem)  │  │ │
│   │  │ Timestamps     │  │    │  │                        │  │ │
│   │  │ Relationships  │  │    │  │ Cluster centroids      │  │ │
│   │  └────────────────┘  │    │  │ (IVF partitions)       │  │ │
│   └──────────────────────┘    │  └────────────────────────┘  │ │
│           │                    │             │                │ │
│           │ ID mapping         │             │ GPU memory     │ │
│           ▼                    │             ▼                │ │
│   ┌──────────────────────┐    │  ┌────────────────────────┐  │ │
│   │ Serialized index     │◄───┼──│ Index serialization    │  │ │
│   │ (disk backup)        │    │  │ (for persistence)      │  │ │
│   └──────────────────────┘    └──┴────────────────────────┘  │ │
│                                                                  │
│   - Two storage systems to coordinate                            │
│   - GPU required for search                                      │
│   - Sync complexity between systems                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Feature Comparison

| Aspect | HNSW2 (RocksDB) | IVF-PQ (cuVS) |
|--------|-----------------|---------------|
| **Storage** | Single (RocksDB) | Dual (RocksDB + GPU/files) |
| **Hardware** | CPU only | GPU required |
| **Integration** | Native RocksDB CFs | External library + sync |
| **Transactions** | RocksDB WriteBatch | Manual coordination |
| **Consistency** | ACID (RocksDB) | Eventually consistent |
| **Online Updates** | Merge operators | Rebuild or append-only |
| **Deletions** | Native support | Requires index rebuild |
| **Persistence** | Automatic (RocksDB) | Manual serialization |
| **Recovery** | RocksDB WAL | Re-import from backup |

### Performance Comparison

| Metric | HNSW2 | IVF-PQ (GPU) | Winner |
|--------|-------|--------------|--------|
| **Build 1M** | 3-5 min | 1-2 min | IVF-PQ |
| **Build 1B** | 5-10 days | 5-10 hours | **IVF-PQ (10-20×)** |
| **Search QPS (batch)** | 500-1K | 50K-100K | **IVF-PQ (100×)** |
| **Search QPS (single)** | 5K-10K | 5K-20K | Similar |
| **Memory 1B** | ~100GB (block cache) | 8-50GB (GPU) | IVF-PQ |
| **Recall** | 95-99% | 90-95% | **HNSW2** |
| **Online insert** | <1ms | Batch only | **HNSW2** |
| **Delete support** | Native | Rebuild | **HNSW2** |

### Integration Complexity

| Task | HNSW2 | IVF-PQ |
|------|-------|--------|
| **Schema setup** | New CFs in existing RocksDB | Separate index files + mapping |
| **ID management** | u32 in RocksDB | u32↔UUID sync layer |
| **Insert flow** | Single WriteBatch | RocksDB + cuVS API calls |
| **Search flow** | RocksDB reads | cuVS search → ID lookup |
| **Consistency** | Automatic | Manual sync required |
| **Crash recovery** | RocksDB handles | Re-import or rebuild |
| **Backup/restore** | RocksDB tooling | Two systems to backup |

### Code Complexity Comparison

**HNSW2 Insert (RocksDB-native):**
```rust
// Everything in RocksDB - single atomic operation
let batch = WriteBatch::new();
batch.put_cf(vectors_cf, &id.to_le_bytes(), &vector_bytes);
batch.merge_cf(edges_cf, &key, EdgeOp::AddBatch(neighbors).serialize());
batch.put_cf(pending_updates_cf, &id.to_le_bytes(), &[1]);
db.write(batch)?;  // Atomic, consistent, durable
```

**IVF-PQ Insert (dual-system coordination):**
```rust
// Two systems to coordinate - potential inconsistency
let internal_id = id_mapper.get_or_create(uuid);  // Mapping layer

// 1. Store metadata in RocksDB
let batch = WriteBatch::new();
batch.put_cf(metadata_cf, &uuid.as_bytes(), &metadata);
db.write(batch)?;

// 2. Add to cuVS index (separate operation)
cuvs_index.add(&[internal_id], &[vector])?;

// 3. Persist cuVS index (manual)
cuvs_index.serialize_to_file(&index_path)?;

// WARNING: If step 2 or 3 fails after step 1, systems are inconsistent!
```

### Update/Delete Handling

**HNSW2 Delete (native support):**
```rust
async fn delete_vector(db: &DB, id: u32) -> Result<()> {
    let batch = WriteBatch::new();

    // Get neighbors, remove reverse edges atomically
    let bitmap = get_edge_bitmap(db, id)?;
    for neighbor in bitmap.iter() {
        batch.merge_cf(edges_cf, &neighbor_key, EdgeOp::Remove(id));
        batch.put_cf(pending_updates_cf, &neighbor.to_le_bytes(), &[1]);
    }

    batch.delete_cf(vectors_cf, &id.to_le_bytes());
    batch.delete_cf(edges_cf, &edge_key);
    db.write(batch)?;  // Atomic delete with graph repair
    Ok(())
}
```

**IVF-PQ Delete (requires workarounds):**
```rust
async fn delete_vector(id: Uuid) -> Result<()> {
    // Option 1: Mark as deleted, filter at search time (degrades performance)
    deleted_set.insert(id);

    // Option 2: Periodic full rebuild (expensive)
    if deleted_set.len() > threshold {
        let valid_ids: Vec<_> = all_ids.difference(&deleted_set).collect();
        let valid_vectors = fetch_vectors(&valid_ids)?;
        cuvs_index = IvfPqIndex::build(&valid_vectors)?;  // Full rebuild!
    }
    Ok(())
}
```

### Ease of Integration Scoring

| Criterion | HNSW2 | IVF-PQ | Weight |
|-----------|-------|--------|--------|
| Fits existing architecture | 9/10 | 4/10 | High |
| Transaction support | 10/10 | 3/10 | High |
| Online updates | 9/10 | 2/10 | Medium |
| Delete support | 9/10 | 2/10 | Medium |
| Crash recovery | 10/10 | 5/10 | High |
| Code complexity | 7/10 | 5/10 | Medium |
| Testing complexity | 8/10 | 4/10 | Medium |
| **Weighted Total** | **8.5/10** | **3.5/10** | - |

### Scale Limits

| Scale | HNSW2 Feasibility | IVF-PQ Feasibility | Recommendation |
|-------|-------------------|--------------------| ---------------|
| < 10M | Excellent | Overkill | **HNSW2** |
| 10M-50M | Good (hours to build) | Good | Either works |
| 50M-100M | Challenging (days) | Good | **IVF-PQ** |
| 100M-1B | Impractical | Required | **IVF-PQ only** |

### When to Choose Each

**Choose HNSW2 when:**
- Need ACID transactions with vector data
- Online updates (insert/delete) are critical
- Want single storage system (operational simplicity)
- No GPU available or GPU cost prohibitive
- Scale: up to ~50-100M vectors
- Recall > 95% required
- Strong consistency required

**Choose IVF-PQ when:**
- Scale: 100M+ to 1B+ vectors
- Batch workloads (build once, query many)
- Can tolerate periodic index rebuilds
- GPU infrastructure already available
- Recall ~90-95% acceptable
- Query throughput > 10K QPS required
- Memory efficiency critical (8-50GB vs 100GB+)

### Hybrid Architecture (Best of Both)

For production systems needing both real-time updates AND billion-scale search, a hybrid architecture combines HNSW2's operational strengths with IVF-PQ's scale advantages.

#### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Hybrid: HNSW2 + IVF-PQ                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Real-time Layer (HNSW2):              Batch Layer (IVF-PQ):   │
│   ┌─────────────────────────┐          ┌─────────────────────┐ │
│   │ Recent vectors (<1M)    │          │ Historical (100M+)  │ │
│   │ - Online insert/delete  │          │ - Rebuilt nightly   │ │
│   │ - RocksDB native        │          │ - GPU accelerated   │ │
│   │ - High recall (95%+)    │          │ - Memory efficient  │ │
│   │ - ACID transactions     │          │ - High throughput   │ │
│   └─────────────────────────┘          └─────────────────────┘ │
│              │                                    │             │
│              └──────────────┬─────────────────────┘             │
│                             ▼                                    │
│                    ┌─────────────────┐                          │
│                    │ Merge Results   │                          │
│                    │ (dedup by ID)   │                          │
│                    └─────────────────┘                          │
│                                                                  │
│   Nightly job: HNSW2 vectors → merge into IVF-PQ batch rebuild  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Detailed Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         INSERT FLOW                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Client ──► Insert Request                                      │
│                    │                                             │
│                    ▼                                             │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                 HNSW2 (RocksDB)                          │   │
│   │  1. Assign u32 ID                                        │   │
│   │  2. Store vector in vectors_cf                           │   │
│   │  3. Build graph edges via merge operators                │   │
│   │  4. Mark in pending_compaction_cf (for batch layer)      │   │
│   │  5. Return success immediately                           │   │
│   └─────────────────────────────────────────────────────────┘   │
│                    │                                             │
│                    ▼                                             │
│   Vector immediately searchable in real-time layer              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         SEARCH FLOW                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Client ──► Search Request (query, k=10)                        │
│                    │                                             │
│          ┌────────┴────────┐                                    │
│          ▼                 ▼                                    │
│   ┌─────────────┐   ┌─────────────┐                             │
│   │   HNSW2     │   │   IVF-PQ    │   (parallel)                │
│   │  (recent)   │   │  (historical)│                            │
│   │  k=10       │   │  k=10       │                             │
│   └─────────────┘   └─────────────┘                             │
│          │                 │                                    │
│          └────────┬────────┘                                    │
│                   ▼                                             │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                   Result Merger                          │   │
│   │  1. Combine results (up to 20 candidates)                │   │
│   │  2. Deduplicate by ID (prefer real-time if conflict)     │   │
│   │  3. Re-rank by distance                                  │   │
│   │  4. Return top k                                         │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      COMPACTION FLOW                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Nightly Job (or when real-time layer > threshold):            │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  1. Scan pending_compaction_cf for vectors to migrate    │   │
│   │  2. Export vectors from HNSW2 (RocksDB read)             │   │
│   │  3. Merge with existing IVF-PQ dataset                   │   │
│   │  4. Rebuild IVF-PQ index on GPU                          │   │
│   │  5. Swap new index atomically                            │   │
│   │  6. Delete migrated vectors from HNSW2                   │   │
│   │  7. Clear pending_compaction_cf entries                  │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Timeline:                                                      │
│   - Export 1M vectors: ~1 min                                   │
│   - Rebuild 100M IVF-PQ: ~30-60 min (GPU)                       │
│   - Total compaction: ~1 hour                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Storage Schema

```
┌─────────────────────────────────────────────────────────────────┐
│                     RocksDB (HNSW2 Layer)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Column Families:                                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ vectors_cf                                                │   │
│  │   Key: u32 ID (4 bytes)                                   │   │
│  │   Value: f32[dim] binary                                  │   │
│  │   Purpose: Store recent vectors for HNSW2                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ edges_cf                                                  │   │
│  │   Key: u32 ID | u8 layer                                  │   │
│  │   Value: RoaringBitmap (neighbors)                        │   │
│  │   Purpose: HNSW2 graph structure                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ pending_compaction_cf                                     │   │
│  │   Key: u32 ID                                             │   │
│  │   Value: timestamp (u64)                                  │   │
│  │   Purpose: Track vectors to migrate to batch layer        │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ id_mapping_cf                                             │   │
│  │   Key: UUID (16 bytes)                                    │   │
│  │   Value: u32 ID | u8 layer_flag (HNSW2=0, IVF-PQ=1)       │   │
│  │   Purpose: External ID → internal ID + location           │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ deleted_cf                                                │   │
│  │   Key: u32 ID                                             │   │
│  │   Value: timestamp (u64)                                  │   │
│  │   Purpose: Tombstones for IVF-PQ layer (filter at search) │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     GPU / Files (IVF-PQ Layer)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Files:                                                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ ivfpq_index.bin                                           │   │
│  │   Content: Serialized cuVS IVF-PQ index                   │   │
│  │   Size: ~8-50GB for 1B vectors                            │   │
│  │   Purpose: Main batch index (loaded to GPU)               │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ ivfpq_vectors.bin                                         │   │
│  │   Content: Original f32 vectors (for refinement)          │   │
│  │   Size: ~4TB for 1B vectors (optional, disk-based)        │   │
│  │   Purpose: Re-ranking after approximate search            │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ ivfpq_metadata.json                                       │   │
│  │   Content: Index parameters, build timestamp, stats       │   │
│  │   Purpose: Track index version for consistency            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### API Design

```rust
/// Hybrid vector index combining HNSW2 (real-time) with IVF-PQ (batch)
pub struct HybridIndex {
    /// Real-time layer: HNSW2 backed by RocksDB
    realtime: Hnsw2Index,

    /// Batch layer: IVF-PQ backed by cuVS (GPU)
    batch: IvfPqIndex,

    /// ID mapping: UUID → (u32, layer)
    id_mapper: IdMapper,

    /// Deleted IDs (for filtering batch layer results)
    deleted: RoaringBitmap,

    /// Configuration
    config: HybridConfig,
}

pub struct HybridConfig {
    /// Max vectors in real-time layer before compaction
    pub realtime_threshold: usize,  // e.g., 1_000_000

    /// How often to run compaction
    pub compaction_interval: Duration,  // e.g., 24 hours

    /// Number of candidates to fetch from each layer
    pub search_k_multiplier: usize,  // e.g., 2 (fetch 2k, return k)
}

impl HybridIndex {
    /// Insert a vector (goes to real-time layer)
    pub async fn insert(&mut self, uuid: Uuid, vector: &[f32]) -> Result<()> {
        // 1. Get or create internal ID
        let id = self.id_mapper.get_or_create(uuid, Layer::Realtime);

        // 2. Insert into HNSW2 (RocksDB)
        self.realtime.insert(id, vector).await?;

        // 3. Mark for future compaction
        self.realtime.mark_pending_compaction(id)?;

        Ok(())
    }

    /// Delete a vector (marks deleted, actual removal on compaction)
    pub async fn delete(&mut self, uuid: Uuid) -> Result<()> {
        let (id, layer) = self.id_mapper.get(uuid)?;

        match layer {
            Layer::Realtime => {
                // Immediate delete from HNSW2
                self.realtime.delete(id).await?;
            }
            Layer::Batch => {
                // Mark as deleted (filter at search time)
                self.deleted.insert(id);
                // Persist to deleted_cf for crash recovery
                self.realtime.db.put_cf(deleted_cf, &id.to_le_bytes(), &now())?;
            }
        }

        self.id_mapper.remove(uuid)?;
        Ok(())
    }

    /// Search across both layers
    pub async fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let fetch_k = k * self.config.search_k_multiplier;

        // 1. Search both layers in parallel
        let (realtime_results, batch_results) = tokio::join!(
            self.realtime.search(query, fetch_k),
            self.batch.search(query, fetch_k, &self.deleted),  // Pass deleted set
        );

        // 2. Merge and deduplicate
        let mut combined: HashMap<u32, (f32, Layer)> = HashMap::new();

        for (dist, id) in realtime_results? {
            combined.insert(id, (dist, Layer::Realtime));
        }

        for (dist, id) in batch_results? {
            // Prefer real-time if same ID exists (fresher data)
            combined.entry(id).or_insert((dist, Layer::Batch));
        }

        // 3. Sort by distance and take top k
        let mut results: Vec<_> = combined.into_iter()
            .map(|(id, (dist, layer))| SearchResult { id, dist, layer })
            .collect();
        results.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        results.truncate(k);

        // 4. Map back to UUIDs
        for r in &mut results {
            r.uuid = self.id_mapper.to_uuid(r.id)?;
        }

        Ok(results)
    }

    /// Run compaction (typically called by background job)
    pub async fn compact(&mut self) -> Result<CompactionStats> {
        let start = Instant::now();

        // 1. Get vectors pending compaction
        let pending = self.realtime.get_pending_compaction()?;
        if pending.is_empty() {
            return Ok(CompactionStats::empty());
        }

        // 2. Export vectors from HNSW2
        let vectors: Vec<(u32, Vec<f32>)> = self.realtime
            .export_vectors(&pending)?;

        // 3. Merge with existing batch vectors
        let existing = self.batch.export_all()?;
        let merged: Vec<_> = existing.into_iter()
            .filter(|(id, _)| !self.deleted.contains(*id))  // Skip deleted
            .chain(vectors.into_iter())
            .collect();

        // 4. Rebuild IVF-PQ index on GPU
        let new_batch = IvfPqIndex::build(&merged, &self.config.ivfpq_params)?;

        // 5. Atomic swap
        let old_batch = std::mem::replace(&mut self.batch, new_batch);

        // 6. Update ID mappings
        for (id, _) in &pending {
            self.id_mapper.update_layer(*id, Layer::Batch)?;
        }

        // 7. Delete from real-time layer
        for (id, _) in &pending {
            self.realtime.delete(*id).await?;
        }

        // 8. Clear pending and deleted sets
        self.realtime.clear_pending_compaction(&pending)?;
        self.deleted.clear();

        Ok(CompactionStats {
            vectors_migrated: pending.len(),
            new_batch_size: merged.len(),
            duration: start.elapsed(),
        })
    }
}
```

#### Consistency Guarantees

| Operation | Consistency | Recovery |
|-----------|-------------|----------|
| **Insert** | Immediately visible (HNSW2) | RocksDB WAL |
| **Delete (real-time)** | Immediately removed | RocksDB WAL |
| **Delete (batch)** | Filtered at search time | deleted_cf persisted |
| **Search** | Sees all non-deleted vectors | N/A |
| **Compaction** | Atomic swap | Checkpoint + retry |

#### Failure Scenarios

```
┌─────────────────────────────────────────────────────────────────┐
│                     Failure Handling                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Scenario 1: Crash during insert                                │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  - RocksDB WAL ensures atomicity                         │   │
│   │  - Either fully inserted or not at all                   │   │
│   │  - Recovery: RocksDB replay                              │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Scenario 2: Crash during compaction (before swap)              │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  - New IVF-PQ index not yet active                       │   │
│   │  - Vectors still in HNSW2                                │   │
│   │  - Recovery: Re-run compaction                           │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Scenario 3: Crash during compaction (after swap, before clean) │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  - New IVF-PQ active, but vectors duplicated in HNSW2    │   │
│   │  - Search deduplicates by ID (no incorrect results)      │   │
│   │  - Recovery: Resume cleanup from pending_compaction_cf   │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Scenario 4: GPU failure during search                          │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  - Batch layer unavailable                               │   │
│   │  - Fallback: Return real-time results only               │   │
│   │  - Or: Use CAGRA → HNSW export for CPU fallback          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Performance Characteristics

| Metric | Real-time Only | Batch Only | Hybrid |
|--------|----------------|------------|--------|
| **Insert latency** | <1ms | N/A (batch) | <1ms |
| **Delete latency** | <1ms | N/A (rebuild) | <1ms |
| **Search latency** | 10-50ms | 1-5ms | 15-55ms (parallel) |
| **Search QPS** | 1-5K | 50-100K | 1-5K (bottleneck: merge) |
| **Memory (1B)** | 100GB+ | 8-50GB | 100GB + 8-50GB |
| **Consistency** | Strong | Eventual | Strong for recent |

#### When to Use Hybrid

| Requirement | Hybrid? | Reason |
|-------------|---------|--------|
| Real-time insert + billion scale | **Yes** | Only way to achieve both |
| Batch-only workload | No | IVF-PQ alone simpler |
| < 50M vectors | No | HNSW2 alone sufficient |
| No GPU available | No | IVF-PQ requires GPU |
| 100% strong consistency | No | Batch layer is eventual |
| Delete-heavy workload | **Yes** | Real-time deletes, batch rebuilt |

#### Implementation Phases

| Phase | Effort | Deliverable |
|-------|--------|-------------|
| 1. HNSW2 core | 4-6 weeks | RocksDB-native HNSW |
| 2. IVF-PQ integration | 3-4 weeks | cuVS wrapper |
| 3. Hybrid orchestration | 2-3 weeks | Merge, compaction, ID mapping |
| 4. Production hardening | 2-3 weeks | Failure handling, monitoring |
| **Total** | **11-16 weeks** | Full hybrid system |

### RocksDB Schema for IVF-PQ

While IVF-PQ is typically stored in GPU memory or serialized files, a RocksDB-backed implementation enables persistence, crash recovery, and integration with existing infrastructure. This schema supports both CPU-only operation and GPU acceleration (load from RocksDB → GPU).

#### Schema Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                  RocksDB IVF-PQ Schema                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ CF: ivfpq_meta                                            │   │
│  │   Purpose: Index configuration and statistics             │   │
│  │   Keys: String identifiers                                │   │
│  │   Size: <1KB total                                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ CF: ivfpq_centroids                                       │   │
│  │   Purpose: IVF cluster centroids (coarse quantizer)       │   │
│  │   Keys: u32 cluster_id                                    │   │
│  │   Size: nlist × dim × 4 bytes (e.g., 4MB for 1024 clusters)│  │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ CF: ivfpq_codebooks                                       │   │
│  │   Purpose: PQ sub-quantizer codebooks                     │   │
│  │   Keys: u8 subvector_id                                   │   │
│  │   Size: M × 256 × (dim/M) × 4 bytes (e.g., 8MB)          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ CF: ivfpq_vectors                                         │   │
│  │   Purpose: Original vectors (optional, for refinement)    │   │
│  │   Keys: u32 vector_id                                     │   │
│  │   Size: n × dim × 4 bytes (4GB per 1M vectors @ 1024-dim) │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ CF: ivfpq_codes                                           │   │
│  │   Purpose: PQ-encoded vectors (compressed)                │   │
│  │   Keys: u32 cluster_id | u32 vector_id                    │   │
│  │   Size: n × M bytes (8MB per 1M vectors @ M=8)           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ CF: ivfpq_cluster_index                                   │   │
│  │   Purpose: Cluster membership (vector_id → cluster_id)    │   │
│  │   Keys: u32 vector_id                                     │   │
│  │   Size: n × 4 bytes (4MB per 1M vectors)                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ CF: ivfpq_cluster_lists                                   │   │
│  │   Purpose: Inverted lists (cluster_id → vector_ids)       │   │
│  │   Keys: u32 cluster_id                                    │   │
│  │   Value: RoaringBitmap of vector_ids                      │   │
│  │   Size: ~1-10MB total (compressed)                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ CF: ivfpq_id_mapping                                      │   │
│  │   Purpose: External UUID → internal u32 mapping           │   │
│  │   Keys: UUID (16 bytes)                                   │   │
│  │   Size: n × 20 bytes                                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Column Family Specifications

##### 1. ivfpq_meta (Index Metadata)

```
┌─────────────────────────────────────────────────────────────────┐
│ CF: ivfpq_meta                                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ Key-Value Pairs:                                                 │
│                                                                  │
│ "config" → {                                                     │
│     dim: u32,           // Vector dimensionality (e.g., 1024)   │
│     nlist: u32,         // Number of IVF clusters (e.g., 1024)  │
│     m: u8,              // Number of PQ subvectors (e.g., 8)    │
│     nbits: u8,          // Bits per PQ code (e.g., 8 = 256 codes)│
│     metric: String,     // "L2" | "IP" | "Cosine"               │
│ }                                                                │
│                                                                  │
│ "stats" → {                                                      │
│     num_vectors: u64,   // Total vectors indexed                │
│     build_time_ms: u64, // Time to build index                  │
│     last_updated: u64,  // Unix timestamp                       │
│ }                                                                │
│                                                                  │
│ "training" → {                                                   │
│     trained: bool,      // Whether codebooks are trained        │
│     training_size: u32, // Vectors used for training            │
│     kmeans_iters: u32,  // Iterations for k-means               │
│ }                                                                │
│                                                                  │
│ Encoding: bincode or msgpack                                     │
│ Size: ~200 bytes total                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Rust struct:
```rust
#[derive(Serialize, Deserialize)]
pub struct IvfPqConfig {
    pub dim: u32,
    pub nlist: u32,
    pub m: u8,
    pub nbits: u8,
    pub metric: DistanceMetric,
}

#[derive(Serialize, Deserialize)]
pub struct IvfPqStats {
    pub num_vectors: u64,
    pub build_time_ms: u64,
    pub last_updated: u64,
}
```

##### 2. ivfpq_centroids (IVF Cluster Centers)

```
┌─────────────────────────────────────────────────────────────────┐
│ CF: ivfpq_centroids                                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ Key:   u32 cluster_id (4 bytes, little-endian)                  │
│ Value: f32[dim] binary (dim × 4 bytes)                          │
│                                                                  │
│ Example (dim=1024, nlist=1024):                                  │
│   Key: 0x00000000 → [f32; 1024] = 4096 bytes                    │
│   Key: 0x00000001 → [f32; 1024] = 4096 bytes                    │
│   ...                                                            │
│   Key: 0x000003FF → [f32; 1024] = 4096 bytes                    │
│                                                                  │
│ Total size: nlist × dim × 4 = 1024 × 1024 × 4 = 4 MB            │
│                                                                  │
│ RocksDB options:                                                 │
│   - No compression (already dense floats)                       │
│   - Small block size (4KB) for random access                    │
│   - High priority in block cache (frequently accessed)          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Rust encoding:
```rust
fn encode_centroid_key(cluster_id: u32) -> [u8; 4] {
    cluster_id.to_le_bytes()
}

fn encode_centroid_value(centroid: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice(centroid).to_vec()
}

fn decode_centroid_value(bytes: &[u8]) -> &[f32] {
    bytemuck::cast_slice(bytes)
}
```

##### 3. ivfpq_codebooks (PQ Sub-Quantizers)

```
┌─────────────────────────────────────────────────────────────────┐
│ CF: ivfpq_codebooks                                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ Key:   u8 subvector_id (1 byte)                                 │
│ Value: f32[256][subdim] binary                                  │
│        where subdim = dim / M                                   │
│                                                                  │
│ Example (dim=1024, M=8, nbits=8):                                │
│   subdim = 1024 / 8 = 128                                       │
│   Each codebook: 256 codes × 128 dims × 4 bytes = 128 KB        │
│                                                                  │
│   Key: 0x00 → f32[256][128] = 128 KB (subvector 0)              │
│   Key: 0x01 → f32[256][128] = 128 KB (subvector 1)              │
│   ...                                                            │
│   Key: 0x07 → f32[256][128] = 128 KB (subvector 7)              │
│                                                                  │
│ Total size: M × 2^nbits × subdim × 4                            │
│           = 8 × 256 × 128 × 4 = 1 MB                            │
│                                                                  │
│ For dim=128, M=8: 8 × 256 × 16 × 4 = 128 KB                     │
│                                                                  │
│ RocksDB options:                                                 │
│   - No compression                                               │
│   - Entire CF fits in block cache                               │
│   - Read-only after training                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Rust struct:
```rust
pub struct PqCodebook {
    pub subvector_id: u8,
    pub codes: Vec<Vec<f32>>,  // [256][subdim]
}

impl PqCodebook {
    pub fn encode(&self) -> Vec<u8> {
        // Flatten to [256 * subdim] f32 array
        let flat: Vec<f32> = self.codes.iter().flatten().copied().collect();
        bytemuck::cast_slice(&flat).to_vec()
    }

    pub fn decode(subvector_id: u8, bytes: &[u8], subdim: usize) -> Self {
        let flat: &[f32] = bytemuck::cast_slice(bytes);
        let codes: Vec<Vec<f32>> = flat.chunks(subdim)
            .map(|c| c.to_vec())
            .collect();
        Self { subvector_id, codes }
    }
}
```

##### 4. ivfpq_vectors (Original Vectors - Optional)

```
┌─────────────────────────────────────────────────────────────────┐
│ CF: ivfpq_vectors                                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ Key:   u32 vector_id (4 bytes)                                  │
│ Value: f32[dim] binary                                          │
│                                                                  │
│ Purpose:                                                         │
│   - Re-ranking after approximate PQ search (refinement)         │
│   - Exact distance computation for top candidates               │
│   - Optional: can be omitted for pure approximate search        │
│                                                                  │
│ Example (dim=1024):                                              │
│   Key: 0x00000000 → [f32; 1024] = 4096 bytes                    │
│   Key: 0x00000001 → [f32; 1024] = 4096 bytes                    │
│   ...                                                            │
│                                                                  │
│ Size per 1M vectors: 1M × 4096 = 4 GB                           │
│ Size per 1B vectors: 1B × 4096 = 4 TB                           │
│                                                                  │
│ RocksDB options:                                                 │
│   - ZSTD compression (vectors compress ~30-40%)                 │
│   - Large block size (64KB) for sequential reads                │
│   - Lower block cache priority (less frequently accessed)       │
│   - Consider: store on separate disk/SSD                        │
│                                                                  │
│ Alternative: Memory-map file for large scale                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

##### 5. ivfpq_codes (PQ-Encoded Vectors)

```
┌─────────────────────────────────────────────────────────────────┐
│ CF: ivfpq_codes                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ Key:   u32 cluster_id | u32 vector_id (8 bytes)                 │
│ Value: u8[M] PQ codes                                           │
│                                                                  │
│ Key format enables efficient cluster scanning:                   │
│   Prefix scan on cluster_id retrieves all vectors in cluster    │
│                                                                  │
│ Example (M=8):                                                   │
│   Key: [0x00000000, 0x00000042] → [42, 17, 89, 234, 56, 78, 12, 90]│
│   Key: [0x00000000, 0x00000099] → [18, 55, 23, 167, 89, 45, 67, 33]│
│   Key: [0x00000001, 0x00000001] → [99, 44, 22, 88, 11, 77, 33, 66]│
│                                                                  │
│ Size per vector: M bytes = 8 bytes                              │
│ Size per 1M vectors: 8 MB                                       │
│ Size per 1B vectors: 8 GB                                       │
│                                                                  │
│ Compression ratio vs original:                                   │
│   Original: 1024 × 4 = 4096 bytes                               │
│   PQ codes: 8 bytes                                             │
│   Ratio: 512× compression!                                      │
│                                                                  │
│ RocksDB options:                                                 │
│   - No compression (already minimal)                            │
│   - Optimize for prefix scans (cluster iteration)               │
│   - Bloom filter on prefix (cluster_id)                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Rust encoding:
```rust
fn encode_code_key(cluster_id: u32, vector_id: u32) -> [u8; 8] {
    let mut key = [0u8; 8];
    key[0..4].copy_from_slice(&cluster_id.to_be_bytes()); // BE for prefix scan
    key[4..8].copy_from_slice(&vector_id.to_le_bytes());
    key
}

fn decode_code_key(key: &[u8]) -> (u32, u32) {
    let cluster_id = u32::from_be_bytes(key[0..4].try_into().unwrap());
    let vector_id = u32::from_le_bytes(key[4..8].try_into().unwrap());
    (cluster_id, vector_id)
}

// PQ codes are just raw bytes
type PqCode = [u8; M];  // e.g., [u8; 8]
```

##### 6. ivfpq_cluster_lists (Inverted Lists)

```
┌─────────────────────────────────────────────────────────────────┐
│ CF: ivfpq_cluster_lists                                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ Key:   u32 cluster_id (4 bytes)                                 │
│ Value: RoaringBitmap of vector_ids                              │
│                                                                  │
│ Purpose:                                                         │
│   - Fast lookup: which vectors belong to cluster X?             │
│   - O(1) cluster size query                                     │
│   - Efficient iteration over cluster members                    │
│                                                                  │
│ Example (1M vectors, 1024 clusters, ~1K per cluster):           │
│   Key: 0x00000000 → RoaringBitmap{0, 42, 99, 156, ...}  ~2KB    │
│   Key: 0x00000001 → RoaringBitmap{1, 55, 789, ...}      ~2KB    │
│   ...                                                            │
│                                                                  │
│ Total size: ~1-10 MB (compressed bitmaps)                       │
│                                                                  │
│ Operations:                                                      │
│   - cluster_lists.get(cluster_id).len() → cluster size          │
│   - cluster_lists.get(cluster_id).iter() → vector IDs           │
│   - cluster_lists.get(cluster_id).contains(id) → membership     │
│                                                                  │
│ RocksDB options:                                                 │
│   - Merge operator for atomic add/remove                        │
│   - Fits entirely in block cache                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Rust merge operator:
```rust
enum ClusterOp {
    Add(u32),        // Add vector to cluster
    Remove(u32),     // Remove vector from cluster
    Replace(Vec<u32>), // Replace entire list
}

fn cluster_merge_operator(
    _key: &[u8],
    existing: Option<&[u8]>,
    operands: &[&[u8]],
) -> Option<Vec<u8>> {
    let mut bitmap = existing
        .map(|b| RoaringBitmap::deserialize_from(b).unwrap())
        .unwrap_or_default();

    for op_bytes in operands {
        match bincode::deserialize::<ClusterOp>(op_bytes).unwrap() {
            ClusterOp::Add(id) => { bitmap.insert(id); }
            ClusterOp::Remove(id) => { bitmap.remove(id); }
            ClusterOp::Replace(ids) => {
                bitmap.clear();
                for id in ids { bitmap.insert(id); }
            }
        }
    }

    let mut buf = Vec::new();
    bitmap.serialize_into(&mut buf).unwrap();
    Some(buf)
}
```

##### 7. ivfpq_cluster_index (Reverse Mapping)

```
┌─────────────────────────────────────────────────────────────────┐
│ CF: ivfpq_cluster_index                                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ Key:   u32 vector_id (4 bytes)                                  │
│ Value: u32 cluster_id (4 bytes)                                 │
│                                                                  │
│ Purpose:                                                         │
│   - Reverse lookup: which cluster contains vector X?            │
│   - Needed for updates/deletes                                  │
│   - O(1) cluster lookup per vector                              │
│                                                                  │
│ Size per vector: 8 bytes (key + value)                          │
│ Size per 1M vectors: 8 MB                                       │
│ Size per 1B vectors: 8 GB                                       │
│                                                                  │
│ Alternative: Encode cluster_id in ivfpq_codes key               │
│   (already done - this CF is optional but faster for lookups)   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

##### 8. ivfpq_id_mapping (External → Internal ID)

```
┌─────────────────────────────────────────────────────────────────┐
│ CF: ivfpq_id_mapping                                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│ Key:   UUID (16 bytes) or String                                │
│ Value: u32 internal_id (4 bytes)                                │
│                                                                  │
│ Purpose:                                                         │
│   - Map external identifiers to internal u32 IDs                │
│   - Required because PQ uses dense u32 IDs                      │
│                                                                  │
│ Reverse mapping stored separately:                               │
│   Key: u32 internal_id                                          │
│   Value: UUID                                                   │
│   (Or use bidirectional encoding in single CF)                  │
│                                                                  │
│ Size per 1M vectors: ~20 MB                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Size Summary

| CF | Per Vector | 1M Vectors | 1B Vectors | Notes |
|----|------------|------------|------------|-------|
| ivfpq_meta | - | <1 KB | <1 KB | Config only |
| ivfpq_centroids | - | 4 MB | 4 MB | nlist=1024 |
| ivfpq_codebooks | - | 1 MB | 1 MB | M=8, dim=1024 |
| ivfpq_vectors | 4 KB | 4 GB | 4 TB | Optional |
| ivfpq_codes | 8 B | 8 MB | 8 GB | **Core index** |
| ivfpq_cluster_lists | ~2 KB/cluster | 2 MB | 2 MB | Compressed |
| ivfpq_cluster_index | 8 B | 8 MB | 8 GB | Optional |
| ivfpq_id_mapping | 20 B | 20 MB | 20 GB | If UUIDs used |
| **Total (minimal)** | **~8 B** | **~15 MB** | **~8 GB** | Codes only |
| **Total (with vectors)** | **~4 KB** | **~4 GB** | **~4 TB** | With refinement |

#### Search Implementation

```rust
impl RocksDbIvfPq {
    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize, nprobe: usize) -> Result<Vec<(f32, u32)>> {
        // 1. Find nprobe nearest clusters (coarse quantization)
        let centroids = self.load_all_centroids()?;
        let mut cluster_dists: Vec<(f32, u32)> = centroids.iter()
            .enumerate()
            .map(|(i, c)| (euclidean_distance(query, c), i as u32))
            .collect();
        cluster_dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let probe_clusters: Vec<u32> = cluster_dists.iter()
            .take(nprobe)
            .map(|(_, id)| *id)
            .collect();

        // 2. Precompute distance tables for PQ
        let codebooks = self.load_codebooks()?;
        let distance_tables = self.compute_distance_tables(query, &codebooks);

        // 3. Scan vectors in probed clusters
        let mut candidates: Vec<(f32, u32)> = Vec::new();

        for cluster_id in probe_clusters {
            // Prefix scan on ivfpq_codes
            let prefix = cluster_id.to_be_bytes();
            let iter = self.db.prefix_iterator_cf(codes_cf, &prefix);

            for item in iter {
                let (key, pq_codes) = item?;
                let (cid, vector_id) = decode_code_key(&key);
                if cid != cluster_id { break; }  // Past this cluster

                // Compute approximate distance using PQ lookup tables
                let approx_dist = self.compute_pq_distance(&pq_codes, &distance_tables);
                candidates.push((approx_dist, vector_id));
            }
        }

        // 4. Sort and take top candidates
        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        candidates.truncate(k * 4);  // Over-fetch for refinement

        // 5. Optional: Refine with exact distances
        if self.config.use_refinement {
            let vector_ids: Vec<u32> = candidates.iter().map(|(_, id)| *id).collect();
            let vectors = self.batch_get_vectors(&vector_ids)?;

            candidates = candidates.iter()
                .zip(vectors.iter())
                .filter_map(|((_, id), vec)| {
                    vec.as_ref().map(|v| (euclidean_distance(query, v), *id))
                })
                .collect();

            candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        }

        candidates.truncate(k);
        Ok(candidates)
    }

    /// Compute PQ distance using precomputed lookup tables
    fn compute_pq_distance(&self, pq_codes: &[u8], tables: &[Vec<f32>]) -> f32 {
        pq_codes.iter()
            .enumerate()
            .map(|(i, &code)| tables[i][code as usize])
            .sum()
    }

    /// Precompute distance tables for each subvector
    fn compute_distance_tables(&self, query: &[f32], codebooks: &[PqCodebook]) -> Vec<Vec<f32>> {
        let subdim = query.len() / codebooks.len();

        codebooks.iter()
            .enumerate()
            .map(|(i, codebook)| {
                let subquery = &query[i * subdim..(i + 1) * subdim];
                codebook.codes.iter()
                    .map(|code| euclidean_distance(subquery, code))
                    .collect()
            })
            .collect()
    }
}
```

#### Insert Implementation

```rust
impl RocksDbIvfPq {
    /// Add a vector to the index
    pub fn insert(&mut self, uuid: Uuid, vector: &[f32]) -> Result<u32> {
        // 1. Assign internal ID
        let vector_id = self.id_allocator.next();

        // 2. Find nearest cluster
        let centroids = self.load_all_centroids()?;
        let cluster_id = centroids.iter()
            .enumerate()
            .map(|(i, c)| (euclidean_distance(vector, c), i as u32))
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .map(|(_, id)| id)
            .unwrap();

        // 3. Compute PQ codes
        let codebooks = self.load_codebooks()?;
        let residual = self.compute_residual(vector, &centroids[cluster_id as usize]);
        let pq_codes = self.encode_pq(&residual, &codebooks);

        // 4. Write to RocksDB
        let batch = WriteBatch::new();

        // Store PQ codes
        let code_key = encode_code_key(cluster_id, vector_id);
        batch.put_cf(codes_cf, &code_key, &pq_codes);

        // Update cluster list
        batch.merge_cf(cluster_lists_cf,
            &cluster_id.to_le_bytes(),
            &bincode::serialize(&ClusterOp::Add(vector_id))?);

        // Store cluster assignment
        batch.put_cf(cluster_index_cf,
            &vector_id.to_le_bytes(),
            &cluster_id.to_le_bytes());

        // Store ID mapping
        batch.put_cf(id_mapping_cf, uuid.as_bytes(), &vector_id.to_le_bytes());

        // Optionally store original vector
        if self.config.store_vectors {
            batch.put_cf(vectors_cf,
                &vector_id.to_le_bytes(),
                bytemuck::cast_slice(vector));
        }

        // Update stats
        self.stats.num_vectors += 1;

        self.db.write(batch)?;
        Ok(vector_id)
    }

    /// Compute residual: vector - centroid
    fn compute_residual(&self, vector: &[f32], centroid: &[f32]) -> Vec<f32> {
        vector.iter()
            .zip(centroid.iter())
            .map(|(v, c)| v - c)
            .collect()
    }

    /// Encode residual using PQ codebooks
    fn encode_pq(&self, residual: &[f32], codebooks: &[PqCodebook]) -> Vec<u8> {
        let subdim = residual.len() / codebooks.len();

        codebooks.iter()
            .enumerate()
            .map(|(i, codebook)| {
                let subvec = &residual[i * subdim..(i + 1) * subdim];
                // Find nearest code
                codebook.codes.iter()
                    .enumerate()
                    .map(|(j, code)| (euclidean_distance(subvec, code), j as u8))
                    .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                    .map(|(_, code)| code)
                    .unwrap()
            })
            .collect()
    }
}
```

#### Training Implementation

```rust
impl RocksDbIvfPq {
    /// Train IVF centroids and PQ codebooks
    pub fn train(&mut self, training_vectors: &[Vec<f32>]) -> Result<()> {
        let dim = training_vectors[0].len();
        let subdim = dim / self.config.m as usize;

        // 1. Train IVF centroids using k-means
        println!("Training {} IVF centroids...", self.config.nlist);
        let centroids = kmeans(training_vectors, self.config.nlist as usize, 25);

        // Store centroids
        let batch = WriteBatch::new();
        for (i, centroid) in centroids.iter().enumerate() {
            batch.put_cf(centroids_cf,
                &(i as u32).to_le_bytes(),
                bytemuck::cast_slice(centroid));
        }

        // 2. Compute residuals for PQ training
        let residuals: Vec<Vec<f32>> = training_vectors.iter()
            .map(|v| {
                let nearest = centroids.iter()
                    .min_by(|a, b| {
                        euclidean_distance(v, a)
                            .partial_cmp(&euclidean_distance(v, b))
                            .unwrap()
                    })
                    .unwrap();
                self.compute_residual(v, nearest)
            })
            .collect();

        // 3. Train PQ codebooks (one per subvector)
        println!("Training {} PQ codebooks...", self.config.m);
        for subvec_id in 0..self.config.m {
            let subvectors: Vec<Vec<f32>> = residuals.iter()
                .map(|r| r[subvec_id as usize * subdim..(subvec_id as usize + 1) * subdim].to_vec())
                .collect();

            let codes = kmeans(&subvectors, 256, 25);  // 256 codes for 8-bit

            let codebook = PqCodebook {
                subvector_id: subvec_id,
                codes,
            };
            batch.put_cf(codebooks_cf, &[subvec_id], &codebook.encode());
        }

        // 4. Update metadata
        let training_meta = TrainingMeta {
            trained: true,
            training_size: training_vectors.len() as u32,
            kmeans_iters: 25,
        };
        batch.put_cf(meta_cf, b"training", &bincode::serialize(&training_meta)?);

        self.db.write(batch)?;
        Ok(())
    }
}
```

### Summary Table

| Question | HNSW2 | IVF-PQ |
|----------|-------|--------|
| **Easier RocksDB integration?** | **Yes** (native CFs, merge operators) | No (external system) |
| **Better for online updates?** | **Yes** (atomic, <1ms) | No (rebuild required) |
| **Better for 1B scale?** | No (5-10 days build) | **Yes** (5-10 hours) |
| **Lower engineering effort?** | **Yes** (4-6 weeks) | No (9-10 weeks) |
| **Higher throughput?** | No (500-1K QPS) | **Yes** (50K-100K QPS) |
| **Higher recall?** | **Yes** (95-99%) | No (90-95%) |
| **Simpler operations?** | **Yes** (single system) | No (dual system) |
| **GPU required?** | No | Yes |

### Final Recommendation

| Scale | Primary Choice | Reason |
|-------|----------------|--------|
| < 50M | **HNSW2** | Native integration, online updates, simpler ops |
| 50M-100M | **HNSW2 or IVF-PQ** | Evaluate based on update frequency vs throughput needs |
| > 100M | **IVF-PQ** | Only practical option for build time |
| Any + real-time | **Hybrid** | HNSW2 for recent + IVF-PQ for historical |

---

## IVF-PQ Performance: CPU vs GPU vs RocksDB

This section analyzes IVF-PQ performance across different deployment scenarios, particularly for environments without GPU acceleration.

### Deployment Scenarios

```
┌─────────────────────────────────────────────────────────────────┐
│              IVF-PQ Deployment Scenarios                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Scenario A: GPU + VRAM                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  GPU VRAM (40-80GB)                                      │    │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐        │    │
│  │  │ Centroids   │ │ Codebooks   │ │ PQ Codes    │        │    │
│  │  │ (4MB)       │ │ (1MB)       │ │ (8GB @ 1B)  │        │    │
│  │  └─────────────┘ └─────────────┘ └─────────────┘        │    │
│  │  - Massively parallel distance computation               │    │
│  │  - 50K-100K QPS                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Scenario B: CPU + RAM                                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  System RAM (64-256GB)                                   │    │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐        │    │
│  │  │ Centroids   │ │ Codebooks   │ │ PQ Codes    │        │    │
│  │  │ (4MB)       │ │ (1MB)       │ │ (8GB @ 1B)  │        │    │
│  │  └─────────────┘ └─────────────┘ └─────────────┘        │    │
│  │  - SIMD vectorization (AVX2/AVX-512)                     │    │
│  │  - 1K-5K QPS                                             │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Scenario C: CPU + RocksDB (Disk)                                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Block Cache (1-10GB)    │    RocksDB (SSD)             │    │
│  │  ┌─────────────┐         │    ┌─────────────────────┐   │    │
│  │  │ Hot clusters│ ◄───────┼───►│ All PQ codes        │   │    │
│  │  │ Centroids   │         │    │ Centroids           │   │    │
│  │  │ Codebooks   │         │    │ Codebooks           │   │    │
│  │  └─────────────┘         │    └─────────────────────┘   │    │
│  │  - Disk I/O for cold clusters                            │    │
│  │  - 100-1K QPS                                            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Performance Comparison

| Metric | GPU + VRAM | CPU + RAM | CPU + RocksDB |
|--------|------------|-----------|---------------|
| **Search QPS (single)** | 10K-50K | 1K-5K | 100-500 |
| **Search QPS (batch 1K)** | 50K-100K | 2K-10K | 500-2K |
| **Search latency (p50)** | 0.1-0.5ms | 1-5ms | 5-20ms |
| **Search latency (p99)** | 1-2ms | 10-20ms | 50-100ms |
| **Build 1M** | 1-2 min | 5-15 min | 10-30 min |
| **Build 1B** | 5-10 hours | 2-5 days | 5-10 days |
| **Memory (1B)** | 8-50GB VRAM | 8-50GB RAM | 1-10GB cache |
| **Max scale** | ~1-5B (multi-GPU) | ~1-5B (if RAM) | Unlimited |

### Why CPU is 10-50× Slower than GPU

```
┌─────────────────────────────────────────────────────────────────┐
│                Distance Computation Comparison                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  GPU (CUDA cores):                                               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  10,000+ parallel threads                                │    │
│  │  Each thread: compute distance for one vector            │    │
│  │  All nprobe clusters processed simultaneously            │    │
│  │  Memory bandwidth: 1-2 TB/s (HBM)                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  CPU (AVX-512):                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  16-64 parallel lanes (SIMD)                             │    │
│  │  Process 16 distances per instruction                    │    │
│  │  Sequential cluster iteration                            │    │
│  │  Memory bandwidth: 50-100 GB/s (DDR5)                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Parallelism gap: 10,000 / 64 = ~150× fewer parallel ops        │
│  Bandwidth gap: 2000 / 100 = ~20× less bandwidth                │
│  Actual gap: 10-50× (memory-bound workload)                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### CPU + RAM Performance by Scale

| Scale | nprobe=10 | nprobe=50 | nprobe=100 |
|-------|-----------|-----------|------------|
| **1M vectors** | | | |
| - QPS (single) | 5,000 | 2,000 | 1,000 |
| - Latency (p50) | 0.5ms | 1ms | 2ms |
| - Recall@10 | 85% | 92% | 95% |
| **100M vectors** | | | |
| - QPS (single) | 2,000 | 800 | 400 |
| - Latency (p50) | 1ms | 3ms | 5ms |
| - Recall@10 | 80% | 90% | 93% |
| **1B vectors** | | | |
| - QPS (single) | 500 | 200 | 100 |
| - Latency (p50) | 5ms | 15ms | 30ms |
| - Recall@10 | 75% | 88% | 92% |

### RocksDB-Backed IVF-PQ: Latency Breakdown

```
┌─────────────────────────────────────────────────────────────────┐
│              RocksDB IVF-PQ Search Breakdown                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Find nearest clusters (centroids in cache):     ~0.1ms      │
│                                                                  │
│  2. Load PQ codes for nprobe clusters:                          │
│     ┌─────────────────────────────────────────────────────┐     │
│     │ If in block cache (hot):   ~0.5ms per cluster       │     │
│     │ If on SSD (cold):          ~2-5ms per cluster       │     │
│     │ nprobe=10, 50% cache hit:  ~15ms                    │     │
│     └─────────────────────────────────────────────────────┘     │
│                                                                  │
│  3. Compute PQ distances:                           ~1-5ms      │
│                                                                  │
│  4. Refinement (optional, if vectors stored):                   │
│     ┌─────────────────────────────────────────────────────┐     │
│     │ Fetch k×4 vectors from disk:  ~10-50ms              │     │
│     │ Compute exact distances:      ~1-2ms                │     │
│     └─────────────────────────────────────────────────────┘     │
│                                                                  │
│  Total (no refinement):   ~20ms                                 │
│  Total (with refinement): ~50-70ms                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### CPU IVF-PQ Optimizations

```rust
// Key optimizations for CPU IVF-PQ

// 1. SIMD distance tables (AVX2/AVX-512)
#[cfg(target_arch = "x86_64")]
fn compute_pq_distance_simd(pq_codes: &[u8], tables: &[&[f32]]) -> f32 {
    use std::arch::x86_64::*;

    unsafe {
        let mut sum = _mm256_setzero_ps();

        // Process 8 subvectors at a time with vectorized gather
        for i in (0..pq_codes.len()).step_by(8) {
            let distances = gather_distances_avx(&pq_codes[i..], &tables[i..]);
            sum = _mm256_add_ps(sum, distances);
        }

        horizontal_sum_avx(sum)
    }
}

// 2. Batch queries to amortize centroid computation
fn search_batch(queries: &[Vec<f32>], k: usize, nprobe: usize) -> Vec<Vec<(f32, u32)>> {
    // Compute all query-centroid distances at once
    let all_cluster_dists = batch_centroid_distances(queries, &centroids);

    // Process queries in parallel using rayon
    queries.par_iter()
        .enumerate()
        .map(|(i, query)| {
            let probe_clusters = top_k_clusters(&all_cluster_dists[i], nprobe);
            search_clusters(query, &probe_clusters, k)
        })
        .collect()
}

// 3. Prefetch PQ codes for next cluster (hide memory latency)
fn search_clusters(query: &[f32], clusters: &[u32], k: usize) -> Vec<(f32, u32)> {
    let mut candidates = Vec::new();

    for (i, &cluster_id) in clusters.iter().enumerate() {
        // Prefetch next cluster while processing current
        if i + 1 < clusters.len() {
            prefetch_cluster(clusters[i + 1]);
        }

        for (vector_id, pq_codes) in get_cluster_codes(cluster_id) {
            let dist = compute_pq_distance_simd(pq_codes, &distance_tables);
            candidates.push((dist, vector_id));
        }
    }

    top_k(&mut candidates, k)
}
```

### RocksDB Optimizations for IVF-PQ

```rust
impl RocksDbIvfPq {
    // 1. Cache centroids and codebooks in memory (small, always hot)
    pub fn new(db: DB) -> Self {
        let centroids = Self::load_all_centroids(&db);  // 4MB, keep in RAM
        let codebooks = Self::load_all_codebooks(&db);  // 1MB, keep in RAM
        Self { db, centroids, codebooks, .. }
    }

    // 2. Parallel cluster loading with async I/O
    pub async fn search(&self, query: &[f32], k: usize, nprobe: usize) -> Result<Vec<(f32, u32)>> {
        let probe_clusters = self.find_nearest_clusters(query, nprobe);

        // Load all clusters in parallel
        let cluster_data: Vec<_> = futures::future::join_all(
            probe_clusters.iter().map(|&cid| self.load_cluster_async(cid))
        ).await;

        // Process all codes with SIMD
        let mut candidates = Vec::new();
        for codes in cluster_data {
            for (vid, pq_codes) in codes? {
                let dist = self.compute_pq_distance_simd(&pq_codes);
                candidates.push((dist, vid));
            }
        }

        top_k(&mut candidates, k);
        Ok(candidates)
    }

    // 3. Tune RocksDB for IVF-PQ workload
    pub fn configure_rocksdb(opts: &mut Options) {
        // Large block cache for PQ codes
        let cache = Cache::new_lru_cache(4 * 1024 * 1024 * 1024);  // 4GB
        let mut block_opts = BlockBasedOptions::default();
        block_opts.set_block_cache(&cache);
        block_opts.set_block_size(16 * 1024);  // 16KB blocks
        block_opts.set_cache_index_and_filter_blocks(true);

        // Bloom filter for cluster lookups
        block_opts.set_bloom_filter(10.0, false);

        opts.set_block_based_table_factory(&block_opts);
    }
}
```

### Comparison: IVF-PQ vs HNSW2 (Both on RocksDB)

| Metric | IVF-PQ (CPU+RAM) | IVF-PQ (RocksDB) | HNSW2 (RocksDB) |
|--------|------------------|------------------|-----------------|
| **QPS @ 1M** | 2,000-5,000 | 200-500 | 1,000-5,000 |
| **Recall@10** | 85-95% | 85-95% | **95-99%** |
| **Build 1M** | 5-15 min | 10-30 min | **3-5 min** |
| **Online insert** | Rebuild | Incremental* | **<1ms** |
| **Memory @ 1M** | 8-50 MB | 10-50 MB | 100 MB |
| **Memory @ 1B** | 8-50 GB | 1-10 GB cache | 100 GB |
| **Integration** | External | Native | **Native** |
| **Compression** | **512×** | **512×** | 1× |

*IVF-PQ can do incremental inserts but recall may degrade without periodic retraining.

### Hardware Cost Comparison

| Hardware | GPU + VRAM | CPU + RAM | CPU + RocksDB |
|----------|------------|-----------|---------------|
| **For 1B vectors** | | | |
| GPU | $10-50K (A100) | - | - |
| RAM | 64GB ($200) | 64GB RAM ($200) | 16GB ($50) |
| Storage | 1TB NVMe ($100) | 1TB NVMe ($100) | 1TB NVMe ($100) |
| **Total** | **$10-50K** | **$300** | **$150** |
| **Performance** | 50K QPS | 500 QPS | 100 QPS |
| **Cost/QPS** | $0.20-1.00 | $0.60 | $1.50 |

### When to Use Each Scenario

| Use Case | Recommended | Why |
|----------|-------------|-----|
| Real-time search, <100M | GPU + VRAM | Best latency |
| Real-time search, >1B | Multi-GPU sharding | Only option for scale |
| Batch analytics, any scale | CPU + RAM | Cost-effective |
| Memory-constrained | CPU + RocksDB | Fits any hardware |
| Need ACID/persistence | CPU + RocksDB | Built-in durability |
| Cost-sensitive | CPU + RocksDB | Cheapest hardware |
| **<50M, online updates needed** | **HNSW2** | Better recall, true online updates |

### Key Takeaways

1. **Without GPU, IVF-PQ is 10-50× slower** but still achieves 500-2000 QPS at 1B scale
2. **RocksDB adds ~10-50ms latency** but enables persistence and unlimited scale
3. **For <50M vectors without GPU, HNSW2 is likely better**:
   - Higher recall (95-99% vs 85-95%)
   - Native RocksDB integration
   - True online updates (<1ms insert)
4. **CPU IVF-PQ best fits**:
   - Scale >100M where HNSW2 build time is impractical
   - Batch workloads where 50-100ms latency is acceptable
   - Memory-constrained environments
   - When you need 512× compression

---

## Recommendations

### Short-term (< 10M vectors): Keep HNSW

Current HNSW with incremental optimizations (batched flush, connection counting) can handle up to 10M vectors:
- Build time: 2-4 hours (acceptable for batch)
- Recall: 95%+ (best in class)
- No GPU dependency

### Medium-term (10M-100M): Add IVF-PQ Option

For 10M-100M vectors, IVF-PQ with cuVS is compelling:
- Build time: 30-60 min vs 38 days
- Memory: 5 GB GPU vs 400 GB RAM
- Recall: 90-95% with refinement

### Long-term (100M-1B): Require GPU

At 100M+ vectors, GPU acceleration is essential:
- CAGRA for highest recall + speed
- IVF-PQ for memory efficiency
- Build on GPU, optionally search on CPU

### Hybrid Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                    Recommended Deployment                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   < 1M vectors:                                                  │
│   └── HNSW (current) - best recall, simple deployment           │
│                                                                  │
│   1M - 10M vectors:                                              │
│   └── HNSW with optimizations OR IVF-PQ (GPU)                   │
│       Choice depends on: GPU availability, latency requirements │
│                                                                  │
│   10M - 100M vectors:                                            │
│   └── IVF-PQ (GPU) - only practical option                      │
│       Optional: CAGRA if recall is critical                     │
│                                                                  │
│   100M+ vectors:                                                 │
│   └── IVF-PQ (GPU) with multi-GPU sharding                      │
│       CAGRA → HNSW export for CPU search fallback               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Hardware Requirements for 1B Vectors

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 1× A100 40GB | 2-4× A100 80GB |
| GPU Memory | 40 GB | 160-320 GB (multi) |
| System RAM | 128 GB | 256 GB |
| Storage | 1 TB NVMe | 2+ TB NVMe |
| CUDA | 12.0+ | 13.0 |

### Cost-Benefit Analysis

| Approach | Hardware Cost | Engineering Effort | Time to 1B |
|----------|---------------|-------------------|------------|
| Current HNSW | $0 | 2 weeks optimize | 460 days |
| HNSW2 (redesign) | $0 | 4-6 weeks | 5-10 days |
| cuVS IVF-PQ | $10-50K (GPU) | 9-10 weeks | 5-10 hours |
| cuVS CAGRA | $10-50K (GPU) | 9-10 weeks | 3-5 hours |

**Recommendation**: For production at 100M+ vectors, the GPU investment pays for itself in engineering time and operational efficiency.

---

## References

### Algorithms

- [IVF-PQ Deep Dive (NVIDIA)](https://developer.nvidia.com/blog/accelerating-vector-search-nvidia-cuvs-ivf-pq-deep-dive-part-1/)
- [CAGRA Paper (arXiv)](https://arxiv.org/abs/2308.15136)
- [Product Quantization (LanceDB)](https://lancedb.com/blog/benchmarking-lancedb-92b01032874a-2/)
- [IVF-PQ Explained (Milvus)](https://milvus.io/docs/ivf-pq.md)

### Libraries

- [cuVS (NVIDIA)](https://developer.nvidia.com/cuvs)
- [cuVS GitHub](https://github.com/rapidsai/cuvs)
- [Faiss (Meta)](https://github.com/facebookresearch/faiss)
- [faiss-rs (Rust bindings)](https://github.com/Enet4/faiss-rs)

### Benchmarks

- [GPU-accelerated Vector Search (OpenSearch)](https://opensearch.org/blog/gpu-accelerated-vector-search-opensearch-new-frontier/)
- [cuVS in Faiss (Meta Engineering)](https://engineering.fb.com/2025/05/08/data-infrastructure/accelerating-gpu-indexes-in-faiss-with-nvidia-cuvs/)
- [HNSW vs CAGRA (Vectroid)](https://www.vectroid.com/resources/hnsw-vs-cagra-gpu-vs-cpu-ann-algorithms)
- [IVF vs HNSW (MyScale)](https://www.myscale.com/blog/hnsw-vs-ivf-explained-powerful-comparison/)

### Comparisons

- [ANN Benchmarks](https://ann-benchmarks.com/)
- [Faiss Indexing 1M Vectors](https://github.com/facebookresearch/faiss/wiki/Indexing-1M-vectors)
- [IVF+PQ+HNSW for Billion Scale](https://towardsdatascience.com/ivfpq-hnsw-for-billion-scale-similarity-search-89ff2f89d90e/)
