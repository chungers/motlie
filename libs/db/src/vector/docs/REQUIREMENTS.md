# Vector Search Requirements

This document defines the ground truth requirements for motlie_db vector search. All design decisions, benchmarks, and implementations should reference these requirements.

## Document References

**Design Documents** (in `examples/vector/`):
- **[POC.md](../../../../../examples/vector/POC.md)** - Current implementation (Phase 1: schema, Flush API, Transaction API)
- **[PERF.md](../../../../../examples/vector/PERF.md)** - Benchmark results and performance analysis
- **[HNSW2.md](../../../../../examples/vector/HNSW2.md)** - HNSW optimization proposal (Phase 2)
- **[HYBRID.md](../../../../../examples/vector/HYBRID.md)** - Production architecture design (Phase 4)
- **[ALTERNATIVES.md](../../../../../examples/vector/ALTERNATIVES.md)** - Alternative architectures analysis (SPFresh, ScaNN, RaBitQ)
- **[ISSUES.md](../../../../../examples/vector/ISSUES.md)** - Known issues and solutions
- **[SIMD.md](../../../../../examples/vector/SIMD.md)** - SIMD acceleration design (implemented in `motlie_core::distance`)

**Implementation Roadmap**:
- **[ROADMAP.md](./ROADMAP.md)** - Detailed implementation plan and task breakdown

---

## 1. Scale Requirements

| ID | Requirement | Target | Priority |
|----|-------------|--------|----------|
| <a id="scale-1"></a>**SCALE-1** | Maximum dataset size | 1 billion vectors | P0 |
| <a id="scale-2"></a>**SCALE-2** | Vector dimensionality | 128-1024 dimensions | P0 |
| <a id="scale-3"></a>**SCALE-3** | Memory footprint at 1B | < 64 GB RAM | P0 |
| <a id="scale-4"></a>**SCALE-4** | Disk footprint at 1B | < 500 GB | P1 |
| <a id="scale-5"></a>**SCALE-5** | Minimum viable scale | 1M vectors | P0 |

### Scale Projections

| Vectors | Memory (Hybrid) | Disk (Hybrid) | Reference |
|---------|-----------------|---------------|-----------|
| 1K | < 100 MB | < 10 MB | Validated |
| 10K | < 200 MB | < 100 MB | Validated |
| 100K | < 500 MB | < 1 GB | Validated |
| 1M | < 2 GB | < 5 GB | Validated |
| 10M | < 8 GB | < 50 GB | Projected |
| 100M | < 20 GB | < 150 GB | Projected |
| 1B | < 64 GB | < 500 GB | Projected (HYBRID.md) |

---

## 2. Latency Requirements

| ID | Requirement | Target | Priority |
|----|-------------|--------|----------|
| <a id="lat-1"></a>**LAT-1** | Search P50 latency at 1M | < 20 ms | P0 |
| <a id="lat-2"></a>**LAT-2** | Search P99 latency at 1M | < 50 ms | P0 |
| <a id="lat-3"></a>**LAT-3** | Search P99 latency at 1B | < 100 ms | P1 |
| <a id="lat-4"></a>**LAT-4** | Insert latency (sync) | < 10 ms P99 | P0 |
| <a id="lat-5"></a>**LAT-5** | Insert latency (async) | < 1 ms P99 | P1 |

### Current Latency Results

| Scale | Algorithm | Search P50 | Search P99 | Reference |
|-------|-----------|------------|------------|-----------|
| 1K | HNSW | 8.6 ms | ~15 ms | PERF.md |
| 10K | HNSW | 15.4 ms | ~25 ms | PERF.md |
| 100K | HNSW | 19.0 ms | ~30 ms | PERF.md |
| 1M | HNSW | 21.5 ms | ~40 ms | PERF.md |
| 1K | Vamana | 4.7 ms | ~8 ms | PERF.md |
| 10K | Vamana | 7.0 ms | ~12 ms | PERF.md |
| 100K | Vamana | 7.1 ms | ~15 ms | PERF.md |

---

## 3. Recall Requirements

| ID | Requirement | Target | Priority |
|----|-------------|--------|----------|
| <a id="rec-1"></a>**REC-1** | Recall@10 at 1M scale | > 95% | P0 |
| <a id="rec-2"></a>**REC-2** | Recall@10 at 1B scale | > 95% | P1 |
| <a id="rec-3"></a>**REC-3** | Recall@10 on random data | > 99% | P0 |
| <a id="rec-4"></a>**REC-4** | Recall@10 on SIFT data | > 90% | P0 |

### Current Recall Results

| Scale | Algorithm | Recall@10 | Notes | Reference |
|-------|-----------|-----------|-------|-----------|
| 1K | HNSW | 52.6% | Below target | PERF.md |
| 10K | HNSW | 80.7% | Below target | PERF.md |
| 100K | HNSW | 81.7% | Below target | PERF.md |
| 1M | HNSW | **95.3%** | **Meets REC-1** | PERF.md |
| 1K | Vamana (L=100) | 61.0% | Below target | PERF.md |
| 10K | Vamana (L=100) | 77.8% | Below target | PERF.md |
| 1M | Vamana (L=200) | 81.9% | Below target | PERF.md |
| 1K | Random | 99.6% | **Meets REC-3** | PERF.md |

### Reference: Industry Baselines

| Implementation | Recall@10 | QPS | Notes |
|----------------|-----------|-----|-------|
| hnswlib (1M) | 98.5% | 16,108 | In-memory, SIMD |
| Faiss HNSW (1M) | 97.8% | ~30,000 | In-memory, SIMD |
| motlie_db HNSW (1M) | 95.3% | 47 | Disk-based |

---

## 4. Throughput Requirements

| ID | Requirement | Target | Priority |
|----|-------------|--------|----------|
| <a id="thr-1"></a>**THR-1** | Insert throughput (online) | > 5,000 vec/s | P0 |
| <a id="thr-2"></a>**THR-2** | Insert throughput (batch) | > 10,000 vec/s | P1 |
| <a id="thr-3"></a>**THR-3** | Search QPS at 1M | > 500 QPS | P0 |
| <a id="thr-4"></a>**THR-4** | Search QPS at 1B | > 100 QPS | P1 |
| <a id="thr-5"></a>**THR-5** | Build throughput at 1M | > 1,000 vec/s | P1 |

### Current Throughput Results

| Scale | Algorithm | Insert | Search QPS | Reference |
|-------|-----------|--------|------------|-----------|
| 1K | HNSW | 103.6/s | 117 | PERF.md |
| 10K | HNSW | 68.1/s | 65 | PERF.md |
| 100K | HNSW | 49.8/s | 53 | PERF.md |
| 1M | HNSW | 39.9/s | 47 | PERF.md |
| 1K | Vamana | 72.7/s | 213 | PERF.md |
| 10K | Vamana | 53.8/s | 143 | PERF.md |

**Gap Analysis**: Current insert throughput (~40-100/s) is 50-100x below THR-1 target (5,000/s). See HYBRID.md for optimization path.

---

## 5. Functionality Requirements

### 5.1 Core Operations

| ID | Requirement | Status | Priority |
|----|-------------|--------|----------|
| <a id="func-1"></a>**FUNC-1** | Vector insert (online) | Partial | P0 |
| <a id="func-2"></a>**FUNC-2** | K-NN search | Complete | P0 |
| <a id="func-3"></a>**FUNC-3** | Vector delete | Not started | P1 |
| <a id="func-4"></a>**FUNC-4** | Vector update | Not started | P2 |
| <a id="func-5"></a>**FUNC-5** | Batch insert | Not started | P1 |
| <a id="func-6"></a>**FUNC-6** | Filtered search | Not started | P2 |
| <a id="func-7"></a>**FUNC-7** | Multi-embedding support | Not started | P0 |

**FUNC-7 Details**: A single document (ULID) may be embedded in multiple embedding spaces simultaneously (e.g., "qwen3", "gemma", "openai-ada-002"). Each embedding space maintains its own HNSW graph, and the same document will have different internal vec_ids and graph positions in each space. This is required to support:
- Embedding model migration (run old and new models in parallel)
- Multi-modal embeddings (text + image for same document)
- A/B testing of embedding strategies
- Ensemble search across multiple embedding models

### 5.2 Consistency Requirements

| ID | Requirement | Status | Priority |
|----|-------------|--------|----------|
| <a id="con-1"></a>**CON-1** | Read-after-write consistency | Complete | P0 |
| <a id="con-2"></a>**CON-2** | Atomic multi-edge transactions | Not started | P1 |
| <a id="con-3"></a>**CON-3** | Concurrent read during write | Not started | P1 |
| <a id="con-4"></a>**CON-4** | Snapshot isolation for search | Not started | P2 |

### 5.3 Storage Requirements

| ID | Requirement | Status | Priority |
|----|-------------|--------|----------|
| <a id="stor-1"></a>**STOR-1** | Disk-based operation (no full memory load) | Complete | P0 |
| <a id="stor-2"></a>**STOR-2** | RocksDB persistence | Complete | P0 |
| <a id="stor-3"></a>**STOR-3** | Crash recovery | Complete | P0 |
| <a id="stor-4"></a>**STOR-4** | Vector compression (training-free) | Not started | P1 |
| <a id="stor-5"></a>**STOR-5** | SIMD distance computation | Complete | P1 |

**Note on STOR-4**: Due to DATA-1 constraint below, compression must be training-free. See Section 5.4.

### 5.4 Data Constraints (Critical)

| ID | Requirement | Description | Priority |
|----|-------------|-------------|----------|
| <a id="data-1"></a>**DATA-1** | No pre-training data | System must operate without representative training data | **P0** |
| <a id="data-2"></a>**DATA-2** | Unknown data distribution | Cannot assume data distribution (clustered vs uniform) | **P0** |
| <a id="data-3"></a>**DATA-3** | Incremental operation | All algorithms must work incrementally (online) | P0 |

**Rationale**: motlie_db is a general-purpose graph database. Vector search must work with any embedding type without prior knowledge of the data distribution. This is fundamentally different from purpose-built vector databases that can pre-train on known datasets.

**Impact on Algorithm Selection**:

| Algorithm/Technique | Training Required | DATA-1 Compliant | Notes |
|---------------------|-------------------|------------------|-------|
| HNSW (graph) | No | **Yes** | Builds incrementally |
| Vamana (graph) | No | **Yes** | Builds incrementally |
| Product Quantization (PQ) | Yes (k-means codebooks) | **No** | Requires training data |
| ScaNN (anisotropic) | Yes (learned loss) | **No** | Requires training data |
| SPANN/SPFresh (clusters) | Yes (k-means centroids) | **No** | Requires training data |
| **RaBitQ (binary)** | **No** (random rotation) | **Yes** | Training-free quantization |
| Scalar Quantization | No | **Yes** | Simple but less effective |

**Implication for STOR-4**: Traditional Product Quantization violates DATA-1. Alternatives:
1. **RaBitQ** - Random rotation matrix (no training), O(1/√D) error bound
2. **Scalar Quantization** - Min/max normalization (no training), simpler but lower compression
3. **Online PQ** - Learn codebooks incrementally (complex, violates spirit of DATA-1)

See **ALTERNATIVES.md** for detailed analysis of training-free approaches.

---

## 6. Algorithm Parameters

### 6.1 HNSW Parameters

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| **M** | 16 | 8-64 | Connections per node. Higher = better recall, more memory |
| **M_max0** | 32 | 2*M | Max connections at layer 0 |
| **ef_construction** | 200 | 50-500 | Build beam width. Higher = better recall, slower build |
| **ef_search** | 100 | 10-500 | Query beam width. Higher = better recall, slower search |
| **m_L** | 1/ln(M) | - | Level multiplier (derived) |

### 6.2 Vamana Parameters

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| **R** | 64 | 32-128 | Max out-degree. Higher = better recall, more memory |
| **L** | 100-200 | 50-500 | Search list size. Higher = better recall, slower build |
| **alpha** | 1.2 | 1.0-2.0 | RNG pruning threshold. Higher = more edges |
| **num_passes** | 3 | 2-5 | Construction passes. Higher = better graph quality |

### 6.3 Hybrid Architecture Parameters (HYBRID.md)

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| **num_subquantizers** | 8 | 4-16 | PQ subvectors. 8 = 8 bytes/vector |
| **num_centroids** | 256 | 64-256 | PQ codebook size |
| **rerank_k** | 2*k | k-10*k | Candidates for exact distance reranking |
| **batch_size** | 100 | 10-1000 | Async updater batch size |
| **batch_timeout** | 100ms | 10-500ms | Max wait before flush |

---

## 7. Use Cases

### 7.1 Primary Use Cases

| ID | Use Case | Scale | Latency | Recall | Priority |
|----|----------|-------|---------|--------|----------|
| <a id="uc-1"></a>**UC-1** | Semantic document search | 1-10M | < 50ms | > 90% | P0 |
| <a id="uc-2"></a>**UC-2** | Image similarity search | 1-100M | < 100ms | > 85% | P1 |
| <a id="uc-3"></a>**UC-3** | Real-time recommendations | 100M-1B | < 50ms | > 80% | P1 |
| <a id="uc-4"></a>**UC-4** | RAG vector store | 1-10M | < 20ms | > 95% | P0 |

### 7.2 Operational Use Cases

| ID | Use Case | Requirement | Priority |
|----|----------|-------------|----------|
| <a id="uc-5"></a>**UC-5** | Online index updates | Insert without rebuild | P0 |
| <a id="uc-6"></a>**UC-6** | Index persistence | Survive restarts | P0 |
| <a id="uc-7"></a>**UC-7** | Horizontal scaling | Shard across nodes | P2 |
| <a id="uc-8"></a>**UC-8** | Index backup/restore | Point-in-time recovery | P2 |

---

## 8. Constraint Summary

### 8.1 Hard Constraints (Must Meet)

| Constraint | Limit | Rationale |
|------------|-------|-----------|
| Memory at 1B vectors | < 64 GB | Single-node deployment |
| Search P99 at 1M | < 50 ms | Interactive UX |
| Recall@10 at 1M | > 95% | Quality threshold |
| Insert durability | Crash-safe | Data integrity |

### 8.2 Soft Constraints (Target)

| Constraint | Target | Stretch Goal |
|------------|--------|--------------|
| Insert throughput | 5,000/s | 10,000/s |
| Search QPS at 1M | 500 | 2,000 |
| Disk at 1B | < 500 GB | < 200 GB |
| Build time at 1M | < 30 min | < 10 min |

---

## 9. Quality Metrics

### 9.1 Definition of Done

A feature is complete when:

1. **Correctness**: Ground truth validation passes (SIFT dataset)
2. **Performance**: Meets latency/throughput targets for its scale tier
3. **Reliability**: No data loss on crash/restart
4. **Documentation**: Updated in PERF.md with benchmark results

### 9.2 Benchmark Protocol

All benchmarks must use:
- **Dataset**: SIFT1M (or SIFT10K for quick tests)
- **Queries**: 100 queries minimum
- **Metrics**: Recall@10, P50/P99 latency, QPS, build time
- **Hardware**: Document CPU, RAM, disk type
- **Reproducibility**: Document exact command and parameters

---

## 10. Architecture Assumptions

These assumptions shape the implementation architecture and are derived from project owner decisions.

### 10.1 Temporal Filtering Strategy

| ID | Assumption | Rationale |
|----|------------|-----------|
| <a id="arch-1"></a>**ARCH-1** | Vector index is **temporal-agnostic** | Simplifies index, avoids sync complexity |
| <a id="arch-2"></a>**ARCH-2** | Temporal visibility enforced during re-ranking | Graph RocksDB is source of truth |
| <a id="arch-3"></a>**ARCH-3** | Over-fetch pattern for filtered queries | Return larger candidate set, filter by `TemporalRange` |

**Design Pattern:**

```
Vector Search (ef=200) -> ID Mapping -> Temporal Filter (Graph) -> Re-rank -> Top-K
```

**Trade-off Analysis:**
- Additional O(1) lookup per candidate after ID mapping (~5us per candidate)
- For 200 candidates: ~1ms overhead for ID mapping + ~1-2ms for temporal MultiGet
- Acceptable given recall > latency preference

**Benefits:**
1. Vector index stays simple and fast
2. Single source of truth (graph) for visibility
3. Extensible to other filters (ACL, soft-delete)
4. Consistent with Tantivy pattern (fulltext defers to graph for visibility)

### 10.2 Internal ID Strategy

| ID | Assumption | Rationale |
|----|------------|-----------|
| <a id="arch-4"></a>**ARCH-4** | Vector IDs are **internal** (similar to Tantivy `doc_id`) | Memory efficiency, roaring bitmap compatibility |
| <a id="arch-5"></a>**ARCH-5** | Bi-directional mapping: internal u32 <-> ULID | Required for graph integration |
| <a id="arch-6"></a>**ARCH-6** | Reverse mapping (u32 -> ULID) is hot path | Optimized via dense array or mmap |

**Implications:**
- Vector index uses 4-byte u32 IDs internally (required for RoaringBitmap)
- API accepts/returns ULIDs; translation is internal
- Forward mapping (ULID -> u32): RocksDB-backed, insert path
- Reverse mapping (u32 -> ULID): Memory-mapped dense array, search path

### 10.3 Hardware Requirements

| ID | Assumption | Rationale |
|----|------------|-----------|
| <a id="arch-7"></a>**ARCH-7** | **SSD required** for production at 1B scale | Re-ranking latency, cold-cache performance |
| <a id="arch-8"></a>**ARCH-8** | 64 GB RAM constraint includes all components | Vector index + graph block cache + ID mapping |

### 10.4 Performance Trade-offs

| ID | Assumption | Rationale |
|----|------------|-----------|
| <a id="arch-9"></a>**ARCH-9** | **Recall > Latency** | Quality of results prioritized over speed |
| <a id="arch-10"></a>**ARCH-10** | Target 98%+ recall, accept 30-50ms latency | Acceptable for semantic search use cases |

**Parameter Implications:**

| Parameter | Latency-Optimized | Recall-Optimized (Chosen) |
|-----------|-------------------|---------------------------|
| ef_search | 50-100 | 200-500 |
| rerank_count | 20-50 | 100-200 |
| Extended-RaBitQ | 1 bit | 2-4 bits |

### 10.5 Vector Scope

| ID | Assumption | Rationale |
|----|------------|-----------|
| <a id="arch-11"></a>**ARCH-11** | Vectors represent node/edge summaries and fragments | Graph entities are the primary data model |
| <a id="arch-12"></a>**ARCH-12** | Graph RocksDB is source of truth for entity visibility | Temporal range, existence, relationships |

### 10.6 Multi-Embedding Storage Strategy

| ID | Assumption | Rationale |
|----|------------|-----------|
| <a id="arch-13"></a>**ARCH-13** | Documents may exist in **multiple embedding spaces** simultaneously | Model migration, multi-modal, A/B testing |
| <a id="arch-14"></a>**ARCH-14** | Embedding namespace uses **u64** (8 bytes) identifier | Register-aligned, <1000 embeddings expected |
| <a id="arch-15"></a>**ARCH-15** | Internal vec_ids use **u32** per embedding space | RoaringBitmap edge storage requires u32 |
| <a id="arch-16"></a>**ARCH-16** | ID mappings require embedding prefix: `[embedding: 8] + [ulid: 16]` | Same ULID maps to different vec_ids per space |
| <a id="arch-17"></a>**ARCH-17** | `Embedding` is a **rich struct** with model, dim, distance, embedder | Direct field access without registry lookup; self-describing |
| <a id="arch-18"></a>**ARCH-18** | `Embedder` trait enables **document-to-vector compute** | Optional behavior; supports Ollama, OpenAI, local models |

**Storage Key Layout** (per FUNC-7):

| CF | Key Format | Size |
|----|------------|------|
| vectors | `[embedding: 8] + [vec_id: 4]` | 12 bytes |
| edges | `[embedding: 8] + [vec_id: 4] + [layer: 1]` | 13 bytes |
| id_forward | `[embedding: 8] + [ulid: 16]` | 24 bytes |
| id_reverse | `[embedding: 8] + [vec_id: 4]` | 12 bytes |

**Why u32 for vec_id**: RoaringBitmap (used for compressed edge storage) only supports u32 values. This limits each embedding space to 4B vectors, which is acceptable given SCALE-1 target of 1B vectors total.

---

## 11. Traceability Matrix

| Requirement | Design Doc | Status |
|-------------|------------|--------|
| SCALE-1 (1B vectors) | HYBRID.md Resource Projections | Design |
| LAT-1 (< 20ms P50) | HYBRID.md Performance Projections | Validated 1M |
| REC-1 (> 95% recall) | PERF.md Benchmark Results | **Achieved** |
| THR-1 (5K inserts/s) | HYBRID.md Async Updater | Not started |
| FUNC-1 (online insert) | HYBRID.md Insert Path | Partial |
| CON-1 (read-after-write) | POC.md Flush API | **Complete** |
| STOR-4 (compression) | HNSW2.md RaBitQ | Not started |
| STOR-5 (SIMD) | `motlie_core::distance` | **Complete** |

---

## Changelog

| Date | Change | Author |
|------|--------|--------|
| 2025-12-24 | Initial requirements document | Claude |
| 2025-12-24 | Added 1M benchmark results | Claude |
| 2025-12-24 | Linked to HYBRID.md architecture | Claude |
| 2025-12-25 | Added Section 10: Architecture Assumptions (ARCH-1 to ARCH-12) | Claude Opus 4.5 |
| 2025-12-25 | Added anchor IDs to all requirements for cross-document linking | Claude Opus 4.5 |
| 2026-01-02 | Copied to libs/db/src/vector/ with updated paths | Claude Opus 4.5 |
| 2026-01-03 | Added FUNC-7 (multi-embedding support) and ARCH-13 to ARCH-16 (storage strategy) | Claude Opus 4.5 |
| 2026-01-03 | Renamed node_id to vec_id throughout for consistency with graph module terminology | Claude Opus 4.5 |
| 2026-01-03 | Added ARCH-17 (rich Embedding struct) and ARCH-18 (Embedder trait) | Claude Opus 4.5 |
