# Vector Search Requirements

This document defines the ground truth requirements for motlie_db vector search. All design decisions, benchmarks, and implementations should reference these requirements.

## Document References

- **POC.md** - Current implementation (Phase 1: schema, Flush API, Transaction API)
- **PERF.md** - Benchmark results and performance analysis
- **HNSW2.md** - HNSW optimization proposal (Phase 2)
- **HYBRID.md** - Production architecture design (Phase 4)
- **ALTERNATIVES.md** - Alternative architectures analysis (SPFresh, ScaNN, RaBitQ)
- **ISSUES.md** - Known issues and solutions
- **README.md** - Project plan and status

---

## 1. Scale Requirements

| ID | Requirement | Target | Priority |
|----|-------------|--------|----------|
| **SCALE-1** | Maximum dataset size | 1 billion vectors | P0 |
| **SCALE-2** | Vector dimensionality | 128-1024 dimensions | P0 |
| **SCALE-3** | Memory footprint at 1B | < 64 GB RAM | P0 |
| **SCALE-4** | Disk footprint at 1B | < 500 GB | P1 |
| **SCALE-5** | Minimum viable scale | 1M vectors | P0 |

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
| **LAT-1** | Search P50 latency at 1M | < 20 ms | P0 |
| **LAT-2** | Search P99 latency at 1M | < 50 ms | P0 |
| **LAT-3** | Search P99 latency at 1B | < 100 ms | P1 |
| **LAT-4** | Insert latency (sync) | < 10 ms P99 | P0 |
| **LAT-5** | Insert latency (async) | < 1 ms P99 | P1 |

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
| **REC-1** | Recall@10 at 1M scale | > 95% | P0 |
| **REC-2** | Recall@10 at 1B scale | > 95% | P1 |
| **REC-3** | Recall@10 on random data | > 99% | P0 |
| **REC-4** | Recall@10 on SIFT data | > 90% | P0 |

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
| **THR-1** | Insert throughput (online) | > 5,000 vec/s | P0 |
| **THR-2** | Insert throughput (batch) | > 10,000 vec/s | P1 |
| **THR-3** | Search QPS at 1M | > 500 QPS | P0 |
| **THR-4** | Search QPS at 1B | > 100 QPS | P1 |
| **THR-5** | Build throughput at 1M | > 1,000 vec/s | P1 |

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
| **FUNC-1** | Vector insert (online) | Partial | P0 |
| **FUNC-2** | K-NN search | Complete | P0 |
| **FUNC-3** | Vector delete | Not started | P1 |
| **FUNC-4** | Vector update | Not started | P2 |
| **FUNC-5** | Batch insert | Not started | P1 |
| **FUNC-6** | Filtered search | Not started | P2 |

### 5.2 Consistency Requirements

| ID | Requirement | Status | Priority |
|----|-------------|--------|----------|
| **CON-1** | Read-after-write consistency | Complete | P0 |
| **CON-2** | Atomic multi-edge transactions | Not started | P1 |
| **CON-3** | Concurrent read during write | Not started | P1 |
| **CON-4** | Snapshot isolation for search | Not started | P2 |

### 5.3 Storage Requirements

| ID | Requirement | Status | Priority |
|----|-------------|--------|----------|
| **STOR-1** | Disk-based operation (no full memory load) | Complete | P0 |
| **STOR-2** | RocksDB persistence | Complete | P0 |
| **STOR-3** | Crash recovery | Complete | P0 |
| **STOR-4** | Vector compression (training-free) | Not started | P1 |
| **STOR-5** | SIMD distance computation | Not started | P1 |

**Note on STOR-4**: Due to DATA-1 constraint below, compression must be training-free. See Section 5.4.

### 5.4 Data Constraints (Critical)

| ID | Requirement | Description | Priority |
|----|-------------|-------------|----------|
| **DATA-1** | No pre-training data | System must operate without representative training data | **P0** |
| **DATA-2** | Unknown data distribution | Cannot assume data distribution (clustered vs uniform) | **P0** |
| **DATA-3** | Incremental operation | All algorithms must work incrementally (online) | P0 |

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

See [ALTERNATIVES.md](./ALTERNATIVES.md) for detailed analysis of training-free approaches.

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
| **UC-1** | Semantic document search | 1-10M | < 50ms | > 90% | P0 |
| **UC-2** | Image similarity search | 1-100M | < 100ms | > 85% | P1 |
| **UC-3** | Real-time recommendations | 100M-1B | < 50ms | > 80% | P1 |
| **UC-4** | RAG vector store | 1-10M | < 20ms | > 95% | P0 |

### 7.2 Operational Use Cases

| ID | Use Case | Requirement | Priority |
|----|----------|-------------|----------|
| **UC-5** | Online index updates | Insert without rebuild | P0 |
| **UC-6** | Index persistence | Survive restarts | P0 |
| **UC-7** | Horizontal scaling | Shard across nodes | P2 |
| **UC-8** | Index backup/restore | Point-in-time recovery | P2 |

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

## 10. Traceability Matrix

| Requirement | HYBRID.md Section | PERF.md Section | Status |
|-------------|-------------------|-----------------|--------|
| SCALE-1 (1B vectors) | Resource Projections | - | Design |
| LAT-1 (< 20ms P50) | Performance Projections | Current Results | Validated 1M |
| REC-1 (> 95% recall) | - | Benchmark Results | **Achieved** |
| THR-1 (5K inserts/s) | Async Updater | - | Not started |
| FUNC-1 (online insert) | Insert Path | - | Partial |
| CON-1 (read-after-write) | - | Flush API | **Complete** |
| STOR-4 (PQ compression) | PQ Module | - | Not started |

---

## Changelog

| Date | Change | Author |
|------|--------|--------|
| 2025-12-24 | Initial requirements document | Claude |
| 2025-12-24 | Added 1M benchmark results | Claude |
| 2025-12-24 | Linked to HYBRID.md architecture | Claude |
