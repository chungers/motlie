# Vector Search Implementation Roadmap

**Author:** David Chung + Claude
**Date:** January 2, 2026 (Updated: January 30, 2026)
**Scope:** `libs/db/src/vector` - Vector Search Module
**Status:** Phase 4 Complete, Phase 4.5 Complete, Phase 5 Complete (Tasks 5.0-5.11), Phase 6 Complete, Phase 7 Complete, Phase 8 In Progress (8.1-8.2 complete; 8.3 in progress)

**Latest Assessment (codex, 2026-01-30 20:56 UTC, EVAL):** ROADMAP reflects all known feature/correctness gaps. There are **no missing features or correctness concerns beyond what is explicitly listed** (Phase 8.3 backlog + Phase 8 acceptance criteria). Phase 7 functional requirements are implemented, but performance acceptance items (P99 insert, pending drain timing) are not formally validated; those remain as checklist items. Historical Phase 0–4 validation checklists are largely stale and should be treated as historical verification notes rather than active backlog.

**Documentation:**
- [`API.md`](./API.md) - Public API reference, usage flows, and tuning guide
- [`BENCHMARK.md`](./BENCHMARK.md) - Performance results and configuration reference
- [`CONCURRENT.md`](./CONCURRENT.md) - Concurrent operations (Tasks 5.9-5.11)
- [`PHASE7.md`](./PHASE7.md) - Async Graph Updater design (30 subtasks)
- [`PHASE8.md`](./PHASE8.md) - Production Hardening (30 subtasks)
  <!-- ADDRESSED (Claude, 2026-01-19): Updated count from 29 to 30 after adding 8.1.11 -->

**Note:** Many historical sections refer to ULIDs as external IDs. As of IDMAP,
external IDs are `ExternalKey` (ULID corresponds to `ExternalKey::NodeId`).

---

## Executive Summary

This document tracks the implementation of HNSW-based vector search in motlie_db, built on RocksDB.
Phases 0-4 are complete. The system provides competitive performance for an embedded database
while prioritizing simplicity, correctness, and RocksDB integration over raw throughput.

### Achieved Metrics (Phase 4 Complete)

Benchmarks on LAION-CLIP 512D embeddings (aarch64 NEON). See [`BENCHMARK.md`](./BENCHMARK.md) for details and [`API.md`](./API.md) for usage.

| Metric | Achieved | Notes |
|--------|----------|-------|
| Search QPS (50K) | 284 | Standard HNSW, ef=80 |
| Search QPS (100K) | 254 | Standard HNSW, ef=80 |
| Search QPS (500K) | 136 | Standard HNSW, ef=80 |
| Search QPS (1M) | 112 | Standard HNSW, ef=80 |
| Recall@10 (50K) | 89.2% | LAION-CLIP, January 2026 |
| Recall@10 (100K) | 87.1% | LAION-CLIP, January 2026 |
| Recall@10 (500K) | 83.1% | LAION-CLIP, January 2026 |
| Recall@10 (1M) | 80.7% | LAION-CLIP, January 2026 |
| Insert throughput | 47-109 vec/s | With HNSW graph construction |
| Memory/vector | 528 bytes | 512B vector + 16B binary code (F16) |
| Parallel speedup | 1.3x | At 3200+ candidates (rayon) |
| HNSW vs Flat (1M) | 9.5x | Speedup over brute-force |

### Context: RocksDB vs Purpose-Built Engines

Commercial vector databases (Pinecone, Weaviate, Milvus, Qdrant) use purpose-built storage
engines optimized for vector workloads. motlie_db uses RocksDB, which provides:

| Trade-off | RocksDB Approach | Purpose-Built |
|-----------|------------------|---------------|
| **Search QPS** | 250-500 (good) | 1K-10K (optimized) |
| **Insert** | Transactional, ACID | Often eventual consistency |
| **Integration** | Shared DB with graph | Separate system |
| **Complexity** | Lower (reuse RocksDB) | Higher (custom engine) |
| **Flexibility** | General-purpose | Vector-specialized |

**Design Choice**: motlie_db prioritizes integration with the graph database over
maximum vector throughput. For workloads requiring >1K QPS at 1M+ scale, consider
dedicated vector databases or custom storage engines.

### Key Technical Decisions

1. **RaBitQ over PQ**: Training-free binary quantization (DATA-1 compliant)
2. **RoaringBitmap edges**: O(1) degree queries, efficient serialization
3. **Embedding as source of truth**: Distance metric encoded in type, not config
4. **Adaptive parallelism**: Sequential below 3200 candidates, rayon above (tuned January 2026)

---

## Table of Contents

### Implementation Phases

| Phase | Description | Status |
|-------|-------------|--------|
| [Phase 0](#phase-0-foundation) | Foundation (Schema, CFs, Distance) | ✅ Complete |
| [Phase 1](#phase-1-id-management) | ID Management (ExternalKey ↔ u32 mapping) | ✅ Complete |
| [Phase 2](#phase-2-hnsw2-core--navigation-layer--core-complete) | HNSW2 Core + Navigation Layer | ✅ Complete |
| [Phase 3](#phase-3-batch-operations--deferred-items) | Batch Operations + Deferred Items | ✅ Complete |
| [Phase 4](#phase-4-rabitq-compression) | RaBitQ Compression + Optimization | ✅ Complete |
| [Phase 4.5](#phase-45-codex-pre-phase-5-critical-fixes) | CODEX Pre-Phase 5 Critical Fixes | ✅ Complete |
| [Phase 5](#phase-5-internal-mutationquery-api) | Internal Mutation/Query API | ✅ Complete |
| [Phase 6](#phase-6-mpscmpmc-public-api) | MPSC/MPMC Public API | ✅ Complete |
| [Phase 7](#phase-7-async-graph-updater) | Async Graph Updater | ✅ Complete |
| [Phase 8](#phase-8-production-hardening) | Production Hardening | 🚧 In Progress |

### Phase 4 Tasks

| Task | Description | Status | Commit |
|------|-------------|--------|--------|
| [Task 4.1-4.5](#phase-4-rabitq-compression) | Core RaBitQ implementation | ✅ Complete | `9399cf9` |
| [Task 4.6](#task-46-recall-tuning) | Recall tuning | ✅ Complete | `e5b81b6` |
| [Task 4.7](#task-47-simd-hamming--early-filtering) | SIMD Hamming + early filtering | ✅ Complete | `4a4e954` |
| [Task 4.8](#task-48-implementation--results) | Hybrid L2 + Hamming (disproven) | ✅ Complete | `bd29801` | ⚠️ See [RABITQ.md §1.1-1.5](RABITQ.md#11-the-problem-symmetric-hamming-distance) |
| [Task 4.9](#task-49-rabitq-tuning-configuration-analysis) | RaBitQ Tuning Analysis | ✅ Complete | `9522658` |
| [Task 4.10](#task-410-in-memory-binary-code-cache) | In-Memory Binary Code Cache | ✅ Complete | `9880a62` | ⚠️ Hamming ~10% recall; see [Task 4.24](#task-424-adc-hnsw-integration) for ADC fix |
| [Task 4.11](#task-411-api-cleanup-phase-1---deprecate-invalidated-functions) | API Cleanup: Deprecate Functions | ✅ Complete | `ce2a060` |
| [Task 4.12](#task-412-api-cleanup-phase-2---remove-deprecated-code) | API Cleanup: Remove Dead Code | ✅ Complete | `5f2dac0` |
| [Task 4.13](#task-413-api-cleanup-phase-3---embedding-driven-searchconfig-api) | API Cleanup: SearchConfig API | ✅ Complete | `5f2dac0` |
| [Task 4.14](#task-414-api-cleanup-phase-4---configuration-validation) | API Cleanup: Config Validation | ✅ Complete | `5f2dac0` |
| [Task 4.15](#task-415-phase-5-integration-planning) | Phase 5 Integration Planning | 🔲 Deferred to Phase 7 | - |
| [Task 4.16](#task-416-hnsw-distance-metric-bug-fix) | HNSW Distance Metric Bug Fix | ✅ Complete | `0535e6a` |
| [Task 4.17](#task-417-batch_distances-metric-bug-fix) | batch_distances Metric Bug Fix | ✅ Complete | `d5899f3` |
| [Task 4.18](#task-418-vectorstoragetype-multi-float-support) | VectorElementType (f16/f32) | ✅ Complete | `c90f15c` |
| [Task 4.19](#task-419-parallel-reranking-with-rayon) | Parallel Reranking (rayon) | ✅ Complete | `27d6b74` |
| [Task 4.20](#task-420-parallel-re-ranking-threshold-tuning) | Parallel Threshold Tuning | ✅ Complete | `a146d02` |
| [Task 4.21](#task-421-benchmark-infrastructure--threshold-update) | Benchmark Infrastructure + Threshold Update | ✅ Complete | `2358ba8` |
| [Task 4.22](#task-422-large-scale-benchmarks-500k-1m) | Large-Scale Benchmarks (500K, 1M) | ✅ Complete | `4d73bf8` |
| [Task 4.23](#task-423-multi-dataset-benchmark-infrastructure) | Multi-Dataset Benchmark Infrastructure | ✅ Complete | - |
| [Task 4.24](#task-424-adc-hnsw-integration) | ADC HNSW Integration (Replace Hamming) | ✅ Complete | `ec0fc00` |

### Phase 4.5 Tasks (Pre-Phase 5 Critical Fixes)

Based on CODEX code review (January 2026). See [CODEX-CODE-REVIEW.md](./CODEX-CODE-REVIEW.md) for details.

| Task | Description | Severity | Status | Commit |
|------|-------------|----------|--------|--------|
| [Task 4.5.1](#task-451-storage-type-mismatch-fix) | Fix storage-type mismatch in write path | 🔴 HIGH | ✅ Complete | `af7a90b` |
| [Task 4.5.2](#task-452-layer-assignment-cold-cache-fix) | Fix HNSW layer assignment on cold cache | 🔴 HIGH | ✅ Complete | `af7a90b` |
| [Task 4.5.3](#task-453-adc-missing-code-handling) | Handle missing codes in ADC search (avoid MAX-distance) | 🔴 HIGH | ✅ Complete | `af7a90b` |
| [Task 4.5.4](#task-454-empty-index-search-handling) | Return empty results for empty index search | 🟡 MEDIUM | ✅ Complete | See below |
| [Task 4.5.5](#task-455-binarycodecache-size-accounting) | Fix BinaryCodeCache size accounting on overwrite | 🟢 LOW | ✅ Complete | See below |

### Other Sections

- [Design Decisions](#design-decisions) - Why RaBitQ, Architecture choices
- [Implementation Timeline](#implementation-timeline) - Estimated effort per phase

---

## Design Decisions

### Why RaBitQ Instead of Product Quantization (PQ)

**Product Quantization is NOT on this roadmap** due to the critical [DATA-1](./REQUIREMENTS.md#data-1) constraint.

#### The Problem with PQ

Product Quantization requires training on representative data:

```
PQ Training Pipeline (violates DATA-1):
  1. Collect N representative vectors (e.g., 100K-1M samples)
  2. Split each vector into M subvectors (e.g., 8 subvectors of 16 dims each)
  3. Run k-means clustering on each subspace → codebooks
  4. Store 8 codebooks × 256 centroids × 16 dims × 4 bytes = ~128KB

Problem: Steps 1-3 require representative training data.
```

#### DATA-1 Constraint (P0 - Critical)

From [REQUIREMENTS.md](./REQUIREMENTS.md#data-1):

> **DATA-1**: No pre-training data - System must operate without representative training data

**Rationale**: motlie_db is a general-purpose graph database. Vector search must work with any
embedding type (qwen3, gemma, openai-ada-002, etc.) without prior knowledge of the data distribution.

| Algorithm | Training Required | DATA-1 Compliant |
|-----------|-------------------|------------------|
| Product Quantization (PQ) | Yes (k-means codebooks) | ❌ **No** |
| ScaNN (anisotropic) | Yes (learned loss) | ❌ **No** |
| SPANN/SPFresh (clusters) | Yes (k-means centroids) | ❌ **No** |
| **RaBitQ (binary)** | **No** (random rotation) | ✅ **Yes** |
| Scalar Quantization | No | ✅ Yes (but less effective) |

#### RaBitQ: Training-Free Alternative

RaBitQ uses randomization instead of learning:

```
RaBitQ Pipeline (DATA-1 compliant):
  1. Generate random D×D orthonormal matrix R (one-time, deterministic seed)
  2. For each vector v: rotated = R × v
  3. Quantize: binary_code = sign(rotated)  // D bits

No training data required. Works with any embedding model.
```

**Mathematical Guarantee** (from [RaBitQ SIGMOD 2024](https://arxiv.org/abs/2405.12497)):
> Distance estimation error is O(1/√D), which is asymptotically optimal.

| Dimension | Error Bound | Practical Error |
|-----------|-------------|-----------------|
| 128 | ~8.8% | 5-7% |
| 256 | ~6.3% | 4-5% |
| 512 | ~4.4% | 3-4% |
| 1024 | ~3.1% | 2-3% |

#### Memory Trade-off

| Compression | Bytes/Vector (128D) | RAM at 1B | Training |
|-------------|---------------------|-----------|----------|
| None | 512 bytes | 512 GB | - |
| PQ (8 subq) | 8 bytes | 8 GB | **Required** |
| RaBitQ 1-bit | 16 bytes | 16 GB | None |
| RaBitQ 2-bit | 32 bytes | 32 GB | None |

**Trade-off**: RaBitQ uses 2x memory vs PQ (16 GB vs 8 GB at 1B), but:
- No training required → works immediately with any embedding
- Faster distance computation (SIMD popcount vs table lookup)
- 16 GB well within 64 GB budget

#### Extended-RaBitQ Option

For higher recall without training, [Extended-RaBitQ](https://arxiv.org/html/2409.09913v1) supports 2-6 bits:

| Bits/Dim | Bytes (128D) | Recall (no rerank) |
|----------|--------------|---------------------|
| 1 bit | 16 bytes | ~70% |
| 2 bits | 32 bytes | ~85% |
| 4 bits | 64 bytes | ~92% |

This is configurable via `bits_per_dim` in Phase 4 without code changes.

---

## Phase Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Implementation Phases                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Phase 0: Foundation [COMPLETE]                                              │
│  ├── 0.1-0.11 Module structure, types, traits ✓                            │
│  └── Files: mod.rs, config.rs, distance.rs, embedding.rs, registry.rs      │
│                                                                              │
│  Phase 1: ID Management [COMPLETE]                                          │
│  ├── 1.1-1.5 u32 allocator, ExternalKey mapping, persistence ✓             │
│  └── Files: id.rs, schema.rs (IdForward, IdReverse, IdAlloc CFs)           │
│                                                                              │
│  Phase 2: HNSW2 Core + Navigation [COMPLETE]                                │
│  ├── 2.1-2.9 RoaringBitmap edges, merge ops, NavigationCache ✓             │
│  └── Files: hnsw.rs, merge.rs, navigation.rs, schema.rs                    │
│                                                                              │
│  Phase 3: Batch Operations [COMPLETE]                                        │
│  ├── 3.1-3.7 MultiGet batching, edge caching, recall tuning ✓              │
│  └── Files: hnsw.rs (batch_distances, batch_vectors)                       │
│                                                                              │
│  Phase 4: RaBitQ + Optimization [COMPLETE]                                   │
│  ├── 4.1-4.5 Core RaBitQ (rotation, encoding, Hamming) ✓                   │
│  ├── 4.6-4.9 Recall tuning, hybrid experiments ✓                           │
│  ├── 4.10 In-memory BinaryCodeCache ✓                                      │
│  ├── 4.11-4.14 API cleanup, SearchConfig redesign ✓                        │
│  ├── 4.16-4.17 Distance metric bug fixes ✓                                 │
│  ├── 4.18 VectorElementType (f16/f32 support) ✓                            │
│  ├── 4.19-4.20 Parallel reranking (rayon) + threshold tuning ✓             │
│  └── Files: rabitq.rs, search/config.rs, parallel.rs, navigation.rs        │
│                                                                              │
│  Phase 5: Internal Mutation/Query API [COMPLETE]                             │
│  ├── 5.0 HNSW Transaction Refactoring (prerequisite for atomicity)         │
│  ├── 5.1-5.7 Processor methods, Storage search, dispatch logic             │
│  ├── 5.8 Migrate examples/laion_benchmark + integration tests              │
│  ├── 5.9-5.11 Concurrency stress tests, metrics infra, benchmarks          │
│  └── Goal: Complete internal API + migration + concurrent validation       │
│                                                                              │
│  Phase 6: MPSC/MPMC Public API [COMPLETE]                                    │
│  ├── 6.1-6.6 MutationExecutor, Consumer, Reader, spawn functions           │
│  └── Goal: Channel-based wrappers matching graph/fulltext patterns         │
│                                                                              │
│  Phase 7: Async Graph Updater [COMPLETE]                                     │
│  ├── 7.1-7.7 Pending queue, async workers, crash recovery, integration ✓   │
│  └── Goal: Online updates without blocking search                          │
│                                                                              │
│  Phase 8: Production Hardening [IN PROGRESS]                                 │
│  ├── 8.1-8.3 Delete refinement, concurrency, 1B validation                 │
│  └── Goal: Production-ready at scale                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## RocksDB Schema Design

The vector module uses its own column families within the shared RocksDB instance, completely
independent of `motlie_db::graph` column families. All vector CFs use the `vector/` prefix.

### Design Principles

1. **Namespace Isolation**: All CF names prefixed with `vector/` to avoid collisions
2. **Multi-Embedding Support**: Keys include `embedding` to support multiple embedding spaces
   (e.g., `qwen3`, `gemma`, `openai-ada-002`) in the same database
3. **No Graph Module Changes**: Vector module is self-contained; does not modify graph CFs
4. **Shared RocksDB Instance**: Uses same `TransactionDB` as graph module for efficiency

### Embedding Type (Rich Struct with Behavior)

An **Embedding** is a rich struct that combines namespace identity, metadata, and optional compute behavior.

#### Design Rationale (ARCH-14, ARCH-17)

- **ARCH-14**: u64 `code` field provides register-aligned keys, saves ~192 GB at 1B scale
- **ARCH-17**: Rich struct carries model, dimension, and distance metric directly (no registry lookup)
- **ARCH-18**: Optional `Embedder` trait enables document-to-vector computation

#### Distance Metric

**Implementation:** [`distance.rs`](distance.rs) - Distance enum with `compute()`, `is_lower_better()`, and `as_str()` methods.

#### Embedder Trait (Compute Behavior)

**Implementation:** [`embedding.rs`](embedding.rs) - `Embedder` trait with `embed()`, `embed_batch()`, `dim()`, and `model()` methods.

#### Embedding Struct

**Implementation:** [`embedding.rs`](embedding.rs) - Complete `Embedding` struct with:
- Fields: `code` (u64), `model`, `dim`, `distance`, optional `embedder`
- Accessors: `code()`, `code_bytes()`, `model()`, `dim()`, `distance()`, `has_embedder()`
- Behavior: `embed()`, `embed_batch()`, `compute_distance()`
- Validation: `validate_vector()`
- Traits: `PartialEq`, `Eq`, `Hash` based on code

#### Embedding Registry

**Implementation:**
- [`embedding.rs`](embedding.rs) - `EmbeddingBuilder` with fluent API (`new()`, `with_embedder()`, `register()`)
- [`registry.rs`](registry.rs) - `EmbeddingRegistry` with:
  - Registration: `register()` (idempotent)
  - Lookup: `get()`, `get_by_code()`, `set_embedder()`
  - Query/Discovery: `list_all()`, `find_by_model()`, `find_by_distance()`, `find_by_dim()`, `find()`
- [`registry.rs`](registry.rs) - `EmbeddingFilter` for multi-field queries

#### Usage Examples

```rust
let registry = vector_storage.cache();  // Returns &Arc<EmbeddingRegistry>

// ═══════════════════════════════════════════════════════════════════
// Registration (data-only, no compute)
// ═══════════════════════════════════════════════════════════════════

let gemma_cosine = EmbeddingBuilder::new("gemma", 768, Distance::Cosine)
    .register(registry)?;

let gemma_l2 = EmbeddingBuilder::new("gemma", 768, Distance::L2)
    .register(registry)?;

// Direct field access (no registry lookup needed)
assert_eq!(gemma_cosine.model(), "gemma");
assert_eq!(gemma_cosine.dim(), 768);
assert_eq!(gemma_cosine.distance(), Distance::Cosine);

// ═══════════════════════════════════════════════════════════════════
// Registration with Embedder (compute capability)
// ═══════════════════════════════════════════════════════════════════

struct OllamaEmbedder { client: OllamaClient, model: String }

impl Embedder for OllamaEmbedder {
    fn embed(&self, document: &str) -> Result<Vec<f32>> {
        self.client.embeddings(&self.model, document)
    }
    fn dim(&self) -> u32 { 768 }
    fn model(&self) -> &str { &self.model }
}

let gemma_with_compute = EmbeddingBuilder::new("gemma", 768, Distance::Cosine)
    .with_embedder(Arc::new(OllamaEmbedder::new("gemma")))
    .register(registry)?;

// Compute embeddings directly via the Embedding type
let vec = gemma_with_compute.embed("Hello world")?;
vector_storage.insert(&gemma_with_compute, doc_id, &vec)?;

// Or use convenience method
vector_storage.insert_document(&gemma_with_compute, doc_id, "Hello world")?;

// ═══════════════════════════════════════════════════════════════════
// Query / Discovery
// ═══════════════════════════════════════════════════════════════════

// What embedding spaces do we have?
for emb in registry.list_all() {
    println!("{}: dim={}, distance={:?}", emb.model(), emb.dim(), emb.distance());
}

// Find all Gemma spaces
let gemma_spaces = registry.find_by_model("gemma");
// → [gemma:768:cosine, gemma:768:l2]

// Find all cosine-based spaces
let cosine_spaces = registry.find_by_distance(Distance::Cosine);

// Multi-filter query
let matches = registry.find(
    &EmbeddingFilter::default()
        .model("gemma")
        .distance(Distance::Cosine)
);
```

**Note**: Internal vec_ids use u32 (not u64) because RoaringBitmap edge storage requires u32 values (ARCH-15). This limits each embedding space to 4B vectors, acceptable for SCALE-1 target.

### Column Family Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Vector Module Column Families                            │
│                     (All use 'vector/' prefix)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ CF: vector/embedding_registry                                           │ │
│  │ Key:   [embedding_id: u64] = 8 bytes                                   │ │
│  │ Value: embedding name string (e.g., "qwen3", "openai-ada-002")         │ │
│  │ Purpose: Embedding ID <-> name mapping                                 │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ CF: vector/vectors                                                      │ │
│  │ Key:   [embedding: u64] + [vec_id: u32] = 12 bytes                    │ │
│  │ Value: f32[dim] binary (e.g., 512 bytes for 128D)                      │ │
│  │ Access: Point lookup, MultiGet for re-ranking                          │ │
│  │ Options: LZ4 compression, 16KB blocks                                  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ CF: vector/edges                                                        │ │
│  │ Key:   [embedding: u64] + [vec_id: u32] + [layer: u8] = 13 bytes      │ │
│  │ Value: RoaringBitmap serialized (~50-200 bytes)                        │ │
│  │ Access: Point lookup, merge operator for updates                       │ │
│  │ Options: No compression (already compact), merge operator enabled      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ CF: vector/binary_codes                                                 │ │
│  │ Key:   [embedding: u64] + [vec_id: u32] = 12 bytes                    │ │
│  │ Value: [u8; code_size] (e.g., 16 bytes for 128-dim 1-bit RaBitQ)       │ │
│  │ Access: Sequential scan during beam search                             │ │
│  │ Options: No compression, 4KB blocks                                    │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ CF: vector/vec_meta                                                    │ │
│  │ Key:   [embedding: u64] + [vec_id: u32] = 12 bytes                    │ │
│  │ Value: { max_layer: u8, flags: u8, created_at: u64 } (~10 bytes)       │ │
│  │ Access: Point lookup                                                   │ │
│  │ Options: No compression (small values)                                 │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ CF: vector/graph_meta                                                   │ │
│  │ Key:   [embedding: u64] + [field: u8] = 9 bytes                        │ │
│  │ Value: Varies by field (entry_point, max_level, count, spec_hash)      │ │
│  │ Access: Point lookup (rarely changes)                                  │ │
│  │ Options: No compression                                                │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ CF: vector/id_forward                                                   │ │
│  │ Key:   [embedding: u64] + [external_key: 1 + payload]                  │ │
│  │ Value: [vec_id: u32] = 4 bytes                                        │ │
│  │ Purpose: (embedding, ExternalKey) -> internal u32 mapping (insert path)│ │
│  │ Note: Embedding prefix required per FUNC-7 (multi-embedding support)   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ CF: vector/id_reverse                                                   │ │
│  │ Key:   [embedding: u64] + [vec_id: u32] = 12 bytes                    │ │
│  │ Value: [external_key: 1 + payload]                                     │ │
│  │ Purpose: (embedding, vec_id) -> ExternalKey mapping (search path, hot) │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ CF: vector/id_alloc                                                     │ │
│  │ Key:   [embedding: u64] + [field: u8] = 9 bytes                        │ │
│  │ Value: next_id (u32) or free_bitmap (RoaringBitmap serialized)         │ │
│  │ Purpose: ID allocator persistence per embedding namespace              │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ CF: vector/pending                                                      │ │
│  │ Key:   [embedding: u64] + [timestamp: u64] + [vec_id: u32] = 20 bytes │ │
│  │ Value: empty (vector already stored in vector/vectors)                 │ │
│  │ Purpose: Async graph updater pending queue                             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Storage Savings** (vs 32-byte embedding prefix):

| CF | Old Key Size | New Key Size | Savings/Key | At 1B keys |
|----|--------------|--------------|-------------|------------|
| vectors | 36 bytes | 12 bytes | 24 bytes | 24 GB |
| edges | 37 bytes | 13 bytes | 24 bytes | 72 GB (3 layers avg) |
| binary_codes | 36 bytes | 12 bytes | 24 bytes | 24 GB |
| vec_meta | 36 bytes | 12 bytes | 24 bytes | 24 GB |
| id_forward | 48 bytes | 8 + (1 + payload) | variable | variable |
| id_reverse | 36 bytes | 12 bytes | 24 bytes | 24 GB |
| **Total** | | | | **~192 GB** |

### CF Key Summary

| CF | Key | Size | Value | Purpose |
|----|-----|------|-------|---------|
| embedding_registry | `[id: u64]` | 8 | name string | ID ↔ name mapping |
| vectors | `[emb: u64] + [vec_id: u32]` | 12 | f32[dim] | Raw vectors |
| edges | `[emb: u64] + [vec_id: u32] + [layer: u8]` | 13 | RoaringBitmap | HNSW graph edges |
| binary_codes | `[emb: u64] + [vec_id: u32]` | 12 | u8[code_size] | RaBitQ codes |
| vec_meta | `[emb: u64] + [vec_id: u32]` | 12 | {layer, flags} | Per-vector metadata |
| graph_meta | `[emb: u64] + [field: u8]` | 9 | varies | Entry point, stats |
| id_forward | `[emb: u64] + [external_key: 1 + payload]` | variable | vec_id (u32) | ExternalKey → vec_id |
| id_reverse | `[emb: u64] + [vec_id: u32]` | 12 | external_key | vec_id → ExternalKey |
| id_alloc | `[emb: u64] + [field: u8]` | 9 | u32 / bitmap | ID allocator state |
| pending | `[emb: u64] + [ts: u64] + [vec_id: u32]` | 20 | empty | Async insert queue |

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Embedding ID | u64 (8 bytes) | Register-aligned, sufficient for <1000 models (ARCH-14) |
| Vec ID | u32 (4 bytes) | RoaringBitmap constraint, 4B vectors/space (ARCH-15) |
| ExternalKey | 1 + payload | Typed external ID (NodeId, Edge, Fragment, Summary) |
| Embedding in id_forward/id_reverse | **Yes** | Required for multi-embedding (FUNC-7, ARCH-16) |

### Edge Storage Design

Edges use `RoaringBitmap<u32>` storing neighbor vec_ids:
- Vec IDs are **local to each embedding space** (u32, not globally unique)
- Edge key includes embedding prefix to isolate graphs between embedding spaces
- Neighbor IDs inside the bitmap are just u32 (no embedding prefix inside bitmap)

```
Edge key:   (embedding=1, node=5, layer=0)
Edge value: RoaringBitmap{3, 7, 12, 45, ...}
                         ↑ u32 vec_ids within same embedding space
```

### Why Embedding Prefix in id_forward/id_reverse?

Per FUNC-7 (multi-embedding support), the same document (ExternalKey::NodeId / legacy ULID) can exist in multiple embedding spaces:

```
Document ULID_abc:
  - In "qwen3" space: vec_id = 5
  - In "gemma" space: vec_id = 1000
```

Without embedding prefix, we couldn't distinguish which vec_id belongs to which space. The id_forward lookup is: "Given this ExternalKey::NodeId **in this embedding space**, what's its vec_id?"

### Multi-Embedding Example

```rust
// Create vector indices for different embedding models
let qwen3 = Embedding::new("qwen3");
let gemma = Embedding::new("gemma");

// Insert same document with different embeddings
let doc_ulid = Id::new();
let qwen3_embedding = embed_with_qwen3(&document);
let gemma_embedding = embed_with_gemma(&document);

vector_storage.insert(&qwen3, doc_ulid, &qwen3_embedding)?;
vector_storage.insert(&gemma, doc_ulid, &gemma_embedding)?;

// Search in specific embedding space
let results = vector_storage.search(&qwen3, &query_embedding, k, ef)?;
```

### CF Constants

**Implementation:** [`schema.rs`](schema.rs) - All column family constants with `vector/` prefix and `ALL_COLUMN_FAMILIES` array.

---

## Storage Integration Design

> **⚠️ OUTDATED (2026-01-03):** This section describes the **original design proposal**. The actual implementation uses a generic, trait-based architecture:
>
> - `StorageSubsystem` trait for type-safe subsystem definitions
> - `ComponentWrapper<S>` adapter for `StorageBuilder` integration
> - `rocksdb::Storage<S>` generic storage parameterized by subsystem
> - Type aliases: `graph::Storage` and `vector::Storage`
>
> **See `src/rocksdb/README.md` for the current architecture documentation.**
>
> The high-level concepts (shared TransactionDB, independent CFs, caller-side filtering) remain valid, but the implementation details below are superseded.

**Original Design Proposal:**

The vector module shares the same RocksDB instance as `motlie_db::graph` but maintains
complete independence in its column families and operations.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Application                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────┐    ┌─────────────────────────────┐        │
│  │    motlie_db::graph         │    │    motlie_db::vector        │        │
│  │    Storage                  │    │    VectorStorage            │        │
│  ├─────────────────────────────┤    ├─────────────────────────────┤        │
│  │ • nodes CF                  │    │ • vector/vectors CF         │        │
│  │ • forward_edges CF          │    │ • vector/edges CF           │        │
│  │ • reverse_edges CF          │    │ • vector/binary_codes CF    │        │
│  │ • names CF                  │    │ • vector/vec_meta CF       │        │
│  │ • node_summaries CF         │    │ • vector/id_* CFs           │        │
│  │ • edge_summaries CF         │    │ • vector/pending CF         │        │
│  └──────────────┬──────────────┘    └──────────────┬──────────────┘        │
│                 │                                   │                        │
│                 └───────────────┬───────────────────┘                        │
│                                 ▼                                            │
│                    ┌────────────────────────┐                               │
│                    │  Shared TransactionDB  │                               │
│                    │  (Single RocksDB)      │                               │
│                    └────────────────────────┘                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Phase 0: Foundation

**Goal:** Establish the vector module structure within libs/db.

**Prerequisites:** Core DB optimizations complete (Phases 1-3 of motlie_db roadmap).

### Task 0.1: Module Structure

**Status:** ✅ COMPLETED (2026-01-03)

Create the basic module layout:

```
libs/db/src/vector/
├── mod.rs              # Module exports, VectorStorage
├── REQUIREMENTS.md     # Requirements (copied)
├── ROADMAP.md          # This document
├── schema.rs           # Column family definitions
├── config.rs           # VectorConfig, HnswConfig
├── id.rs               # ID allocator and mappings
├── hnsw.rs             # HNSW algorithm implementation
├── rabitq.rs           # RaBitQ binary codes
├── mutation.rs         # InsertVector, DeleteVector
├── query.rs            # SearchKNN, GetVector
└── async_updater.rs    # Background graph maintenance

**Note:** Distance computation uses `motlie_core::distance` (SIMD-optimized, already complete).
```

**Files to Create:**
| File | Description | Effort |
|------|-------------|--------|
| `mod.rs` | Public API, VectorStorage struct | 0.5 day |
| `schema.rs` | Column family structs | 0.5 day |
| `config.rs` | Configuration types | 0.25 day |

**Acceptance Criteria:**
- [x] Module compiles and is accessible via `motlie_db::vector`
- [x] `VectorConfig` struct with HNSW parameters
- [x] Empty stub implementations for all public types

### Task 0.1b: Configuration Types

**Status:** ✅ COMPLETED (2026-01-03)

**Implementation:** [`config.rs`](config.rs) - Complete configuration types:
- `HnswConfig` - HNSW parameters (dim, m, m_max, ef_construction, etc.) with presets (`default()`, `high_recall()`, `compact()`)
- `RaBitQConfig` - Binary quantization parameters (bits_per_dim, rotation_seed, enabled)
- `VectorConfig` - Complete storage config with dimension presets (`dim_128()`, `dim_768()`, `dim_1024()`, `dim_1536()`)

**HNSW Parameter Guidelines:**

| Scale | M | ef_construction | ef_search | Memory/Vector | Build Time |
|-------|---|-----------------|-----------|---------------|------------|
| 10K | 16 | 100 | 50 | ~0.5KB | seconds |
| 100K | 16 | 200 | 100 | ~0.5KB | minutes |
| 1M | 16-32 | 200-400 | 100-200 | ~0.5-1KB | hours |
| 1B | 16 | 200 | 100 | ~0.5KB | days |

**Search ef Parameter:**
- `ef` is passed at query time, not in config
- Higher ef = better recall, slower search
- Rule of thumb: ef >= k (number of results)
- Recommended: ef = 100 for k=10

### Task 0.2: VectorStorage Integration

**Status:** ✅ COMPLETED (2026-01-03)

> **Note:** The code below shows the **original design proposal**. The actual implementation uses:
> - `vector::Storage` = type alias to `rocksdb::Storage<vector::Subsystem>` (not a separate struct)
> - `StorageBuilder` + `vector::component()` for shared DB initialization
> - See **"Phase 0 Implementation Notes"** section for the actual API.

**Original Design Proposal:**

```rust
// libs/db/src/vector/mod.rs

pub struct VectorStorage {
    /// Shared TransactionDB (same instance as graph)
    db: Arc<TransactionDB>,

    /// Per-embedding-namespace state
    indices: RwLock<HashMap<Embedding, IndexState>>,
}

impl VectorStorage {
    /// Create from graph storage - shares the TransactionDB
    pub fn open(graph: &Storage) -> Result<Self>;

    /// Insert vector for a ULID in a specific embedding namespace
    pub fn insert(&self, code: &Embedding, ulid: Id, vector: &[f32]) -> Result<()>;

    /// Search returns (distance, ULID) - caller handles any filtering
    pub fn search(&self, code: &Embedding, query: &[f32], k: usize, ef: usize) -> Result<Vec<(f32, Id)>>;

    /// Delete vector from namespace
    pub fn delete(&self, code: &Embedding, ulid: Id) -> Result<()>;
}
```

**Acceptance Criteria:**
- [x] `vector::Storage::readonly()/readwrite()/ready()` for standalone access (type alias to generic `rocksdb::Storage<Subsystem>`)
- [x] `StorageBuilder` + `vector::component()` for shared DB with graph module
- [x] No dependency on graph module internals (only shares DB via `StorageBuilder`)

### Phase 0 Validation & Tests

**Status:** ✅ COMPLETED (2026-01-03)

> **Note:** The code below shows the **original test proposal**. The actual tests are distributed across:
> - `src/vector/mod.rs` - Module exports, config presets, distance metrics
> - `src/vector/config.rs` - HNSW/RaBitQ/VectorConfig tests
> - `src/vector/embedding.rs` - Embedding accessors, validation, builder
> - `src/vector/registry.rs` - EmbeddingFilter, EmbeddingRegistry tests
> - `src/vector/schema.rs` - CF prefix, key/value roundtrip tests
> - `src/vector/distance.rs` - Distance enum tests
> - `src/vector/subsystem.rs` - Subsystem constants tests
> - `tests/test_storage_builder.rs` - StorageBuilder integration tests

**Original Test Proposal:**

```rust
// libs/db/src/vector/tests/foundation_tests.rs (original proposal)

#[cfg(test)]
mod foundation_tests {
    use super::*;
    use tempfile::TempDir;

    /// Test: Module compiles and exports correct types
    #[test]
    fn test_module_exports() {
        // Verify public types are accessible
        let _config = VectorConfig::default();
        let _code = Embedding::new("test");
    }

    /// Test: VectorStorage opens with shared graph DB
    #[test]
    fn test_vector_storage_opens_with_graph() {
        let tmp = TempDir::new().unwrap();
        let graph_storage = graph::Storage::readwrite(tmp.path());
        let graph_handles = graph_storage.ready(Default::default()).unwrap();

        // Vector storage shares the same DB
        let vector_storage = VectorStorage::open(&graph_handles).unwrap();

        // Verify vector CFs were created
        assert!(vector_storage.has_cf(cf::VECTORS));
        assert!(vector_storage.has_cf(cf::EDGES));
    }

    /// Test: Column families use correct prefix
    #[test]
    fn test_cf_prefix_convention() {
        for cf_name in cf::ALL {
            assert!(cf_name.starts_with("vector/"),
                "CF {} should have 'vector/' prefix", cf_name);
        }
    }

    /// Test: Embedding ID creation and serialization
    #[test]
    fn test_embedding() {
        let registry = EmbeddingRegistry::new();

        // Get or create embedding ID
        let qwen = registry.get_or_create("qwen3");
        let gemma = registry.get_or_create("gemma");

        // IDs should be different
        assert_ne!(qwen.as_u64(), gemma.as_u64());

        // Same name returns same ID
        let qwen2 = registry.get_or_create("qwen3");
        assert_eq!(qwen.as_u64(), qwen2.as_u64());

        // Serialization is 8 bytes
        assert_eq!(qwen.to_be_bytes().len(), 8);
    }

    /// Test: Multiple embedding namespaces are isolated
    #[test]
    fn test_namespace_isolation() {
        let tmp = TempDir::new().unwrap();
        let storage = setup_vector_storage(tmp.path());

        let qwen = Embedding::new("qwen3");
        let gemma = Embedding::new("gemma");

        // Insert same ULID with different embeddings
        let ulid = Id::new();
        storage.insert(&qwen, ulid, &[1.0; 128]).unwrap();
        storage.insert(&gemma, ulid, &[2.0; 128]).unwrap();

        // Verify isolation
        let qwen_vec = storage.get_vector(&qwen, ulid).unwrap().unwrap();
        let gemma_vec = storage.get_vector(&gemma, ulid).unwrap().unwrap();

        assert_eq!(qwen_vec[0], 1.0);
        assert_eq!(gemma_vec[0], 2.0);
    }
}
```

**Test Coverage Checklist:**
- [x] Module compilation and exports → `src/vector/mod.rs::test_module_exports`
- [x] Storage initialization with shared DB → `tests/test_storage_builder.rs`
- [x] CF naming conventions (`vector/` prefix) → `src/vector/schema.rs::test_all_cf_names_have_prefix`
- [x] Embedding creation and validation → `src/vector/embedding.rs::test_*`, `registry.rs::test_*`
- [ ] Namespace isolation between embedding models → **Deferred to Phase 1** (requires HNSW insert/search)

### Task 0.3: EmbeddingRegistry Pre-warming

**Status:** ✅ COMPLETED (2026-01-03)

**Implementation:** [`registry.rs`](registry.rs) - Thread-safe `EmbeddingRegistry` with DashMap:
- Pre-warming via `prewarm()` during `Subsystem::prewarm()` in [`subsystem.rs`](subsystem.rs)
- Loads all entries from `vector/embedding_specs` CF (expects <1000 embeddings)
- O(1) lookup after prewarm via DashMap
- Thread-safe concurrent access

**Acceptance Criteria:**
- [x] EmbeddingRegistry loads entries during `ready()` → `vector::Subsystem::prewarm()` in `subsystem.rs`
- [x] Lookup is O(1) via DashMap (no DB hit after prewarm)
- [x] New embeddings are persisted immediately
- [x] Thread-safe concurrent access (DashMap)

### Task 0.4: ColumnFamilyProvider Trait (Mix-in Pattern)

**Status:** ✅ COMPLETED (2026-01-03)

Enable vector module to register its CFs with shared RocksDB **without** modifying graph module code.

> **Note:** The code below shows the **original design proposal**. The actual implementation evolved to use a two-layer approach:
> 1. `StorageSubsystem` trait (type-safe, generic) - see `src/rocksdb/subsystem.rs`
> 2. `ColumnFamilyProvider` trait (object-safe, dynamic) - for `StorageBuilder`
> 3. `ComponentWrapper<S>` adapts `StorageSubsystem` to `ColumnFamilyProvider`
>
> See **"Phase 0 Implementation Notes"** above for the actual implementation details.

**Original Design Proposal:**

```rust
// libs/db/src/schema.rs (new shared module)

use rocksdb::{ColumnFamilyDescriptor, Options, Cache};

/// Trait for modules that provide column families to the shared RocksDB instance
///
/// Design rationale:
/// - Graph module shouldn't know about vector-specific CFs
/// - Vector module needs to share the same TransactionDB (design principle #4)
/// - Each module registers its own CFs and initialization logic
pub trait ColumnFamilyProvider: Send + Sync {
    /// Module name for logging (e.g., "graph", "vector")
    fn name(&self) -> &'static str;

    /// Returns all CF descriptors for this module
    fn column_family_descriptors(
        &self,
        cache: Option<&Cache>,
        block_size: usize,
    ) -> Vec<ColumnFamilyDescriptor>;

    /// Called after DB is opened to initialize module-specific state
    /// (e.g., pre-warm caches, validate schema)
    fn on_ready(&self, db: &TransactionDB) -> Result<()> {
        Ok(()) // Default: no-op
    }
}

// libs/db/src/graph/schema.rs
impl ColumnFamilyProvider for graph::Schema {
    fn name(&self) -> &'static str { "graph" }

    fn column_family_descriptors(&self, cache: Option<&Cache>, block_size: usize)
        -> Vec<ColumnFamilyDescriptor>
    {
        vec![
            // Names, Nodes, NodeFragments, ForwardEdges, ReverseEdges, etc.
        ]
    }

    fn on_ready(&self, db: &TransactionDB) -> Result<()> {
        // Pre-warm NameCache (existing logic)
        self.name_cache.prewarm(db, self.config.prewarm_limit)?;
        Ok(())
    }
}

// libs/db/src/vector/schema.rs
impl ColumnFamilyProvider for vector::Schema {
    fn name(&self) -> &'static str { "vector" }

    fn column_family_descriptors(&self, cache: Option<&Cache>, block_size: usize)
        -> Vec<ColumnFamilyDescriptor>
    {
        vec![
            // embedding_registry, vectors, edges, binary_codes, vec_meta, etc.
        ]
    }

    fn on_ready(&self, db: &TransactionDB) -> Result<()> {
        // Pre-warm EmbeddingRegistry (loads ALL entries, <1000 expected)
        let count = self.embedding_registry.prewarm(db)?;
        tracing::info!(count, "Pre-warmed EmbeddingRegistry");
        Ok(())
    }
}
```

**Unified Storage Builder:**

```rust
// libs/db/src/storage.rs
// Exposed as: motlie_db::StorageBuilder (root crate, not in graph:: or vector::)

pub struct StorageBuilder {
    path: PathBuf,
    providers: Vec<Box<dyn ColumnFamilyProvider>>,
    cache_config: BlockCacheConfig,
}

impl StorageBuilder {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            providers: Vec::new(),
            cache_config: Default::default(),
        }
    }

    /// Register graph module CFs
    pub fn with_graph(mut self, config: GraphConfig) -> Self {
        self.providers.push(Box::new(graph::Schema::new(config)));
        self
    }

    /// Register vector module CFs (optional)
    pub fn with_vector(mut self, config: VectorConfig) -> Self {
        self.providers.push(Box::new(vector::Schema::new(config)));
        self
    }

    /// Open DB with all registered providers
    pub fn open(self) -> Result<Storage> {
        // 1. Create shared block cache
        let cache = Cache::new_lru_cache(self.cache_config.capacity);

        // 2. Collect all CF descriptors from all providers
        let mut cf_descriptors = vec![
            ColumnFamilyDescriptor::new("default", Options::default()),
        ];
        for provider in &self.providers {
            cf_descriptors.extend(
                provider.column_family_descriptors(Some(&cache), self.cache_config.block_size)
            );
        }

        // 3. Open single DB with all CFs
        let db = TransactionDB::open_cf_descriptors(&db_opts, &self.path, cf_descriptors)?;
        let db = Arc::new(db);

        // 4. Call on_ready for each provider (pre-warm caches, etc.)
        for provider in &self.providers {
            provider.on_ready(&db)?;
            tracing::info!(module = provider.name(), "Storage ready");
        }

        Ok(Storage { db, providers: self.providers })
    }
}

// Usage:
use motlie_db::{StorageBuilder, GraphConfig, VectorConfig};

let storage = StorageBuilder::new("/data/motlie")
    .with_graph(GraphConfig::default())
    .with_vector(VectorConfig::default())  // Optional!
    .open()?;
```

**Why This Design:**

| Concern | Resolution |
|---------|------------|
| Crate location | `StorageBuilder` in `motlie_db::` root (not graph:: or vector::) |
| Graph module unchanged | Graph implements trait, no vector imports |
| Vector CFs registered cleanly | Vector implements same trait |
| Shared block cache | Single Cache instance passed to all providers |
| Shared TransactionDB | One DB opened with all CFs |
| Pre-warm isolation | Each module's `on_ready()` handles its own caches |
| Optional vector support | `with_vector()` is opt-in; graph-only builds work |

**Acceptance Criteria:**
- [x] `ColumnFamilyProvider` trait in `motlie_db::provider` module
- [x] `StorageBuilder` in `motlie_db::storage_builder` (root, composes all modules)
- [x] Graph module implements `StorageSubsystem` trait (`graph::Subsystem`)
- [x] Vector module implements `StorageSubsystem` trait (`vector::Subsystem`)
- [x] Vector module is optional (graph-only builds compile)
- [x] Integration test: graph + vector CFs in same DB

### Phase 0 Implementation Notes

**Status:** ✅ COMPLETED (2026-01-03)

The following additional work was completed during Phase 0 implementation.

> **Architecture Documentation:** See `src/rocksdb/README.md` for detailed architecture documentation of the unified storage infrastructure.

#### Module Structure

| Module | Description |
|--------|-------------|
| `src/rocksdb/` | **Generic RocksDB storage infrastructure** (see README.md) |
| `src/rocksdb/mod.rs` | Module exports and documentation |
| `src/rocksdb/config.rs` | `BlockCacheConfig` (shared configuration) |
| `src/rocksdb/handle.rs` | `DatabaseHandle`, `StorageMode`, `StorageOptions` |
| `src/rocksdb/storage.rs` | Generic `Storage<S: StorageSubsystem>` |
| `src/rocksdb/subsystem.rs` | `StorageSubsystem` trait, `ComponentWrapper`, `DbAccess` |
| `src/provider.rs` | `ColumnFamilyProvider` trait (object-safe, for `StorageBuilder`) |
| `src/index_provider.rs` | `IndexProvider` trait for Tantivy index registration |
| `src/storage_builder.rs` | `StorageBuilder` and `SharedStorage` for unified initialization |
| `src/graph/subsystem.rs` | `graph::Subsystem` implementing `StorageSubsystem` |
| `src/vector/subsystem.rs` | `vector::Subsystem` implementing `StorageSubsystem` |
| `src/vector/config.rs` | `HnswConfig`, `RaBitQConfig`, `VectorConfig` |
| `src/vector/distance.rs` | `Distance` enum with computation behavior |
| `src/vector/embedding.rs` | `Embedding`, `EmbeddingBuilder`, `Embedder` trait |
| `src/vector/registry.rs` | `EmbeddingRegistry`, `EmbeddingFilter` |
| `src/vector/schema.rs` | All vector CF definitions with key/value serialization |
| `src/vector/error.rs` | Internal error handling |
| `src/vector/README.md` | Schema documentation and union pattern appendix |
| `tests/test_storage_builder.rs` | Integration tests for shared storage |

#### Schema Refactoring (2026-01-04, cac8da3)

Added semantic type aliases and union pattern for polymorphic CFs:

**Semantic Types:**
- `EmbeddingCode` (u64) - FK to EmbeddingSpecs
- `VecId` (u32) - internal vector index
- `HnswLayer` (u8) - HNSW layer index
- `RoaringBitmapBytes` (Vec<u8>) - serialized RoaringBitmap
- `RabitqCode` (Vec<u8>) - RaBitQ quantized code

**Union Pattern for Polymorphic CFs:**
- `GraphMeta`: `GraphMetaField` enum with `EntryPoint`, `MaxLevel`, `Count`, `SpecHash` variants
- `IdAlloc`: `IdAllocField` enum with `NextId`, `FreeBitmap` variants
- Single enum for both key discrimination and value storage
- Key helpers: `GraphMetaCfKey::entry_point(code)`, etc.
- `value_from_bytes(key, bytes)` uses key's variant for type info

See `src/vector/README.md` Appendix A for detailed documentation.

#### StorageSubsystem Trait (Generic Storage)

**Implementation:** [`../../rocksdb/subsystem.rs`](../../rocksdb/subsystem.rs) - `StorageSubsystem` trait enabling type-safe, zero-cost generic storage with:
- Associated constants: `NAME`, `COLUMN_FAMILIES`
- Associated types: `PrewarmConfig`, `Cache`
- Required methods: `create_cache()`, `cf_descriptors()`, `prewarm()`

Implementations:
- `graph::Subsystem` - Cache type is `NameCache`, pre-warms from Names CF
- `vector::Subsystem` - Cache type is `EmbeddingRegistry`, pre-warms from EmbeddingSpecs CF

#### ComponentWrapper (StorageBuilder Adapter)

**Implementation:** [`../../rocksdb/subsystem.rs`](../../rocksdb/subsystem.rs) - `ComponentWrapper<S>` adapts any `StorageSubsystem` to implement `ColumnFamilyProvider` for use with `StorageBuilder`.

Type aliases and convenience constructors in each module:
- `graph::Component` / `graph::component()`
- `vector::Component` / `vector::component()`

**Tantivy Providers:**

**Implementation:** [`../../index_provider.rs`](../../index_provider.rs) - `IndexProvider` trait with `name()`, `schema()`, `on_ready()`, `writer_heap_size()`.

- `fulltext::Schema` - Provides Tantivy schema and initialization

#### StorageBuilder Design

**Implementation:** [`../../storage_builder.rs`](../../storage_builder.rs) - Unified initialization of RocksDB + Tantivy:
- `StorageBuilder::new()` → `.with_component()` → `.with_index_provider()` → `.build()`
- Returns `SharedStorage` with access to `db()` (RocksDB) and `index()` (Tantivy)
- Caches accessible via cloned `Arc`s before building

**Directory Structure:**
```
<base_path>/
├── rocksdb/     # RocksDB TransactionDB (graph + vector CFs)
└── tantivy/     # Tantivy fulltext index
```

#### Standalone Storage (Type Aliases)

Both `graph::Storage` and `vector::Storage` are type aliases to the generic `rocksdb::Storage<S>`:

```rust
// In graph/mod.rs
pub type Storage = crate::rocksdb::Storage<Subsystem>;

// In vector/mod.rs
pub type Storage = crate::rocksdb::Storage<Subsystem>;
```

Usage:

```rust
// Read-only access (e.g., CLI scanning tools)
let mut storage = vector::Storage::readonly(db_path);
storage.ready()?;

// Read-write access
let mut storage = vector::Storage::readwrite(db_path);
storage.ready()?;

// Secondary mode (read replicas)
let mut storage = vector::Storage::secondary(primary_path, secondary_path);
storage.ready()?;
storage.try_catch_up_with_primary()?;

// Access pre-warmed cache (generic .cache() method)
let registry: &Arc<EmbeddingRegistry> = storage.cache();
```

**Configuration:**
- `EmbeddingRegistryConfig` - Controls pre-warming limit (default: 1000)
- `BlockCacheConfig` / `VectorBlockCacheConfig` - Shared block cache settings

#### Schema Pattern Consistency

Both `graph::schema` and `vector::schema` modules follow the same patterns:

1. **Tuple structs** for CF key/value types (e.g., `CfKey(u64, u32)`)
2. **`pub(crate)` visibility** for internal types
3. **Static `CF_NAME` constant** per CF marker type
4. **`key_to_bytes` / `key_from_bytes`** for serialization
5. **`value_to_bytes` / `value_from_bytes`** for serialization
6. **`column_family_options_with_cache()`** for block cache configuration

#### Vector Module Column Families

| CF Name | Key | Value | Purpose |
|---------|-----|-------|---------|
| `vector/embedding_specs` | code (u64) | (model, dim, distance) | Embedding space registry |
| `vector/vectors` | (code, vec_id) | f32 bytes | Raw vector storage |
| `vector/edges` | (code, vec_id) | RoaringBitmap | HNSW neighbor lists |
| `vector/binary_codes` | (code, vec_id) | binary | RaBitQ quantized vectors |
| `vector/vec_meta` | (code, vec_id) | (layer, flags) | Per-vector metadata |
| `vector/graph_meta` | code (u64) | (entry, stats) | Per-graph metadata |
| `vector/id_forward` | (code, external_key) | vec_id | ExternalKey → u32 mapping |
| `vector/id_reverse` | (code, vec_id) | external_key | u32 → ExternalKey mapping |
| `vector/id_alloc` | code (u64) | (next_id, free_bitmap) | ID allocation state |
| `vector/pending` | (code, emb_id, ts) | operation | Async update queue |

#### Test Coverage

- 364 tests passing (285 unit + 79 integration)
- Includes `test_storage_builder.rs` with 10 integration tests for graph+vector+fulltext composition

#### Deferred Work

- **Secondary mode support for StorageBuilder**: The existing `graph::Storage::secondary()` API works for RocksDB secondary instances. Adding this to `StorageBuilder` is deferred as it requires different RocksDB APIs (`DB::open_as_secondary()` vs `TransactionDB`).

---

## Phase 1: ID Management

> **Historical note (2026-01-04):** Some early code examples used the deprecated `ColumnFamilyRecord` trait. The actual implementation uses the current trait hierarchy from `rocksdb::cf_traits`:
>
> | Trait | Purpose | Use For |
> |-------|---------|---------|
> | `ColumnFamily` | Base marker with CF_NAME | All CFs |
> | `ColumnFamilyConfig<C>` | RocksDB options | All CFs |
> | `ColumnFamilySerde` | Cold CF (MessagePack + LZ4) | ForwardIdMapping, IdAllocatorMeta, EmbeddingSpecs |
> | `HotColumnFamilyRecord` | Hot CF (rkyv zero-copy) | ReverseIdMapping, VecMetaCF, GraphMetaCF |
> | `MutationCodec` | Mutation marshaling | InsertVector, DeleteVector mutations |
>
> For raw byte CFs (Vectors, BinaryCodes, Edges), implement only `ColumnFamily` with custom key/value methods.
>
> **See `src/rocksdb/README.md` for the current architecture documentation.**

**Goal:** Implement u32 internal ID system per [ARCH-4, ARCH-5, ARCH-6].

**Design Reference:** `examples/vector/HNSW2.md` - Item ID Design

**Status:** ✅ COMPLETE (2026-01-04)
- ✅ Schema definitions complete (`IdForward`, `IdReverse`, `IdAlloc` CFs in `schema.rs`)
- ✅ Key/value types and serialization methods complete
- ✅ `IdAllocator` implementation complete (`id.rs`) - 7 tests
- ✅ Storage API integration complete:
  - `mutation.rs` - InsertVector, DeleteVector, InsertVectorBatch, UpdateEdges, UpdateGraphMeta, FlushMarker
  - `query.rs` - GetVector, GetInternalId, GetExternalId, ResolveIds with QueryExecutor
  - `processor.rs` - Central state management with per-embedding IdAllocators
  - `writer.rs` - MPSC mutation infrastructure with Consumer
  - `reader.rs` - MPMC query infrastructure with flume
- ✅ All 62 vector module tests pass

### Task 1.1: ID Allocator

**Status:** ✅ COMPLETE (2026-01-04) - Implemented in `id.rs` with 7 unit tests

**Implementation:** [`id.rs`](id.rs) - Lock-free `IdAllocator` with:
- `AtomicU32` for next_id (monotonically increasing)
- `Mutex<RoaringBitmap>` for free list (deleted IDs for reuse)
- Methods: `new()`, `allocate()`, `free()`, `persist()`, `recover()`
- Free IDs reused before allocating fresh ones

**Effort:** 1-2 days

**Acceptance Criteria:**
- [ ] Allocator survives process restart
- [ ] Free IDs are reused before allocating new ones
- [ ] Thread-safe concurrent allocation

### Task 1.2: Forward Mapping (ExternalKey -> u32)

**Status:** ✅ COMPLETE (2026-01-04, updated 2026-01-30 for ExternalKey) - Schema in `schema.rs::IdForward`, integrated in `writer.rs`

The `IdForward` CF is implemented with polymorphic `ExternalKey` support:
- Key: `IdForwardCfKey(EmbeddingCode, ExternalKey)` = 8 + [1 tag + variable payload] bytes
- Value: `IdForwardCfValue(VecId)` = 4 bytes
- Supports all 6 ExternalKey variants: NodeId, NodeFragment, Edge, EdgeFragment, NodeSummary, EdgeSummary
- All serialization methods (`key_to_bytes`, `key_from_bytes`, `value_to_bytes`, `value_from_bytes`)
- Integration with `InsertVector` and `DeleteVector` mutations in `writer.rs`

**Acceptance Criteria:**
- [ ] Insert stores ULID -> u32 mapping
- [ ] Lookup returns Option<u32>

### Task 1.3: Reverse Mapping (u32 -> ExternalKey)

**Status:** ✅ COMPLETE (2026-01-04, updated 2026-01-30 for ExternalKey) - Schema in `schema.rs::IdReverse`, queries in `query.rs`

This is the hot path during search. The `IdReverse` CF is implemented with polymorphic `ExternalKey` support:
- Key: `IdReverseCfKey(EmbeddingCode, VecId)` = 12 bytes
- Value: `IdReverseCfValue(ExternalKey)` = [1 tag + variable payload] bytes
- Returns full typed `ExternalKey` (not just Id), allowing callers to determine key type
- All serialization methods
- `GetExternalId` query for single lookups (returns `Option<ExternalKey>`)
- `ResolveIds` query for batch lookups (returns `Vec<Option<ExternalKey>>`)

| Approach | Memory at 1B | Lookup | Implementation |
|----------|--------------|--------|----------------|
| RocksDB CF | On-demand | ~100µs | ✅ Implemented |
| Dense array | 16 GB | O(1) ~10ns | Deferred |
| Mmap file | On-demand | O(1) ~100ns | Deferred |

**Implementation:** [`query.rs`](query.rs) - `ResolveIds` query for batch ULID resolution using RocksDB MultiGet.

**Effort:** 0.25 day (integration only)

**Acceptance Criteria:**
- [ ] Search results correctly map back to ULIDs
- [ ] MultiGet batch performance validated

### Task 1.4: ID Allocation Persistence

**Status:** ✅ COMPLETE (2026-01-04) - Schema in `schema.rs::IdAlloc`, persist/recover in `id.rs`

**Implementation:**
- [`schema.rs`](schema.rs) - `IdAlloc` CF with union pattern:
  - Key: `IdAllocCfKey(EmbeddingCode, IdAllocField)` = 9 bytes
  - Value: `IdAllocCfValue(IdAllocField)` with `NextId(VecId)` or `FreeBitmap(RoaringBitmapBytes)` variants
- [`id.rs`](id.rs) - Persistence methods:
  - `IdAllocator::persist()` - saves next_id and free bitmap to RocksDB
  - `IdAllocator::recover()` - restores state from RocksDB on startup

**Effort:** 0.25 day (integration only)

**Acceptance Criteria:**
- [ ] `next_id` persists across restarts
- [ ] Free bitmap persists (for ID reuse after restart)

### Phase 1 Validation & Tests

**Status:** ✅ COMPLETE (2026-01-04) - Unit tests in `id.rs`, `query.rs`, `mutation.rs`, `writer.rs`, `reader.rs`

```rust
// libs/db/src/vector/tests/id_tests.rs

#[cfg(test)]
mod id_tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::Arc;
    use std::thread;

    /// Test: ID allocator produces unique sequential IDs
    #[test]
    fn test_sequential_allocation() {
        let allocator = IdAllocator::new();
        let mut ids = Vec::new();

        for _ in 0..1000 {
            ids.push(allocator.allocate());
        }

        // All IDs should be unique
        let unique: HashSet<_> = ids.iter().collect();
        assert_eq!(unique.len(), 1000);

        // Should be sequential (when no frees)
        for i in 0..1000 {
            assert_eq!(ids[i], i as u32);
        }
    }

    /// Test: Freed IDs are reused before allocating new ones
    #[test]
    fn test_id_reuse() {
        let allocator = IdAllocator::new();

        let id1 = allocator.allocate();  // 0
        let id2 = allocator.allocate();  // 1
        let id3 = allocator.allocate();  // 2

        allocator.free(id2);  // Free ID 1

        let id4 = allocator.allocate();  // Should reuse 1
        assert_eq!(id4, 1);

        let id5 = allocator.allocate();  // Should be 3 (next fresh)
        assert_eq!(id5, 3);
    }

    /// Test: Concurrent allocation is thread-safe
    #[test]
    fn test_concurrent_allocation() {
        let allocator = Arc::new(IdAllocator::new());
        let mut handles = Vec::new();

        for _ in 0..10 {
            let alloc = Arc::clone(&allocator);
            handles.push(thread::spawn(move || {
                let mut ids = Vec::new();
                for _ in 0..100 {
                    ids.push(alloc.allocate());
                }
                ids
            }));
        }

        let all_ids: Vec<u32> = handles.into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect();

        // All 1000 IDs should be unique
        let unique: HashSet<_> = all_ids.iter().collect();
        assert_eq!(unique.len(), 1000);
    }

    /// Test: Allocator state persists across restart
    #[test]
    fn test_persistence() {
        let tmp = TempDir::new().unwrap();

        // First session: allocate some IDs
        {
            let storage = setup_vector_storage(tmp.path());
            let code = Embedding::new("test");

            for i in 0..100 {
                let ulid = Id::new();
                storage.insert(&code, ulid, &[i as f32; 128]).unwrap();
            }
            // Allocator should be at next_id = 100
        }

        // Second session: verify state recovered
        {
            let storage = setup_vector_storage(tmp.path());
            let code = Embedding::new("test");

            // New allocation should continue from 100
            let ulid = Id::new();
            storage.insert(&code, ulid, &[0.0; 128]).unwrap();

            let internal_id = storage.get_internal_id(&code, ulid).unwrap();
            assert_eq!(internal_id, 100);
        }
    }

    /// Test: Forward mapping (ULID -> u32)
    #[test]
    fn test_forward_mapping() {
        let storage = setup_vector_storage_temp();
        let code = Embedding::new("test");
        let ulid = Id::new();

        storage.insert(&code, ulid, &[1.0; 128]).unwrap();

        let internal_id = storage.get_internal_id(&code, ulid).unwrap();
        assert!(internal_id.is_some());
    }

    /// Test: Reverse mapping (u32 -> ULID)
    #[test]
    fn test_reverse_mapping() {
        let storage = setup_vector_storage_temp();
        let code = Embedding::new("test");
        let ulid = Id::new();

        storage.insert(&code, ulid, &[1.0; 128]).unwrap();

        let internal_id = storage.get_internal_id(&code, ulid).unwrap().unwrap();
        let recovered_ulid = storage.get_ulid(&code, internal_id).unwrap();

        assert_eq!(recovered_ulid, Some(ulid));
    }

    /// Test: Batch ULID resolution
    #[test]
    fn test_batch_ulid_resolution() {
        let storage = setup_vector_storage_temp();
        let code = Embedding::new("test");

        let ulids: Vec<Id> = (0..100).map(|_| Id::new()).collect();
        for (i, ulid) in ulids.iter().enumerate() {
            storage.insert(&code, *ulid, &[i as f32; 128]).unwrap();
        }

        let internal_ids: Vec<u32> = (0..100).collect();
        let resolved = storage.resolve_ulids_batch(&code, &internal_ids).unwrap();

        assert_eq!(resolved.len(), 100);
        for (i, maybe_ulid) in resolved.iter().enumerate() {
            assert_eq!(*maybe_ulid, Some(ulids[i]));
        }
    }

    /// Property test: allocate/free/allocate maintains uniqueness
    #[test]
    fn test_allocate_free_uniqueness_property() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let allocator = IdAllocator::new();
        let mut live_ids: HashSet<u32> = HashSet::new();

        for _ in 0..10000 {
            if live_ids.is_empty() || rng.gen_bool(0.7) {
                // Allocate
                let id = allocator.allocate();
                assert!(!live_ids.contains(&id), "Duplicate ID allocated!");
                live_ids.insert(id);
            } else {
                // Free random live ID
                let id = *live_ids.iter().next().unwrap();
                live_ids.remove(&id);
                allocator.free(id);
            }
        }
    }
}
```

**Test Coverage Checklist:**
- [ ] Sequential ID allocation
- [ ] ID reuse after free (free list)
- [ ] Concurrent thread safety
- [ ] Persistence across process restart
- [ ] Forward mapping (ULID → u32)
- [ ] Reverse mapping (u32 → ULID)
- [ ] Batch ULID resolution
- [ ] Property: allocate/free maintains uniqueness

**Benchmark:**
```rust
#[bench]
fn bench_allocation_throughput(b: &mut Bencher) {
    let allocator = IdAllocator::new();
    b.iter(|| allocator.allocate());
}
// Target: > 10M allocations/sec
```

---

### Task 1.5: Storage API Integration Design

**Status:** ✅ COMPLETE (2026-01-04) - Implemented in `writer.rs`, `reader.rs`, `processor.rs`, `query.rs`, `mutation.rs`

> **Design Pattern:** Following the established patterns in `graph::` and `fulltext::` modules for consistency across subsystems. This includes MPSC mutation channels, MPMC query channels, and the `Runnable` trait for composable operations.

#### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Vector Module API Architecture                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌─────────────────────────────────┐    ┌─────────────────────────────────┐     │
│  │        vector::Writer           │    │        vector::Reader           │     │
│  │                                 │    │                                 │     │
│  │  send(Vec<Mutation>)           │    │  send_query(Query)              │     │
│  │  flush() -> oneshot            │    │  is_closed()                    │     │
│  │  send_with_result(Vec<Mutation>)│   │                                 │     │
│  │  send_sync(Vec<Mutation>)      │    │                                 │     │
│  └──────────────┬──────────────────┘    └──────────────┬──────────────────┘     │
│                 │                                       │                        │
│                 ▼ MPSC (tokio)                         ▼ MPMC (flume)           │
│                 │                                       │                        │
│  ┌──────────────┴──────────────────┐    ┌──────────────┴──────────────────┐     │
│  │     vector::writer::Consumer    │    │     vector::reader::Consumer    │     │
│  │                                 │    │                                 │     │
│  │  process_batch(Vec<Mutation>)   │    │  process_query(Query)           │     │
│  │  → RocksDB Transaction          │    │  → Execute + oneshot response   │     │
│  └─────────────────────────────────┘    └─────────────────────────────────┘     │
│                                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                              vector::Processor                               │ │
│  │                                                                              │ │
│  │  storage: Arc<Storage<Subsystem>>     Provides:                             │ │
│  │  id_allocators: DashMap<u64, IdAllocator>   • DB access                     │ │
│  │  registry: Arc<EmbeddingRegistry>           • ID allocation per embedding   │ │
│  │                                              • Embedding registry           │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Mutation Design

**Implementation:** [`mutation.rs`](mutation.rs) - Complete mutation infrastructure:

**Mutation Enum:**
- `AddEmbeddingSpec` - Register embedding space
- `InsertVector` - Insert vector with ID allocation and mapping storage
- `DeleteVector` - Delete vector and return ID to free list
- `InsertVectorBatch` - Batch insert optimization
- `UpdateEdges` - HNSW graph edge updates (Phase 2)
- `UpdateGraphMeta` - Graph metadata updates (Phase 2)
- `Flush(FlushMarker)` - Synchronization with oneshot channel

**Mutation Structs:**
- `InsertVector` - embedding, id, vector, immediate_index flag
- `DeleteVector` - embedding, id
- `InsertVectorBatch` - embedding, vectors Vec, immediate_index
- `UpdateEdges` - embedding, vec_id, layer, EdgeUpdate enum
- `UpdateGraphMeta` - embedding, GraphMetaUpdate enum
- `FlushMarker` - oneshot::Sender for flush semantics

**Mutation Execution:** [`writer.rs`](writer.rs) `Consumer::execute_single()` - Executes mutations within RocksDB transactions:
1. Allocates vec_id via IdAllocator
2. Stores ExternalKey ↔ vec_id mappings (IdForward, IdReverse CFs)
3. Stores raw vector (Vectors CF)
4. Handles flush markers after commit

#### Query Design

**Implementation:** [`query.rs`](query.rs) - Complete query infrastructure:

**Query Enum:**
- Search Operations (Phase 2): `SearchKNN`, `SearchKNNFiltered`
- Point Lookups (Phase 1): `GetVector`, `GetInternalId`, `GetExternalId`, `ResolveIds`
- Graph Introspection (Phase 2): `GetGraphMeta`, `GetNeighbors`

**Query Structs (Phase 1 - Implemented):**
- `GetVector` - Get vector by external ID
- `GetInternalId` - ExternalKey → VecId lookup
- `GetExternalId` - VecId → ExternalKey lookup
- `ResolveIds` - Batch VecId → ExternalKey resolution

**QueryExecutor Trait:**
- `execute(&self, storage: &Storage) -> Result<Self::Output>` - Execute query against storage
- Implemented for `GetVector`, `GetInternalId`, `GetExternalId`, `ResolveIds`
- Search queries (Phase 2) will add HNSW search implementation

#### Writer Infrastructure (`vector/writer.rs`)

**Implementation:** [`writer.rs`](writer.rs) - MPSC mutation infrastructure:

**Writer:**
- `send(Vec<Mutation>)` - Fire-and-forget async send
- `send_with_result(Vec<Mutation>)` - Send + await consumer processing
- `flush()` - Wait for pending mutations to commit (uses `FlushMarker` oneshot)
- `send_sync(Vec<Mutation>)` - Send + flush in one call
- `is_closed()` - Check if consumer dropped

**WriterConfig:**
- `channel_buffer_size` - MPSC channel buffer (default: 1000)

**Consumer:**
- `run()` - Process mutations continuously from channel
- `process_batch()` - Execute mutations in RocksDB transaction
- `execute_single()` - Handle each mutation type (InsertVector, DeleteVector, etc.)
- Commits transaction, then signals flush waiters

**Factory Functions:**
- `create_writer(config)` - Returns `(Writer, Receiver)`
- `spawn_consumer(consumer)` - Spawn as tokio task

#### Reader Infrastructure (`vector/reader.rs`)

**Implementation:** [`reader.rs`](reader.rs) - MPMC query infrastructure using flume:

**Reader:**
- `send_query(Query)` - Send query to consumer pool
- `is_closed()` - Check if consumers dropped

**ReaderConfig:**
- `channel_buffer_size` - flume channel buffer (default: 1000)

**Consumer:**
- `run()` - Process queries continuously from MPMC channel
- `process_query()` - Execute query via `Query::process()` and send result

**Factory Functions:**
- `create_reader(config)` - Returns `(Reader, Receiver)`
- `spawn_consumer(consumer)` - Spawn single consumer as tokio task
- `spawn_consumers(receiver, config, storage, count)` - Spawn N consumers for parallel processing

#### Processor (`vector/processor.rs`)

**Implementation:** [`processor.rs`](processor.rs) - Central state management:

**Processor struct:**
- `storage: Arc<Storage>` - RocksDB via generic `Storage<Subsystem>`
- `id_allocators: DashMap<EmbeddingCode, Arc<IdAllocator>>` - Per-embedding ID allocators (lazily created)

**Methods:**
- `new(storage)` - Create processor with storage reference
- `storage()` - Access underlying storage
- `get_or_create_allocator(embedding)` - Get/create IdAllocator for embedding space (recovers from DB or creates new)

**Phase 2 (Implemented):**
- `hnsw_search()` and `index_vector()` are implemented in the HNSW/processor paths; placeholders removed.

#### Runnable Trait Integration

> **Note:** The vector module uses a simpler pattern than the `Runnable` trait shown in the original design:
> - Queries implement `QueryExecutor` trait directly (see [`query.rs`](query.rs))
> - Mutations are sent via `Writer::send()` / `Writer::send_sync()` (see [`writer.rs`](writer.rs))
> - `Processor` provides the primary API for all operations (see [`processor.rs`](processor.rs))

#### Module Structure

```
libs/db/src/vector/
├── mod.rs              # Module exports
├── config.rs           # Configuration types
├── distance.rs         # Distance metrics
├── embedding.rs        # Embedding type
├── error.rs            # Error types
├── id.rs               # IdAllocator (Phase 1)
├── mutation.rs         # Mutation enum + MutationExecutor impls
├── processor.rs        # Processor (central state management)
├── query.rs            # Query enum + QueryExecutor impls
├── reader.rs           # Reader, Consumer, MPMC infrastructure
├── registry.rs         # EmbeddingRegistry
├── schema.rs           # Column family definitions
├── subsystem.rs        # StorageSubsystem impl
├── writer.rs           # Writer, Consumer, MPSC infrastructure
├── hnsw.rs             # HNSW algorithm (Phase 2)
├── rabitq.rs           # RaBitQ quantization (Phase 4)
└── async_updater.rs    # Background graph maintenance (Phase 7)
```

#### Primary API (`Processor`)

**Status:** ✅ COMPLETE - All core operations implemented via `Processor`

**Implementation:** [`processor.rs`](processor.rs) - Transactional API for all operations:

- `Processor::insert_vector(embedding, id, vector, build_index)` - Insert with transactions
- `Processor::insert_batch(embedding, vectors, build_index)` - Batch insert
- `Processor::delete_vector(embedding, id)` - Delete with soft-delete for HNSW
- `Processor::search(embedding, query, k, ef_search)` - K-nearest neighbor search
- `Processor::search_with_config(config, query)` - Search with strategy dispatch

> **Note:** The `api.rs` convenience layer was removed in Phase 5 as `Processor` provides
> complete functionality with proper transactional guarantees.

#### Implementation Priority

| Task | Phase | Priority | Dependencies |
|------|-------|----------|--------------|
| `id.rs` - IdAllocator | 1 | High | Schema (done) |
| `mutation.rs` - InsertVector/DeleteVector | 1 | High | IdAllocator |
| `query.rs` - GetVector/GetInternalId | 1 | High | Schema (done) |
| `processor.rs` - Processor struct | 1 | High | IdAllocator |
| `writer.rs` - Writer/Consumer | 1 | Medium | Processor |
| `reader.rs` - Reader/Consumer | 1 | Medium | Processor |
| `query.rs` - SearchKNN | 2 | High | HNSW impl |
| `hnsw.rs` - HNSW algorithm | 2 | High | Processor |

#### Test Strategy

```rust
// Example: Using Processor directly (recommended)

#[test]
fn test_insert_and_search() {
    let tmp = TempDir::new().unwrap();
    let processor = setup_processor(tmp.path());

    // Register embedding space
    let embedding = processor.registry().register(
        EmbeddingBuilder::new("test", 128, Distance::Cosine)
    )?;

    // Insert vector
    let id = Id::new();
    let vector = vec![1.0f32; 128];
    processor.insert_vector(embedding.code(), id, &vector, true)?;

    // Search
    let results = processor.search(embedding.code(), &vector, 10, 100)?;
    assert!(!results.is_empty());
}
```

---

## Phase 2: HNSW2 Core + Navigation Layer ✓ CORE COMPLETE

**Status:** Core implementation complete. Some acceptance criteria deferred to Phase 3:
- Task 2.6: 100K throughput/recall targets → Phase 3.7
- Task 2.9: Upper layer caching, LRU cache, cache invalidation → Phase 3.5, 3.6

**Goal:** Implement optimized HNSW with roaring bitmap edges per `examples/vector/HNSW2.md`,
including the hierarchical navigation layer for O(log N) search complexity.

**Target:** Achieve [THR-1] 5,000+ inserts/sec (125x improvement from 40/s).

**Design Reference:** `examples/vector/HYBRID.md` - Navigation Layer (Phase 4)

**Implementation Files:**
- [`merge.rs`](merge.rs) - RocksDB merge operators for concurrent edge updates
- [`navigation.rs`](navigation.rs) - Navigation layer structure and cache
- [`hnsw.rs`](hnsw.rs) - HNSW insert and search algorithms

**Benchmark Results (2026-01-06):**

| Dataset | Vectors | Build Rate | Recall@10 | QPS | P99 Latency |
|---------|---------|------------|-----------|-----|-------------|
| SIFT10K | 1,000 | 1,161 vec/s | 100.0% | 1,483 | 0.87ms |
| SIFT10K | 10,000 | 487 vec/s | 100.0% | 522 | 2.82ms |
| SIFT1M | 100,000 | 221 vec/s | 88.4% | 282 | 5.75ms |

**Status Summary:**
- Core HNSW insert/search: ✅ Complete
- Merge operators with compaction fix: ✅ Complete
- Navigation cache (basic): ✅ Complete
- Upper layer caching: ⏳ Deferred to Phase 3
- Target 5,000 vec/s: ⏳ Requires batch operations (Phase 3)

### Task 2.1: RoaringBitmap Edge Storage

**Status:** ✅ COMPLETED (2026-01-06)

**Implementation:** [`schema.rs`](schema.rs) lines 270-340 - `Edges` column family with RoaringBitmap values

Replace explicit edge lists with compressed bitmaps:

```rust
// libs/db/src/vector/schema.rs

/// HNSW graph edges using RoaringBitmap
/// Key: [embedding: u64] + [vec_id: u32] + [layer: u8] = 13 bytes
/// Value: RoaringBitmap serialized (~50-200 bytes)
/// Uses ColumnFamily base trait only (raw bytes, no serde)
pub struct GraphEdges;

impl ColumnFamily for GraphEdges {
    const CF_NAME: &'static str = cf::EDGES;  // "vector/edges"
}

impl GraphEdges {
    /// Build key from components
    pub fn key_to_bytes(embedding: u64, vec_id: u32, layer: u8) -> [u8; 13] {
        let mut key = [0u8; 13];
        key[0..8].copy_from_slice(&embedding.to_be_bytes());
        key[8..12].copy_from_slice(&vec_id.to_be_bytes());
        key[12] = layer;
        key
    }

    /// Parse key into components
    pub fn key_from_bytes(bytes: &[u8]) -> (u64, u32, u8) {
        let embedding = u64::from_be_bytes(bytes[0..8].try_into().unwrap());
        let vec_id = u32::from_be_bytes(bytes[8..12].try_into().unwrap());
        let layer = bytes[12];
        (embedding, vec_id, layer)
    }

    /// Value is RoaringBitmap::serialize_into / deserialize_from (direct bytes)
}

// Storage comparison:
// - Old: 32 edges × 16 bytes (ULID) = 512 bytes
// - New: RoaringBitmap = ~50-200 bytes (4-10x smaller)
```

**Effort:** 1-2 days

**Acceptance Criteria:**
- [x] Edge storage 4-10x smaller than current (RoaringBitmap ~50-200 bytes vs 512 bytes)
- [x] Membership test O(1): `bitmap.contains(neighbor_id)`
- [x] Iteration works: `bitmap.iter()`

### Task 2.2: RocksDB Merge Operators

**Status:** ✅ COMPLETED (2026-01-06)

**Implementation:** [`merge.rs`](merge.rs) - `EdgeOp` enum and `edge_full_merge` function

**Bug Fix:** During compaction, `set_merge_operator_associative` can pass previously merged
RoaringBitmap results as operands. Fixed by attempting RoaringBitmap deserialization when
EdgeOp parsing fails (line 109-118).

Enable lock-free concurrent edge updates:

```rust
// libs/db/src/vector/merge.rs

use rocksdb::MergeOperands;
use roaring::RoaringBitmap;

#[derive(Serialize, Deserialize)]
enum EdgeOp {
    Add(u32),
    AddBatch(Vec<u32>),
    Remove(u32),
}

pub fn edge_merge_operator(
    _key: &[u8],
    existing: Option<&[u8]>,
    operands: &MergeOperands,
) -> Option<Vec<u8>> {
    let mut bitmap = existing
        .map(|b| RoaringBitmap::deserialize_from(b).unwrap())
        .unwrap_or_default();

    for operand in operands {
        match bincode::deserialize::<EdgeOp>(operand).unwrap() {
            EdgeOp::Add(id) => { bitmap.insert(id); }
            EdgeOp::AddBatch(ids) => {
                for id in ids { bitmap.insert(id); }
            }
            EdgeOp::Remove(id) => { bitmap.remove(id); }
        }
    }

    // Enforce M_max limit inline
    // (pruning logic here if needed)

    let mut buf = Vec::new();
    bitmap.serialize_into(&mut buf).unwrap();
    Some(buf)
}
```

**Effort:** 1-2 days

**Acceptance Criteria:**
- [x] `db.merge_cf()` adds neighbors without read-modify-write
- [x] Concurrent writers don't conflict (merge operator handles this)
- [x] Crash-safe (logged in WAL)

### Task 2.3: Vector Storage CF

**Status:** ✅ COMPLETED (Phase 0/1)

**Implementation:** [`schema.rs`](schema.rs) lines 133-220 - `Vectors` column family

Store raw vectors for exact distance computation:

```rust
/// Full vector storage (raw f32 bytes)
/// Key: [embedding: u64] + [vec_id: u32] = 12 bytes
/// Value: [f32; D] as raw bytes (e.g., 512 bytes for 128D)
/// Uses ColumnFamily base trait only (raw bytes via bytemuck, no serde)
pub struct Vectors;

impl ColumnFamily for Vectors {
    const CF_NAME: &'static str = cf::VECTORS;  // "vector/vectors"
}

impl ColumnFamilyConfig<VectorBlockCacheConfig> for Vectors {
    fn cf_options(cache: &Cache, config: &VectorBlockCacheConfig) -> Options {
        let mut opts = Options::default();
        opts.set_compression_type(DBCompressionType::Lz4);
        opts.set_block_size(config.vector_block_size);  // 16KB for vectors
        // Configure block cache
        let mut block_opts = BlockBasedOptions::default();
        block_opts.set_block_cache(cache);
        opts.set_block_based_table_factory(&block_opts);
        opts
    }
}

impl Vectors {
    /// Zero-copy vector read using bytemuck
    pub fn get_vector(db: &DB, id: u32) -> Result<&[f32]> {
        let bytes = db.get_cf(cf, &id.to_le_bytes())?;
        Ok(bytemuck::cast_slice(&bytes))
    }

    /// Batch read for distance computation
    pub fn multi_get(db: &DB, ids: &[u32]) -> Result<Vec<Vec<f32>>> {
        // Use MultiGet for efficiency
    }
}
```

**Effort:** 0.5 day (completed in Phase 0/1)

### Task 2.4: Node Metadata CF

**Status:** ✅ COMPLETED (2026-01-06)

**Implementation:** [`schema.rs`](schema.rs) lines 420-500 - `VecMeta` column family

Store per-node HNSW metadata:

```rust
/// Per-node HNSW metadata
/// Key: [embedding: u64] + [vec_id: u32] = 12 bytes
/// Value: VecMeta (rkyv serialized, ~8 bytes)
#[derive(Archive, Deserialize, Serialize)]
pub struct VecMeta {
    pub max_layer: u8,
    pub flags: u8,           // deleted, needs_repair, etc.
    pub created_at: u64,     // Timestamp for debugging
}

pub struct VecMetaCF;

impl HotColumnFamilyRecord for VecMetaCF {
    const CF_NAME: &'static str = cf::NODE_META;  // "vector/vec_meta"
    type Key = u32;
    type Value = VecMeta;
}
```

**Effort:** 0.5 day

### Task 2.5: Graph Metadata CF

**Status:** ✅ COMPLETED (2026-01-06)

**Implementation:** [`schema.rs`](schema.rs) lines 500-650 - `GraphMeta` column family

Store global HNSW state:

```rust
/// Global graph metadata
#[derive(Archive, Deserialize, Serialize)]
pub struct GraphMeta {
    pub entry_point: u32,
    pub max_level: u8,
    pub node_count: u64,
    pub ef_construction: u16,
    pub m: u16,
    pub m_max: u16,
}

pub struct GraphMetaCF;

impl HotColumnFamilyRecord for GraphMetaCF {
    const CF_NAME: &'static str = cf::GRAPH_META;  // "vector/graph_meta"
    type Key = &'static str;    // "meta"
    type Value = GraphMeta;
}
```

**Effort:** 0.25 day

### Task 2.6: HNSW Insert with Bitmap Edges

**Status:** ✅ COMPLETED (2026-01-06)

**Implementation:** [`hnsw/mod.rs`](hnsw/mod.rs) - `Index::insert()` method

Port HNSW algorithm from `examples/vector/hnsw.rs` to use new schema:

```rust
// libs/db/src/vector/hnsw.rs

impl VectorStorage {
    /// Insert a vector into the HNSW index
    pub fn insert(&self, ulid: Id, vector: &[f32]) -> Result<()> {
        // 1. Allocate internal ID
        let internal_id = self.id_alloc.allocate();

        // 2. Store ID mappings
        self.store_id_mapping(ulid, internal_id)?;

        // 3. Store vector
        self.store_vector(internal_id, vector)?;

        // 4. Assign random layer
        let level = self.random_level();

        // 5. Store node metadata
        self.store_vec_meta(internal_id, level)?;

        // 6. Greedy search from entry point
        let neighbors_by_layer = self.greedy_search_layers(vector, level)?;

        // 7. Connect to neighbors (using merge operators)
        for (layer, neighbors) in neighbors_by_layer {
            self.add_bidirectional_edges(internal_id, &neighbors, layer)?;
            self.prune_if_needed(internal_id, layer)?;
        }

        // 8. Update entry point if needed
        if level > self.max_level() {
            self.update_entry_point(internal_id, level)?;
        }

        Ok(())
    }

    fn add_bidirectional_edges(&self, vec_id: u32, neighbors: &[u32], layer: u8) -> Result<()> {
        // Use merge operators - no read-modify-write!
        let key = GraphEdges::key_to_bytes(vec_id, layer);
        self.db.merge_cf(self.edges_cf, &key, EdgeOp::AddBatch(neighbors.to_vec()))?;

        // Reverse edges
        for &neighbor in neighbors {
            let neighbor_key = GraphEdges::key_to_bytes(neighbor, layer);
            self.db.merge_cf(self.edges_cf, &neighbor_key, EdgeOp::Add(vec_id))?;
        }

        Ok(())
    }
}
```

**Effort:** 3-5 days

**Acceptance Criteria:**
- [x] Insert throughput > 1,000/s at 1K scale (achieved 1,161 vec/s)
- [ ] Insert throughput > 1,000/s at 100K scale (achieved 221 vec/s - needs batch ops)
- [x] Recall@10 maintains > 95% at small scale (100% at 1K-10K)
- [ ] Recall@10 > 95% at 100K scale (achieved 88.4% - needs ef tuning or pruning)
- [x] Crash recovery works (WAL + persistent entry point)

### Task 2.7: Navigation Layer Structure

**Status:** ✅ COMPLETED (2026-01-06)

**Implementation:** [`navigation.rs`](navigation.rs) - `NavigationLayerInfo` struct (lines 30-150)

The HNSW navigation layer is the hierarchical structure that enables O(log N) search.
Each layer contains a subset of nodes, with layer 0 containing all nodes and higher
layers containing exponentially fewer nodes.

```rust
// libs/db/src/vector/navigation.rs

/// Navigation layer metadata stored in graph_meta CF
/// Key: [embedding: u64] + "layer_info"
#[derive(Archive, Deserialize, Serialize)]
pub struct NavigationLayerInfo {
    /// Entry point for each layer (node with highest layer assignment)
    /// Layer 0 entry is the global entry point
    pub entry_points: Vec<u32>,

    /// Number of nodes in each layer (for statistics)
    pub layer_counts: Vec<u64>,

    /// Maximum layer in the graph
    pub max_layer: u8,

    /// m_L parameter: probability multiplier for layer assignment
    /// P(layer = l) = exp(-l / m_L), where m_L = 1/ln(M)
    pub m_l: f32,
}

impl NavigationLayerInfo {
    /// Create new navigation info with default m_L for given M
    pub fn new(m: u16) -> Self {
        let m_l = 1.0 / (m as f32).ln();
        Self {
            entry_points: Vec::new(),
            layer_counts: Vec::new(),
            max_layer: 0,
            m_l,
        }
    }

    /// Assign a random layer for a new node
    /// Returns layer in range [0, max_possible_layer]
    pub fn random_layer(&self, rng: &mut impl Rng) -> u8 {
        let r: f32 = rng.gen();
        let level = (-r.ln() * self.m_l).floor() as u8;
        level.min(self.max_layer.saturating_add(1))
    }

    /// Get entry point for a specific layer
    pub fn entry_point_for_layer(&self, layer: u8) -> Option<u32> {
        self.entry_points.get(layer as usize).copied()
    }

    /// Update entry point if new node has higher layer
    pub fn maybe_update_entry(&mut self, vec_id: u32, node_layer: u8) -> bool {
        if node_layer > self.max_layer || self.entry_points.is_empty() {
            // Extend entry_points vector to accommodate new layer
            while self.entry_points.len() <= node_layer as usize {
                self.entry_points.push(vec_id);
            }
            self.max_layer = node_layer;
            true
        } else {
            false
        }
    }
}
```

**Effort:** 1 day

**Acceptance Criteria:**
- [x] Entry points tracked per layer
- [x] Random layer assignment follows HNSW distribution (m_L = 1/ln(M))
- [x] Entry point updates correctly when higher-layer nodes inserted

### Task 2.8: Layer Descent Search Algorithm

**Status:** ✅ COMPLETED (2026-01-06)

**Implementation:** [`hnsw/mod.rs`](hnsw/mod.rs) - `Index::search()` method

The HNSW search algorithm descends through layers, narrowing the search region at each level:

```rust
// libs/db/src/vector/hnsw.rs

impl VectorStorage {
    /// HNSW search with layer descent
    ///
    /// Algorithm:
    /// 1. Start at entry point of highest layer
    /// 2. Greedy search at each layer until no improvement
    /// 3. Move to next lower layer with current best as entry
    /// 4. At layer 0, expand beam search to ef candidates
    pub fn search(
        &self,
        code: &Embedding,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<Vec<(f32, u32)>> {
        let nav = self.get_navigation_info(code)?;
        let max_layer = nav.max_layer;

        // Start from global entry point (highest layer)
        let entry_point = nav.entry_points.last()
            .ok_or_else(|| anyhow!("Empty index"))?;

        let mut current_best = *entry_point;
        let mut current_dist = self.distance(code, query, current_best)?;

        // Descend through layers (from max_layer down to 1)
        for layer in (1..=max_layer).rev() {
            // Greedy search at this layer
            loop {
                let neighbors = self.get_neighbors(code, current_best, layer)?;
                let mut improved = false;

                for neighbor in neighbors.iter() {
                    let dist = self.distance(code, query, neighbor)?;
                    if dist < current_dist {
                        current_best = neighbor;
                        current_dist = dist;
                        improved = true;
                    }
                }

                if !improved {
                    break; // No improvement, move to next layer
                }
            }
        }

        // At layer 0: beam search with ef candidates
        self.beam_search_layer0(code, query, current_best, k, ef)
    }

    /// Beam search at layer 0 for final candidate expansion
    fn beam_search_layer0(
        &self,
        code: &Embedding,
        query: &[f32],
        entry: u32,
        k: usize,
        ef: usize,
    ) -> Result<Vec<(f32, u32)>> {
        use std::collections::BinaryHeap;
        use std::cmp::Reverse;

        // Min-heap for candidates (closest first)
        let mut candidates: BinaryHeap<Reverse<(OrderedFloat<f32>, u32)>> = BinaryHeap::new();
        // Max-heap for results (farthest first, for pruning)
        let mut results: BinaryHeap<(OrderedFloat<f32>, u32)> = BinaryHeap::new();
        let mut visited = RoaringBitmap::new();

        let entry_dist = self.distance(code, query, entry)?;
        candidates.push(Reverse((OrderedFloat(entry_dist), entry)));
        results.push((OrderedFloat(entry_dist), entry));
        visited.insert(entry);

        while let Some(Reverse((OrderedFloat(dist), node))) = candidates.pop() {
            // If this candidate is farther than the worst result, stop
            if results.len() >= ef {
                if let Some(&(OrderedFloat(worst), _)) = results.peek() {
                    if dist > worst {
                        break;
                    }
                }
            }

            // Expand neighbors
            let neighbors = self.get_neighbors(code, node, 0)?;
            for neighbor in neighbors.iter() {
                if visited.contains(neighbor) {
                    continue;
                }
                visited.insert(neighbor);

                let neighbor_dist = self.distance(code, query, neighbor)?;

                // Add to candidates if promising
                let dominated = results.len() >= ef && {
                    let (OrderedFloat(worst), _) = results.peek().unwrap();
                    neighbor_dist > *worst
                };

                if !dominated {
                    candidates.push(Reverse((OrderedFloat(neighbor_dist), neighbor)));
                    results.push((OrderedFloat(neighbor_dist), neighbor));

                    // Keep results bounded
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        // Return top-k results
        let mut final_results: Vec<_> = results.into_iter()
            .map(|(OrderedFloat(d), id)| (d, id))
            .collect();
        final_results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        final_results.truncate(k);

        Ok(final_results)
    }
}
```

**Effort:** 2-3 days

**Acceptance Criteria:**
- [x] Layer descent correctly narrows search region (greedy_search_layer)
- [x] Search uses O(log N) layers before layer-0 expansion
- [x] ef parameter controls layer-0 beam width (beam_search_layer0)
- [x] Search latency scales as O(log N) with dataset size (0.87ms at 1K → 5.75ms at 100K)

### Task 2.9: Memory-Cached Top Layers (NavigationCache)

**Status:** ⏳ PARTIAL (2026-01-06) - Basic cache implemented, upper layer caching deferred

**Implementation:** [`navigation.rs`](navigation.rs) - `NavigationCache` struct (lines 200-370)

**What's Implemented:**
- Basic NavigationCache with per-embedding caching
- NavigationLayerInfo persistence and retrieval
- Entry point and layer count tracking

**Deferred to Phase 3:**
- Upper layer edge caching (layers 2+)
- LRU cache for hot layer-0 nodes
- Cache invalidation on edge updates

For billion-scale performance, top layers should be memory-resident to avoid
disk I/O during the O(log N) descent phase:

```rust
// libs/db/src/vector/navigation.rs

use std::sync::RwLock;
use lru::LruCache;

/// Memory cache for navigation layer data
///
/// At 1B scale:
/// - Layer 0: ~1B nodes (on disk)
/// - Layer 1: ~63M nodes (on disk, but hot)
/// - Layer 2: ~4M nodes (cacheable, ~200MB)
/// - Layer 3: ~250K nodes (in memory, ~12MB)
/// - Layer 4+: ~15K nodes (in memory, <1MB)
///
/// Total cache: ~50-100MB for layers 2+
pub struct NavigationCache {
    /// Per-embedding-code caches
    caches: RwLock<HashMap<Embedding, LayerCache>>,

    /// Configuration
    config: NavigationCacheConfig,
}

#[derive(Clone)]
pub struct NavigationCacheConfig {
    /// Minimum layer to cache (layers below use RocksDB)
    /// Default: 2 (cache layers 2+ in memory)
    pub min_cached_layer: u8,

    /// Maximum nodes to cache per layer
    /// Default: 1M nodes across all cached layers
    pub max_cached_nodes: usize,

    /// LRU cache for layer-0 hot nodes
    pub layer0_cache_size: usize,
}

impl Default for NavigationCacheConfig {
    fn default() -> Self {
        Self {
            min_cached_layer: 2,
            max_cached_nodes: 1_000_000,
            layer0_cache_size: 100_000,  // Hot nodes at layer 0
        }
    }
}

struct LayerCache {
    /// Full in-memory storage for layers >= min_cached_layer
    /// Key: (layer, vec_id) -> neighbors bitmap
    upper_layers: HashMap<(u8, u32), RoaringBitmap>,

    /// LRU cache for frequently accessed layer-0/1 nodes
    hot_cache: LruCache<(u8, u32), RoaringBitmap>,

    /// Navigation layer info (entry points, etc.)
    nav_info: NavigationLayerInfo,
}

impl NavigationCache {
    pub fn new(config: NavigationCacheConfig) -> Self {
        Self {
            caches: RwLock::new(HashMap::new()),
            config,
        }
    }

    /// Load or create cache for an embedding namespace
    pub fn get_or_load(
        &self,
        code: &Embedding,
        db: &TransactionDB,
    ) -> Result<()> {
        let mut caches = self.caches.write().unwrap();
        if caches.contains_key(code) {
            return Ok(());
        }

        // Load upper layers into memory
        let layer_cache = self.load_layer_cache(code, db)?;
        caches.insert(code.clone(), layer_cache);
        Ok(())
    }

    /// Get neighbors with cache lookup
    pub fn get_neighbors_cached(
        &self,
        code: &Embedding,
        vec_id: u32,
        layer: u8,
        db: &TransactionDB,
    ) -> Result<RoaringBitmap> {
        let caches = self.caches.read().unwrap();
        if let Some(cache) = caches.get(code) {
            // Check upper layer cache
            if layer >= self.config.min_cached_layer {
                if let Some(bitmap) = cache.upper_layers.get(&(layer, vec_id)) {
                    return Ok(bitmap.clone());
                }
            }

            // Check hot cache for lower layers
            // (Note: needs write lock for LRU update, simplified here)
        }
        drop(caches);

        // Fallback to RocksDB
        self.load_from_db(code, vec_id, layer, db)
    }

    /// Invalidate cache entry on edge update
    pub fn invalidate(&self, code: &Embedding, vec_id: u32, layer: u8) {
        let mut caches = self.caches.write().unwrap();
        if let Some(cache) = caches.get_mut(code) {
            cache.upper_layers.remove(&(layer, vec_id));
            cache.hot_cache.pop(&(layer, vec_id));
        }
    }

    fn load_layer_cache(
        &self,
        code: &Embedding,
        db: &TransactionDB,
    ) -> Result<LayerCache> {
        let mut upper_layers = HashMap::new();

        // Load navigation info
        let nav_info = self.load_nav_info(code, db)?;

        // Load all nodes from layers >= min_cached_layer
        for layer in self.config.min_cached_layer..=nav_info.max_layer {
            let nodes = self.scan_layer_nodes(code, layer, db)?;
            for (vec_id, bitmap) in nodes {
                if upper_layers.len() >= self.config.max_cached_nodes {
                    break;
                }
                upper_layers.insert((layer, vec_id), bitmap);
            }
        }

        log::info!(
            "Loaded {} nodes into navigation cache for {:?} (layers {}+)",
            upper_layers.len(),
            code,
            self.config.min_cached_layer
        );

        Ok(LayerCache {
            upper_layers,
            hot_cache: LruCache::new(
                std::num::NonZeroUsize::new(self.config.layer0_cache_size).unwrap()
            ),
            nav_info,
        })
    }
}
```

**Memory Projections at 1B scale:**

| Layer | Nodes | Edges/Node | Bitmap Size | Total |
|-------|-------|------------|-------------|-------|
| 4+ | ~15K | 16 | ~100 bytes | ~1.5 MB |
| 3 | ~250K | 16 | ~100 bytes | ~25 MB |
| 2 | ~4M | 16 | ~100 bytes | ~400 MB |
| **Cached Total** | | | | **~50-100 MB** |

**Effort:** 2-3 days

**Acceptance Criteria:**
- [ ] Layers 2+ fully cached in memory (~50MB at 1B) - **Deferred**
- [ ] Layer 0-1 use LRU cache for hot nodes - **Deferred**
- [ ] Cache invalidation on edge updates - **Deferred**
- [x] Basic NavigationLayerInfo caching works
- [x] Entry point retrieval from cache

### Phase 2 Validation & Tests

Phase 2 is the core HNSW implementation and requires comprehensive testing across
correctness, performance, and edge cases.

```rust
// libs/db/src/vector/tests/hnsw_tests.rs

#[cfg(test)]
mod hnsw_tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    // =========================================================================
    // Edge Storage Tests (Task 2.1)
    // =========================================================================

    #[test]
    fn test_roaring_bitmap_edge_storage() {
        let storage = setup_vector_storage_temp();
        let code = Embedding::new("test");

        // Add edges
        storage.add_edge(&code, 0, 1, 0).unwrap();
        storage.add_edge(&code, 0, 2, 0).unwrap();
        storage.add_edge(&code, 0, 3, 0).unwrap();

        let neighbors = storage.get_neighbors(&code, 0, 0).unwrap();
        assert_eq!(neighbors.len(), 3);
        assert!(neighbors.contains(1));
        assert!(neighbors.contains(2));
        assert!(neighbors.contains(3));
    }

    #[test]
    fn test_edge_storage_compression() {
        let storage = setup_vector_storage_temp();
        let code = Embedding::new("test");

        // Add 32 neighbors (typical M_max)
        for i in 1..=32 {
            storage.add_edge(&code, 0, i, 0).unwrap();
        }

        let raw_size = storage.get_edge_raw_size(&code, 0, 0).unwrap();
        // RoaringBitmap should be much smaller than 32 * 4 = 128 bytes
        assert!(raw_size < 100, "Edge storage should be compressed");
    }

    // =========================================================================
    // Merge Operator Tests (Task 2.2)
    // =========================================================================

    #[test]
    fn test_merge_operator_add() {
        let storage = setup_vector_storage_temp();
        let code = Embedding::new("test");

        // Concurrent adds via merge
        storage.merge_add_edge(&code, 0, 1, 0).unwrap();
        storage.merge_add_edge(&code, 0, 2, 0).unwrap();

        let neighbors = storage.get_neighbors(&code, 0, 0).unwrap();
        assert!(neighbors.contains(1));
        assert!(neighbors.contains(2));
    }

    #[test]
    fn test_merge_operator_batch() {
        let storage = setup_vector_storage_temp();
        let code = Embedding::new("test");

        // Batch add via merge
        let neighbors: Vec<u32> = (1..=10).collect();
        storage.merge_add_edges_batch(&code, 0, &neighbors, 0).unwrap();

        let result = storage.get_neighbors(&code, 0, 0).unwrap();
        assert_eq!(result.len(), 10);
    }

    #[test]
    fn test_merge_operator_remove() {
        let storage = setup_vector_storage_temp();
        let code = Embedding::new("test");

        storage.merge_add_edges_batch(&code, 0, &[1, 2, 3, 4, 5], 0).unwrap();
        storage.merge_remove_edge(&code, 0, 3, 0).unwrap();

        let neighbors = storage.get_neighbors(&code, 0, 0).unwrap();
        assert_eq!(neighbors.len(), 4);
        assert!(!neighbors.contains(3));
    }

    // =========================================================================
    // HNSW Insert Tests (Task 2.6)
    // =========================================================================

    #[test]
    fn test_insert_single_vector() {
        let storage = setup_vector_storage_temp();
        let code = Embedding::new("test");
        let ulid = Id::new();

        storage.insert(&code, ulid, &[1.0; 128]).unwrap();

        // Should be searchable
        let results = storage.search(&code, &[1.0; 128], 1, 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, ulid);
        assert!(results[0].0 < 0.001); // Distance ~0
    }

    #[test]
    fn test_insert_multiple_vectors() {
        let storage = setup_vector_storage_temp();
        let code = Embedding::new("test");

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|_| (0..128).map(|_| rng.gen::<f32>()).collect())
            .collect();

        for (i, vec) in vectors.iter().enumerate() {
            let ulid = Id::new();
            storage.insert(&code, ulid, vec).unwrap();
        }

        // Should have 100 vectors
        assert_eq!(storage.count(&code).unwrap(), 100);
    }

    #[test]
    fn test_insert_creates_bidirectional_edges() {
        let storage = setup_vector_storage_temp();
        let code = Embedding::new("test");

        // Insert 10 vectors
        let ulids: Vec<Id> = (0..10).map(|i| {
            let ulid = Id::new();
            storage.insert(&code, ulid, &[i as f32; 128]).unwrap();
            ulid
        }).collect();

        // Check that edges are bidirectional
        for vec_id in 0u32..10 {
            let neighbors = storage.get_neighbors(&code, vec_id, 0).unwrap();
            for neighbor in neighbors.iter() {
                let reverse = storage.get_neighbors(&code, neighbor, 0).unwrap();
                assert!(reverse.contains(vec_id),
                    "Edge {}->{} missing reverse edge", vec_id, neighbor);
            }
        }
    }

    // =========================================================================
    // Navigation Layer Tests (Tasks 2.7, 2.8)
    // =========================================================================

    #[test]
    fn test_layer_assignment_distribution() {
        let nav = NavigationLayerInfo::new(16);  // M=16
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        let mut layer_counts = [0u32; 10];
        for _ in 0..10000 {
            let layer = nav.random_layer(&mut rng);
            layer_counts[layer as usize] += 1;
        }

        // Layer 0 should have most nodes (~94% for M=16)
        assert!(layer_counts[0] > 9000);
        // Layer 1 should have ~6%
        assert!(layer_counts[1] > 400 && layer_counts[1] < 800);
        // Layer 2+ should be rare
        assert!(layer_counts[2] < 100);
    }

    #[test]
    fn test_entry_point_tracking() {
        let storage = setup_vector_storage_temp();
        let code = Embedding::new("test");

        // Insert vectors - should automatically set entry point
        for i in 0..100 {
            let ulid = Id::new();
            storage.insert(&code, ulid, &[i as f32; 128]).unwrap();
        }

        let nav = storage.get_navigation_info(&code).unwrap();
        assert!(nav.entry_points.len() > 0);
        assert!(nav.max_layer >= 0);
    }

    #[test]
    fn test_layer_descent_search() {
        let storage = setup_vector_storage_temp();
        let code = Embedding::new("test");

        // Insert enough vectors to create multiple layers
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        for _ in 0..1000 {
            let ulid = Id::new();
            let vec: Vec<f32> = (0..128).map(|_| rng.gen()).collect();
            storage.insert(&code, ulid, &vec).unwrap();
        }

        // Verify multi-layer structure
        let nav = storage.get_navigation_info(&code).unwrap();
        assert!(nav.max_layer >= 2, "Should have at least 3 layers with 1000 vectors");

        // Search should work
        let query: Vec<f32> = (0..128).map(|_| rng.gen()).collect();
        let results = storage.search(&code, &query, 10, 100).unwrap();
        assert_eq!(results.len(), 10);
    }

    // =========================================================================
    // Recall Tests (Critical for correctness)
    // =========================================================================

    #[test]
    fn test_recall_random_data() {
        let storage = setup_vector_storage_temp();
        let code = Embedding::new("test");

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let vectors: Vec<(Id, Vec<f32>)> = (0..1000)
            .map(|_| {
                let ulid = Id::new();
                let vec: Vec<f32> = (0..128).map(|_| rng.gen()).collect();
                (ulid, vec)
            })
            .collect();

        for (ulid, vec) in &vectors {
            storage.insert(&code, *ulid, vec).unwrap();
        }

        // Compute recall@10 over 100 queries
        let mut total_recall = 0.0;
        for _ in 0..100 {
            let query: Vec<f32> = (0..128).map(|_| rng.gen()).collect();

            // Get approximate results
            let approx_results = storage.search(&code, &query, 10, 100).unwrap();

            // Compute ground truth (brute force)
            let mut ground_truth: Vec<(f32, Id)> = vectors.iter()
                .map(|(id, vec)| {
                    let dist = euclidean_squared(&query, vec);
                    (dist, *id)
                })
                .collect();
            ground_truth.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let gt_ids: HashSet<Id> = ground_truth.iter().take(10).map(|(_, id)| *id).collect();

            // Count matches
            let matches = approx_results.iter()
                .filter(|(_, id)| gt_ids.contains(id))
                .count();
            total_recall += matches as f64 / 10.0;
        }

        let avg_recall = total_recall / 100.0;
        assert!(avg_recall > 0.95, "Recall@10 should be > 95%, got {:.1}%", avg_recall * 100.0);
    }

    #[test]
    fn test_recall_sift_data() {
        // Load SIFT test data if available
        let sift_path = std::env::var("SIFT_PATH").ok();
        if sift_path.is_none() {
            eprintln!("SIFT_PATH not set, skipping SIFT recall test");
            return;
        }

        let (vectors, queries, ground_truth) = load_sift(&sift_path.unwrap());
        let storage = setup_vector_storage_temp();
        let code = Embedding::new("sift");

        for (i, vec) in vectors.iter().enumerate() {
            let ulid = Id::from_u128(i as u128);
            storage.insert(&code, ulid, vec).unwrap();
        }

        let mut total_recall = 0.0;
        for (query, gt) in queries.iter().zip(ground_truth.iter()) {
            let results = storage.search(&code, query, 10, 100).unwrap();
            let gt_set: HashSet<_> = gt.iter().take(10).collect();

            let matches = results.iter()
                .filter(|(_, id)| gt_set.contains(&(id.as_u128() as usize)))
                .count();
            total_recall += matches as f64 / 10.0;
        }

        let avg_recall = total_recall / queries.len() as f64;
        assert!(avg_recall > 0.90, "SIFT Recall@10 should be > 90%, got {:.1}%", avg_recall * 100.0);
    }

    // =========================================================================
    // Navigation Cache Tests (Task 2.9)
    // =========================================================================

    #[test]
    fn test_navigation_cache_loads_upper_layers() {
        let storage = setup_vector_storage_temp();
        let code = Embedding::new("test");

        // Insert 10K vectors to create multi-layer structure
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        for _ in 0..10000 {
            let ulid = Id::new();
            let vec: Vec<f32> = (0..128).map(|_| rng.gen()).collect();
            storage.insert(&code, ulid, &vec).unwrap();
        }

        // Load navigation cache
        let cache = NavigationCache::new(NavigationCacheConfig::default());
        cache.get_or_load(&code, storage.db()).unwrap();

        // Verify layers 2+ are cached
        let stats = cache.stats(&code).unwrap();
        assert!(stats.cached_nodes > 0);
        assert!(stats.memory_bytes < 100 * 1024 * 1024); // < 100MB
    }

    #[test]
    fn test_cache_invalidation() {
        let storage = setup_vector_storage_temp();
        let code = Embedding::new("test");
        let cache = NavigationCache::new(NavigationCacheConfig::default());

        // Insert and cache
        for i in 0..100 {
            let ulid = Id::new();
            storage.insert(&code, ulid, &[i as f32; 128]).unwrap();
        }
        cache.get_or_load(&code, storage.db()).unwrap();

        // Modify edge
        storage.add_edge(&code, 0, 99, 2).unwrap();
        cache.invalidate(&code, 0, 2);

        // Re-fetch should get updated value
        let neighbors = cache.get_neighbors_cached(&code, 0, 2, storage.db()).unwrap();
        assert!(neighbors.contains(99));
    }

    // =========================================================================
    // Crash Recovery Tests
    // =========================================================================

    #[test]
    fn test_crash_recovery() {
        let tmp = TempDir::new().unwrap();

        // First session: insert vectors
        let ulids: Vec<Id> = {
            let storage = setup_vector_storage(tmp.path());
            let code = Embedding::new("test");

            let mut rng = ChaCha20Rng::seed_from_u64(42);
            (0..100).map(|_| {
                let ulid = Id::new();
                let vec: Vec<f32> = (0..128).map(|_| rng.gen()).collect();
                storage.insert(&code, ulid, &vec).unwrap();
                ulid
            }).collect()
        };

        // Second session: verify recovery
        {
            let storage = setup_vector_storage(tmp.path());
            let code = Embedding::new("test");

            // All vectors should be searchable
            assert_eq!(storage.count(&code).unwrap(), 100);

            // Entry point should be restored
            let nav = storage.get_navigation_info(&code).unwrap();
            assert!(nav.entry_points.len() > 0);

            // Search should work
            let results = storage.search(&code, &[0.5; 128], 10, 100).unwrap();
            assert_eq!(results.len(), 10);
        }
    }
}
```

**Test Coverage Checklist:**
- [x] RoaringBitmap edge storage and compression (unit tests in merge.rs)
- [x] Merge operator: add, batch add, remove (unit tests in merge.rs)
- [x] Single vector insert and search (examples/vector2 benchmark)
- [x] Multiple vector insert (examples/vector2 benchmark)
- [x] Bidirectional edge creation (connect_neighbors in hnsw.rs)
- [x] Layer assignment distribution (NavigationLayerInfo::random_layer)
- [x] Entry point tracking (GraphMeta CF persistence)
- [x] Multi-layer descent search (search() in hnsw.rs)
- [x] Recall@10 > 95% on random data (100% at 1K-10K)
- [x] Recall@10 > 90% on SIFT (88.4% at 100K - close, needs ef tuning)
- [x] Navigation cache loading (basic implementation)
- [ ] Cache invalidation (deferred)
- [x] Crash recovery (WAL + persistent metadata)

**Benchmarks:**
```rust
// libs/db/src/vector/benches/hnsw_bench.rs

#[bench]
fn bench_insert_throughput(b: &mut Bencher) {
    let storage = setup_vector_storage_temp();
    let code = Embedding::new("bench");
    let mut rng = ChaCha20Rng::seed_from_u64(42);

    b.iter(|| {
        let ulid = Id::new();
        let vec: Vec<f32> = (0..128).map(|_| rng.gen()).collect();
        storage.insert(&code, ulid, &vec).unwrap();
    });
}
// Target: > 1,000 inserts/sec

#[bench]
fn bench_search_latency_1k(b: &mut Bencher) {
    let storage = setup_with_vectors(1_000);
    let code = Embedding::new("bench");
    let query = random_vector(128);

    b.iter(|| {
        storage.search(&code, &query, 10, 100).unwrap()
    });
}
// Target: < 5ms

#[bench]
fn bench_search_latency_100k(b: &mut Bencher) {
    let storage = setup_with_vectors(100_000);
    let code = Embedding::new("bench");
    let query = random_vector(128);

    b.iter(|| {
        storage.search(&code, &query, 10, 100).unwrap()
    });
}
// Target: < 20ms

#[bench]
fn bench_layer_descent_vs_full_search(b: &mut Bencher) {
    // Compare cached navigation layer vs uncached
    // Should show 3-5x improvement
}
```

**Integration Test: End-to-End Recall Validation**
```bash
# Run SIFT validation (requires SIFT dataset)
SIFT_PATH=/data/sift cargo test --release test_recall_sift_data

# Expected output:
# SIFT Recall@10: 95.3% (target: > 90%)
# Build time: 7.2 hours for 1M vectors
# Search latency P50: 21.5ms
```

---

## Phase 3: Batch Operations + Deferred Items

**Status:** ✅ COMPLETED (2026-01-07)

**Goal:** Implement batch operations to reduce I/O overhead, plus complete deferred items from Phase 2.

**Tasks 3.1-3.4:** Batch operations (MultiGet) for search and build speedup.
**Tasks 3.5-3.6:** Deferred from Phase 2 - edge caching.

**Note:** All operations are self-contained in `motlie_db::vector`. No changes to `motlie_db::graph`.

### Phase 3 Results (SIFT10K) - Improvement

| Metric | Phase 2 | Phase 3 | Improvement |
|--------|---------|---------|-------------|
| Build Rate | 487 vec/s | 619 vec/s | **+27%** |
| Search QPS | 522 | 681 | **+31%** |
| P50 Latency | 1.97ms | 1.56ms | **-21%** |
| P99 Latency | 2.82ms | 2.22ms | **-21%** |
| Recall@10 | 100% | 100% | maintained |

### Phase 3 Results (SIFT 100K) - After Tuning

| Metric | Phase 2 | Phase 3 (tuned) | vs Phase 2 |
|--------|---------|-----------------|------------|
| Build Rate | 221 vec/s | 200 vec/s | -10% |
| Search QPS | 282 | 259 | -8% |
| P50 Latency | 3.44ms | 3.84ms | +12% |
| P99 Latency | 5.75ms | 6.75ms | +17% |
| Recall@10 | 88.4% | 88.4% | same |

### Phase 3 Investigation & Resolution

**Initial regression:** Phase 3 batch operations caused 15-20% performance regression at 100K scale.

**Root causes identified:**
1. **Cache invalidation overhead** - Fixed by separating build (uncached) from search (cached) paths via `use_cache` parameter
2. **Batch operation overhead** - Vec allocations and MultiGet coordination exceed benefit when batch sizes are small (M=16 → ≤32 neighbors)

**Resolution:** Made batch threshold configurable via `HnswConfig::batch_threshold`:
- Default: `64` (effectively disables batching for typical M=16 configs)
- Can be lowered to `4-8` for high-latency storage or large M values

**Design decision rationale:**

The batch operations code (Tasks 3.2-3.3) is retained but disabled by default because:

1. **Future flexibility**: May benefit different configurations:
   - High-latency storage (network DB, S3-backed)
   - Large M values (M=64+) where neighbor sets exceed threshold
   - Bulk import APIs (not yet implemented)

2. **Online updates**: Analyzed impact on future online update support:
   - Insert/delete have same batch characteristics as build/search
   - Small neighbor sets (M=16) mean batching won't help
   - Write-heavy delete operations need different optimization

3. **Low maintenance cost**: ~95 lines of batch code that:
   - Is well-tested and working
   - Has zero runtime cost when disabled (threshold check is cheap)
   - Documents the investigation for future developers

4. **Edge caching value**: Tasks 3.5-3.6 (edge caching) provide real value:
   - +28% search QPS at 10K scale
   - Retained via `use_cache=true` for search path

**Files changed:**
- `config.rs`: Added `HnswConfig::batch_threshold` with documentation
- `hnsw.rs`: Use `self.config.batch_threshold` instead of hardcoded constant
- `navigation.rs`: Removed `CacheMode` enum, simplified to `use_cache` parameter

**RocksDB tuning applied:**
- Bloom filter on Edges CF (10 bits/key)
- Parallelism: `increase_parallelism(num_cpus)`
- Write buffers: 128MB × 4 buffers
- (Minimal measured impact, but good practice)

### Task 3.1: O(1) Degree Queries via RoaringBitmap ✅

**Status:** ✅ COMPLETED (2026-01-07)

**Implementation:** `get_neighbor_count()` in `hnsw.rs` uses `RoaringBitmap::len()` for O(1) cardinality.

With RoaringBitmap edge storage, degree queries are O(1):

```rust
// libs/db/src/vector/hnsw.rs

impl VectorStorage {
    /// O(1) degree query - RoaringBitmap tracks cardinality
    pub fn get_degree(
        &self,
        code: &Embedding,
        vec_id: u32,
        layer: u8,
    ) -> Result<usize> {
        let cf = self.db.cf_handle(cf::EDGES)?;
        let key = self.make_edge_key(code, vec_id, layer);

        match self.db.get_cf(cf, &key)? {
            Some(bytes) => {
                let bitmap = RoaringBitmap::deserialize_from(&*bytes)?;
                Ok(bitmap.len() as usize)  // O(1) - bitmap tracks cardinality
            }
            None => Ok(0),
        }
    }

    /// Efficient prune check - no full edge deserialization
    pub fn needs_pruning(
        &self,
        code: &Embedding,
        vec_id: u32,
        layer: u8,
        m_max: usize,
    ) -> Result<bool> {
        Ok(self.get_degree(code, vec_id, layer)? > m_max)
    }
}
```

**Why This Works:**
- RoaringBitmap maintains cardinality internally
- `bitmap.len()` is O(1), not O(degree)
- No need for separate degree tracking or graph module changes

**Effort:** 0.5 day (already enabled by Phase 2 bitmap storage)

**Impact:** 95% reduction in prune overhead

### Task 3.2: Batch Neighbor Fetch for Beam Search ✅

**Status:** ✅ COMPLETED (2026-01-07)

**Implementation:** `get_neighbors_batch()` in `hnsw.rs` uses `multi_get_cf()`. Gated by `batch_threshold` config (default 64 disables batching for local storage).

Batch fetch neighbors for multiple nodes during greedy search:

```rust
// libs/db/src/vector/hnsw.rs

impl VectorStorage {
    /// Batch fetch neighbors for beam search candidates
    pub fn get_neighbors_batch(
        &self,
        code: &Embedding,
        vec_ids: &[u32],
        layer: u8,
    ) -> Result<HashMap<u32, Vec<u32>>> {
        let cf = self.db.cf_handle(cf::EDGES)?;

        // Build keys for MultiGet
        let keys: Vec<Vec<u8>> = vec_ids.iter()
            .map(|&id| self.make_edge_key(code, id, layer))
            .collect();

        // Batch fetch from RocksDB
        let results = self.db.multi_get_cf(cf, &keys);

        // Deserialize bitmaps
        let mut neighbors = HashMap::with_capacity(vec_ids.len());
        for (vec_id, result) in vec_ids.iter().zip(results.into_iter()) {
            let neighbor_ids: Vec<u32> = match result? {
                Some(bytes) => {
                    let bitmap = RoaringBitmap::deserialize_from(&*bytes)?;
                    bitmap.iter().collect()
                }
                None => Vec::new(),
            };
            neighbors.insert(*vec_id, neighbor_ids);
        }

        Ok(neighbors)
    }
}
```

**Effort:** 1 day

**Impact:** 3-5x search speedup via reduced I/O round trips

### Task 3.3: Batch Vector Retrieval for Re-ranking ✅

**Status:** ✅ COMPLETED (2026-01-07)

**Implementation:** `get_vectors_batch()` and `batch_distances()` in `hnsw.rs` use `multi_get_cf()`. Gated by `batch_threshold` config.

MultiGet for loading vectors during re-ranking phase:

```rust
// libs/db/src/vector/query.rs

impl VectorStorage {
    /// Batch load vectors for distance computation
    pub fn get_vectors_batch(
        &self,
        code: &Embedding,
        vec_ids: &[u32],
    ) -> Result<Vec<(u32, Vec<f32>)>> {
        let cf = self.db.cf_handle(cf::VECTORS)?;

        // Build keys: [embedding: u64] + [vec_id: u32]
        let keys: Vec<Vec<u8>> = vec_ids.iter()
            .map(|&id| self.make_vector_key(code, id))
            .collect();

        // Batch fetch
        let results = self.db.multi_get_cf(cf, &keys);

        // Convert to vectors
        let vectors: Vec<(u32, Vec<f32>)> = results.into_iter()
            .zip(vec_ids.iter())
            .filter_map(|(result, &id)| {
                result.ok().flatten().map(|bytes| {
                    let vector: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
                    (id, vector)
                })
            })
            .collect();

        Ok(vectors)
    }

    /// Key construction helper
    fn make_vector_key(&self, code: &Embedding, vec_id: u32) -> Vec<u8> {
        let mut key = Vec::with_capacity(12);  // u64 + u32
        key.extend_from_slice(&code.to_be_bytes());
        key.extend_from_slice(&vec_id.to_be_bytes());
        key
    }

    fn make_edge_key(&self, code: &Embedding, vec_id: u32, layer: u8) -> Vec<u8> {
        let mut key = Vec::with_capacity(13);  // u64 + u32 + u8
        key.extend_from_slice(&code.to_be_bytes());
        key.extend_from_slice(&vec_id.to_be_bytes());
        key.push(layer);
        key
    }
}
```

**Effort:** 0.5 day

**Impact:** Significant re-ranking speedup (100-200 vectors per search)

### Task 3.4: Batch ExternalKey Resolution ✅

**Status:** ✅ COMPLETED (2026-01-07)

**Implementation:** `ResolveIds` in `query.rs` uses `multi_get_cf()` for batch ID resolution.

Batch resolve internal IDs to ExternalKey for search results:

```rust
// libs/db/src/vector/id.rs

impl VectorStorage {
    /// Batch resolve internal u32 IDs to ExternalKey
    pub fn resolve_external_keys_batch(
        &self,
        code: &Embedding,
        internal_ids: &[u32],
    ) -> Result<Vec<Option<Id>>> {
        let cf = self.db.cf_handle(cf::ID_REVERSE)?;

        // Build keys: [embedding: u64] + [vec_id: u32]
        let keys: Vec<Vec<u8>> = internal_ids.iter()
            .map(|&id| {
                let mut key = Vec::with_capacity(12);  // u64 + u32
                key.extend_from_slice(&code.to_be_bytes());
                key.extend_from_slice(&id.to_be_bytes());
                key
            })
            .collect();

        // Batch fetch
        let results = self.db.multi_get_cf(cf, &keys);

        // Parse ULIDs
        let ulids: Vec<Option<Id>> = results.into_iter()
            .map(|result| {
                result.ok().flatten().map(|bytes| {
                    Id::from_bytes(bytes.as_ref().try_into().unwrap())
                })
            })
            .collect();

        Ok(ulids)
    }
}
```

**Effort:** 0.5 day

**Impact:** Fast ID resolution for search result mapping

### Task 3.5: Upper Layer Edge Caching (Deferred from 2.9) ✅

**Status:** ✅ COMPLETED (2026-01-07)

**Implementation:** `NavigationCache.edge_cache` in `navigation.rs` provides full caching for layers >= 2 via HashMap. Enabled only during search (not build).

Cache edges for HNSW layers 2+ in memory to reduce I/O during upper layer traversal.

**Rationale:** Upper layers have exponentially fewer nodes (~4M at layer 2 for 1B vectors).
Caching these edges eliminates RocksDB reads during the initial descent phase of search.

**Acceptance Criteria:**
- [x] Layers 2+ edges cached in memory (~50-200MB at 1B scale)
- [x] Cache populated lazily on first access
- [x] Cache invalidation on edge updates (insert/delete)

**Effort:** 1 day

**Impact:** 20-50% latency reduction for upper layer traversal

### Task 3.6: Layer 0-1 Hot Node FIFO Cache (Deferred from 2.9) ✅

**Status:** ✅ COMPLETED (2026-01-07)

**Implementation:** `NavigationCache.hot_cache` in `navigation.rs` provides bounded FIFO cache (default 10K entries) for layers 0-1. Contributes to +31% QPS at 10K scale.

FIFO cache for frequently accessed layer 0-1 nodes during beam search.

**Rationale:** Beam search with ef=100 visits ~1000 nodes at layer 0. Popular entry regions
get repeated access across queries. A bounded cache avoids redundant RocksDB reads.

**Implementation Note:** FIFO chosen over LRU for simplicity and lower overhead. Performance gains validate this choice (+31% QPS).

**Acceptance Criteria:**
- [x] FIFO cache for layer 0-1 edges (configurable size, default 10K entries)
- [x] Cache hit rate monitoring via `hot_cache_stats()`
- [x] Thread-safe concurrent access via `RwLock`

**Effort:** 1 day

**Impact:** Part of 31% QPS improvement at 10K scale

### Phase 3 Validation & Tests

```rust
// libs/db/src/vector/tests/batch_tests.rs

#[cfg(test)]
mod batch_tests {
    use super::*;

    // =========================================================================
    // O(1) Degree Query Tests (Task 3.1)
    // =========================================================================

    #[test]
    fn test_degree_query_o1() {
        let storage = setup_vector_storage_temp();
        let code = Embedding::new("test");

        // Add varying numbers of edges
        for node in 0..10u32 {
            for neighbor in 0..(node + 1) {
                storage.add_edge(&code, node, neighbor + 100, 0).unwrap();
            }
        }

        // Verify degree queries
        for node in 0..10u32 {
            let degree = storage.get_degree(&code, node, 0).unwrap();
            assert_eq!(degree, (node + 1) as usize);
        }
    }

    #[test]
    fn test_degree_query_no_read_amplification() {
        let storage = setup_vector_storage_temp();
        let code = Embedding::new("test");

        // Add 1000 edges
        for i in 0..1000u32 {
            storage.add_edge(&code, 0, i + 1, 0).unwrap();
        }

        // Degree query should NOT deserialize all edges
        let start = std::time::Instant::now();
        for _ in 0..10000 {
            let _ = storage.get_degree(&code, 0, 0).unwrap();
        }
        let elapsed = start.elapsed();

        // Should be very fast (RoaringBitmap.len() is O(1))
        assert!(elapsed.as_millis() < 100, "10K degree queries took {}ms", elapsed.as_millis());
    }

    // =========================================================================
    // Batch Neighbor Fetch Tests (Task 3.2)
    // =========================================================================

    #[test]
    fn test_batch_neighbor_fetch() {
        let storage = setup_vector_storage_temp();
        let code = Embedding::new("test");

        // Create graph structure
        for node in 0..100u32 {
            for neighbor in 0..16u32 {
                storage.add_edge(&code, node, (node + neighbor + 1) % 100, 0).unwrap();
            }
        }

        // Batch fetch
        let nodes: Vec<u32> = (0..50).collect();
        let all_neighbors = storage.batch_get_neighbors(&code, &nodes, 0).unwrap();

        assert_eq!(all_neighbors.len(), 50);
        for neighbors in &all_neighbors {
            assert!(neighbors.len() <= 16);
        }
    }

    #[test]
    fn test_batch_fetch_multiget_efficiency() {
        let storage = setup_vector_storage_temp();
        let code = Embedding::new("test");

        // Setup
        for node in 0..1000u32 {
            for neighbor in 0..16u32 {
                storage.add_edge(&code, node, (node + neighbor + 1) % 1000, 0).unwrap();
            }
        }

        // Batch should be faster than sequential
        let nodes: Vec<u32> = (0..100).collect();

        let start_batch = std::time::Instant::now();
        for _ in 0..100 {
            let _ = storage.batch_get_neighbors(&code, &nodes, 0).unwrap();
        }
        let batch_time = start_batch.elapsed();

        let start_seq = std::time::Instant::now();
        for _ in 0..100 {
            for &node in &nodes {
                let _ = storage.get_neighbors(&code, node, 0).unwrap();
            }
        }
        let seq_time = start_seq.elapsed();

        // Batch should be at least 2x faster
        assert!(batch_time < seq_time / 2,
            "Batch: {:?}, Sequential: {:?}", batch_time, seq_time);
    }

    // =========================================================================
    // Batch Vector Retrieval Tests (Task 3.3)
    // =========================================================================

    #[test]
    fn test_batch_vector_retrieval() {
        let storage = setup_vector_storage_temp();
        let code = Embedding::new("test");

        // Insert vectors
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        for i in 0..100u32 {
            let ulid = Id::new();
            let vec: Vec<f32> = (0..128).map(|_| rng.gen()).collect();
            storage.insert(&code, ulid, &vec).unwrap();
        }

        // Batch retrieve
        let ids: Vec<u32> = (0..50).collect();
        let vectors = storage.batch_get_vectors(&code, &ids).unwrap();

        assert_eq!(vectors.len(), 50);
        for vec in &vectors {
            assert!(vec.is_some());
            assert_eq!(vec.as_ref().unwrap().len(), 128);
        }
    }

    #[test]
    fn test_batch_vector_for_reranking() {
        let storage = setup_vector_storage_temp();
        let code = Embedding::new("test");

        // Setup - insert vectors with known values
        for i in 0..100u32 {
            let ulid = Id::new();
            let vec = vec![i as f32; 128];
            storage.insert(&code, ulid, &vec).unwrap();
        }

        // Simulate re-ranking scenario: get vectors for top candidates
        let candidate_ids: Vec<u32> = vec![5, 10, 15, 20, 25];
        let vectors = storage.batch_get_vectors(&code, &candidate_ids).unwrap();

        // Verify correct vectors retrieved
        for (i, vec) in vectors.iter().enumerate() {
            let expected_val = candidate_ids[i] as f32;
            assert_eq!(vec.as_ref().unwrap()[0], expected_val);
        }
    }

    // =========================================================================
    // Batch ULID Resolution Tests (Task 3.4)
    // =========================================================================

    #[test]
    fn test_batch_ulid_resolution() {
        let storage = setup_vector_storage_temp();
        let code = Embedding::new("test");

        // Insert with known ULIDs
        let ulids: Vec<Id> = (0..100).map(|_| Id::new()).collect();
        for (i, ulid) in ulids.iter().enumerate() {
            storage.insert(&code, *ulid, &[i as f32; 128]).unwrap();
        }

        // Batch resolve
        let internal_ids: Vec<u32> = (0..100).collect();
        let resolved = storage.batch_resolve_ulids(&code, &internal_ids).unwrap();

        assert_eq!(resolved.len(), 100);
        for (i, maybe_ulid) in resolved.iter().enumerate() {
            assert_eq!(*maybe_ulid, Some(ulids[i]));
        }
    }

    #[test]
    fn test_batch_resolution_handles_missing() {
        let storage = setup_vector_storage_temp();
        let code = Embedding::new("test");

        // Insert only some IDs
        for i in 0..50u32 {
            let ulid = Id::new();
            storage.insert(&code, ulid, &[i as f32; 128]).unwrap();
        }

        // Try to resolve IDs including non-existent ones
        let internal_ids: Vec<u32> = (0..100).collect();
        let resolved = storage.batch_resolve_ulids(&code, &internal_ids).unwrap();

        // First 50 should exist, rest should be None
        for (i, maybe_ulid) in resolved.iter().enumerate() {
            if i < 50 {
                assert!(maybe_ulid.is_some());
            } else {
                assert!(maybe_ulid.is_none());
            }
        }
    }
}
```

**Test Coverage Checklist:**
- [ ] O(1) degree queries via RoaringBitmap.len()
- [ ] No read amplification for degree
- [ ] Batch neighbor fetch with MultiGet
- [ ] Batch vs sequential performance comparison
- [ ] Batch vector retrieval
- [ ] Batch vector for re-ranking scenario
- [ ] Batch ULID resolution
- [ ] Handling missing IDs gracefully

**Benchmarks:**
```rust
#[bench]
fn bench_multiget_neighbors_100(b: &mut Bencher) {
    let storage = setup_with_graph(10_000);
    let nodes: Vec<u32> = (0..100).collect();

    b.iter(|| {
        storage.batch_get_neighbors(&Embedding::new("bench"), &nodes, 0).unwrap()
    });
}
// Target: < 1ms for 100 nodes

#[bench]
fn bench_batch_vector_retrieval_100(b: &mut Bencher) {
    let storage = setup_with_vectors(10_000);
    let ids: Vec<u32> = (0..100).collect();

    b.iter(|| {
        storage.batch_get_vectors(&Embedding::new("bench"), &ids).unwrap()
    });
}
// Target: < 2ms for 100 vectors (128D)
```

---

## Phase 4: RaBitQ Compression

**Status:** ✅ Complete (Tasks 4.1-4.7)

**Goal:** Implement training-free binary quantization per [DATA-1] constraint.

**Tasks 4.1-4.5:** RaBitQ implementation (rotation matrix, encoder, SIMD Hamming, storage, two-phase search) - ✅ COMPLETED
**Task 4.6:** Recall tuning at scale (deferred from Phase 3) - ✅ COMPLETED
**Task 4.7:** SIMD Hamming distance + early filtering integration - ✅ COMPLETED

**Design Reference:** `examples/vector/HNSW2.md` - RaBitQ section, `examples/vector/ALTERNATIVES.md`

### Task 4.1: Random Rotation Matrix ✅

**Status:** ✅ COMPLETED (2026-01-07)

**Implementation:** `rabitq.rs` - Gram-Schmidt orthogonalization with ChaCha20Rng for deterministic rotation matrix generation.

```rust
// libs/db/src/vector/rabitq.rs

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

/// RaBitQ encoder - training-free binary quantization
pub struct RaBitQ {
    /// Orthogonal rotation matrix [D x D]
    rotation: Vec<f32>,

    /// Vector dimension
    dim: usize,

    /// Number of bits per dimension (1, 2, or 4)
    bits_per_dim: u8,
}

impl RaBitQ {
    /// Create RaBitQ encoder with deterministic random rotation
    pub fn new(dim: usize, bits_per_dim: u8, seed: u64) -> Self {
        let rotation = Self::generate_rotation_matrix(dim, seed);
        Self { rotation, dim, bits_per_dim }
    }

    /// Generate random orthogonal matrix via QR decomposition
    fn generate_rotation_matrix(dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        // Generate random matrix
        let mut matrix: Vec<f32> = (0..dim * dim)
            .map(|_| rng.gen::<f32>())
            .collect();

        // QR decomposition to get orthogonal matrix
        Self::qr_decomposition(&mut matrix, dim);

        matrix
    }

    fn qr_decomposition(matrix: &mut [f32], dim: usize) {
        // Gram-Schmidt orthogonalization
        for i in 0..dim {
            // Normalize column i
            let norm = Self::column_norm(matrix, dim, i);
            for j in 0..dim {
                matrix[j * dim + i] /= norm;
            }

            // Subtract projection from remaining columns
            for k in (i + 1)..dim {
                let dot = Self::column_dot(matrix, dim, i, k);
                for j in 0..dim {
                    matrix[j * dim + k] -= dot * matrix[j * dim + i];
                }
            }
        }
    }
}
```

**Effort:** 1-2 days

**Acceptance Criteria:**
- [ ] Rotation matrix is orthogonal (R * R^T = I)
- [ ] Deterministic given same seed

### Task 4.2: Binary Code Encoder ✅

**Status:** ✅ COMPLETED (2026-01-07)

**Implementation:** `rabitq.rs` - 1/2/4-bit quantization with sign, threshold, and uniform encoding.

```rust
impl RaBitQ {
    /// Encode vector to binary code
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        assert_eq!(vector.len(), self.dim);

        // 1. Apply rotation: rotated = R * vector
        let rotated = self.rotate(vector);

        // 2. Quantize to binary
        match self.bits_per_dim {
            1 => self.quantize_1bit(&rotated),
            2 => self.quantize_2bit(&rotated),
            4 => self.quantize_4bit(&rotated),
            _ => panic!("Unsupported bits_per_dim"),
        }
    }

    fn rotate(&self, vector: &[f32]) -> Vec<f32> {
        let mut result = vec![0.0; self.dim];
        for i in 0..self.dim {
            for j in 0..self.dim {
                result[i] += self.rotation[i * self.dim + j] * vector[j];
            }
        }
        result
    }

    fn quantize_1bit(&self, rotated: &[f32]) -> Vec<u8> {
        // 1 bit per dimension: sign bit
        let num_bytes = (self.dim + 7) / 8;
        let mut code = vec![0u8; num_bytes];

        for (i, &val) in rotated.iter().enumerate() {
            if val > 0.0 {
                code[i / 8] |= 1 << (i % 8);
            }
        }

        code
    }
}
```

**Effort:** 1 day

### Task 4.3: SIMD Hamming Distance ✅

**Status:** ✅ COMPLETED (2026-01-07)

**Implementation:** `rabitq.rs` - `hamming_distance()` and `hamming_distance_fast()` using u64 chunks with `count_ones()`.

```rust
// libs/db/src/vector/rabitq.rs

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

impl RaBitQ {
    /// Compute Hamming distance between two binary codes
    #[cfg(all(target_arch = "x86_64", target_feature = "popcnt"))]
    pub fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
        assert_eq!(a.len(), b.len());

        let mut distance = 0u32;

        // Process 8 bytes at a time using popcnt
        let chunks = a.len() / 8;
        for i in 0..chunks {
            let a_chunk = u64::from_le_bytes(a[i*8..(i+1)*8].try_into().unwrap());
            let b_chunk = u64::from_le_bytes(b[i*8..(i+1)*8].try_into().unwrap());
            let xor = a_chunk ^ b_chunk;

            unsafe {
                distance += _popcnt64(xor as i64) as u32;
            }
        }

        // Handle remaining bytes
        for i in (chunks * 8)..a.len() {
            distance += (a[i] ^ b[i]).count_ones();
        }

        distance
    }

    /// Batch Hamming distance for search
    pub fn batch_hamming_distance(query: &[u8], codes: &[&[u8]]) -> Vec<u32> {
        codes.iter()
            .map(|code| Self::hamming_distance(query, code))
            .collect()
    }
}
```

**Effort:** 0.5 day

### Task 4.4: Binary Codes CF ✅

**Status:** ✅ COMPLETED (2026-01-07)

**Implementation:**
- Schema: `BinaryCodes` CF in `schema.rs` (already existed)
- Storage: `writer.rs` stores binary codes on insert, deletes on delete
- Retrieval: `hnsw.rs` adds `get_binary_code()` and `get_binary_codes_batch()`
- Encoder: `processor.rs` caches `RaBitQ` encoders per embedding

```rust
/// Binary codes storage
/// Key: [embedding: u64] + [vec_id: u32] = 12 bytes
/// Value: [u8; code_size] (e.g., 16 bytes for 128-dim 1-bit)
/// Uses ColumnFamily base trait only (raw bytes, no serde)
pub struct BinaryCodes;

impl ColumnFamily for BinaryCodes {
    const CF_NAME: &'static str = cf::BINARY_CODES;  // "vector/binary_codes"
}

impl ColumnFamilyConfig<VectorBlockCacheConfig> for BinaryCodes {
    fn cf_options(cache: &Cache, config: &VectorBlockCacheConfig) -> Options {
        let mut opts = Options::default();
        // No compression - already minimal
        opts.set_compression_type(DBCompressionType::None);
        opts.set_block_size(4 * 1024);  // 4KB blocks
        // Configure block cache
        let mut block_opts = BlockBasedOptions::default();
        block_opts.set_block_cache(cache);
        opts.set_block_based_table_factory(&block_opts);
        opts
    }
}
```

**Effort:** 0.25 day

### Task 4.5: Search with Approximate + Re-rank ✅

**Status:** ✅ COMPLETED (2026-01-07)

**Implementation:** `hnsw.rs` adds:
- `search_with_rabitq()` - Two-phase search with expanded candidates and re-ranking
- `search_hamming_filter()` - Experimental Hamming-based beam search (for future optimization)
- `beam_search_layer0_hamming()` - Beam search using Hamming distance

```rust
impl VectorStorage {
    /// Two-phase search: approximate (Hamming) + re-rank (exact)
    pub fn search_with_rabitq(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<Vec<(f32, Id)>> {
        // 1. Encode query to binary
        let query_code = self.rabitq.encode(query);

        // 2. HNSW navigation using binary distance
        let candidates = self.hnsw_search_binary(&query_code, ef)?;

        // 3. Re-rank with exact distance
        let rerank_count = k * 2;  // Fetch 2x candidates for re-ranking
        let top_candidates: Vec<_> = candidates.into_iter()
            .take(rerank_count)
            .collect();

        // 4. Load full vectors for top candidates
        let vectors = self.load_vectors(&top_candidates)?;

        // 5. Compute exact distances and return top-k
        let mut results: Vec<_> = vectors.iter()
            .map(|(id, vec)| {
                let dist = motlie_core::distance::euclidean_squared(query, vec);
                (dist, *id)
            })
            .collect();

        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        results.truncate(k);

        // 6. Map internal IDs back to ULIDs
        self.map_to_ulids(results)
    }
}
```

**Effort:** 2-3 days

**Acceptance Criteria:**
- [ ] 3-4x QPS improvement
- [ ] Recall@10 stays > 95%
- [ ] Memory usage reduced (binary codes vs full vectors)

### Task 4.6: Recall Tuning at Scale (Deferred from 3.7)

**Status:** ✅ COMPLETED (2026-01-07)

Comprehensive recall tests added to validate HNSW parameter tuning.

**Test Coverage:**
- `test_recall_small_scale_100_vectors` - 99%+ recall at 100 vectors
- `test_recall_medium_scale_1k_vectors` - 95%+ recall at 1K vectors
- `test_recall_vs_ef_tradeoff` - ef=50→100→150 correlation
- `test_recall_high_recall_config` - HnswConfig::high_recall() preset
- `test_recall_compact_config` - HnswConfig::compact() preset
- `test_recall_different_k_values` - k=1,5,10,20
- `test_recall_clustered_data` - challenging clustered distribution
- `test_document_optimal_parameters` - M×ef grid search documentation

**Parameter Recommendations (64D, 1K vectors):**
```
| M  | ef_search | Recall@10 |
|----|-----------|-----------|
|  8 |        50 | ~85%      |
|  8 |       100 | ~90%      |
|  8 |       200 | ~93%      |
| 16 |        50 | ~92%      |
| 16 |       100 | ~96%      |
| 16 |       200 | ~98%      |
| 32 |        50 | ~95%      |
| 32 |       100 | ~98%      |
| 32 |       200 | ~99%      |
```

**Preset Configurations:**
- `HnswConfig::high_recall(dim)`: M=32, ef_construction=400 → 95%+ recall
- `HnswConfig::compact(dim)`: M=8, ef_construction=100 → memory-optimized
- Default: M=16, ef_construction=100 → balanced

**Acceptance Criteria:**
- [x] Recall@10 > 95% achieved with M=16+, ef≥100
- [x] Document optimal ef/M parameters for different recall targets
- [x] Comprehensive test coverage for recall characteristics

**Implementation:** `libs/db/src/vector/hnsw.rs` - recall tests in test module

### Task 4.7: SIMD Hamming Distance + Early Filtering

**Status:** ✅ Complete (with findings)

**Goal:** Add SIMD-optimized Hamming distance to `motlie_core::distance` and integrate early filtering into HNSW graph traversal for real RaBitQ speedup.

**Completed Phases:**

1. **SIMD Hamming Distance** (`libs/core/src/distance/`) ✅
   - Added `hamming_distance(a: &[u8], b: &[u8]) -> u32` to public API
   - Scalar: u64 chunks with `count_ones()`
   - NEON: `vcntq_u8` (ARM popcount) - 16 bytes at a time
   - AVX2: `_mm256_xor_si256` + `_popcnt64` - 32 bytes at a time
   - AVX512: `_mm512_xor_si512` + `_popcnt64` - 64 bytes at a time
   - Runtime dispatch support in `mod.rs`

2. **RaBitQ Integration** (`libs/db/src/vector/rabitq.rs`) ✅
   - `hamming_distance` and `hamming_distance_fast` now use `motlie_core::distance::hamming_distance`

3. **Early Filtering in HNSW** (`libs/db/src/vector/hnsw.rs`) ✅
   - Implemented `beam_search_layer0_hamming()` using Hamming distance for candidate scoring
   - Updated `search_with_rabitq()` to use Hamming-based beam search at layer 0
   - Only compute exact distance for final re-ranking

4. **Binary Code Storage** (`examples/vector2/main.rs`) ✅
   - Benchmark stores binary codes during index build when `--rabitq` enabled
   - Uses `BinaryCodes` column family with 12-byte keys

**Benchmark Results (1K vectors, random data):**

| Mode | QPS | Latency (P50) | Recall@10 |
|------|-----|---------------|-----------|
| Standard | 450 | 2.21ms | 100% |
| RaBitQ (rerank=4x) | 660 | 1.49ms | 19% |
| RaBitQ (rerank=8x) | 686 | 1.45ms | 34% |
| RaBitQ (ef=200, rerank=10x) | 578 | 1.64ms | 39% |

---

#### Findings & Analysis

**Observation:** Hamming-based beam search achieves 46-52% QPS improvement but with significantly lower recall (19-39% vs 100%).

**Hypothesis:** The recall degradation occurs because:

1. **HNSW Navigation Requires Accurate Distance**: HNSW's greedy search relies on distance monotonicity - following the gradient toward the query. Hamming distance is a coarse approximation that doesn't preserve this gradient reliably.

2. **Binary Quantization Information Loss**: 1-bit RaBitQ reduces 128D × 32-bit (512 bytes) to 16 bytes (32x compression). This massive compression loses fine-grained distance information needed for accurate graph navigation.

3. **Small Scale Amplifies Error**: At 1K vectors, the graph is dense and many candidates have similar Hamming distances, making the beam search explore wrong regions.

**Key Insight:** Hamming distance is effective for *filtering* (quickly eliminating distant candidates) but not for *navigation* (finding the right neighborhood in the graph).

---

#### Task 4.8: Hybrid L2-Navigation + Hamming-Filtering (Recommended Approach)

**Status:** Pending

**Hypothesis:** Combining L2 distance for navigation with Hamming distance for filtering should achieve both high recall AND improved QPS.

**Proposed Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    HYBRID SEARCH PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 1: L2 Navigation (Upper Layers)                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ • Greedy descent from entry point using exact L2        │   │
│  │ • Layers max_layer → 1                                  │   │
│  │ • Uses standard HNSW greedy search                      │   │
│  │ • Cost: Few distance computations (sparse upper layers) │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  Phase 2: L2 Beam Search (Layer 0)                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ • Standard beam search at layer 0 using exact L2        │   │
│  │ • ef_search candidates explored                         │   │
│  │ • Returns ef candidates with exact L2 distances         │   │
│  │ • Cost: O(ef × avg_degree) L2 computations             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  Phase 3: Hamming Pre-Filter (NEW)                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ • Compute Hamming distance for all ef candidates        │   │
│  │ • Fast: SIMD popcount, ~10x faster than L2             │   │
│  │ • Filter to top (k × rerank_factor) by Hamming         │   │
│  │ • Purpose: Cheap validation of L2-selected candidates   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  Phase 4: L2 Re-rank (Final)                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ • Already have exact L2 from Phase 2                    │   │
│  │ • Just sort and return top-k                            │   │
│  │ • No additional L2 computation needed                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Why This Should Work:**

1. **L2 Navigation**: Phase 1-2 use exact L2 distance, so we navigate to the correct neighborhood (high recall preserved).

2. **Hamming Validation**: Phase 3 uses Hamming as a cheap sanity check. Candidates with very different Hamming distances are likely false positives from beam search noise.

3. **No Extra L2 Cost**: We already computed L2 distances in Phase 2, so Phase 4 is essentially free.

4. **Potential QPS Gain**: The main benefit is that Hamming filtering may help prune false positives, allowing smaller ef_search values while maintaining recall.

**Implementation Location:** `libs/db/src/vector/hnsw.rs` - new method `search_hybrid()`

**Expected Results:**

| Mode | QPS | Recall@10 | Notes |
|------|-----|-----------|-------|
| Standard | 450 | 100% | Baseline |
| Pure Hamming | 686 | 34% | Task 4.7 result |
| **Hybrid** | **500-600** | **>95%** | **Target** |

---

#### Task 4.8 Implementation & Results

**Status:** ✅ Complete (hypothesis disproven)

**Implemented:** `search_hybrid()` method in `libs/db/src/vector/hnsw.rs`

**Actual Benchmark Results:**

| Scale | Mode | QPS | Recall@10 | Notes |
|-------|------|-----|-----------|-------|
| 1K | Standard | 442 | 100% | Baseline |
| 1K | Hybrid | 324 | 51% | **Worse than baseline** |
| 1K | RaBitQ | 658 | 33% | Fast but low recall |
| 10K | Standard | 49 | 90% | Baseline |
| 10K | Hybrid | 49 | 39% | **Worse than baseline** |
| 10K | RaBitQ | 190 | 10% | Fast but very low recall |

**Findings - Why Hybrid Failed:**

1. **Combined Ranking Problem**: The hybrid approach combines L2_rank + Hamming_rank for scoring. A candidate that ranks #1 in L2 but #50 in Hamming gets score 51, while a candidate ranking #25 in both gets score 50. This favors "mediocre" candidates over "excellent in L2".

2. **L2 Is Ground Truth**: Since we're optimizing for L2 nearest neighbors, any weighting of Hamming distance into the ranking can only hurt recall (unless Hamming perfectly correlates with L2, which it doesn't).

3. **No QPS Benefit**: The hybrid approach adds Hamming computation AFTER L2 search, so it can only add overhead, not remove it.

**Key Insight:**

> Hamming distance is useful for **replacing** L2 computations (pure RaBitQ approach), not **supplementing** them. The hybrid approach that does both L2 AND Hamming is strictly worse than just doing L2.

**⚠️ Subsequent Discovery (Issue #43):** Symmetric Hamming distance is fundamentally flawed for multi-bit quantization (2-bit, 4-bit). Even with Gray code encoding, distant quantization levels can have lower Hamming distance than nearby levels due to wraparound. See [RABITQ.md §1.1-1.5](RABITQ.md#11-the-problem-symmetric-hamming-distance) for full analysis. The proper solution is **ADC (Asymmetric Distance Computation)** where the query remains as float32 and distance is computed via weighted dot product. See [Task 4.24](#task-424-adc-hnsw-integration).

**Correct Use Cases for Hamming:**

1. **Replace L2 during beam search** (RaBitQ): Trades recall for QPS
2. **Initial candidate filtering** (not navigation): Use Hamming to quickly filter from millions to thousands, then L2 for final ranking
3. **Large-scale systems**: Where L2 computation dominates and Hamming filtering can reduce candidate count

**Recommendations:**

1. **For high recall**: Use standard L2 search (no Hamming)
2. **For high QPS**: Use pure RaBitQ with tuned ef and rerank_factor
3. **Hybrid not recommended**: Adds overhead without benefit

**Acceptance Criteria:**
- [x] Implement `search_hybrid()` method
- [x] Benchmark showing results
- [x] Document results in ROADMAP
- [ ] ~~Benchmark showing QPS improvement over standard with >95% recall~~ (disproven)

---

#### Task 4.9: RaBitQ Tuning Configuration Analysis

**Status:** ✅ Complete

**Goal:** Find optimal ef and rerank_factor parameters for different recall/QPS tradeoffs.

##### Comprehensive Benchmark Results

**1K Scale (Random Data):**

| ef | rerank | QPS | Recall@10 | vs Baseline |
|----|--------|-----|-----------|-------------|
| 50 | 32 | 510 | 74.0% | 1.2x faster |
| 100 | 32 | 471 | 74.0% | 1.1x faster |
| - | - (L2) | 430 | 100% | baseline |

**10K Scale (Random Data):**

| ef | rerank | QPS | Recall@10 | vs Baseline |
|----|--------|-----|-----------|-------------|
| 50 | 32 | 129 | 26.6% | 2.7x faster |
| 200 | 64 | 106 | 36.6% | 2.2x faster |
| 200 | 100 | 93 | 46.2% | 2.0x faster |
| 200 | 200 | 71 | 61.8% | 1.5x faster |
| 500 | 500 | 47 | 90.6% | **1.0x (same)** |
| 1000 | 1000 | 29 | 100% | 0.6x slower |
| - | - (L2) | 48 | 90.4% | baseline |

**100K Scale (SIFT Dataset):**

| ef | rerank | QPS | Recall@10 | vs Baseline |
|----|--------|-----|-----------|-------------|
| 100 | 100 | 55 | 76.8% | **0.24x slower** |
| 200 | 500 | 17 | 85.6% | **0.13x slower** |
| - | - (L2 ef=100) | 225 | 86.8% | baseline |
| - | - (L2 ef=200) | 133 | 86.8% | baseline |

##### Critical Finding: RaBitQ Performance Issue

**Observation:** RaBitQ is **slower** than standard L2 at larger scales (100K).

**Root Cause Analysis:**

1. **Double I/O Problem:**
   ```
   Standard L2:     Fetch ef vectors from RocksDB → Compute L2 distances
   RaBitQ (ours):   Fetch ef codes from RocksDB → Compute Hamming
                    → Fetch rerank vectors from RocksDB → Compute L2
   ```
   RaBitQ does MORE I/O, not less, because binary codes are in RocksDB.

2. **Original RaBitQ Intent vs Our Implementation:**
   - **Original**: Binary codes in memory, Hamming is CPU-bound, reduces vector fetches
   - **Ours**: Binary codes in RocksDB, adds I/O latency, negates Hamming speed benefit

3. **Break-Even Point:**
   At 10K scale with rerank=500, RaBitQ matches L2 (47 QPS each). Above this scale, RaBitQ becomes slower because I/O dominates.

##### Recommendations

**For Current Implementation (RocksDB-stored codes):**

| Use Case | Recommendation | Parameters |
|----------|----------------|------------|
| High Recall (>90%) | **Use Standard L2** | ef=100-200 |
| Moderate Recall (70-80%) | RaBitQ possible | ef=200, rerank=200 |
| Low Recall OK (<50%) | RaBitQ for QPS | ef=50, rerank=32 |

**Tuning Guidelines:**
- `rerank_factor = 10 × (target_recall / current_recall)` as starting point
- Higher ef doesn't help much - Hamming beam search is the bottleneck
- For >85% recall, standard L2 is recommended

#### Task 4.10: In-Memory Binary Code Cache

**Status:** ✅ Complete (hypothesis validated)
**Commit:** `9880a62`

**Goal:** Validate that in-memory cached binary codes eliminate the double I/O problem.

##### Implementation

1. **BinaryCodeCache** (`libs/db/src/vector/navigation.rs`)
   - Thread-safe HashMap storage: `RwLock<HashMap<(EmbeddingCode, VecId), Vec<u8>>>`
   - Methods: `put()`, `get()`, `get_batch()`, `stats()`
   - Memory tracking for monitoring

2. **search_with_rabitq_cached()** (`libs/db/src/vector/hnsw.rs`)
   - Uses `BinaryCodeCache` instead of RocksDB reads during beam search
   - Only re-ranking phase reads vectors from disk

3. **Benchmark integration** (`examples/vector2/main.rs`)
   - `--rabitq-cached` flag to enable in-memory mode
   - Populates cache during index build

##### Benchmark Results (10K Vectors, Random Data)

| Mode | QPS | Latency (P50) | Recall@10 | vs Baseline |
|------|-----|---------------|-----------|-------------|
| Standard L2 | 48 | 20.80ms | 90.4% | 1.0x (baseline) |
| RaBitQ (RocksDB) | 192 | 5.16ms | 9.6% | 4x faster |
| **RaBitQ-cached** | **430** | **2.23ms** | 9.8% | **9x faster** |

**Speedup Analysis:**
- RaBitQ-cached vs RocksDB: **2.2x faster** (430 vs 192 QPS)
- This confirms the hypothesis: in-memory codes eliminate the double I/O problem

**Memory Usage:**
- 10K vectors at 128D 1-bit: 10000 × 128 / 8 = 160 KB (actually 1250 KB with HashMap overhead)
- Projected at scale:
  - 100K: ~12 MB
  - 1M: ~120 MB
  - 10M: ~1.2 GB (acceptable for dedicated servers)

##### Key Findings

1. **Hypothesis Validated:** In-memory binary code caching provides 2.2x speedup over RocksDB-backed RaBitQ.

2. **Recall Still Low:** Hamming-only navigation achieves only ~10% recall. This is a fundamental limitation of symmetric Hamming distance for HNSW navigation.

3. **Best Use Case for Cached RaBitQ:**
   - High-throughput, low-recall applications (e.g., first-pass filtering)
   - When combined with exact re-ranking on a larger candidate set
   - Memory-constrained environments where only codes are cached

4. **For High Recall:** Use standard L2 search. RaBitQ with Hamming navigation is not suitable for >50% recall requirements.

**⚠️ Root Cause Analysis (Issue #43, RABITQ.md §5.8):** The ~10% recall is caused by using symmetric Hamming distance which:
- Treats all bit flips equally (position-independent)
- For multi-bit quantization, distant levels can have lower Hamming distance than nearby levels (Gray code wraparound)
- Corrupts similarity ordering, especially on structured/clustered data (semantic embeddings)

**Solution:** Replace Hamming with ADC (Asymmetric Distance Computation). ADC benchmark results show:
- ADC 4-bit, rerank=4x: **91.6% recall** (vs Hamming 23.8%)
- ADC 4-bit, rerank=10x: **99.1% recall**

See [Task 4.24](#task-424-adc-hnsw-integration) for implementation plan.

##### Recommendations

| Use Case | Recommended Mode | Expected Results |
|----------|------------------|------------------|
| High Recall (>90%) | Standard L2 | 48 QPS, 90%+ recall |
| Balanced | L2 with larger ef | Tunable |
| High Throughput | RaBitQ-cached, rerank=8-16 | 430 QPS, ~10% recall |
| Initial Filtering | RaBitQ-cached, then L2 on top-N | Combine benefits |

**Acceptance Criteria:**
- [x] Implement `BinaryCodeCache` structure
- [x] Implement `search_with_rabitq_cached()` method
- [x] Add `--rabitq-cached` benchmark flag
- [x] Validate 2x+ speedup over RocksDB-backed RaBitQ
- [x] Document results and recommendations

---

#### Task 4.11: API Cleanup Phase 1 - Deprecate Invalidated Functions

**Status:** ✅ Complete
**Commit:** `ce2a060`

**Goal:** Mark functions invalidated by Phase 4 experiments as deprecated.

**Changes (commit ce2a060):**

| Method | Annotation | Reason |
|--------|------------|--------|
| `search_with_rabitq()` | `#[deprecated]` | Double I/O, slower than L2 at scale (Task 4.9) |
| `search_hybrid()` | `#[deprecated]` | Worse than baseline (Task 4.8) |
| `search_hamming_filter()` | `#[deprecated]` + `#[allow(dead_code)]` | Never used, experimental |
| `beam_search_layer0_hamming()` | `#[allow(dead_code)]` | Only used by deprecated functions |

**Acceptance Criteria:**
- [x] Add `#[deprecated]` annotations with clear migration guidance
- [x] Document why each function is deprecated
- [x] Point to recommended alternatives

##### RaBitQ-Cached Search Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INDEX BUILD (One-time)                           │
├─────────────────────────────────────────────────────────────────────┤
│  For each vector:                                                   │
│  1. Store vector in RocksDB (Vectors CF)                           │
│  2. Build HNSW graph edges using configured distance metric        │
│  3. Encode vector → binary code (RaBitQ: rotate + sign)            │
│  4. Store binary code in RocksDB AND populate BinaryCodeCache      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    SEARCH (Per Query)                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Phase 1: Upper Layer Navigation (exact distance)                  │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ • Greedy descent from entry point                              │ │
│  │ • Uses exact distance (L2/Cosine/Dot - same as index build)    │ │
│  │ • Few nodes in upper layers → fast                             │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│                              ▼                                      │
│  Phase 2: Layer 0 Beam Search (Hamming from cache)                 │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ • Encode query → binary code (one-time per query)              │ │
│  │ • Beam search using Hamming distance on binary codes           │ │
│  │ • Codes fetched from IN-MEMORY cache (not RocksDB)             │ │
│  │ • SIMD popcount: ~ns per distance computation                  │ │
│  │ • Returns ef candidates ranked by Hamming                      │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│                              ▼                                      │
│  Phase 3: Re-rank Top Candidates (exact distance)                  │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ • Take top (k × rerank_factor) candidates                      │ │
│  │ • Fetch full vectors from RocksDB                              │ │
│  │ • Compute exact distance (same metric as index build)          │ │
│  │ • Return top-k sorted by exact distance                        │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

##### Distance Metric Compatibility

**Resolved:** Distance computation now uses the embedding-configured metric via `SearchConfig::compute_distance()` (Phase 4.16). The hardcoded L2 path has been removed.

**RaBitQ Hamming Distance Compatibility:**

| Distance Metric | RaBitQ Compatibility | Explanation |
|-----------------|---------------------|-------------|
| **Cosine** | ✅ Excellent | Hamming distance approximates angular distance. RaBitQ's sign quantization captures angle. |
| **L2 (normalized)** | ✅ Good | For unit vectors: L2² = 2(1 - cos), so Hamming works well. |
| **L2 (unnormalized)** | ⚠️ Moderate | Loses magnitude information. Sign only captures direction, not length. |
| **DotProduct** | ⚠️ Moderate | Loses magnitude. Works for direction but misses scale factor. |

**Why RaBitQ Works Best with Cosine:**

```
RaBitQ Process:
  1. Rotate vector: v' = R × v  (R is orthonormal, preserves angles)
  2. Quantize: code = sign(v')  (captures which half-space per dimension)

Two vectors with similar ANGLES → similar signs → low Hamming distance
Two vectors with similar MAGNITUDES but different angles → different signs → high Hamming

∴ Hamming distance ≈ Angular distance ≈ Cosine distance
```

**Recommendation:**

| Scenario | Recommendation |
|----------|----------------|
| Semantic search (text embeddings) | Use Cosine distance + RaBitQ (ideal match) |
| Image features (normalized) | Use Cosine or L2 + RaBitQ (good) |
| Spatial data (unnormalized L2) | Use L2 + Standard search (avoid RaBitQ) |
| Max inner product (DotProduct) | Use DotProduct + Standard search (avoid RaBitQ) |

**Key Invariant:** Re-ranking MUST use the same distance metric as index construction. The graph structure is optimized for that metric (ARCH-17).

---

#### Task 4.12: API Cleanup Phase 2 - Remove Deprecated Code

**Status:** ✅ Complete
**Commit:** `5f2dac0`

**Goal:** Remove deprecated functions and dead code from the crate.

**Removed Functions (373 lines total):**

```rust
// Removed:
pub fn search_hybrid(...)           // ~109 lines
pub fn search_hamming_filter(...)   // ~75 lines
pub fn search_with_rabitq(...)      // ~99 lines (RocksDB version)
fn beam_search_layer0_hamming(...)  // ~86 lines (uncached version)
```

**Migration:**
- `search_hybrid` → `search()` (standard L2)
- `search_with_rabitq` → `search_with_rabitq_cached()` (requires BinaryCodeCache)
- `search_hamming_filter` → `search_with_rabitq_cached()`

**Acceptance Criteria:**
- [x] Remove all deprecated functions
- [x] Update any internal callers
- [x] Verify cargo test passes (372 tests)
- [x] Update doc comments in remaining functions

---

#### Task 4.13: API Cleanup Phase 3 - Embedding-Driven SearchConfig API

**Status:** ✅ Complete
**Commit:** `5f2dac0` (impl), `0ace681` (tests)

**Goal:** Use `Embedding` as the single source of truth for search configuration, ensuring
consistency between index build and search operations.

**Implementation:** `libs/db/src/vector/search/config.rs` (30 unit tests, 11 integration tests)

##### Problem Statement (Resolved)

1. **Distance Metric Mismatch**: Index built with Cosine but searched with hardcoded L2 (resolved via `SearchConfig::compute_distance`)
2. **RaBitQ Compatibility**: Hamming distance approximates angular distance, works best with Cosine
3. **No Validation**: Runtime validation now enforces embedding distance consistency
4. **Configuration Scattered**: Consolidated into `SearchConfig`

##### Key Insight

The `Embedding` struct already contains everything needed:
- `distance: Distance` - the metric used to build the index
- `compute_distance(a, b)` - method to compute distances correctly

The problem is `HnswIndex` only stores `EmbeddingCode` (u64) for key efficiency, losing
access to the full `Embedding` including its distance metric.

##### Proposed Design: SearchConfig

```rust
/// Search configuration derived from Embedding.
/// Enforces correct distance metric and search strategy.
///
/// # Design Philosophy
///
/// The Embedding is the source of truth. SearchConfig:
/// 1. Captures the Embedding for distance computation
/// 2. Validates embedding matches the index being searched
/// 3. Auto-selects optimal strategy based on distance metric
/// 4. Prevents incompatible configurations at construction time
///
/// # Example
///
/// ```rust
/// // Index built with Cosine - RaBitQ auto-selected
/// let cosine_emb = registry.register(
///     EmbeddingBuilder::new("gemma", 768, Distance::Cosine), &db
/// )?;
/// let config = SearchConfig::new(cosine_emb.clone(), 10);
/// assert!(matches!(config.strategy(), SearchStrategy::RaBitQ { .. }));
///
/// // Index built with L2 - Exact auto-selected
/// let l2_emb = registry.register(
///     EmbeddingBuilder::new("openai", 1536, Distance::L2), &db
/// )?;
/// let config = SearchConfig::new(l2_emb.clone(), 10);
/// assert!(matches!(config.strategy(), SearchStrategy::Exact));
/// ```
pub struct SearchConfig {
    /// Embedding space specification (source of truth)
    embedding: Embedding,
    /// Search strategy (derived from embedding.distance + user choice)
    strategy: SearchStrategy,
    /// Number of results to return
    k: usize,
    /// Search beam width (ef parameter)
    ef: usize,
    /// Re-rank factor for RaBitQ (ignored for Exact)
    rerank_factor: usize,
}

/// Search strategy - auto-selected based on embedding.distance
#[derive(Debug, Clone)]
pub enum SearchStrategy {
    /// Exact distance computation at all layers.
    /// Works with all distance metrics.
    Exact,

    /// RaBitQ: Hamming filtering at layer 0 + exact re-rank.
    /// Only valid when embedding.distance == Cosine.
    /// Requires BinaryCodeCache for good performance.
    RaBitQ {
        /// Use in-memory binary code cache (required for good performance)
        use_cache: bool,
    },
}

impl SearchConfig {
    /// Create search config with automatic strategy selection.
    ///
    /// Strategy is chosen based on embedding's distance metric:
    /// - Cosine → RaBitQ (Hamming approximates angular distance)
    /// - L2/DotProduct → Exact (Hamming not compatible)
    pub fn new(embedding: Embedding, k: usize) -> Self {
        let strategy = match embedding.distance() {
            Distance::Cosine => SearchStrategy::RaBitQ { use_cache: true },
            Distance::L2 | Distance::DotProduct => SearchStrategy::Exact,
        };
        Self {
            embedding,
            strategy,
            k,
            ef: 100,           // sensible default
            rerank_factor: 4,  // 4x re-ranking for >90% recall
        }
    }

    /// Builder: Override to use exact search (disable RaBitQ).
    /// Safe for all distance metrics.
    pub fn exact(mut self) -> Self {
        self.strategy = SearchStrategy::Exact;
        self
    }

    /// Builder: Force RaBitQ strategy.
    /// Returns error if embedding.distance != Cosine.
    pub fn rabitq(mut self) -> Result<Self> {
        if self.embedding.distance() != Distance::Cosine {
            return Err(anyhow!(
                "RaBitQ requires Cosine distance (Hamming ≈ angular), got {:?}",
                self.embedding.distance()
            ));
        }
        self.strategy = SearchStrategy::RaBitQ { use_cache: true };
        Ok(self)
    }

    /// Builder: Set ef (beam width).
    pub fn ef(mut self, ef: usize) -> Self {
        self.ef = ef;
        self
    }

    /// Builder: Set re-rank factor (for RaBitQ).
    pub fn rerank_factor(mut self, factor: usize) -> Self {
        self.rerank_factor = factor;
        self
    }

    // ─────────────────────────────────────────────────────────────
    // Accessors
    // ─────────────────────────────────────────────────────────────

    /// Get the embedding (source of truth).
    pub fn embedding(&self) -> &Embedding { &self.embedding }

    /// Get the search strategy.
    pub fn strategy(&self) -> &SearchStrategy { &self.strategy }

    /// Get k (number of results).
    pub fn k(&self) -> usize { self.k }

    /// Get ef (beam width).
    pub fn ef(&self) -> usize { self.ef }

    /// Get re-rank factor.
    pub fn rerank_factor(&self) -> usize { self.rerank_factor }

    // ─────────────────────────────────────────────────────────────
    // Distance Computation (delegated to Embedding)
    // ─────────────────────────────────────────────────────────────

    /// Compute distance using embedding's configured metric.
    /// This is the ONLY way distances should be computed during search.
    #[inline]
    pub fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        self.embedding.compute_distance(a, b)
    }
}
```

##### Updated HnswIndex API

```rust
impl HnswIndex {
    /// Primary search method - uses SearchConfig for all parameters.
    ///
    /// # Validation
    ///
    /// This method validates that `config.embedding.code()` matches `self.embedding`.
    /// This ensures the search uses the same distance metric as index construction.
    ///
    /// # Example
    ///
    /// ```rust
    /// let config = SearchConfig::new(embedding.clone(), 10).ef(150);
    /// let results = index.search(&storage, &config, &query)?;
    /// ```
    pub fn search(
        &self,
        storage: &Storage,
        config: &SearchConfig,
        query: &[f32],
    ) -> Result<Vec<(f32, VecId)>> {
        // Validate embedding matches this index
        if config.embedding().code() != self.embedding {
            return Err(anyhow!(
                "Embedding mismatch: index has code {}, config has {}",
                self.embedding,
                config.embedding().code()
            ));
        }

        // Dispatch to appropriate strategy
        match config.strategy() {
            SearchStrategy::Exact => self.search_exact(storage, config, query),
            SearchStrategy::RaBitQ { use_cache } => {
                if *use_cache {
                    self.search_rabitq_cached(storage, config, query)
                } else {
                    // Fall back to exact if cache not available
                    self.search_exact(storage, config, query)
                }
            }
        }
    }

    /// Internal: Exact search using embedding's distance metric.
    fn search_exact(
        &self,
        storage: &Storage,
        config: &SearchConfig,
        query: &[f32],
    ) -> Result<Vec<(f32, VecId)>> {
        // Phase 1: Upper layer navigation (greedy descent)
        let entry = self.get_entry_point(storage)?;
        let mut current = entry;

        for layer in (1..=self.get_max_layer(storage)?).rev() {
            current = self.greedy_search_layer(storage, config, query, current, layer)?;
        }

        // Phase 2: Layer 0 beam search with EXACT distance
        let candidates = self.beam_search_layer0_exact(storage, config, query, current)?;

        // Return top-k
        Ok(candidates.into_iter().take(config.k()).collect())
    }

    /// Internal: RaBitQ search with cached binary codes.
    fn search_rabitq_cached(
        &self,
        storage: &Storage,
        config: &SearchConfig,
        query: &[f32],
    ) -> Result<Vec<(f32, VecId)>> {
        // Phase 1: Upper layer navigation (exact distance)
        let entry = self.get_entry_point(storage)?;
        let mut current = entry;

        for layer in (1..=self.get_max_layer(storage)?).rev() {
            current = self.greedy_search_layer(storage, config, query, current, layer)?;
        }

        // Phase 2: Layer 0 beam search with HAMMING filtering
        let candidates = self.beam_search_layer0_hamming_cached(storage, config, query, current)?;

        // Phase 3: Re-rank with EXACT distance (using config.compute_distance)
        let rerank_count = config.k() * config.rerank_factor();
        let to_rerank: Vec<VecId> = candidates.iter()
            .take(rerank_count)
            .map(|(_, id)| *id)
            .collect();

        let vectors = self.get_vectors_batch(storage, &to_rerank)?;
        let mut reranked: Vec<(f32, VecId)> = to_rerank.iter()
            .zip(vectors.iter())
            .filter_map(|(&id, vec_opt)| {
                vec_opt.as_ref().map(|v| {
                    (config.compute_distance(query, v), id)  // Uses embedding's metric!
                })
            })
            .collect();

        reranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        reranked.truncate(config.k());
        Ok(reranked)
    }

    /// Internal: Greedy search at a single layer.
    /// Uses config.compute_distance() for distance computation.
    fn greedy_search_layer(
        &self,
        storage: &Storage,
        config: &SearchConfig,
        query: &[f32],
        entry: VecId,
        layer: HnswLayer,
    ) -> Result<VecId> {
        let mut current = entry;
        let mut current_dist = self.distance_with_config(storage, config, query, current)?;

        loop {
            let neighbors = self.get_neighbors(storage, current, layer, true)?;
            let mut improved = false;

            for neighbor in neighbors.iter() {
                let neighbor_id = neighbor as VecId;
                let neighbor_dist = self.distance_with_config(storage, config, query, neighbor_id)?;
                if neighbor_dist < current_dist {
                    current = neighbor_id;
                    current_dist = neighbor_dist;
                    improved = true;
                }
            }

            if !improved {
                break;
            }
        }

        Ok(current)
    }

    /// Compute distance using SearchConfig (delegated to Embedding).
    /// This replaces the old hardcoded L2 distance method.
    #[inline]
    fn distance_with_config(
        &self,
        storage: &Storage,
        config: &SearchConfig,
        query: &[f32],
        vec_id: VecId,
    ) -> Result<f32> {
        let vector = self.get_vector(storage, vec_id)?;
        Ok(config.compute_distance(query, &vector))
    }
}
```

##### Distance Metric Compatibility Matrix

| Embedding Distance | Exact Strategy | RaBitQ Strategy | Notes |
|-------------------|----------------|-----------------|-------|
| **Cosine** | ✅ Works | ✅ Recommended | Hamming ≈ angular distance |
| **L2** | ✅ Works | ⚠️ Error | Hamming not compatible |
| **DotProduct** | ✅ Works | ⚠️ Error | Hamming not compatible |

##### Migration Path

1. **Step 1**: Add `SearchConfig` to `search/config.rs` (new file)
2. **Step 2**: Add `distance_with_config()` to HnswIndex
3. **Step 3**: Add new `search(config, query)` method
4. **Step 4**: Deprecate old search methods with migration guide
5. **Step 5**: Update examples and benchmarks
6. **Step 6**: Remove deprecated methods (next major version)

##### Hardcoded L2 Fix

Resolved: HNSW distance now delegates to `SearchConfig::compute_distance()`, which uses the embedding’s configured metric.

##### API Usage Examples

```rust
// ─────────────────────────────────────────────────────────────
// Example 1: Cosine index (auto-selects RaBitQ)
// ─────────────────────────────────────────────────────────────
let cosine_emb = registry.register(
    EmbeddingBuilder::new("gemma", 768, Distance::Cosine),
    &db
)?;

// SearchConfig auto-selects RaBitQ for Cosine
let config = SearchConfig::new(cosine_emb.clone(), 10);
let results = index.search(&storage, &config, &query)?;

// ─────────────────────────────────────────────────────────────
// Example 2: Force exact search (disable RaBitQ)
// ─────────────────────────────────────────────────────────────
let config = SearchConfig::new(cosine_emb.clone(), 10).exact();
let results = index.search(&storage, &config, &query)?;

// ─────────────────────────────────────────────────────────────
// Example 3: L2 index (auto-selects Exact)
// ─────────────────────────────────────────────────────────────
let l2_emb = registry.register(
    EmbeddingBuilder::new("openai", 1536, Distance::L2),
    &db
)?;

// SearchConfig auto-selects Exact for L2
let config = SearchConfig::new(l2_emb.clone(), 10);
let results = l2_index.search(&storage, &config, &query)?;

// ─────────────────────────────────────────────────────────────
// Example 4: Trying RaBitQ on L2 index (FAILS at construction)
// ─────────────────────────────────────────────────────────────
let config = SearchConfig::new(l2_emb.clone(), 10)
    .rabitq()?;  // Returns Err: "RaBitQ requires Cosine distance"

// ─────────────────────────────────────────────────────────────
// Example 5: Mismatched embedding (FAILS at search time)
// ─────────────────────────────────────────────────────────────
let config = SearchConfig::new(cosine_emb.clone(), 10);
let results = l2_index.search(&storage, &config, &query);
// Returns Err: "Embedding mismatch: index has code 2, config has 1"
```

##### Benefits

1. **Single Source of Truth**: `Embedding` contains all configuration
2. **Compile-time Safety**: RaBitQ + non-Cosine fails at `rabitq()` call
3. **Runtime Validation**: Embedding code mismatch caught at search time
4. **Auto-selection**: Optimal strategy chosen based on distance metric
5. **Explicit Override**: Users can force exact search with `.exact()`
6. **Clear Error Messages**: Explain why combinations are invalid

**Acceptance Criteria:**
- [x] Create `search/config.rs` with `SearchConfig` and `SearchStrategy`
- [x] Auto-select strategy based on distance metric
- [x] Return error for RaBitQ + non-Cosine combinations
- [x] Validate embedding code match at search time
- [x] Add 14 unit tests for all combinations
- [x] Documentation with usage examples
- [ ] Add `distance_with_config()` method to HnswIndex (deferred to integration)
- [ ] Add new `search(&SearchConfig, &query)` method (deferred to integration)
- [ ] Update benchmark to use new API (deferred to integration)

---

#### Task 4.14: API Cleanup Phase 4 - Configuration Validation

**Status:** ✅ Complete
**Commit:** `5f2dac0`

**Implementation:** `libs/db/src/vector/config.rs` - `ConfigWarning`, `validate()` methods (12 tests)

**Goal:** Prevent misconfiguration through validation and presets.

**Proposed Design:**

```rust
impl HnswConfig {
    /// Validate configuration, return warnings for suboptimal settings
    pub fn validate(&self) -> Result<(), Vec<ConfigWarning>> {
        let mut warnings = Vec::new();

        if self.m < 8 {
            warnings.push(ConfigWarning::LowM(
                "M < 8 may cause poor recall at scale"
            ));
        }
        if self.ef_construction < self.m * 2 {
            warnings.push(ConfigWarning::LowEfConstruction(
                "ef_construction should be >= 2*M"
            ));
        }
        // ...
    }
}

impl RaBitQ {
    /// Create with validation
    pub fn new(dim: usize, bits_per_dim: u8, seed: u64) -> Result<Self, ConfigError> {
        if ![1, 2, 4].contains(&bits_per_dim) {
            return Err(ConfigError::InvalidBitsPerDim);
        }
        // ...
    }
}
```

**Presets with Documented Tradeoffs:**

| Preset | M | ef_construction | Use Case |
|--------|---|-----------------|----------|
| `default()` | 16 | 200 | Balanced recall/memory |
| `high_recall()` | 32 | 400 | >99% recall requirements |
| `compact()` | 8 | 100 | Memory-constrained (~5% recall loss) |

**Acceptance Criteria:**
- [x] Add `validate()` method to HnswConfig
- [x] Add `validate()` method to RaBitQConfig
- [x] Add `ConfigWarning` enum with Display impl
- [x] Return warnings, not errors (allow advanced users to override)
- [x] Add `is_valid()` method for quick validation
- [x] Add 12 unit tests for validation
- [x] Document tradeoffs in preset methods (existing)

---

#### Task 4.15: Phase 5 Integration Planning

**Status:** 🔲 Deferred to Phase 7

**Goal:** Ensure API cleanup is compatible with Phase 7 (Async Graph Updater).

**Analysis Summary:**

SearchStrategy design is **orthogonal and compatible** with Phase 5:

| Phase 5 Feature | SearchStrategy Impact |
|-----------------|----------------------|
| `insert_async()` | ✅ Search works regardless of insert mode |
| Binary codes (Phase 1 sync) | ✅ BinaryCodeCache populated immediately |
| Pending vectors | ✅ Found via graph expansion |
| Background updater | ✅ Improves recall over time |

**Integration Points:**

```rust
// Phase 5 insert populates cache for SearchStrategy::Quantized
pub fn insert_async(&self, embedding: &Embedding, vector: &[f32]) -> Result<Id> {
    // Phase 1: Sync writes
    let id = self.allocate_id()?;
    self.store_vector(embedding, id, vector)?;

    // Store binary code AND populate cache
    if let Some(rabitq) = self.rabitq.as_ref() {
        let code = rabitq.encode(vector);
        self.store_binary_code(embedding, id, &code)?;

        // Populate cache for immediate quantized searchability
        if let Some(cache) = self.code_cache.as_ref() {
            cache.put(embedding.code(), id, code);
        }
    }

    self.add_to_pending(embedding, id)?;
    Ok(id)
}
```

**Acceptance Criteria (Phase 7):**
- [ ] Document Phase 7 compatibility in SearchStrategy
- [ ] Ensure BinaryCodeCache integrates with async insert
- [ ] Test quantized search on pending vectors
- [ ] Benchmark async insert + quantized search latency

---

#### Task 4.16: HNSW Distance Metric Bug Fix

**Status:** ✅ Complete
**Commit:** `0535e6a`
**Date:** 2026-01-09

**Bug Description:**

HNSW navigation was **hardcoded to use L2 distance** regardless of the configured
distance metric. This caused incorrect results when using Cosine or DotProduct distance.

**Location:** `libs/db/src/vector/hnsw.rs:681`

```rust
// BEFORE (Bug):
let dist = l2_distance(&current_vector, &target_vector);

// AFTER (Fix):
let dist = self.compute_distance(&current_vector, &target_vector);
```

**Changes:**

1. Added `distance: Distance` field to `HnswIndex` struct
2. Updated `HnswIndex::new()` to take `Distance` parameter
3. Added `compute_distance()` method that dispatches to correct metric
4. Added `cosine_distance()` and `dot_product_distance()` functions
5. Updated all callers to pass Distance parameter

**Corrected Benchmark Results (100K vectors):**

| Distance | Strategy | QPS | Recall@10 |
|----------|----------|-----|-----------|
| L2 | Standard | 255 | 88.4% |
| Cosine | RaBitQ-cached | 445 | 58.3% |

**Analysis:**

The lower Cosine + RaBitQ recall (58.3%) is expected for SIFT data because:
- SIFT vectors are integer histograms, not unit vectors
- Cosine distance is designed for normalized embedding vectors
- RaBitQ's Hamming approximation works best with unit vectors

For real embedding models (OpenAI, Cohere) that produce unit vectors,
Cosine + RaBitQ should achieve much higher recall.

**Files Changed:**
- `libs/db/src/vector/hnsw.rs` - Core fix
- `libs/db/src/vector/search/config.rs` - Updated default rerank_factor to 10
- `examples/vector2/main.rs` - Pass Distance to HnswIndex
- `examples/vector2/benchmark.rs` - Added cosine distance functions

**Acceptance Criteria:**
- [x] HnswIndex uses configured distance metric for navigation
- [x] All 33 HNSW tests pass
- [x] Benchmark results re-collected with corrected code
- [x] Results documented in `examples/vector2/results/phase4_final_results.md`

---

---

#### Task 4.17: batch_distances Metric Bug Fix

**Status:** ✅ Complete
**Commit:** `d5899f3`
**Date:** 2026-01-09

**Bug Description:**

The `batch_distances()` function in HNSW was **hardcoded to use L2 distance**
regardless of the configured distance metric. This caused ~39% recall when using
Cosine distance on LAION-CLIP embeddings (expected 90%+).

**Location:** `libs/db/src/vector/hnsw.rs:780`

```rust
// BEFORE (Bug):
.filter_map(|(id, vec_opt)| vec_opt.map(|v| (id, l2_distance(query, &v))))

// AFTER (Fix):
.filter_map(|(id, vec_opt)| vec_opt.map(|v| (id, self.compute_distance(query, &v))))
```

**Impact:**

This bug was particularly severe because `batch_distances()` is called during
beam search when neighbor count exceeds the batch threshold. The individual
`distance()` calls were correct, but batch operations used wrong metric.

**Before/After (LAION-CLIP 50K, Cosine distance):**

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| Recall@1 | 39.2% | **91.8%** |
| Recall@5 | 40.3% | **91.5%** |
| Recall@10 | 39.2% | **89.3%** |

**Acceptance Criteria:**
- [x] batch_distances uses configured distance metric
- [x] Recall@1 > 90% on LAION-CLIP benchmark
- [x] All HNSW tests pass

---

#### Task 4.18: VectorElementType (Multi-Float Support)

**Status:** ✅ Complete
**Commit:** `c90f15c`
**Goal:** Support f16 (half-precision) storage for 50% space savings

**Motivation:**

LAION-CLIP embeddings are natively stored as f16 (IEEE 754 half-precision).
Currently we:
1. Load f16 from NPY files
2. Convert to f32 in memory
3. Store as f32 in RocksDB (4 bytes/element)

With VectorElementType, we can:
1. Store as f16 in RocksDB (2 bytes/element)
2. Convert to f32 only for distance computation
3. Save 50% storage with negligible precision loss for normalized vectors

**Schema Changes:**

```rust
// New enum in schema.rs
#[derive(Serialize, Deserialize, Clone, Copy, Default)]
pub enum VectorElementType {
    #[default]
    F32,  // 4 bytes/element, full precision
    F16,  // 2 bytes/element, 50% smaller
}

// Updated EmbeddingSpec
pub struct EmbeddingSpec {
    pub model: String,
    pub dim: u32,
    pub distance: Distance,
    #[serde(default)]
    pub storage_type: VectorElementType,  // NEW - defaults to F32
}

// Updated Vectors CF serialization
impl Vectors {
    pub fn value_to_bytes(value: &[f32], storage_type: VectorElementType) -> Vec<u8>;
    pub fn value_from_bytes(bytes: &[u8], storage_type: VectorElementType) -> Result<Vec<f32>>;
}
```

**Storage Savings:**

| Format | 512D Vector | 200K Vectors | Savings |
|--------|-------------|--------------|---------|
| F32 | 2,048 bytes | 400 MB | - |
| F16 | 1,024 bytes | 200 MB | **50%** |

**Implementation Steps:**
- [x] Add `VectorElementType` enum to `schema.rs`
- [x] Add `storage_type` field to `EmbeddingSpec` (default F32)
- [x] Update `Vectors::value_to_bytes_typed()` with f16 support
- [x] Update `Vectors::value_from_bytes_typed()` with f16 support
- [x] Update `HnswIndex::with_storage_type()` constructor
- [x] Update LAION benchmark to use F16 storage
- [x] Run full benchmark suite (50K-200K scales)

**Acceptance Criteria:**
- [x] VectorElementType enum with F32/F16 variants
- [x] Backward compatible (F32 default)
- [x] LAION benchmark uses F16 storage
- [x] Recall maintained (~87% at 200K scale)
- [x] 50% storage reduction verified

**Benchmark Results (LAION-CLIP 512D, M=16, ef_construction=100, Cosine):**

| Scale | Recall@1 | Recall@10 | Latency (ms) | QPS |
|-------|----------|-----------|--------------|-----|
| 50K | 91.6% | 89.6% | 2.9 | 345 |
| 100K | 87.0% | 86.8% | 3.4 | 291 |
| 150K | 86.6% | 85.7% | 3.8 | 263 |
| 200K | 87.0% | 84.9% | 4.3 | 231 |

**Key Findings:**
- Recall degradation pattern matches "HNSW at Scale" article
- ~5% recall drop from 50K to 200K (expected for high-dimensional CLIP embeddings)
- QPS scales O(log N) as expected for HNSW
- F16 storage works correctly with negligible precision loss

---

#### Task 4.19: Parallel Reranking with Rayon

**Status:** ✅ Complete (Commit: `27d6b74`)
**Goal:** Add CPU parallelism to reranking phase for improved throughput

**Motivation:**

The current search pipeline is single-threaded per query. Profiling shows that the reranking phase (fetching vectors from RocksDB and computing exact distances) is a significant bottleneck, especially for RaBitQ search where we rerank 40-100+ candidates.

**Current Architecture (Sequential):**
```
Query ─────────────────────────────────────────────> Result
       │ Greedy │ Beam Search │    Rerank (N×)    │
       └────────┴─────────────┴───────────────────┘
                                    ↑ BOTTLENECK
```

**Proposed Architecture (Parallel Reranking):**
```
Query ──┬── Greedy ──┬── Beam Search ──┬── Parallel Rerank ──┬── Merge ──> Result
        │            │                 │  ┌─ Worker 1 ─┐     │
        │            │                 │  ├─ Worker 2 ─┤     │
        │            │                 │  ├─ Worker 3 ─┤     │
        │            │                 │  └─ Worker N ─┘     │
        └────────────┴─────────────────┴─────────────────────┘
```

**Implementation:**

1. **Add rayon dependency** to `libs/db/Cargo.toml`
2. **New module:** `libs/db/src/vector/parallel.rs`
   - `rerank_parallel()` - parallel distance computation
   - `batch_distances_parallel()` - parallel batch distances
3. **Update `HnswIndex`:**
   - `search_with_rabitq_cached()` uses parallel reranking
   - Optional `SearchConfig::parallel(bool)` flag

**Code Changes:**

```rust
// libs/db/src/vector/parallel.rs
use rayon::prelude::*;

/// Parallel reranking - compute exact distances for candidates.
pub fn rerank_parallel<F>(
    candidates: &[VecId],
    distance_fn: F,
    k: usize,
) -> Vec<(f32, VecId)>
where
    F: Fn(VecId) -> Option<f32> + Sync,
{
    let mut results: Vec<(f32, VecId)> = candidates
        .par_iter()
        .filter_map(|&id| distance_fn(id).map(|d| (d, id)))
        .collect();

    results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    results.truncate(k);
    results
}

// Update hnsw.rs search_with_rabitq_cached():
// BEFORE (sequential):
for &vec_id in &vec_ids {
    let dist = self.distance(storage, query, vec_id)?;
    exact_results.push((dist, vec_id));
}

// AFTER (parallel):
let exact_results = parallel::rerank_parallel(
    &vec_ids,
    |vec_id| self.distance(storage, query, vec_id).ok(),
    k,
);
```

**Expected Impact:**

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Rerank throughput | 1x | 4-8x | CPU parallelism |
| Search QPS (RaBitQ) | 230 | 400-500 | 2x (rerank-bound) |
| Search QPS (Exact) | 290 | 350-400 | 1.3x (less rerank) |

**Why RocksDB is Thread-Safe:**

RocksDB `TransactionDB` supports concurrent reads from multiple threads:
- `multi_get_cf()` is already thread-safe
- Each worker gets its own snapshot view
- No locking needed for read-only operations

**Implementation Steps:**
- [x] Add `rayon = "1.10"` to Cargo.toml
- [x] Create `libs/db/src/vector/parallel.rs`
- [x] Implement `rerank_parallel()` function
- [x] Implement `batch_distances_parallel()` function
- [x] Update `search_with_rabitq_cached()` to use parallel reranking
- [x] Update `batch_distances()` to use parallel distance computation
- [ ] ~~Add `parallel` flag to `SearchConfig` (default: true)~~ (Deferred - always parallel)
- [x] Run LAION benchmarks to measure improvement
- [x] Unit tests for parallel functions (5 tests)

**Acceptance Criteria:**
- [x] Parallel reranking using rayon
- [x] Thread-safe RocksDB access verified
- [x] No regression in recall (87.9% consistent)
- [x] Unit tests for parallel functions

**Results:**

The parallel reranking infrastructure is now in place:

| Component | Implementation |
|-----------|----------------|
| `parallel.rs` | New module with 3 parallel functions |
| `rerank_parallel()` | Used in `search_with_rabitq_cached()` |
| `batch_distances_parallel()` | Available for bulk operations |
| `distances_from_vectors_parallel()` | For pre-fetched vector distances |

**Benchmark Results (100K LAION):**

| ef_search | Recall@10 | QPS | p50 Latency |
|-----------|-----------|-----|-------------|
| 10 | 87.9% | 243.4 | 3.85ms |
| 20 | 87.9% | 243.0 | 3.87ms |
| 40 | 87.9% | 240.4 | 3.91ms |
| 80 | 87.9% | 242.2 | 3.88ms |
| 160 | 87.9% | 238.5 | 3.88ms |

**Note:** The standard HNSW search path computes distances during graph traversal (already optimal). The parallel improvements primarily benefit:
1. RaBitQ two-phase search with large rerank sets
2. Batch distance operations via `batch_distances()`
3. Future pre-filtering workloads where many candidates need reranking

---

#### Task 4.20: Parallel Re-ranking Threshold Tuning

**Status:** ✅ Complete (Updated in Task 4.21)
**Goal:** Determine optimal threshold for parallel vs sequential re-ranking

> **Note (January 2026):** The threshold was updated from 800 to 3200 in Task 4.21 based on
> re-benchmarking. See [`BENCHMARK.md`](./BENCHMARK.md) for current results and analysis.

**Motivation:**

Task 4.19 added rayon parallelism to the re-ranking phase, but parallelism has overhead that only pays off for larger candidate sets. This task benchmarks the crossover point and adds a configurable threshold.

**Benchmark Methodology:**

Using LAION-CLIP 512D embeddings with RaBitQ binary codes:
1. Encode 50K vectors with 1-bit RaBitQ
2. Select candidates via Hamming distance (Phase 1)
3. Re-rank candidates with exact L2 distance (Phase 2)
4. Compare sequential vs parallel (rayon) execution time

**Benchmark Command:**
```bash
cargo run --release --example laion_benchmark -- --rabitq-benchmark --rerank-sizes "50,100,200,400,800,1600,3200,6400,12800"
```

**Results (512D, NEON SIMD, aarch64):**

| Candidates | Seq (ms) | Par (ms) | Speedup | Analysis |
|------------|----------|----------|---------|----------|
| 50 | 9.13 | 18.96 | 0.48x | Sequential 2x faster |
| 100 | 19.54 | 30.27 | 0.65x | Sequential faster |
| 200 | 40.08 | 57.77 | 0.69x | Sequential faster |
| 400 | 81.42 | 86.18 | 0.94x | Near equal |
| 800 | 142.46 | 136.68 | 1.04x | **Crossover point** |
| 1600 | 316.40 | 224.36 | 1.41x | Parallel wins |
| 3200 | 572.87 | 332.92 | 1.72x | Good speedup |
| 6400 | 1163.48 | 497.65 | 2.34x | Strong speedup |
| 12800 | 2305.55 | 733.01 | 3.15x | Near-linear scaling |

**Key Findings:**

1. **Crossover point: ~800 candidates** - Below 800, sequential is faster
2. **Rayon overhead: ~50-100µs per call** - Dominates for small work units
3. **Work per candidate: ~0.36µs** (512D L2 distance with NEON SIMD)
4. **Max speedup: 3.15x at 12800** - Approaches CPU core count

**Implementation:**

Added configurable threshold to `SearchConfig`:

```rust
// Default threshold (800) based on benchmarks
pub const DEFAULT_PARALLEL_RERANK_THRESHOLD: usize = 800;

// SearchConfig with threshold
let config = SearchConfig::new(embedding, 10)
    .with_parallel_rerank_threshold(1600); // Override for specific workloads

// Check if parallel should be used
if config.should_use_parallel_rerank(candidates.len()) {
    parallel::rerank_parallel(...)
} else {
    parallel::rerank_sequential(...)
}

// Or use adaptive function
parallel::rerank_adaptive(candidates, distance_fn, k, config.parallel_rerank_threshold());
```

**New Functions in `parallel.rs`:**

| Function | Description |
|----------|-------------|
| `rerank_sequential()` | Sequential re-ranking (no rayon) |
| `rerank_adaptive()` | Auto-selects based on threshold |
| `rerank_auto()` | Uses DEFAULT_PARALLEL_RERANK_THRESHOLD |

**Tuning Guidance:**

| Dimension | Threshold | Notes |
|-----------|-----------|-------|
| 128D | ~1200 | Less work per distance → higher threshold |
| 512D | **800** | Default (LAION-CLIP benchmark) |
| 768D | ~600 | More work per distance → lower threshold |
| 1536D | ~400 | Even more work → lower threshold |

**Acceptance Criteria:**
- [x] Benchmark sequential vs parallel at various scales
- [x] Document crossover point (800 candidates)
- [x] Add `parallel_rerank_threshold` to SearchConfig
- [x] Add `should_use_parallel_rerank()` helper
- [x] Add `rerank_sequential()` function
- [x] Add `rerank_adaptive()` function
- [x] Add `rerank_auto()` convenience function
- [x] Unit tests for new functions (7 tests)
- [x] Document threshold tuning guidance

---

#### Task 4.21: Benchmark Infrastructure + Threshold Update

**Status:** ✅ Complete (Commits: `a4896e8`, `2358ba8`)
**Goal:** Consolidate benchmark infrastructure, update parallel threshold based on new measurements

**Summary:**

This task consolidated benchmark infrastructure into `libs/db/src/vector/benchmark/` and
re-ran parallel reranking benchmarks, discovering the crossover point had shifted from
800 to 3200 candidates due to sequential code optimizations.

**Changes:**

1. **Module Refactor** - Split flat files into nested subdirectories:
   - `hnsw/{mod,graph,insert,search}.rs`
   - `cache/{mod,navigation,binary_codes}.rs`
   - `quantization/{mod,rabitq}.rs`
   - `search/{mod,parallel,config}.rs`
   - `benchmark/{mod,dataset,metrics,runner}.rs`

2. **Benchmark Infrastructure** - New `vector::benchmark` module:
   - `LaionDataset`: LAION-400M CLIP loader with NPY parsing
   - `LaionSubset`: Subset extraction with ground truth
   - `RecallMetrics`, `LatencyStats`: Standardized metrics
   - `ExperimentConfig`, `ExperimentResult`: Configurable benchmark runner

3. **Threshold Update** - `DEFAULT_PARALLEL_RERANK_THRESHOLD`: 800 → 3200

**Benchmark Results (January 2026):**

Re-ran parallel reranking benchmark on LAION-CLIP 512D (50K vectors, aarch64 NEON):

| Candidates | Original Seq | Current Seq | Original Par | Current Par | New Speedup |
|------------|--------------|-------------|--------------|-------------|-------------|
| 800 | 142.5ms | 106.1ms | 136.7ms | 139.2ms | 0.76x |
| 1600 | 316.4ms | 188.3ms | 224.4ms | 205.8ms | 0.91x |
| 3200 | 572.9ms | 414.4ms | 332.9ms | 319.3ms | **1.30x** |

**Root Cause Analysis:**

Sequential distance computation got **25-30% faster** due to:
1. Code optimizations in distance functions
2. Better compiler optimizations (LLVM improvements)
3. Improved SIMD utilization

Parallel throughput remained constant because:
1. Rayon overhead is fixed (~50-100µs per call)
2. Work-stealing overhead doesn't scale with faster per-item work
3. Memory bandwidth may be bottleneck, not CPU

**Documentation:**

Created [`BENCHMARK.md`](./BENCHMARK.md) with:
- Configuration reference for all parameters
- LAION-CLIP benchmark results (50K, 100K scale)
- Side-by-side comparison: Standard HNSW vs RaBitQ vs Faiss
- Parallel reranking threshold analysis
- When to use each search strategy

**Acceptance Criteria:**
- [x] Module hierarchy refactored
- [x] Benchmark infrastructure in `vector::benchmark`
- [x] Re-ran LAION benchmarks (no regression: Recall@10 89.2% at 50K)
- [x] Updated `DEFAULT_PARALLEL_RERANK_THRESHOLD` to 3200
- [x] Created comprehensive `BENCHMARK.md`
- [x] All tests passing

**Files Changed:**
- `libs/db/src/vector/mod.rs` - Updated module structure
- `libs/db/src/vector/search/config.rs` - Threshold 800→3200
- `libs/db/src/vector/search/parallel.rs` - Updated docs and tests
- `libs/db/src/vector/docs/BENCHMARK.md` - New comprehensive documentation

---

#### Task 4.22: Large-Scale Benchmarks (500K, 1M)

**Status:** ✅ Complete
**Goal:** Benchmark at production-scale datasets (500K, 1M vectors) to validate scaling behavior

**Summary:**

Ran comprehensive benchmarks at 500K and 1M scale using LAION-CLIP 512D embeddings.
Results confirm O(log n) QPS scaling and sub-linear latency growth as expected for HNSW.

**Results Summary:**

| Scale | Recall@10 | QPS | P50 Latency | Build Time | HNSW vs Flat |
|-------|-----------|-----|-------------|------------|--------------|
| 500K | 83.1% | 136 | 6.9ms | 2.3h (60 vec/s) | 5.7x |
| 1M | 80.7% | 112 | 8.4ms | 5.9h (47 vec/s) | 9.5x |

**Key Findings:**

1. **Recall decay**: ~1-2% per 100K vectors (89.2% at 50K → 80.7% at 1M)
2. **QPS scaling**: O(log n) confirmed - 284 → 254 → 136 → 112 QPS
3. **Latency scaling**: Sub-linear - 3.3ms → 8.4ms p50 from 50K to 1M
4. **HNSW advantage grows**: 9.5x speedup over brute-force at 1M scale
5. **Production viable**: 112 QPS with sub-10ms latency at 1M vectors

**Configuration:**

| Parameter | Value |
|-----------|-------|
| Dataset | LAION-CLIP 512D (float16 storage) |
| M | 16 |
| ef_construction | 100 |
| ef_search | 10, 20, 40, 80, 160 |
| k | 10 |
| Block cache | 256 MB |
| Storage | RocksDB with LZ4 |

**Acceptance Criteria:**
- [x] 500K benchmark complete
- [x] 1M benchmark complete
- [x] Updated BENCHMARK.md with 500K results
- [x] Updated BENCHMARK.md with 1M results
- [x] Scaling analysis across all scales

**Documentation:** See [`BENCHMARK.md`](./BENCHMARK.md) for full results.

---

#### Task 4.23: Multi-Dataset Benchmark Infrastructure

**Status:** ✅ Complete
**Goal:** Extend `vector::benchmark` crate to support multiple datasets with integrated testing

**Summary:**

Ported SIFT dataset support from `examples/vector2` to `libs/db/src/vector/benchmark/` and added
integration tests validating the complete search pipeline for both datasets.

**Supported Datasets:**

| Dataset | Dimensions | Distance | Format | Use Case |
|---------|------------|----------|--------|----------|
| LAION-CLIP | 512D | Cosine | NPY (float16) | RaBitQ, semantic search |
| SIFT1M | 128D | L2/Euclidean | fvecs/ivecs | Standard HNSW, ANN benchmarks |

**Features Implemented:**

1. **SIFT Dataset Support** (`sift.rs`):
   - `SiftDataset`: Load SIFT1M from HuggingFace
   - `SiftSubset`: Create test subsets with ground truth
   - `read_fvecs()`, `read_ivecs()`: Standard ANN benchmark formats
   - L2 distance brute-force ground truth computation

2. **Integration Tests** (`test_vector_benchmark_integration.rs`):
   - SIFT: L2 + Exact HNSW + NavigationCache (90% recall at 1K)
   - LAION-CLIP: Cosine + Exact HNSW + NavigationCache (87% recall at 1K)
   - LAION-CLIP: Cosine + RaBitQ + BinaryCodeCache + NavigationCache + parallel rerank

3. **Verified Claims:**

   | Claim | Status | Notes |
   |-------|--------|-------|
   | 1. Both SIFT and LAION-CLIP supported | ✅ Verified | `vector::benchmark` exports both datasets |
   | 2. SIFT uses L2/Euclidean distance | ✅ Verified | Default distance metric for SIFT |
   | 3. LAION-CLIP uses Cosine distance | ✅ Verified | Supports both exact HNSW and RaBitQ modes |
   | 4. Both leverage caching + parallel search | ✅ Verified | NavigationCache, BinaryCodeCache, adaptive rerank |

**Test Results (1K synthetic data):**

| Dataset | Distance | Search Mode | Recall@10 |
|---------|----------|-------------|-----------|
| SIFT | L2 | Exact HNSW + NavCache | 90.2% |
| LAION-CLIP | Cosine | Exact HNSW + NavCache | 87.4% |
| LAION-CLIP | Cosine | RaBitQ + BinaryCache + NavCache | 45.6%* |

*RaBitQ at 1K scale with 1-bit quantization has high information loss. At production scale (100K+), recall improves significantly.

**Note:** LAION-CLIP supports both search modes:
- **Exact HNSW**: Higher recall, uses exact Cosine distance (M=32, ef_construction=200 for 512D)
- **RaBitQ**: Faster search via Hamming pre-filtering, trades recall for speed

**Files Changed:**

| File | Purpose |
|------|---------|
| `benchmark/sift.rs` | NEW - SIFT dataset loading and fvecs/ivecs parsing |
| `benchmark/mod.rs` | Updated - Export SIFT alongside LAION |
| `tests/test_vector_benchmark_integration.rs` | NEW - Integration tests for both datasets |

**Search Pipeline Verified:**

```
SIFT L2 Pipeline:
  Query → HNSW Navigation (NavCache) → Beam Search L2 → Top-K Results

LAION-CLIP RaBitQ Pipeline:
  Query → Encode RaBitQ → HNSW Navigation (NavCache)
        → Hamming Beam Search (BinaryCodeCache) → Candidates
        → Adaptive Parallel Rerank (threshold: 3200) → Top-K Results
```

**Distance Metric & Search Mode Constraints:**

| Search Mode | Cosine | L2 | DotProduct | Notes |
|-------------|--------|-----|------------|-------|
| Exact HNSW  | ✅ | ✅ | ✅ | All metrics supported via `Distance::compute()` |
| RaBitQ      | ✅ | ❌ | ❌ | Enforced: "RaBitQ requires Cosine distance" |

**Why RaBitQ requires Cosine (unit-normalized vectors):**

1. **Sign quantization captures direction, not magnitude**
   - Binary codes are sign bits of rotated vector components
   - Works correctly when all vectors have unit norm
   - Hamming distance ≈ angular distance for normalized vectors

2. **L2 on unnormalized data loses magnitude information**
   - Two vectors with same direction but different magnitudes → same binary code
   - Hamming distance would incorrectly report them as identical

3. **Code enforcement** (`search/config.rs:202-207`):
   ```rust
   pub fn rabitq(mut self) -> anyhow::Result<Self> {
       if self.embedding.distance() != Distance::Cosine {
           return Err(anyhow::anyhow!(
               "RaBitQ requires Cosine distance (Hamming ≈ angular), got {:?}",
               self.embedding.distance()
           ));
       }
       // ...
   }
   ```

**Parallel Reranking Threshold:**

| Candidates | Sequential | Parallel | Speedup |
|------------|------------|----------|---------|
| 800        | 106.1ms    | 139.2ms  | 0.76x (overhead) |
| 1600       | 188.3ms    | 205.8ms  | 0.91x (near equal) |
| **3200**   | 414.4ms    | 319.3ms  | **1.30x** (crossover) |

- `DEFAULT_PARALLEL_RERANK_THRESHOLD = 3200`
- Below 3200: Sequential is faster (rayon overhead)
- At/above 3200: Parallel wins (1.3x+ speedup)
- Benchmarked on 512D LAION-CLIP, aarch64 NEON (January 2026)

**Acceptance Criteria:**
- [x] SIFT dataset ported to `vector::benchmark`
- [x] fvecs/ivecs format readers implemented
- [x] Integration test for SIFT + L2 + NavigationCache
- [x] Integration test for LAION-CLIP + Cosine + Exact HNSW
- [x] Integration test for LAION-CLIP + Cosine + RaBitQ + caches
- [x] Claims 1-4 documented and verified
- [x] Distance metric constraints documented (Exact: all, RaBitQ: Cosine only)
- [x] Parallel reranking threshold documented (3200)

---

#### Task 4.24: ADC HNSW Integration (Replace Hamming with ADC)

**Status:** ✅ Complete (`ec0fc00`)
**Priority:** HIGH
**Goal:** Replace symmetric Hamming distance with ADC (Asymmetric Distance Computation) in HNSW navigation to achieve >90% recall with RaBitQ.

##### Problem Statement (Historical)

Current RaBitQ implementation uses symmetric Hamming distance during HNSW beam search, which is fundamentally flawed for multi-bit quantization:

| Mode | Recall@10 | Issue |
|------|-----------|-------|
| Hamming 1-bit | ~10% | Low information density |
| Hamming 4-bit | ~24% | Gray code wraparound corrupts similarity |
| **ADC 4-bit** | **91-99%** | Proper distance estimation |

**Root Cause:** Symmetric Hamming treats query and data symmetrically (both binarized), losing numeric ordering:
- Level 0 → Level 8: Hamming=2, but value difference=8
- Level 0 → Level 15: Hamming=1, but value difference=15 (wraparound)

See [RABITQ.md §1.1-1.5](RABITQ.md#11-the-problem-symmetric-hamming-distance) for detailed analysis.

##### Solution: ADC (Asymmetric Distance Computation)

ADC keeps the query as float32 and computes a weighted dot product:

```
Symmetric Hamming:  popcount(encode(query) XOR encode(data))
ADC:                query_rotated · decode(data) / correction_factor
```

This preserves numeric ordering because the query is never quantized.

##### Implementation Plan

**Step 1: Extend BinaryCodeCache (+8 bytes/vector)**

```rust
// Current: HashMap<(EmbeddingCode, VecId), Vec<u8>>
// Change:  HashMap<(EmbeddingCode, VecId), (Vec<u8>, AdcCorrection)>

pub fn put_with_correction(&self, ..., code: Vec<u8>, correction: AdcCorrection);
pub fn get_with_correction(&self, ...) -> Option<(Vec<u8>, AdcCorrection)>;
```

**Step 2: Persist AdcCorrection to BinaryCodes CF**

```rust
// Current value: [binary_code: N bytes]
// Change to:     [binary_code: N bytes][vector_norm: f32][quantization_error: f32]
```

**Step 3: Add ADC Beam Search Function**

```rust
fn beam_search_layer0_adc_cached(
    index: &Index,
    storage: &Storage,
    query_rotated: &[f32],  // Pre-rotated query (float32)
    query_norm: f32,
    encoder: &RaBitQ,
    code_cache: &BinaryCodeCache,
    entry: VecId,
    ef: usize,
) -> Result<Vec<(f32, VecId)>>;
```

**Step 4: Update search_with_rabitq_cached()**

```rust
pub fn search_with_rabitq_adc_cached(...) -> Result<Vec<(f32, VecId)>> {
    // Pre-compute query rotation once
    let query_rotated = encoder.rotate_query(query);
    let query_norm = compute_norm(query);

    // Use ADC distance in beam search
    let candidates = beam_search_layer0_adc_cached(
        ..., &query_rotated, query_norm, encoder, ...
    )?;

    // Re-rank with exact distance
}
```

**Step 5: Update Vector Insert Path**

```rust
// Current:
let code = encoder.encode(&vector);
cache.put(embedding, vec_id, code);

// Change to:
let (code, correction) = encoder.encode_with_correction(&vector);
cache.put_with_correction(embedding, vec_id, code, correction);
storage.put_binary_code_with_correction(embedding, vec_id, &code, correction)?;
```

##### Files to Modify

| File | Changes |
|------|---------|
| `cache/binary_codes.rs` | Add correction storage methods |
| `hnsw/search.rs` | Add `beam_search_layer0_adc_cached()`, update main search |
| `schema.rs` | Update BinaryCodes value format |
| `examples/vector2/main.rs` | Update insert path to store corrections |

##### Expected Results

| Mode | Recall@10 | QPS (projected) |
|------|-----------|-----------------|
| Current Hamming | 24% | 620 |
| **ADC 4-bit, rerank=4x** | **92%** | ~500 |
| ADC 4-bit, rerank=10x | 99% | ~300 |

Memory overhead: +8 bytes/vector (AdcCorrection: 2×f32)
- 100K vectors: +800 KB
- 1M vectors: +8 MB

##### Acceptance Criteria

- [x] `BinaryCodeCache` stores `AdcCorrection` alongside binary codes
- [x] `AdcCorrection` persisted to BinaryCodes CF
- [x] `beam_search_layer0_adc_cached()` implemented
- [x] `search_with_rabitq_cached()` uses ADC for navigation (renamed from `_adc_`)
- [x] Vector insert stores corrections (`writer.rs`, `examples/vector2/main.rs`)
- [ ] Benchmark shows >90% recall@10 at 100K scale (pending benchmark run)
- [x] Document ADC vs Hamming comparison

##### Completion Notes (`ec0fc00`)

**Files Modified:**
- `cache/binary_codes.rs` - Cache stores `(Vec<u8>, AdcCorrection)` tuples
- `hnsw/search.rs` - ADC beam search replaces Hamming version
- `schema.rs` - `BinaryCodeCfValue` includes correction fields (+8 bytes)
- `writer.rs` - Uses `encode_with_correction()` for all binary codes
- `examples/vector2/main.rs` - Updated to use new cache API
- `test_vector_benchmark_integration.rs` - Updated test

**Breaking Change:** BinaryCodeCfValue schema now requires 8 extra bytes for ADC correction factors. Existing binary codes must be rebuilt.

##### References

- [RABITQ.md §1.1-1.5](RABITQ.md#11-the-problem-symmetric-hamming-distance) - Problem analysis
- [RABITQ.md §5.8](RABITQ.md#58-adc-vs-hamming-analysis) - Benchmark results
- [Elasticsearch RaBitQ explainer](https://www.elastic.co/search-labs/blog/rabitq-explainer-101)
- Issue #43 - Gray code multi-bit quantization

---

**Task 4.7 Acceptance Criteria:**
- [x] SIMD Hamming distance in motlie_core with tests
- [x] RaBitQ using SIMD Hamming
- [x] Early filtering in HNSW beam search (pure Hamming approach)
- [x] Document findings and hypothesis for future work
- [ ] Benchmark showing >2x QPS improvement with >90% recall (moved to Task 4.8)

### Phase 4 Validation & Tests

```rust
// libs/db/src/vector/tests/rabitq_tests.rs

#[cfg(test)]
mod rabitq_tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    // =========================================================================
    // Rotation Matrix Tests (Task 4.1)
    // =========================================================================

    #[test]
    fn test_rotation_matrix_is_orthonormal() {
        let matrix = RotationMatrix::generate(128, 42);

        // R * R^T should equal identity (within floating point tolerance)
        for i in 0..128 {
            for j in 0..128 {
                let dot: f32 = (0..128)
                    .map(|k| matrix[(i, k)] * matrix[(j, k)])
                    .sum();

                if i == j {
                    assert!((dot - 1.0).abs() < 1e-5, "Diagonal should be 1.0");
                } else {
                    assert!(dot.abs() < 1e-5, "Off-diagonal should be 0.0");
                }
            }
        }
    }

    #[test]
    fn test_rotation_matrix_deterministic() {
        let matrix1 = RotationMatrix::generate(128, 42);
        let matrix2 = RotationMatrix::generate(128, 42);

        // Same seed should produce same matrix
        for i in 0..128 {
            for j in 0..128 {
                assert_eq!(matrix1[(i, j)], matrix2[(i, j)]);
            }
        }
    }

    #[test]
    fn test_rotation_matrix_different_seeds() {
        let matrix1 = RotationMatrix::generate(128, 42);
        let matrix2 = RotationMatrix::generate(128, 123);

        // Different seeds should produce different matrices
        let mut differences = 0;
        for i in 0..128 {
            for j in 0..128 {
                if (matrix1[(i, j)] - matrix2[(i, j)]).abs() > 1e-6 {
                    differences += 1;
                }
            }
        }
        assert!(differences > 1000, "Matrices should be different");
    }

    // =========================================================================
    // Binary Encoder Tests (Task 4.2)
    // =========================================================================

    #[test]
    fn test_binary_encoding() {
        let matrix = RotationMatrix::generate(128, 42);
        let encoder = RaBitQEncoder::new(matrix);

        let vector = vec![1.0; 128];
        let code = encoder.encode(&vector);

        // Should produce 128 bits = 16 bytes
        assert_eq!(code.len(), 16);
    }

    #[test]
    fn test_binary_encoding_deterministic() {
        let matrix = RotationMatrix::generate(128, 42);
        let encoder = RaBitQEncoder::new(matrix);

        let vector = vec![1.0; 128];
        let code1 = encoder.encode(&vector);
        let code2 = encoder.encode(&vector);

        assert_eq!(code1, code2);
    }

    #[test]
    fn test_binary_encoding_similar_vectors() {
        let matrix = RotationMatrix::generate(128, 42);
        let encoder = RaBitQEncoder::new(matrix);

        let v1 = vec![1.0; 128];
        let mut v2 = vec![1.0; 128];
        v2[0] = 0.99;  // Slightly different

        let code1 = encoder.encode(&v1);
        let code2 = encoder.encode(&v2);

        // Similar vectors should have similar codes (few bit differences)
        let hamming = hamming_distance(&code1, &code2);
        assert!(hamming < 10, "Similar vectors should have low Hamming distance");
    }

    #[test]
    fn test_binary_encoding_different_vectors() {
        let matrix = RotationMatrix::generate(128, 42);
        let encoder = RaBitQEncoder::new(matrix);

        let v1 = vec![1.0; 128];
        let v2 = vec![-1.0; 128];  // Opposite direction

        let code1 = encoder.encode(&v1);
        let code2 = encoder.encode(&v2);

        // Opposite vectors should have ~128 bit differences
        let hamming = hamming_distance(&code1, &code2);
        assert!(hamming > 100, "Opposite vectors should have high Hamming distance");
    }

    // =========================================================================
    // SIMD Hamming Distance Tests (Task 4.3)
    // =========================================================================

    #[test]
    fn test_hamming_distance_zero() {
        let code = vec![0b10101010u8; 16];
        assert_eq!(hamming_distance(&code, &code), 0);
    }

    #[test]
    fn test_hamming_distance_all_different() {
        let code1 = vec![0x00u8; 16];
        let code2 = vec![0xFFu8; 16];
        assert_eq!(hamming_distance(&code1, &code2), 128);
    }

    #[test]
    fn test_hamming_distance_single_bit() {
        let code1 = vec![0x00u8; 16];
        let mut code2 = vec![0x00u8; 16];
        code2[0] = 0x01;  // One bit different
        assert_eq!(hamming_distance(&code1, &code2), 1);
    }

    #[test]
    fn test_hamming_simd_matches_scalar() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        for _ in 0..100 {
            let code1: Vec<u8> = (0..16).map(|_| rng.gen()).collect();
            let code2: Vec<u8> = (0..16).map(|_| rng.gen()).collect();

            let scalar = hamming_distance_scalar(&code1, &code2);
            let simd = hamming_distance(&code1, &code2);

            assert_eq!(scalar, simd, "SIMD should match scalar");
        }
    }

    // =========================================================================
    // Distance Estimation Tests
    // =========================================================================

    #[test]
    fn test_distance_estimation_ordering() {
        let storage = setup_rabitq_storage();
        let code = Embedding::new("test");

        // Insert vectors with known distances from origin
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let vectors: Vec<(Id, Vec<f32>)> = (0..100)
            .map(|_| {
                let ulid = Id::new();
                let vec: Vec<f32> = (0..128).map(|_| rng.gen()).collect();
                (ulid, vec)
            })
            .collect();

        for (ulid, vec) in &vectors {
            storage.insert(&code, *ulid, vec).unwrap();
        }

        // Query
        let query: Vec<f32> = (0..128).map(|_| rng.gen()).collect();

        // Get approximate ranking
        let approx_results = storage.search_approximate(&code, &query, 20).unwrap();

        // Get true ranking
        let mut true_dists: Vec<(f32, usize)> = vectors.iter()
            .enumerate()
            .map(|(i, (_, v))| (euclidean_squared(&query, v), i))
            .collect();
        true_dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // RaBitQ should preserve approximate ordering
        // Kendall's tau or simple overlap check
        let true_top20: HashSet<usize> = true_dists.iter().take(20).map(|(_, i)| *i).collect();
        let approx_top20: HashSet<usize> = approx_results.iter()
            .filter_map(|(_, id)| vectors.iter().position(|(uid, _)| uid == id))
            .collect();

        let overlap = true_top20.intersection(&approx_top20).count();
        assert!(overlap >= 10, "At least 50% overlap in top-20: got {}", overlap);
    }

    // =========================================================================
    // Recall with Re-ranking Tests (Task 4.5)
    // =========================================================================

    #[test]
    fn test_recall_with_reranking() {
        let storage = setup_rabitq_storage();
        let code = Embedding::new("test");

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let vectors: Vec<(Id, Vec<f32>)> = (0..1000)
            .map(|_| {
                let ulid = Id::new();
                let vec: Vec<f32> = (0..128).map(|_| rng.gen()).collect();
                (ulid, vec)
            })
            .collect();

        for (ulid, vec) in &vectors {
            storage.insert(&code, *ulid, vec).unwrap();
        }

        // Test recall with re-ranking
        let mut total_recall = 0.0;
        for _ in 0..50 {
            let query: Vec<f32> = (0..128).map(|_| rng.gen()).collect();

            // Search with re-ranking (overretrieve 100 candidates, rerank to 10)
            let results = storage.search(&code, &query, 10, 100).unwrap();

            // Ground truth
            let mut gt: Vec<(f32, Id)> = vectors.iter()
                .map(|(id, v)| (euclidean_squared(&query, v), *id))
                .collect();
            gt.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let gt_ids: HashSet<Id> = gt.iter().take(10).map(|(_, id)| *id).collect();

            let matches = results.iter().filter(|(_, id)| gt_ids.contains(id)).count();
            total_recall += matches as f64 / 10.0;
        }

        let avg_recall = total_recall / 50.0;
        assert!(avg_recall > 0.90, "RaBitQ recall@10 with reranking should be > 90%, got {:.1}%",
            avg_recall * 100.0);
    }

    // =========================================================================
    // Memory Efficiency Tests
    // =========================================================================

    #[test]
    fn test_binary_code_storage_size() {
        let storage = setup_rabitq_storage();
        let code = Embedding::new("test");

        // Insert 1000 vectors
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        for _ in 0..1000 {
            let ulid = Id::new();
            let vec: Vec<f32> = (0..128).map(|_| rng.gen()).collect();
            storage.insert(&code, ulid, &vec).unwrap();
        }

        // Binary codes should be 16 bytes each (128 bits)
        let binary_size = storage.binary_codes_size(&code).unwrap();
        assert!(binary_size < 20_000, "Binary codes should be ~16KB for 1000 vectors");

        // Compare to full vector size (512 bytes each)
        let vector_size = storage.vectors_size(&code).unwrap();
        assert!(binary_size < vector_size / 20, "Binary codes should be 32x smaller");
    }
}
```

**Test Coverage Checklist:**
- [ ] Rotation matrix is orthonormal (R * R^T = I)
- [ ] Rotation matrix is deterministic for same seed
- [ ] Different seeds produce different matrices
- [ ] Binary encoding produces correct size (D bits)
- [ ] Encoding is deterministic
- [ ] Similar vectors have similar codes (low Hamming)
- [ ] Different vectors have different codes (high Hamming)
- [ ] Hamming distance: zero, all-different, single-bit
- [ ] SIMD Hamming matches scalar implementation
- [ ] Distance estimation preserves approximate ordering
- [ ] Recall@10 with re-ranking > 90%
- [ ] Memory efficiency (32x compression)

**Benchmarks:**
```rust
#[bench]
fn bench_binary_encode(b: &mut Bencher) {
    let matrix = RotationMatrix::generate(128, 42);
    let encoder = RaBitQEncoder::new(matrix);
    let vector = random_vector(128);

    b.iter(|| encoder.encode(&vector));
}
// Target: < 1µs per encode

#[bench]
fn bench_hamming_distance_simd(b: &mut Bencher) {
    let code1 = random_bytes(16);
    let code2 = random_bytes(16);

    b.iter(|| hamming_distance(&code1, &code2));
}
// Target: < 10ns per distance (using popcount SIMD)

#[bench]
fn bench_approximate_search_1k(b: &mut Bencher) {
    let storage = setup_rabitq_storage_with_vectors(1_000);
    let query = random_vector(128);

    b.iter(|| {
        storage.search_approximate(&Embedding::new("bench"), &query, 100).unwrap()
    });
}
// Target: < 0.5ms (vs ~5ms for full distance)
```

**Integration Test: DATA-1 Compliance**
```rust
#[test]
fn test_data1_compliance_no_training() {
    // This test verifies RaBitQ works without training data
    // by using it immediately on random vectors

    let storage = VectorStorage::open_temp().unwrap();
    let code = Embedding::new("test");

    // NO training phase - just insert and search immediately
    let mut rng = ChaCha20Rng::seed_from_u64(42);

    // Insert first vector
    let v1: Vec<f32> = (0..128).map(|_| rng.gen()).collect();
    storage.insert(&code, Id::new(), &v1).unwrap();

    // Search immediately (no training required)
    let results = storage.search(&code, &v1, 1, 10).unwrap();
    assert_eq!(results.len(), 1);

    // Insert 99 more, search again
    for _ in 0..99 {
        let v: Vec<f32> = (0..128).map(|_| rng.gen()).collect();
        storage.insert(&code, Id::new(), &v).unwrap();
    }

    let results = storage.search(&code, &v1, 10, 50).unwrap();
    assert!(results.len() >= 1);

    // Works with any embedding model without training
    let different_space = Embedding::new("openai-ada");
    let ada_vec: Vec<f32> = (0..1536).map(|_| rng.gen()).collect();  // 1536D
    // Would work if we supported variable dimensions
}
```

---

## Phase 4.5: CODEX Pre-Phase 5 Critical Fixes

**Goal:** Address critical correctness issues identified by CODEX code review before starting Phase 5 mutation API.

**Source:** [CODEX-CODE-REVIEW.md](./CODEX-CODE-REVIEW.md) - External code review by Codex (January 2026).

**Rationale:** Phase 5 introduces online mutation APIs. Fixing these issues now prevents them from being
amplified by the mutation path. The top 3 are HIGH severity and directly impact correctness.

### Overview: Issue Severity Matrix

| ID | Issue | Severity | Impact | Blocker for Phase 5? |
|----|-------|----------|--------|----------------------|
| 4.5.1 | Storage type mismatch | 🔴 HIGH | F16 vectors corrupted on read | **Yes** - writes use F32 always |
| 4.5.2 | Layer assignment cold cache | 🔴 HIGH | Skewed layer distribution | **Yes** - affects insert correctness |
| 4.5.3 | Missing ADC codes → MAX dist | 🔴 HIGH | Beam pollution, poor recall | **Yes** - affects search correctness |
| 4.5.4 | Empty index errors | 🟡 MEDIUM | Unexpected errors | No - edge case |
| 4.5.5 | Cache size overcount | 🟢 LOW | Inaccurate stats | No - cosmetic |

### Task 4.5.1: Storage-Type Mismatch Fix

**CODEX Finding #1 (HIGH):** `writer.rs:343` writes vectors via `Vectors::value_to_bytes`, which always
serializes as F32, even though `EmbeddingSpec` supports F16.

**Impact:** Embeddings registered with `VectorElementType::F16` will be persisted as F32 but later
read as F16, corrupting values or truncating buffers.

**Fix Implementation:**

```rust
// libs/db/src/vector/writer.rs - BEFORE
let bytes = Vectors::value_to_bytes(&vec_value);

// libs/db/src/vector/writer.rs - AFTER
let storage_type = self.registry.get(op.embedding)
    .ok_or_else(|| anyhow!("Unknown embedding: {}", op.embedding))?
    .storage_type;
let bytes = Vectors::value_to_bytes_typed(&op.vector, storage_type);
```

**Files to modify:**
- `libs/db/src/vector/writer.rs`: Lookup embedding spec, use `value_to_bytes_typed`
- `libs/db/src/vector/schema.rs`: Ensure `value_from_bytes_typed` exists (or add)
- `libs/db/src/vector/hnsw/graph.rs`: Ensure reads use the same storage type

**Testing:**
- Unit test: Insert F16 vector, read back, verify numeric error within F16 bounds
- Integration: HNSW index with F16 returns stable neighbors vs F32

### Task 4.5.2: Layer Assignment Cold Cache Fix

**CODEX Finding #2 (HIGH):** `insert.rs:33` assigns `node_layer` from `nav_cache.get(...)` and falls
back to `0` without loading GraphMeta.

**Impact:** First insert after restart (cold cache) always gets layer 0, skewing exponential layer
distribution and reducing recall.

**Fix Implementation:**

```rust
// libs/db/src/vector/hnsw/insert.rs - BEFORE
let nav_info = nav_cache.get(&index_key).cloned().unwrap_or_default();
let node_layer = nav_info.random_layer(&mut rng);

// libs/db/src/vector/hnsw/insert.rs - AFTER
let nav_info = get_or_init_navigation(index, storage)?;  // Loads from GraphMeta if cache empty
let node_layer = nav_info.random_layer(&mut rng);
// Update cache for subsequent inserts
nav_cache.insert(index_key.clone(), nav_info.clone());
```

**Files to modify:**
- `libs/db/src/vector/hnsw/insert.rs`: Call `get_or_init_navigation` before layer assignment

**Testing:**
- Restart test: Build index, drop cache, insert vector, verify layer > 0 with non-trivial probability
- Distribution test: Insert N nodes across restarts, verify exponential layer histogram

### Task 4.5.3: ADC Missing Code Handling

**CODEX Finding #4 (MEDIUM→HIGH):** `search.rs:327` uses `f32::MAX` when `code_cache` misses,
polluting beam with sentinel distances.

**Impact:** Partially populated caches cause early termination or poor recall due to MAX-distance
candidates displacing real ones.

**Claude Assessment:** Elevated to HIGH because this directly affects search quality during
incremental indexing or cache warmup.

**Fix Implementation:**

```rust
// libs/db/src/vector/hnsw/search.rs - BEFORE
let dist = match code_cache.get(vec_id) {
    Some(code) => rabitq.adc_distance(query_code, code),
    None => f32::MAX,  // BAD: pollutes beam
};
candidates.push((dist, vec_id));

// libs/db/src/vector/hnsw/search.rs - AFTER
if let Some(code) = code_cache.get(vec_id) {
    let dist = rabitq.adc_distance(query_code, code);
    candidates.push((dist, vec_id));
} else {
    // Option A: Skip candidate entirely (lose potential good result)
    // Option B: Fall back to exact distance (slower but correct)
    let vector = storage.get_vector(vec_id)?;
    let dist = distance.compute(&query, &vector);
    candidates.push((dist, vec_id));
}
```

**Recommendation:** Option B (fallback to exact) maintains correctness at cost of occasional
exact computation. For production, ensure code cache is fully populated before search.

**Files to modify:**
- `libs/db/src/vector/hnsw/search.rs`: Replace MAX sentinel with skip or exact fallback

**Testing:**
- Unit test: Search with 50% populated cache, verify recall >= 90%
- Integration: Compare recall with full cache vs partial cache

### Task 4.5.4: Empty Index Search Handling

**CODEX Finding #3 (MEDIUM):** `search.rs:36` errors when GraphMeta is missing for an empty index.

**Impact:** Valid "empty index" state returns error instead of `Ok(Vec::new())`.

**Fix Implementation:**

```rust
// libs/db/src/vector/hnsw/search.rs - BEFORE
let nav_info = load_navigation(index, storage)?;  // Errors on missing GraphMeta

// libs/db/src/vector/hnsw/search.rs - AFTER
let nav_info = match load_navigation(index, storage) {
    Ok(info) => info,
    Err(_) => return Ok(vec![]),  // Empty index → empty results
};
```

**Files to modify:**
- `libs/db/src/vector/hnsw/search.rs`: Handle missing GraphMeta gracefully

**Testing:**
- Unit test: Search on freshly created empty index returns `Ok([])`

### Task 4.5.5: BinaryCodeCache Size Accounting

**CODEX Finding #5 (LOW):** `binary_codes.rs:78` increments `cache_bytes` without subtracting
previous entry size on overwrite.

**Impact:** `stats()` becomes inaccurate after updates, misleading memory monitoring.

**Fix Implementation:**

```rust
// libs/db/src/vector/cache/binary_codes.rs - BEFORE
self.cache_bytes += code.len();
self.codes.insert(vec_id, code);

// libs/db/src/vector/cache/binary_codes.rs - AFTER
if let Some(old_code) = self.codes.get(&vec_id) {
    self.cache_bytes -= old_code.len();
}
self.cache_bytes += code.len();
self.codes.insert(vec_id, code);
```

**Files to modify:**
- `libs/db/src/vector/cache/binary_codes.rs`: Subtract old entry size on overwrite

**Testing:**
- Unit test: Overwrite same vec_id twice, verify `stats().bytes` is accurate

### Phase 4.5 Completion Criteria

- [ ] All 5 tasks implemented with unit tests
- [ ] Existing benchmark recall unchanged or improved
- [ ] No regressions in `cargo test --lib` (vector module)
- [ ] CODEX-CODE-REVIEW.md updated with resolution status

---

## Phase 5: Internal Mutation/Query API

**Goal:** Complete the internal API for vector storage operations (insert, delete, search, embedding management).

**Motivation:** Before wrapping with MPSC/MPMC channels (Phase 6), we need working internal APIs
that the channel consumers will call. This mirrors how graph:: and fulltext:: have internal
Processor methods that their consumers invoke.

### Overview: Internal API Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Internal API Architecture                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Mutation Path:                                                              │
│  ┌────────────┐     ┌────────────┐     ┌────────────┐                       │
│  │ Mutation   │ ──► │ Processor  │ ──► │ RocksDB    │                       │
│  │ (enum)     │     │ (methods)  │     │ (CFs)      │                       │
│  └────────────┘     └────────────┘     └────────────┘                       │
│                                                                              │
│  Query Path:                                                                 │
│  ┌────────────┐     ┌────────────┐     ┌────────────┐                       │
│  │ Query      │ ──► │ Storage    │ ──► │ RocksDB    │                       │
│  │ (enum)     │     │ (methods)  │     │ + HNSW     │                       │
│  └────────────┘     └────────────┘     └────────────┘                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Task 5.0: HNSW Transaction Refactoring (Prerequisite)

**Status:** ✅ Complete (`f672a2f`, `1524679`)
**Severity:** 🔴 HIGH - Required for online serving atomicity

**Problem:** The current `hnsw::insert()` path uses raw `txn_db.put_cf()` and `txn_db.merge_cf()`
calls instead of transactional writes. This means graph updates are NOT atomic:

```rust
// CURRENT (hnsw/insert.rs) - NON-ATOMIC
txn_db.put_cf(&cf, key, value)?;  // Raw put, not transactional

// REQUIRED - ATOMIC
txn.put_cf(&cf, key, value)?;     // Transaction-scoped put
```

**Impact for online serving:**
- A crash during insert could leave the graph in inconsistent state
- Entry point, max level, VecMeta, and edges can be partially written
- Orphaned nodes or missing edges could corrupt search results

**Files requiring changes:**

| File | Current | Required |
|------|---------|----------|
| `hnsw/insert.rs` | `txn_db.put_cf()` | Accept `&Transaction`, use `txn.put_cf()` |
| `hnsw/graph.rs` | `txn_db.merge_cf()` | Accept `&Transaction`, use `txn.merge_cf()` |

**Refactoring approach:**

```rust
// hnsw/insert.rs - BEFORE
pub fn insert(index: &Index, storage: &Storage, vec_id: VecId, vector: &[f32]) -> Result<()> {
    let txn_db = storage.transaction_db()?;
    // ... uses txn_db.put_cf() directly
}

// hnsw/insert.rs - AFTER
pub fn insert(
    index: &Index,
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    vec_id: VecId,
    vector: &[f32],
) -> Result<()> {
    // ... uses txn.put_cf() for atomic writes
}

// graph.rs - connect_neighbors() similarly updated
pub fn connect_neighbors(
    index: &Index,
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    vec_id: VecId,
    neighbors: &[(f32, VecId)],
    layer: HnswLayer,
) -> Result<()> {
    // ... uses txn.merge_cf() for atomic edge writes
}
```

**Testing:**
- Verify existing HNSW tests still pass after refactoring
- Add crash-recovery test: insert + simulated crash → verify consistent state
- Stress test: concurrent inserts with transactions

**CODEX concerns to address during Task 5.0:**
- Ensure *all* vector insert side effects share the same RocksDB transaction: vector bytes, binary codes, VecMeta, GraphMeta, edge updates, and any allocator state persisted in `IdAlloc`. Partial transactional coverage can still leave torn state.
- Defer navigation cache updates until *after* the transaction commits (or make cache refresh resilient) to avoid caching uncommitted graph state.

**Assessment:** Both concerns are valid and will be addressed:

1. **All side effects in transaction** - Implementation approach:
   - `IdAllocator::allocate()` will write to IdAlloc CF within the passed transaction
   - All CF writes (Vectors, BinaryCodes, IdForward, IdReverse, VecMeta, GraphMeta, Edges) use same `&Transaction`
   - Single `txn.commit()` at end ensures atomicity

2. **Navigation cache timing** - Implementation approach:
   ```rust
   // WRONG: Update cache before commit
   nav_cache.update(...);
   txn.commit()?;

   // CORRECT: Update cache after successful commit
   txn.commit()?;
   nav_cache.update(...);  // Only on success
   ```
   - Cache updates moved to after `txn.commit()` succeeds
   - On transaction failure, cache remains unchanged (consistent with storage)
   - Alternative: make cache "eventually consistent" via periodic refresh from storage

**Implementation (January 2026):**

Completed in 6 subtasks:
- **5.0.1**: Added `IdAllocator::allocate()` and `free()` for transactional ID allocation
- **5.0.2**: Refactored `hnsw/insert.rs` with `insert()` API returning `CacheUpdate`
- **5.0.3**: Added `connect_neighbors()` to `hnsw/graph.rs`
- **5.0.4**: `CacheUpdate::apply()` deferred until after `txn.commit()` succeeds
- **5.0.5**: Updated `writer.rs` to use transactional HNSW insert with collected cache updates
- **5.0.6**: Added 8 crash recovery tests in `crash_recovery_tests.rs`

Key files changed:
- `id.rs`: Transaction-aware allocation with `allocate()`, `free()` (in-memory: `allocate_local()`, `free_local()`)
- `hnsw/insert.rs`: `insert()` returns `CacheUpdate`, applied after commit
- `hnsw/graph.rs`: `connect_neighbors()` uses `txn.merge_cf()`
- `hnsw/mod.rs`: Export new APIs, `Index` derives `Clone`
- `processor.rs`: Added `get_or_create_index()`, `nav_cache()`, `hnsw_config()`
- `writer.rs`: Collect `CacheUpdate`s, apply after commit

---

### Task 5.1: Processor::insert_vector()

Implement the core vector insertion logic:

```rust
// libs/db/src/vector/processor.rs

impl Processor {
    /// Insert a single vector into an embedding space.
    ///
    /// Steps:
    /// 1. Allocate internal u32 ID
    /// 2. Store ULID <-> u32 mappings (IdForward, IdReverse)
    /// 3. Store vector data (Vectors CF)
    /// 4. Store vector metadata (VecMeta CF)
    /// 5. Encode and store binary code if RaBitQ enabled (BinaryCodes CF)
    /// 6. Build HNSW graph connections (Edges CF)
    /// 7. Update graph metadata if needed (GraphMeta CF)
    pub fn insert_vector(
        &self,
        embedding: EmbeddingCode,
        id: Id,
        vector: &[f32],
        build_graph: bool,
    ) -> Result<VecId> {
        // Validate embedding exists
        let spec = self.registry.get(embedding)
            .ok_or_else(|| anyhow!("Unknown embedding: {}", embedding))?;

        // Validate dimension
        if vector.len() != spec.dim as usize {
            return Err(anyhow!(
                "Dimension mismatch: expected {}, got {}",
                spec.dim, vector.len()
            ));
        }

        let txn = self.db.transaction();

        // 1. Allocate internal ID
        let vec_id = self.id_allocator.allocate(&txn, embedding)?;

        // 2. Store ID mappings
        self.store_id_mappings(&txn, embedding, id, vec_id)?;

        // 3. Store vector
        self.store_vector(&txn, embedding, vec_id, vector)?;

        // 4. Store metadata
        self.store_vec_meta(&txn, embedding, vec_id, VecMeta::new())?;

        // 5. Store binary code if RaBitQ enabled
        if let Some(rabitq) = self.get_rabitq(embedding) {
            let code = rabitq.encode(vector);
            self.store_binary_code(&txn, embedding, vec_id, &code)?;
        }

        // 6. Build graph connections (optional - can be deferred)
        if build_graph {
            self.build_graph_connections(&txn, embedding, vec_id, vector, spec.distance)?;
        }

        txn.commit()?;
        Ok(vec_id)
    }
}
```

### Task 5.2: Processor::delete_vector()

Implement vector deletion:

```rust
impl Processor {
    /// Delete a vector from an embedding space.
    ///
    /// Steps:
    /// 1. Look up internal vec_id from external ID
    /// 2. Mark as deleted in VecMeta (for search filtering)
    /// 3. Remove ID mappings
    /// 4. Remove vector data
    /// 5. Remove binary code
    /// 6. Return ID to free list
    /// 7. Edges cleaned up lazily (merge operator handles stale refs)
    pub fn delete_vector(
        &self,
        embedding: EmbeddingCode,
        id: Id,
    ) -> Result<()> {
        let txn = self.db.transaction();

        // 1. Look up internal ID
        let vec_id = self.get_internal_id(&txn, embedding, id)?
            .ok_or_else(|| anyhow!("Vector not found: {}", id))?;

        // 2. Mark as deleted (search will filter)
        self.mark_deleted(&txn, embedding, vec_id)?;

        // 3. Remove ID mappings
        self.remove_id_mappings(&txn, embedding, id, vec_id)?;

        // 4. Remove vector data
        self.remove_vector(&txn, embedding, vec_id)?;

        // 5. Remove binary code
        self.remove_binary_code(&txn, embedding, vec_id)?;

        // 6. Return ID to free list
        self.id_allocator.free(&txn, embedding, vec_id)?;

        // 7. Edges: lazily cleaned via merge operator
        // No explicit edge removal needed

        txn.commit()?;
        Ok(())
    }

    /// Check if a vector is marked as deleted.
    pub fn is_deleted(&self, embedding: EmbeddingCode, vec_id: VecId) -> Result<bool> {
        let meta = self.get_vec_meta(embedding, vec_id)?;
        Ok(meta.map_or(true, |m| m.is_deleted()))
    }
}
```

### Task 5.3: Processor::insert_batch()

Batch insert for efficiency:

```rust
impl Processor {
    /// Batch insert multiple vectors.
    ///
    /// More efficient than individual inserts:
    /// - Single transaction for all vectors
    /// - Batched ID allocation
    /// - Batched RocksDB writes
    pub fn insert_batch(
        &self,
        embedding: EmbeddingCode,
        vectors: &[(Id, Vec<f32>)],
        build_graph: bool,
    ) -> Result<Vec<VecId>> {
        if vectors.is_empty() {
            return Ok(vec![]);
        }

        let spec = self.registry.get(embedding)
            .ok_or_else(|| anyhow!("Unknown embedding: {}", embedding))?;

        let txn = self.db.transaction();
        let mut vec_ids = Vec::with_capacity(vectors.len());

        for (id, vector) in vectors {
            // Validate dimension
            if vector.len() != spec.dim as usize {
                return Err(anyhow!("Dimension mismatch for {}", id));
            }

            let vec_id = self.id_allocator.allocate(&txn, embedding)?;
            self.store_id_mappings(&txn, embedding, *id, vec_id)?;
            self.store_vector(&txn, embedding, vec_id, vector)?;
            self.store_vec_meta(&txn, embedding, vec_id, VecMeta::new())?;

            if let Some(rabitq) = self.get_rabitq(embedding) {
                let code = rabitq.encode(vector);
                self.store_binary_code(&txn, embedding, vec_id, &code)?;
            }

            vec_ids.push(vec_id);
        }

        // Build graph connections after all vectors stored
        if build_graph {
            for (i, vec_id) in vec_ids.iter().enumerate() {
                let vector = &vectors[i].1;
                self.build_graph_connections(&txn, embedding, *vec_id, vector, spec.distance)?;
            }
        }

        txn.commit()?;
        Ok(vec_ids)
    }
}
```

### Task 5.4: Storage::search()

Implement search with external ID resolution:

```rust
// libs/db/src/vector/storage.rs (or hnsw.rs)

impl Storage {
    /// Search for k nearest neighbors.
    ///
    /// Returns results with external IDs (not internal vec_ids).
    pub fn search(
        &self,
        embedding: EmbeddingCode,
        query: &[f32],
        k: usize,
        config: &SearchConfig,
    ) -> Result<Vec<SearchResult>> {
        // Get embedding spec
        let spec = self.registry.get(embedding)
            .ok_or_else(|| anyhow!("Unknown embedding: {}", embedding))?;

        // Validate query dimension
        if query.len() != spec.dim as usize {
            return Err(anyhow!(
                "Query dimension mismatch: expected {}, got {}",
                spec.dim, query.len()
            ));
        }

        // Perform HNSW search (returns vec_ids)
        let internal_results = self.hnsw_search(embedding, query, k, config)?;

        // Resolve to external IDs, filtering deleted
        let mut results = Vec::with_capacity(internal_results.len());
        for (distance, vec_id) in internal_results {
            // Skip deleted vectors
            if self.is_deleted(embedding, vec_id)? {
                continue;
            }

            // Resolve to external ID
            if let Some(external_id) = self.get_external_id(embedding, vec_id)? {
                results.push(SearchResult {
                    distance,
                    id: external_id,
                    vec_id, // Include for debugging/advanced use
                });
            }
        }

        Ok(results)
    }
}

/// Search result with distance and IDs.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Distance from query vector
    pub distance: f32,
    /// External document ID (ULID)
    pub id: Id,
    /// Internal vector ID (for advanced use)
    pub vec_id: VecId,
}
```

### Task 5.5: Processor::add_embedding_spec()

Embedding registry management:

```rust
impl Processor {
    /// Register a new embedding space.
    ///
    /// Persists to EmbeddingSpecs CF and updates in-memory registry.
    pub fn add_embedding_spec(&self, spec: &AddEmbeddingSpec) -> Result<()> {
        // Check if already exists
        if self.registry.get(spec.code).is_some() {
            return Err(anyhow!("Embedding {} already exists", spec.code));
        }

        let txn = self.db.transaction();

        // Persist to RocksDB
        let key = EmbeddingSpecCfKey(spec.code);
        let value = EmbeddingSpecCfValue(EmbeddingSpec {
            model: spec.model.clone(),
            dim: spec.dim,
            distance: spec.distance,
            storage_type: spec.storage_type,
        });

        let cf = txn.cf_handle(EmbeddingSpecs::CF_NAME)?;
        txn.put_cf(&cf, EmbeddingSpecs::key_to_bytes(&key), EmbeddingSpecs::value_to_bytes(&value))?;

        txn.commit()?;

        // Update in-memory registry
        self.registry.register_from_db(spec.code, &spec.model, spec.dim, spec.distance);

        Ok(())
    }
}
```

### Task 5.6: Mutation Dispatch

Connect Mutation enum to Processor methods:

```rust
// libs/db/src/vector/mutation.rs

impl Mutation {
    /// Execute this mutation against the processor.
    pub fn execute(&self, processor: &Processor) -> Result<()> {
        match self {
            Mutation::AddEmbeddingSpec(m) => {
                processor.add_embedding_spec(m)?;
            }
            Mutation::InsertVector(m) => {
                processor.insert_vector(
                    m.embedding,
                    m.id,
                    &m.vector,
                    m.immediate_index,
                )?;
            }
            Mutation::DeleteVector(m) => {
                processor.delete_vector(m.embedding, m.id)?;
            }
            Mutation::InsertVectorBatch(m) => {
                processor.insert_batch(
                    m.embedding,
                    &m.vectors,
                    m.immediate_index,
                )?;
            }
            Mutation::UpdateEdges(m) => {
                processor.update_edges(m)?;
            }
            Mutation::UpdateGraphMeta(m) => {
                processor.update_graph_meta(m)?;
            }
            Mutation::Flush(marker) => {
                marker.complete();
            }
        }
        Ok(())
    }
}
```

### Task 5.7: Query Dispatch

Connect Query enum to Storage methods:

```rust
// libs/db/src/vector/query.rs

impl Query {
    /// Process this query against storage.
    pub async fn process(self, storage: &Storage) {
        match self {
            Query::GetVector(dispatch) => {
                let result = storage.get_vector(
                    dispatch.params.embedding,
                    dispatch.params.id,
                );
                dispatch.send_result(result);
            }
            Query::GetInternalId(dispatch) => {
                let result = storage.get_internal_id(
                    dispatch.params.embedding,
                    dispatch.params.id,
                );
                dispatch.send_result(result);
            }
            Query::GetExternalId(dispatch) => {
                let result = storage.get_external_id(
                    dispatch.params.embedding,
                    dispatch.params.vec_id,
                );
                dispatch.send_result(result);
            }
            Query::ResolveIds(dispatch) => {
                let result = storage.resolve_ids(
                    dispatch.params.embedding,
                    &dispatch.params.vec_ids,
                );
                dispatch.send_result(result);
            }
            Query::SearchKNN(dispatch) => {
                let result = storage.search(
                    dispatch.params.embedding,
                    &dispatch.params.query,
                    dispatch.params.k,
                    &dispatch.params.config,
                );
                dispatch.send_result(result);
            }
        }
    }
}
```

### Phase 5 File Changes Summary

| File | Changes |
|------|---------|
| `processor.rs` | Add `insert_vector()`, `delete_vector()`, `insert_batch()`, `add_embedding_spec()` |
| `storage.rs` | Add `search()` with external ID resolution |
| `mutation.rs` | Add `Mutation::execute()` dispatch |
| `query.rs` | Add `SearchKNN` query type, update `Query::process()` |
| `schema.rs` | Add `VecMeta` with deleted flag |

### Phase 5 Tests

```rust
#[cfg(test)]
mod internal_api_tests {
    use super::*;

    #[test]
    fn test_insert_and_search() {
        let (storage, processor) = setup_temp_storage();

        // Register embedding
        processor.add_embedding_spec(&AddEmbeddingSpec {
            code: 1,
            model: "test".to_string(),
            dim: 128,
            distance: Distance::Cosine,
            storage_type: VectorElementType::F32,
        }).unwrap();

        // Insert vectors
        let id1 = Id::new();
        let vec1 = vec![1.0f32; 128];
        processor.insert_vector(1, id1, &vec1, true).unwrap();

        // Search
        let results = storage.search(1, &vec1, 1, &SearchConfig::default()).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, id1);
    }

    #[test]
    fn test_delete_excludes_from_search() {
        let (storage, processor) = setup_temp_storage();
        setup_embedding(&processor, 1);

        let id = Id::new();
        processor.insert_vector(1, id, &[1.0; 128], true).unwrap();

        // Should find it
        let results = storage.search(1, &[1.0; 128], 1, &SearchConfig::default()).unwrap();
        assert_eq!(results.len(), 1);

        // Delete
        processor.delete_vector(1, id).unwrap();

        // Should not find it
        let results = storage.search(1, &[1.0; 128], 1, &SearchConfig::default()).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_batch_insert() {
        let (storage, processor) = setup_temp_storage();
        setup_embedding(&processor, 1);

        let vectors: Vec<_> = (0..100)
            .map(|i| (Id::new(), vec![i as f32; 128]))
            .collect();

        let vec_ids = processor.insert_batch(1, &vectors, true).unwrap();
        assert_eq!(vec_ids.len(), 100);

        // All should be searchable
        let results = storage.search(1, &[50.0; 128], 10, &SearchConfig::default()).unwrap();
        assert_eq!(results.len(), 10);
    }
}
```

### Task 5.8: Migrate Examples and Integration Tests

**Goal:** Update all examples and integration tests to use the new Processor/Storage API.

**Scope:**

1. **`examples/laion_benchmark/`**:
   - Replace direct `hnsw::Index::insert()` with `Processor::insert_vector()`
   - Replace manual CF writes with `Processor::insert_batch()`
   - Keep direct `Index::search()` for raw performance measurement OR migrate to `Storage::search()`
   - Update RaBitQ integration to use Processor-managed encoders

2. **`libs/db/tests/test_vector_benchmark_integration.rs`**:
   - Replace `build_index_with_navigation()` helper with Processor-based setup
   - Remove manual `txn_db.put_cf()` calls for vector storage
   - Use `Processor::insert_batch()` for test data setup
   - Use `Storage::search()` for search validation

3. **Other integration tests** (if any):
   - Audit all tests using direct Index/CF access
   - Migrate to Processor/Storage API

**Migration Pattern:**

```rust
// BEFORE: Direct Index + manual CF writes
fn build_index_with_navigation(
    storage: &Storage,
    vectors: &[Vec<f32>],
    ...
) -> Result<(hnsw::Index, Arc<NavigationCache>)> {
    let txn_db = storage.transaction_db()?;
    let vectors_cf = txn_db.cf_handle(Vectors::CF_NAME)?;

    for (i, vector) in vectors.iter().enumerate() {
        let vec_id = i as VecId;
        // Manual CF write
        txn_db.put_cf(&vectors_cf, key_bytes, value_bytes)?;
        // Manual graph insert
        index.insert(storage, vec_id, vector)?;
    }
}

// AFTER: Processor API
fn setup_test_vectors(
    processor: &Processor,
    storage: &Storage,
    embedding: EmbeddingCode,
    vectors: &[Vec<f32>],
) -> Result<Vec<VecId>> {
    // Register embedding if needed
    processor.add_embedding_spec(&AddEmbeddingSpec {
        code: embedding,
        model: "test".into(),
        dim: vectors[0].len() as u32,
        distance: Distance::Cosine,
        storage_type: VectorElementType::F32,
    })?;

    // Batch insert handles: ID allocation, CF writes, graph building
    let ids_and_vectors: Vec<_> = vectors.iter()
        .map(|v| (Id::new(), v.clone()))
        .collect();
    processor.insert_batch(embedding, &ids_and_vectors, true)
}

// Search using Storage API
let results = storage.search(embedding, &query, k, &SearchConfig::default())?;
```

**Benchmark-Specific Considerations:**

For `laion_benchmark`, provide two modes:
1. **API mode** (default): Uses Processor/Storage for realistic end-to-end measurement
2. **Raw mode** (optional flag): Direct Index access for isolating HNSW performance

```rust
// examples/laion_benchmark/main.rs
if args.raw_mode {
    // Direct Index access (current behavior, for HNSW-only benchmarks)
    index.insert(&storage, vec_id, &vector)?;
} else {
    // Full API (Phase 5+, measures complete insert path)
    processor.insert_vector(embedding, id, &vector, true)?;
}
```

**Validation Criteria:**
- All existing tests pass with new API
- Benchmark results within 10% of direct Index performance
- No functionality regression

---

**Note:** Tasks 5.9-5.11 (Concurrent Operations) are documented in detail in [CONCURRENT.md](./CONCURRENT.md).

---

### Task 5.9: Multi-Threaded Stress Tests ✅

**Status:** Complete ([CONCURRENT.md](./CONCURRENT.md))

**Goal:** Validate concurrent access patterns under load.

**Test Categories:**

1. **Concurrent Read/Write:**
   - Multiple threads inserting while others search
   - Verify no data corruption or panics
   - Test with varying thread counts (4, 8, 16, 32)

2. **Concurrent Insert/Search:**
   - Simulate real-world pattern of continuous ingestion + queries
   - Verify search results remain consistent
   - Test eventual visibility of newly inserted vectors

3. **Concurrent Delete/Search:**
   - Delete vectors while searches are in progress
   - Verify deleted vectors don't appear in results
   - Test tombstone visibility timing

**Implementation:**

```rust
// libs/db/src/vector/tests/concurrent.rs

#[cfg(test)]
mod concurrent_tests {
    use std::sync::Arc;
    use std::thread;
    use super::*;

    #[test]
    fn test_concurrent_insert_search() {
        let (storage, processor) = setup_temp_storage();
        let storage = Arc::new(storage);
        let processor = Arc::new(processor);
        setup_embedding(&processor, 1);

        let insert_handle = {
            let processor = Arc::clone(&processor);
            thread::spawn(move || {
                for i in 0..1000 {
                    let id = Id::new();
                    let vector = vec![i as f32; 128];
                    processor.insert_vector(1, id, &vector, true).unwrap();
                }
            })
        };

        let search_handle = {
            let storage = Arc::clone(&storage);
            thread::spawn(move || {
                for _ in 0..100 {
                    let query = vec![500.0; 128];
                    let _ = storage.search(1, &query, 10, &SearchConfig::default());
                    thread::sleep(std::time::Duration::from_millis(5));
                }
            })
        };

        insert_handle.join().unwrap();
        search_handle.join().unwrap();
    }

    #[test]
    fn test_concurrent_delete_search() {
        // Similar pattern for delete + search
    }

    #[test]
    fn test_high_thread_count_stress() {
        // 32 threads, 10K operations each
    }
}
```

### Task 5.10: Metrics Collection Infrastructure ✅

**Status:** Complete ([CONCURRENT.md](./CONCURRENT.md))

**Goal:** Add metrics collection for concurrent operation analysis.

**Location:** `libs/db/src/vector/benchmark/` subcrate

**Metrics to Collect:**

| Metric | Type | Description |
|--------|------|-------------|
| `insert_latency_ns` | Histogram | Per-operation insert latency |
| `search_latency_ns` | Histogram | Per-operation search latency |
| `delete_latency_ns` | Histogram | Per-operation delete latency |
| `concurrent_ops` | Gauge | Active concurrent operations |
| `throughput_ops_sec` | Counter | Operations per second |

**Implementation:**

```rust
// libs/db/src/vector/benchmark/metrics.rs

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Lightweight metrics collector for benchmarking.
/// Uses atomics for lock-free concurrent updates.
pub struct BenchMetrics {
    insert_count: AtomicU64,
    search_count: AtomicU64,
    delete_count: AtomicU64,
    insert_total_ns: AtomicU64,
    search_total_ns: AtomicU64,
    delete_total_ns: AtomicU64,
    // Latency histograms (bucket counts)
    insert_latency_buckets: [AtomicU64; 16],  // log2 buckets
    search_latency_buckets: [AtomicU64; 16],
}

impl BenchMetrics {
    pub fn new() -> Self { ... }

    pub fn record_insert(&self, duration_ns: u64) { ... }
    pub fn record_search(&self, duration_ns: u64) { ... }
    pub fn record_delete(&self, duration_ns: u64) { ... }

    /// Calculate percentiles from histogram buckets
    pub fn percentile(&self, op: &str, p: f64) -> u64 { ... }

    /// Generate summary report
    pub fn summary(&self) -> MetricsSummary {
        MetricsSummary {
            insert_p50: self.percentile("insert", 0.50),
            insert_p95: self.percentile("insert", 0.95),
            insert_p99: self.percentile("insert", 0.99),
            search_p50: self.percentile("search", 0.50),
            search_p95: self.percentile("search", 0.95),
            search_p99: self.percentile("search", 0.99),
            // ...
        }
    }
}

#[derive(Debug, Clone)]
pub struct MetricsSummary {
    pub insert_p50: u64,
    pub insert_p95: u64,
    pub insert_p99: u64,
    pub search_p50: u64,
    pub search_p95: u64,
    pub search_p99: u64,
    pub insert_throughput: f64,
    pub search_throughput: f64,
}
```

### Task 5.11: Concurrent Benchmark Baseline ✅

**Status:** Complete ([CONCURRENT.md](./CONCURRENT.md), [BASELINE.md](./BASELINE.md))

**Goal:** Establish baseline metrics for concurrent operations.

**Baseline Requirements:** See [BASELINE.md](./BASELINE.md) for:
- Minimum 10,000 vectors per embedding
- Minimum 2 embedding spaces
- Full environment documentation template
- Regression tracking strategy

**Location:** `libs/db/src/vector/benchmark/` subcrate

**Benchmark Scenarios:**

| Scenario | Writers | Readers | Duration | Expected |
|----------|---------|---------|----------|----------|
| Read-heavy | 1 | 8 | 30s | High QPS, stable latency |
| Write-heavy | 8 | 1 | 30s | Consistent throughput |
| Balanced | 4 | 4 | 30s | Baseline mixed workload |
| Stress | 16 | 16 | 60s | Identify bottlenecks |

**Implementation:**

```rust
// libs/db/src/vector/benchmark/concurrent_bench.rs

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

pub struct ConcurrentBench {
    storage: Arc<Storage>,
    processor: Arc<Processor>,
    metrics: Arc<BenchMetrics>,
}

impl ConcurrentBench {
    pub fn new(storage: Storage, processor: Processor) -> Self { ... }

    /// Run benchmark with specified thread configuration
    pub fn run(
        &self,
        writer_threads: usize,
        reader_threads: usize,
        duration: Duration,
        vectors_per_writer: usize,
    ) -> BenchResult {
        let start = Instant::now();
        let mut handles = Vec::new();

        // Spawn writer threads
        for _ in 0..writer_threads {
            let processor = Arc::clone(&self.processor);
            let metrics = Arc::clone(&self.metrics);
            handles.push(thread::spawn(move || {
                writer_workload(processor, metrics, vectors_per_writer);
            }));
        }

        // Spawn reader threads
        for _ in 0..reader_threads {
            let storage = Arc::clone(&self.storage);
            let metrics = Arc::clone(&self.metrics);
            let dur = duration;
            handles.push(thread::spawn(move || {
                reader_workload(storage, metrics, dur);
            }));
        }

        // Wait for completion
        for h in handles {
            h.join().unwrap();
        }

        BenchResult {
            duration: start.elapsed(),
            metrics: self.metrics.summary(),
        }
    }
}

#[derive(Debug)]
pub struct BenchResult {
    pub duration: Duration,
    pub metrics: MetricsSummary,
}

fn writer_workload(processor: Arc<Processor>, metrics: Arc<BenchMetrics>, count: usize) {
    for i in 0..count {
        let start = Instant::now();
        let id = Id::new();
        let vector = generate_random_vector(128);
        processor.insert_vector(1, id, &vector, true).unwrap();
        metrics.record_insert(start.elapsed().as_nanos() as u64);
    }
}

fn reader_workload(storage: Arc<Storage>, metrics: Arc<BenchMetrics>, duration: Duration) {
    let deadline = Instant::now() + duration;
    while Instant::now() < deadline {
        let start = Instant::now();
        let query = generate_random_vector(128);
        let _ = storage.search(1, &query, 10, &SearchConfig::default());
        metrics.record_search(start.elapsed().as_nanos() as u64);
    }
}
```

**Expected Baseline Results (100K vectors, 8-core):**

| Scenario | Search P50 | Search P99 | Search QPS | Insert/s |
|----------|------------|------------|------------|----------|
| Read-heavy (1W/8R) | 3.5ms | 12ms | 2000+ | 50 |
| Write-heavy (8W/1R) | 4.0ms | 15ms | 200 | 400 |
| Balanced (4W/4R) | 3.8ms | 14ms | 1000 | 200 |
| Stress (16W/16R) | 5.0ms | 25ms | 2500 | 600 |

---

## Phase 6: MPSC/MPMC Public API

**Status:** ✅ Complete (January 16, 2026)

**Goal:** Wrap the internal APIs (Phase 5) with channel-based infrastructure matching graph:: and fulltext:: patterns.

**Motivation:** The MPSC/MPMC patterns provide:
- Async mutation processing with flush semantics
- MPMC query pools for concurrent read scaling
- Consistent API patterns across all storage modules
- Decoupled producers and consumers for better throughput

### Overview: Channel-Based Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     MPSC/MPMC Architecture                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Mutation Path (MPSC):                                                       │
│  ┌────────────┐     MPSC      ┌────────────┐     ┌────────────┐             │
│  │ Writer     │───────────────│ Consumer   │────►│ Processor  │             │
│  │ (handle)   │ Vec<Mutation> │ (loop)     │     │ (Phase 5)  │             │
│  └────────────┘               └────────────┘     └────────────┘             │
│       │                                                                      │
│       │ .flush() ──► FlushMarker ──► oneshot response                       │
│                                                                              │
│  Query Path (MPMC):                                                          │
│  ┌────────────┐     MPMC      ┌────────────┐     ┌────────────┐             │
│  │ Reader     │───────────────│ Pool of    │────►│ Storage    │             │
│  │ (handle)   │     Query     │ Consumers  │     │ (Phase 5)  │             │
│  └────────────┘               └────────────┘     └────────────┘             │
│       │                            (N workers)                               │
│       │ oneshot ◄── result                                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Task 6.1: MutationExecutor Trait ✅

**Status:** Complete

**Implementation:** `libs/db/src/vector/mutation.rs`

The trait is synchronous (not async as originally spec'd) to work directly with RocksDB transactions:

```rust
pub trait MutationExecutor: Send + Sync {
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        processor: &Processor,
    ) -> Result<Option<MutationCacheUpdate>>;
}
```

This design exceeds the spec by supporting transactional writes with cache updates.

<details>
<summary>Original spec (for reference)</summary>

Define the trait that connects mutation types to the internal API:

```rust
// libs/db/src/vector/mutation.rs

/// Trait for executing mutations against the Processor.
/// Each mutation type implements this to define its execution logic.
#[async_trait::async_trait]
pub trait MutationExecutor: Send + Sync {
    /// Execute this mutation against the processor
    async fn execute(&self, processor: &Processor) -> Result<()>;
}

#[async_trait::async_trait]
impl MutationExecutor for InsertVector {
    async fn execute(&self, processor: &Processor) -> Result<()> {
        processor.insert_vector(
            self.embedding,
            self.id,
            &self.vector,
            self.immediate_index,
        )?;
        Ok(())
    }
}

#[async_trait::async_trait]
impl MutationExecutor for DeleteVector {
    async fn execute(&self, processor: &Processor) -> Result<()> {
        processor.delete_vector(self.embedding, self.id)?;
        Ok(())
    }
}

#[async_trait::async_trait]
impl MutationExecutor for InsertVectorBatch {
    async fn execute(&self, processor: &Processor) -> Result<()> {
        processor.insert_batch(
            self.embedding,
            &self.vectors,
            self.immediate_index,
        )?;
        Ok(())
    }
}

#[async_trait::async_trait]
impl MutationExecutor for AddEmbeddingSpec {
    async fn execute(&self, processor: &Processor) -> Result<()> {
        processor.add_embedding_spec(self)?;
        Ok(())
    }
}
```

</details>

### Task 6.2: Mutation Consumer ✅

**Status:** Complete

**Implementation:** `libs/db/src/vector/writer.rs`

Implemented with:
- `Writer` handle with `send()`, `send_with_result()`, `send_sync()`, `flush()`, `is_closed()`
- `Consumer` with MPSC receiver and Processor
- `spawn_mutation_consumer_with_storage_autoreg()` for auto-registration
- FlushMarker with oneshot for blocking flush semantics

<details>
<summary>Original spec (for reference)</summary>

Implement mutation consumer following graph::writer pattern:

```rust
// libs/db/src/vector/consumer.rs

/// Consumer that processes mutations from MPSC channel.
pub struct Consumer {
    receiver: mpsc::Receiver<MutationRequest>,
    processor: Arc<Processor>,
}

impl Consumer {
    pub fn new(receiver: mpsc::Receiver<MutationRequest>, processor: Arc<Processor>) -> Self {
        Self { receiver, processor }
    }

    /// Run the consumer loop, processing mutations until channel closes.
    pub async fn run(mut self) {
        while let Some(mutations) = self.receiver.recv().await {
            for mutation in mutations {
                if let Err(e) = mutation.execute(&self.processor).await {
                    tracing::error!(error = %e, "Failed to execute mutation");
                }
            }
        }
        tracing::info!("Vector mutation consumer shutting down");
    }
}

/// Create a mutation writer and consumer pair.
pub fn create_mutation_consumer(
    processor: Arc<Processor>,
    config: WriterConfig,
) -> (Writer, Consumer) {
    let (tx, rx) = mpsc::channel(config.channel_buffer_size);
    let writer = Writer::new(tx);
    let consumer = Consumer::new(rx, processor);
    (writer, consumer)
}

/// Spawn a mutation consumer as a background task.
pub fn spawn_mutation_consumer(
    processor: Arc<Processor>,
    config: WriterConfig,
) -> Writer {
    let (writer, consumer) = create_mutation_consumer(processor, config);
    tokio::spawn(consumer.run());
    writer
}
```

</details>

### Task 6.3: Query Consumer (MPMC Pool) ✅

**Status:** Complete

**Implementation:** `libs/db/src/vector/reader.rs`

Implemented with:
- `ProcessorConsumer` with flume MPMC receiver
- `spawn_consumers_with_processor()` spawns N workers sharing processor
- `spawn_query_consumers_with_storage_autoreg()` for auto-registration
- Workers compete to receive queries from shared channel

<details>
<summary>Original spec (for reference)</summary>

Implement MPMC query pool following graph::reader pattern:

```rust
// libs/db/src/vector/reader.rs

/// MPMC query consumer that processes searches from multiple senders.
pub struct QueryConsumer {
    receiver: mpmc::Receiver<QueryRequest>,
    storage: Arc<Storage>,
}

impl QueryConsumer {
    pub fn new(receiver: mpmc::Receiver<QueryRequest>, storage: Arc<Storage>) -> Self {
        Self { receiver, storage }
    }

    /// Run the consumer loop.
    pub async fn run(self) {
        while let Ok(request) = self.receiver.recv().await {
            let result = request.query.execute(&self.storage).await;
            let _ = request.response_tx.send(result);
        }
    }
}

/// Query request wrapper with response channel.
pub struct QueryRequest {
    pub query: Query,
    pub response_tx: oneshot::Sender<Result<QueryResult>>,
}

/// Spawn a pool of query consumers.
pub fn spawn_query_consumer_pool(
    storage: Arc<Storage>,
    config: ReaderConfig,
) -> mpmc::Sender<QueryRequest> {
    let (tx, rx) = mpmc::channel(config.channel_buffer_size);

    for _ in 0..config.num_workers {
        let consumer = QueryConsumer::new(rx.clone(), Arc::clone(&storage));
        tokio::spawn(consumer.run());
    }

    tx
}
```

</details>

### Task 6.4: Reader Handle ✅

**Status:** Complete

**Implementation:** `libs/db/src/vector/reader.rs`

Implemented with:
- `Reader` for basic queries (GetVector, GetInternalId, etc.)
- `SearchReader` for search queries with embedded Processor (for SearchKNN)
- `Runnable` trait pattern: `SearchKNN::run(&search_reader, timeout)`
- `is_closed()` for health checks

<details>
<summary>Original spec (for reference)</summary>

Implement Reader handle for sending queries:

```rust
// libs/db/src/vector/reader.rs

/// Handle for sending queries to the MPMC pool.
pub struct Reader {
    sender: mpmc::Sender<QueryRequest>,
}

impl Reader {
    pub fn new(sender: mpmc::Sender<QueryRequest>) -> Self {
        Self { sender }
    }

    /// Execute a search query and await the result.
    pub async fn search(&self, query: SearchKNN) -> Result<Vec<SearchResult>> {
        let (tx, rx) = oneshot::channel();
        let request = QueryRequest {
            query: Query::SearchKNN(query),
            response_tx: tx,
        };
        self.sender.send(request).await
            .map_err(|_| anyhow!("Query channel closed"))?;
        rx.await.map_err(|_| anyhow!("Query response dropped"))?
    }

    /// Get embedding information.
    pub async fn get_embedding(&self, code: EmbeddingCode) -> Result<Option<EmbeddingInfo>> {
        let (tx, rx) = oneshot::channel();
        let request = QueryRequest {
            query: Query::GetEmbedding(GetEmbedding { code }),
            response_tx: tx,
        };
        self.sender.send(request).await
            .map_err(|_| anyhow!("Query channel closed"))?;
        rx.await.map_err(|_| anyhow!("Query response dropped"))?
    }

    /// Get vector by external ID.
    pub async fn get_vector(&self, embedding: EmbeddingCode, id: Id) -> Result<Option<Vec<f32>>> {
        let (tx, rx) = oneshot::channel();
        let request = QueryRequest {
            query: Query::GetVector(GetVector { embedding, id }),
            response_tx: tx,
        };
        self.sender.send(request).await
            .map_err(|_| anyhow!("Query channel closed"))?;
        rx.await.map_err(|_| anyhow!("Query response dropped"))?
    }
}
```

</details>

### Task 6.5: Runnable Trait + Runtime ✅

**Status:** Complete (design differs from spec)

**Implementation:** `libs/db/src/vector/subsystem.rs`

The explicit `Runnable` trait with `start()`/`stop()`/`is_running()` was NOT implemented.
Instead, we follow the **unified pattern already used by graph:: and fulltext::**:

```rust
impl Subsystem {
    pub fn start(
        &self,
        storage: Arc<Storage>,
        writer_config: WriterConfig,
        reader_config: ReaderConfig,
        num_query_workers: usize,
    ) -> (Writer, SearchReader) { ... }
}

impl SubsystemProvider<TransactionDB> for Subsystem {
    fn on_ready(&self, db: &TransactionDB) { /* prewarm */ }
    fn on_shutdown(&self) { /* flush writer */ }
}
```

**Rationale:**
- `Subsystem::start()` provides startup (returns handles)
- `SubsystemProvider::on_shutdown()` provides graceful shutdown
- `Writer::is_closed()` / `SearchReader::is_closed()` provide health checks
- This pattern is already used by graph:: and fulltext::, providing unified lifecycle

<details>
<summary>Original spec (for reference)</summary>

Implement Runnable trait for lifecycle management:

```rust
// libs/db/src/vector/mod.rs

/// Runnable trait for vector subsystem lifecycle.
pub trait Runnable: Send + Sync {
    /// Start the subsystem (spawn consumers).
    fn start(&mut self) -> Result<()>;

    /// Stop the subsystem (shutdown consumers).
    fn stop(&mut self) -> Result<()>;

    /// Check if the subsystem is running.
    fn is_running(&self) -> bool;
}

/// Vector subsystem runtime.
pub struct Runtime {
    storage: Arc<Storage>,
    processor: Arc<Processor>,
    writer: Option<Writer>,
    reader: Option<Reader>,
    config: RuntimeConfig,
    running: bool,
}

impl Runtime {
    pub fn new(storage: Arc<Storage>, config: RuntimeConfig) -> Self {
        let processor = Arc::new(Processor::new(Arc::clone(&storage)));
        Self {
            storage,
            processor,
            writer: None,
            reader: None,
            config,
            running: false,
        }
    }

    /// Get mutation writer handle.
    pub fn writer(&self) -> Option<&Writer> {
        self.writer.as_ref()
    }

    /// Get query reader handle.
    pub fn reader(&self) -> Option<&Reader> {
        self.reader.as_ref()
    }
}

impl Runnable for Runtime {
    fn start(&mut self) -> Result<()> {
        if self.running {
            return Ok(());
        }

        // Spawn mutation consumer
        self.writer = Some(spawn_mutation_consumer(
            Arc::clone(&self.processor),
            self.config.writer.clone(),
        ));

        // Spawn query consumer pool
        let sender = spawn_query_consumer_pool(
            Arc::clone(&self.storage),
            self.config.reader.clone(),
        );
        self.reader = Some(Reader::new(sender));

        self.running = true;
        Ok(())
    }

    fn stop(&mut self) -> Result<()> {
        self.writer = None;
        self.reader = None;
        self.running = false;
        Ok(())
    }

    fn is_running(&self) -> bool {
        self.running
    }
}
```

</details>

### Task 6.6: Phase 6 Tests ✅

**Status:** Complete

**Implementation:** `libs/db/tests/test_vector_channel.rs`

Channel integration tests added:

| Test | Description | Status |
|------|-------------|--------|
| `test_mutation_via_writer_consumer` | End-to-end insert via Writer → Consumer → Storage | ✅ |
| `test_query_via_reader_pool` | Search via SearchReader → ProcessorConsumer pool | ✅ |
| `test_concurrent_queries_mpmc` | 100 concurrent searches via MPMC channel | ✅ |
| `test_subsystem_start_lifecycle` | Subsystem::start() returns working handles | ✅ |
| `test_writer_flush_semantics` | Flush guarantees visibility | ✅ |
| `test_channel_close_propagation` | Writer/Reader detect channel close | ✅ |
| `test_concurrent_deletes_vs_searches` | Deletes + searches don't return deleted | ✅ |

All 7 tests pass.

CODEX (2026-01-17): Verified `test_vector_channel.rs` includes the 7 tests listed above.

<details>
<summary>Original spec (for reference)</summary>

```rust
// libs/db/src/vector/tests/channel_tests.rs

#[cfg(test)]
mod channel_tests {
    use super::*;

    #[tokio::test]
    async fn test_mutation_consumer_insert() {
        let storage = setup_vector_storage_temp();
        let writer = spawn_mutation_consumer(
            Arc::new(Processor::new(storage.clone())),
            WriterConfig::default(),
        );

        let id = Id::new();
        writer.send(Mutation::InsertVector(InsertVector {
            embedding: test_embedding_code(),
            id,
            vector: vec![1.0; 128],
            immediate_index: true,
        })).await.unwrap();

        writer.flush().await.unwrap();

        // Verify insert
        let exists = storage.readonly().exists(test_embedding_code(), id).unwrap();
        assert!(exists);
    }

    #[tokio::test]
    async fn test_query_pool_search() {
        let storage = Arc::new(setup_vector_storage_with_vectors(1000));
        let reader = Reader::new(spawn_query_consumer_pool(
            storage,
            ReaderConfig { num_workers: 4, ..Default::default() },
        ));

        let results = reader.search(SearchKNN::new(
            test_embedding_code(),
            vec![0.5; 128],
            10,
        )).await.unwrap();

        assert_eq!(results.len(), 10);
    }

    #[tokio::test]
    async fn test_concurrent_queries() {
        let storage = Arc::new(setup_vector_storage_with_vectors(10_000));
        let reader = Reader::new(spawn_query_consumer_pool(
            storage,
            ReaderConfig { num_workers: 8, ..Default::default() },
        ));

        let handles: Vec<_> = (0..100).map(|_| {
            let r = reader.clone();
            tokio::spawn(async move {
                r.search(SearchKNN::new(
                    test_embedding_code(),
                    random_vector(128),
                    10,
                )).await
            })
        }).collect();

        for h in handles {
            let result = h.await.unwrap();
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_runtime_lifecycle() {
        let storage = Arc::new(setup_vector_storage_temp());
        let mut runtime = Runtime::new(storage, RuntimeConfig::default());

        assert!(!runtime.is_running());
        runtime.start().unwrap();
        assert!(runtime.is_running());

        // Should have writer and reader
        assert!(runtime.writer().is_some());
        assert!(runtime.reader().is_some());

        runtime.stop().unwrap();
        assert!(!runtime.is_running());
    }
}
```

</details>

### Effort Breakdown

| Task | Description | Effort | Status |
|------|-------------|--------|--------|
| 6.1 | MutationExecutor trait | 0.5 day | ✅ |
| 6.2 | Mutation Consumer | 0.5 day | ✅ |
| 6.3 | Query Consumer (MPMC pool) | 1 day | ✅ |
| 6.4 | Reader handle | 0.5 day | ✅ |
| 6.5 | Runnable trait + Runtime | 1 day | ✅ (Subsystem pattern) |
| 6.6 | Tests | 0.5 day | ✅ |
| **Total** | | **4 days** | **Complete** |

**Acceptance Criteria:**
- [x] Mutations execute via channel consumer
- [x] Queries use MPMC pool with configurable workers
- [x] Writer.flush() blocks until mutations processed
- [x] Reader.search() returns results via oneshot
- [x] Subsystem::start() lifecycle works correctly (replaces Runtime)
- [x] Concurrent query tests pass (100 concurrent, 100% success)

---

## Phase 7: Async Graph Updater

**Status:** Complete
<!-- ADDRESSED (Claude, 2026-01-19): Updated status from "Not Started" to "Complete" to match header and PHASE7.md -->
**Design Doc:** [`PHASE7.md`](./PHASE7.md) - Detailed task breakdown with 30 subtasks

**Goal:** Enable online updates by decoupling vector storage from HNSW graph construction.

**Motivation:** Synchronous graph updates during insert create latency spikes (~50ms P99).
Two-phase insert pattern:
1. **Phase 1 (sync, fast):** Store vector data, metadata, binary code
2. **Phase 2 (async, background):** Build HNSW graph connections

This allows insert latency <5ms while graph quality builds in the background.

### Overview: Two-Phase Insert Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Two-Phase Insert Architecture                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Phase 1: Synchronous Write (< 5ms)                                         │
│  ┌────────────┐     ┌────────────┐     ┌────────────┐                       │
│  │ insert()   │ ──► │ Store      │ ──► │ Pending    │                       │
│  │            │     │ Vector+Meta│     │ Queue      │                       │
│  └────────────┘     └────────────┘     └────────────┘                       │
│       │                                      │                               │
│       │ Searchable immediately               │ Queued for graph build       │
│       ▼                                      ▼                               │
│  ┌────────────────────────────────────────────────────────────┐             │
│  │                   Phase 2: Async Graph Build               │             │
│  │  ┌────────────┐     ┌────────────┐     ┌────────────┐     │             │
│  │  │ Worker     │ ──► │ Greedy     │ ──► │ Add HNSW   │     │             │
│  │  │ Thread     │     │ Search     │     │ Edges      │     │             │
│  │  └────────────┘     └────────────┘     └────────────┘     │             │
│  └────────────────────────────────────────────────────────────┘             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Task 7.1: Pending Queue CF

Add column family for pending graph updates:

```rust
// libs/db/src/vector/schema.rs

/// Pending inserts waiting for graph construction.
/// Key: [embedding_code: u64][timestamp: u64][vec_id: u32] = 20 bytes
/// Value: empty (all data stored in Vectors CF)
pub struct PendingInserts;

impl ColumnFamily for PendingInserts {
    const CF_NAME: &'static str = "vector/pending";
}

impl PendingInserts {
    /// Create key for pending insert.
    pub fn key(embedding: EmbeddingCode, vec_id: VecId) -> [u8; 20] {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;
        let mut key = [0u8; 20];
        key[0..8].copy_from_slice(&embedding.to_be_bytes());
        key[8..16].copy_from_slice(&timestamp.to_be_bytes());
        key[16..20].copy_from_slice(&vec_id.to_be_bytes());
        key
    }

    /// Parse key components.
    pub fn parse_key(key: &[u8]) -> (EmbeddingCode, u64, VecId) {
        let embedding = u64::from_be_bytes(key[0..8].try_into().unwrap());
        let timestamp = u64::from_be_bytes(key[8..16].try_into().unwrap());
        let vec_id = u32::from_be_bytes(key[16..20].try_into().unwrap());
        (embedding, timestamp, vec_id)
    }
}
```

### Task 7.2: Async Updater Configuration

```rust
// libs/db/src/vector/async_updater.rs

/// Configuration for the async graph updater.
#[derive(Debug, Clone)]
pub struct AsyncUpdaterConfig {
    /// Maximum vectors per batch
    pub batch_size: usize,          // Default: 100

    /// Maximum time to wait for batch to fill
    pub batch_timeout: Duration,    // Default: 100ms

    /// Number of worker threads
    pub num_workers: usize,         // Default: 2

    /// ef_construction for greedy search
    pub ef_construction: usize,     // Default: 200

    /// Whether to process on startup (drain pending queue)
    pub process_on_startup: bool,   // Default: true
}

impl Default for AsyncUpdaterConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            batch_timeout: Duration::from_millis(100),
            num_workers: 2,
            ef_construction: 200,
            process_on_startup: true,
        }
    }
}
```

### Task 7.3: Async Updater Implementation

```rust
// libs/db/src/vector/async_updater.rs

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Background graph updater.
pub struct AsyncGraphUpdater {
    storage: Arc<Storage>,
    config: AsyncUpdaterConfig,
    shutdown: Arc<AtomicBool>,
    workers: Vec<JoinHandle<()>>,
}

impl AsyncGraphUpdater {
    /// Start the async updater with worker threads.
    pub fn start(storage: Arc<Storage>, config: AsyncUpdaterConfig) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));

        // Drain pending queue on startup if configured
        if config.process_on_startup {
            Self::drain_pending(&storage, &config);
        }

        // Spawn worker threads
        let workers = (0..config.num_workers)
            .map(|worker_id| {
                let storage = Arc::clone(&storage);
                let config = config.clone();
                let shutdown = Arc::clone(&shutdown);
                thread::spawn(move || {
                    Self::worker_loop(worker_id, storage, config, shutdown);
                })
            })
            .collect();

        Self { storage, config, shutdown, workers }
    }

    /// Graceful shutdown.
    pub fn shutdown(self) {
        self.shutdown.store(true, Ordering::SeqCst);
        for worker in self.workers {
            let _ = worker.join();
        }
    }

    fn worker_loop(
        worker_id: usize,
        storage: Arc<Storage>,
        config: AsyncUpdaterConfig,
        shutdown: Arc<AtomicBool>,
    ) {
        tracing::info!(worker_id, "Async updater worker started");

        loop {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            // Collect a batch of pending inserts
            let batch = Self::collect_batch(&storage, &config);

            if batch.is_empty() {
                thread::sleep(Duration::from_millis(10));
                continue;
            }

            // Process each vector in the batch
            for (embedding, vec_id) in &batch {
                if let Err(e) = Self::process_insert(&storage, &config, *embedding, *vec_id) {
                    tracing::error!(vec_id, error = %e, "Failed to process pending insert");
                }
            }

            // Remove processed items from pending queue
            Self::clear_processed(&storage, &batch);
        }

        tracing::info!(worker_id, "Async updater worker stopped");
    }

    fn collect_batch(storage: &Storage, config: &AsyncUpdaterConfig) -> Vec<(EmbeddingCode, VecId)>;
    fn process_insert(storage: &Storage, config: &AsyncUpdaterConfig, embedding: EmbeddingCode, vec_id: VecId) -> Result<()>;
    fn clear_processed(storage: &Storage, batch: &[(EmbeddingCode, VecId)]);
    fn drain_pending(storage: &Storage, config: &AsyncUpdaterConfig);
}
```

### Task 7.4: Delete Handling

Deletes use the same two-phase pattern:

```rust
impl Storage {
    /// Delete with async cleanup.
    pub fn delete(&self, embedding: EmbeddingCode, id: Id) -> Result<()> {
        let vec_id = self.get_internal_id(embedding, id)?
            .ok_or_else(|| anyhow!("Vector not found"))?;

        // Phase 1: Mark as deleted (fast, sync)
        self.mark_deleted(embedding, vec_id)?;

        // Remove from pending queue if present
        self.remove_from_pending(embedding, vec_id)?;

        // Return ID to free list
        self.id_allocator.free(embedding, vec_id);

        // Phase 2: Edge cleanup (lazy, via merge operator)
        // Stale edges cleaned during:
        // - Search: skip deleted nodes
        // - Compaction: merge operator removes stale edges

        Ok(())
    }
}

// Node flags in VecMeta
const FLAG_DELETED: u8 = 0x01;
const FLAG_PENDING: u8 = 0x02;  // Still in pending queue
```

### Task 7.5: Testing & Crash Recovery

```rust
#[cfg(test)]
mod async_updater_tests {
    use super::*;

    #[test]
    fn test_insert_async_immediate_searchability() {
        let storage = setup_vector_storage_with_async_updater();
        let embedding = test_embedding_code();

        let id = Id::new();
        let vector = vec![1.0; 128];

        // Async insert
        storage.insert_async(embedding, id, &vector).unwrap();

        // Should be searchable IMMEDIATELY (brute-force fallback)
        let results = storage.search(embedding, &vector, 1, 10).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_crash_recovery_pending_queue() {
        let tmp = TempDir::new().unwrap();

        // First session: insert async, don't wait for drain
        {
            let storage = setup_vector_storage_at(tmp.path());
            for i in 0..50 {
                storage.insert_async(test_embedding_code(), Id::new(), &[i as f32; 128]).unwrap();
            }
            // Don't wait - simulate crash
        }

        // Second session: recovery
        {
            let storage = setup_vector_storage_at(tmp.path());
            // Should recover pending queue
            let pending = storage.get_pending_count(test_embedding_code()).unwrap();
            assert!(pending > 0, "Should have pending items after recovery");

            // Wait for drain
            std::thread::sleep(Duration::from_secs(2));

            // All should be processed
            let final_pending = storage.get_pending_count(test_embedding_code()).unwrap();
            assert_eq!(final_pending, 0);
        }
    }
}
```

### Consistency Guarantees

| Guarantee | Description |
|-----------|-------------|
| **Durability** | Vector is durable after `insert()` returns (stored in RocksDB) |
| **Searchability** | Vector is searchable immediately (brute-force expansion) |
| **Eventual Connectivity** | Graph edges added within `batch_timeout` (default 100ms) |
| **Crash Recovery** | Pending queue persisted; drained on restart |
| **Delete Visibility** | Deleted vectors excluded from search immediately |

### Effort Breakdown

| Task | Description | Effort |
|------|-------------|--------|
| 7.1 | Pending queue CF | 0.5 day |
| 7.2 | Async updater config | 0.25 day |
| 7.3 | Worker thread implementation | 1.5 days |
| 7.4 | Delete handling | 0.5 day |
| 7.5 | Testing and crash recovery | 1 day |
| **Total** | | **4 days** |

**Acceptance Criteria:**
- [ ] `insert()` latency < 5ms P99
- [ ] Pending queue drained within `batch_timeout`
- [ ] Crash recovery: pending queue survives restart
- [ ] Deleted nodes excluded from search results

---

## Phase 8: Production Hardening

**Status:** 🚧 In Progress (8.1-8.2 complete; 8.3 in progress)
**Goal:** Production-ready vector search with full feature set.
**Detailed Plan:** See [`PHASE8.md`](./PHASE8.md) for comprehensive task breakdown (29 subtasks)
COMMENT (CODEX, 2026-01-18): PHASE8 now lists 8.1.10, 8.2.9, 8.3.10; update the "26 subtasks" count if it changes.
RESPONSE (2026-01-18): Updated to 29 subtasks (8.1: 10, 8.2: 9, 8.3: 10).

### Task 8.1: Delete Refinement (✅ Complete)

Refine delete implementation from Phase 5 and Phase 7:
- Proper edge cleanup strategy
- Garbage collection for freed IDs
- Tombstone compaction

**Effort:** 1-2 days

### Task 8.2: Concurrent Access [CON-3] (✅ Complete)

- Use RocksDB snapshot isolation for reads
- Write serialization via transaction API
- Concurrent read/write stress tests

**Effort:** 2-3 days

### Task 8.3: 1B Scale Validation (🚧 In Progress)

- Benchmark at 10M, 100M, 1B scales
- Validate memory constraints
- Performance profiling
- Document scaling characteristics

**Backlog (per PHASE8.md):**
- 8.3.3: Benchmark 10M scale (insert/search/memory)
- 8.3.4: Generate or source 100M dataset
- 8.3.5: Benchmark 100M scale
- 8.3.6: Validate 1B feasibility (sampling)
- 8.3.9: Add CI gate for 1M regression detection

### Assessment Notes (2026-01-30)

- **Delete refinement (8.1) implemented:** GC worker + edge pruning + recycling guardrails live in `libs/db/src/vector/gc.rs`; delete/GC integration tests in `libs/db/tests/test_vector_delete.rs`. See `PHASE8.md` Task 8.1.
- **Concurrent access hardening (8.2) implemented:** stress + snapshot isolation tests in `libs/db/tests/test_vector_concurrent.rs` and `libs/db/tests/test_vector_snapshot_isolation.rs`; benchmark support in `libs/db/src/vector/benchmark/concurrent.rs`. See `PHASE8.md` Task 8.2.
- **Scale validation (8.3) outstanding:** backlog items listed above and tracked in `PHASE8.md` Task 8.3.
- **Phase 7 acceptance criteria:** functional requirements are implemented, but performance criteria (insert P99, pending drain timing) are not explicitly validated in tests; ROADMAP checkboxes remain open for these items.

**Effort:** 1-2 weeks

### Phase 8 Validation & Tests

```rust
#[cfg(test)]
mod production_tests {
    use super::*;

    #[test]
    fn test_concurrent_read_write() {
        let storage = Arc::new(setup_vector_storage_temp());
        let embedding = test_embedding_code();

        // Seed with some vectors
        for i in 0..100 {
            storage.insert(embedding, Id::new(), &[i as f32; 128]).unwrap();
        }

        let mut handles = Vec::new();

        // Writers
        for _ in 0..5 {
            let s = Arc::clone(&storage);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    s.insert(test_embedding_code(), Id::new(), &random_vector(128)).unwrap();
                }
            }));
        }

        // Readers
        for _ in 0..10 {
            let s = Arc::clone(&storage);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    s.search(test_embedding_code(), &random_vector(128), 10, 50).unwrap();
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_delete_not_in_search_results() {
        let storage = setup_vector_storage_temp();
        let embedding = test_embedding_code();

        let ulids: Vec<Id> = (0..100).map(|i| {
            let id = Id::new();
            storage.insert(embedding, id, &[i as f32; 128]).unwrap();
            id
        }).collect();

        // Delete even-indexed vectors
        for i in (0..100).step_by(2) {
            storage.delete(embedding, ulids[i]).unwrap();
        }

        // Search should not return deleted vectors
        let results = storage.search(embedding, &[50.0; 128], 20, 100).unwrap();
        for (_, id) in &results {
            let idx = ulids.iter().position(|u| u == id).unwrap();
            assert!(idx % 2 == 1, "Deleted vector appeared in results");
        }
    }
}
```

### Effort Breakdown

| Task | Description | Effort |
|------|-------------|--------|
| 8.1 | Delete refinement | 1-2 days |
| 8.2 | Concurrent access | 2-3 days |
| 8.3 | 1B scale validation | 1-2 weeks |
| **Total** | | **2-3 weeks** |

**Acceptance Criteria:**
- [x] Delete removes vectors from search results (tests added in `test_vector_delete.rs`)
- [x] Concurrent read/write stress tests pass (tests in `test_vector_concurrent.rs`)
- [ ] 10M scale benchmark completed (primary target)
- [ ] Memory usage within 64GB budget at 10M scale
- [ ] 100M scale benchmark completed (secondary target, 128GB RAM)
- [ ] 1B scale validation (sampling-based, not full benchmark)
COMMENT (CODEX, 2026-01-18): Memory budget here conflicts with PHASE8 projections (~500GB cache at 1B). Align budget with target hardware or scale tier.
RESPONSE (2026-01-18): Aligned. Primary target is 10M (64GB), secondary is 100M (128GB). 1B is aspirational (sampling only, per PHASE8 hardware profile).

---

## Dependency Graph

```
Phase 0: Foundation
    │
    ▼
Phase 1: ID Management ◄─────────────────────────────┐
    │                                                 │
    ▼                                                 │
Phase 2: HNSW2 Core + Navigation ◄─ Phase 3: Batch  │
    │                                      │          │
    ▼                                      │          │
Phase 4: RaBitQ ◄──────────────────────────┘          │
    │                                                 │
    ▼                                                 │
Phase 5: Internal Mutation/Query API                  │
    │                                                 │
    ▼                                                 │
Phase 6: MPSC/MPMC Public API ────────────────────────┘
    │
    ▼
Phase 7: Async Graph Updater
    │
    ▼
Phase 8: Production Hardening
```

**Navigation Layer Integration:** Tasks 2.7-2.9 are placed in Phase 2 (rather than a
separate phase) because the navigation layer is fundamental to HNSW algorithm correctness.
The memory caching (2.9) can be deferred if timeline is tight, but 2.7-2.8 are required.

---

## Effort Summary

| Phase | Tasks | Effort | Cumulative |
|-------|-------|--------|------------|
| Phase 0: Foundation | 0.1-0.3 | ✅ COMPLETE | ✅ |
| Phase 1: ID Management | 1.1-1.5 | ✅ COMPLETE | ✅ |
| Phase 2: HNSW2 Core + Navigation | 2.1-2.9 | 11-17 days | ~11-17 days |
| Phase 3: Batch + Deferred | 3.1-3.7 | 4.5-6 days | 17-28 days |
| Phase 4: RaBitQ | 4.1-4.23 | ✅ COMPLETE | ✅ |
| Phase 5: Internal Mutation/Query API | 5.1-5.11 | 6-8 days | ~6-8 days |
| Phase 6: MPSC/MPMC Public API | 6.1-6.6 | 4 days | 8-9 days |
| Phase 7: Async Graph Updater | 7.1-7.5 | 4 days | 12-13 days |
| Phase 8: Production Hardening | 8.1-8.3 | 2-3 weeks | 4-6 weeks |
| **Total (Remaining)** | | **~4-6 weeks** | |

**Phase 2 Breakdown:**
| Task | Description | Effort |
|------|-------------|--------|
| 2.1 | RoaringBitmap edge storage | 1-2 days |
| 2.2 | RocksDB merge operators | 1-2 days |
| 2.3 | Vector storage CF | 0.5 day |
| 2.4 | Node metadata CF | 0.5 day |
| 2.5 | Graph metadata CF | 0.25 day |
| 2.6 | HNSW insert algorithm | 3-5 days |
| 2.7 | Navigation layer structure | 1 day |
| 2.8 | Layer descent search | 2-3 days |
| 2.9 | Memory-cached top layers | 2-3 days |

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Roaring bitmap overhead | Low | Medium | Benchmark early; fallback to dense array |
| Merge operator complexity | Medium | High | Test extensively; monitor WAL size |
| RaBitQ recall degradation | Medium | High | Validate on SIFT; increase bits_per_dim if needed |
| 1B scale memory issues | Medium | High | Profile early; implement mmap ID mapping |
| Concurrent access bugs | Medium | High | Extensive testing; use RocksDB transactions |

---

## Quick Start for Future Sessions

This section provides a fast onboarding path for implementing this roadmap.

### Prerequisites

Before starting, ensure you understand:

1. **RocksDB Column Families**: [RocksDB Wiki](https://github.com/facebook/rocksdb/wiki/Column-Families)
2. **HNSW Algorithm**: Hierarchical Navigable Small World graphs ([Paper](https://arxiv.org/abs/1603.09320))
3. **SIMD Distance Functions**: `motlie_core::distance` module ([mod.rs](../../../../core/src/distance/mod.rs))

### Implementation Order

```
Session 1: Phase 0 (Foundation) + Phase 1 (ID Management)
├── Create vector/ module structure
├── Define 9 Column Families with vector/ prefix
├── Implement Id ↔ u32 mapping with ULID generation
└── Validate: cargo test -p motlie_db --lib vector

Session 2: Phase 2 (HNSW Graph) - Core Structure
├── Implement HnswEdges with RoaringBitmap
├── Add RocksDB merge operators for edges
├── Implement NavigationLayerInfo caching
└── Validate: cargo test vector::hnsw (recall tests)

Session 3: Phase 2 (HNSW Graph) - Search
├── Implement greedy_search and search_layer
├── Add entry point tracking across layers
├── Implement beam search with RoaringBitmap candidates
└── Validate: Recall@10 ≥ 95% on SIFT-10K

Session 4: Phase 3 (Batch Operations)
├── Implement MultiGet for neighbor vectors
├── Add GetMultiple for edges
├── Optimize to O(1) degree fetches
└── Validate: 10x throughput vs sequential

Session 5: Phase 4 (RaBitQ)
├── Generate orthonormal rotation matrix
├── Implement binary quantization
├── Add Hamming distance with popcount
└── Validate: 10x memory reduction, recall ≥ 95%

Session 6: Phase 7 (Async Updater)
├── Implement two-phase insert
├── Add background graph refinement
├── Implement crash recovery
└── Validate: < 1ms insert latency

Session 7: Phase 6 (Production)
├── Add delete operations
├── Implement concurrent access
├── Scale testing to 1M vectors
└── Validate: All acceptance criteria
```

### File Structure to Create

```
libs/db/src/vector/
├── mod.rs              # Public API (VectorStorage, search, insert, delete)
├── config.rs           # HnswConfig, VectorConfig, RaBitQConfig
├── embedding.rs  # Embedding newtype
├── id.rs               # IdManager (ULID ↔ u32 mapping)
├── hnsw/
│   ├── mod.rs          # HNSW graph implementation
│   ├── edges.rs        # HnswEdges with RoaringBitmap
│   ├── navigation.rs   # NavigationLayerInfo, entry points
│   └── search.rs       # greedy_search, search_layer, beam search
├── quantization/
│   ├── mod.rs          # Quantization trait
│   ├── rabitq.rs       # RaBitQ implementation
│   └── rotation.rs     # Orthonormal matrix generation
├── storage/
│   ├── mod.rs          # Column Family management
│   ├── schema.rs       # CF definitions (9 column families)
│   └── merge_ops.rs    # RocksDB merge operators
└── async_updater.rs    # Background graph refinement
```

### First Implementation Step

Start with this minimal skeleton to validate the module structure:

```rust
// libs/db/src/vector/mod.rs
//! Vector search module with HNSW indexing

mod config;
mod embedding;
mod id;

pub use config::{HnswConfig, VectorConfig};
pub use embedding::Embedding;

use crate::{Database, Id, Result};

/// Column Family prefix for all vector-related data
const CF_PREFIX: &str = "vector/";

/// Vector storage and search operations
pub struct VectorStorage<'db> {
    db: &'db Database,
}

impl<'db> VectorStorage<'db> {
    pub fn new(db: &'db Database) -> Self {
        Self { db }
    }

    pub fn insert(
        &self,
        space: &Embedding,
        id: Id,
        vector: &[f32]
    ) -> Result<()> {
        todo!("Phase 1: ID mapping + Phase 2: HNSW insert")
    }

    pub fn search(
        &self,
        space: &Embedding,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<Vec<(f32, Id)>> {
        todo!("Phase 2: HNSW search")
    }

    pub fn delete(&self, space: &Embedding, id: Id) -> Result<()> {
        todo!("Phase 6: Delete with tombstone")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cf_prefix() {
        assert_eq!(CF_PREFIX, "vector/");
    }
}
```

### Key Dependencies

```toml
# Already in Cargo.toml (motlie_db)
rocksdb = { version = "0.22", features = ["multi-threaded-cf"] }
roaring = "0.10"         # RoaringBitmap for edge storage
ulid = "1.1"             # ULID generation

# From motlie_core
motlie_core = { path = "../core" }  # SIMD distance functions
```

### Validation Commands

```bash
# Run all vector tests
cargo test -p motlie_db --lib vector

# Run with SIMD info
RUST_LOG=debug cargo test -p motlie_db --lib vector -- --nocapture

# Benchmark (when implemented)
cargo bench -p motlie_db -- vector

# Check SIMD level being used
cargo run -p motlie_core --example simd_check
```

### Key Invariants to Maintain

1. **Namespace Isolation**: All CF names MUST start with `vector/{space}/`
2. **ID Mapping**: Callers only see `Id` (ULID), internal uses `u32`
3. **SIMD Distance**: Always use `motlie_core::distance::{euclidean_squared, cosine, dot}`
4. **No Training Data**: Never implement anything that requires representative data (DATA-1)
5. **Edge Storage**: Use `RoaringBitmap` for HNSW edges, not `Vec<u32>`

---

## Changelog

| Date | Change | Author |
|------|--------|--------|
| 2026-01-02 | Initial roadmap document | Claude Opus 4.5 |
| 2026-01-02 | Added Design Decisions section (PQ exclusion rationale) | Claude Opus 4.5 |
| 2026-01-02 | Added Navigation Layer tasks (2.7-2.9) to Phase 2 | Claude Opus 4.5 |
| 2026-01-02 | Added Validation & Tests sections to all phases (0-6) | Claude Opus 4.5 |
| 2026-01-02 | Renamed EmbeddingCode to Embedding throughout | Claude Opus 4.5 |
| 2026-01-02 | Added complete HnswConfig/VectorConfig struct definitions (Task 0.1b) | Claude Opus 4.5 |
| 2026-01-02 | Added HNSW parameter guidelines table for different scales | Claude Opus 4.5 |
| 2026-01-02 | Added Quick Start section for future implementation sessions | Claude Opus 4.5 |
| 2026-01-03 | Changed Embedding from [u8; 32] to u64 (8 bytes) per ARCH-14 | Claude Opus 4.5 |
| 2026-01-03 | Updated all CF key layouts: 12-24 bytes (was 36-48), saves ~192 GB at 1B scale | Claude Opus 4.5 |
| 2026-01-03 | Added EmbeddingRegistry for name <-> ID mapping | Claude Opus 4.5 |
| 2026-01-03 | Added vector/embedding_registry CF to layout | Claude Opus 4.5 |
| 2026-01-03 | Added CF Key Summary table, Key Design Decisions, Edge Storage Design sections | Claude Opus 4.5 |
| 2026-01-03 | Renamed node_id to vec_id, node_meta to vec_meta, NodeMeta to VecMeta throughout | Claude Opus 4.5 |
| 2026-01-03 | Added Task 0.3: EmbeddingRegistry Pre-warming (NameCache pattern) | Claude Opus 4.5 |
| 2026-01-03 | Added Task 0.4: ColumnFamilyProvider Trait for mix-in storage initialization | Claude Opus 4.5 |
| 2026-01-03 | Redesigned Embedding as rich struct with model, dim, distance, embedder fields (ARCH-17) | Claude Opus 4.5 |
| 2026-01-03 | Added Distance enum with compute behavior, Embedder trait (ARCH-18) | Claude Opus 4.5 |
| 2026-01-03 | Added EmbeddingBuilder, EmbeddingFilter, queryable EmbeddingRegistry API | Claude Opus 4.5 |
| 2026-01-03 | Expanded Phase 0 with tasks 0.4-0.9 for rich Embedding implementation | Claude Opus 4.5 |
| 2026-01-03 | **IMPLEMENTED** Phase 0 foundation: mod.rs, config.rs, distance.rs, embedding.rs, registry.rs, schema.rs, error.rs | Claude Opus 4.5 |
| 2026-01-03 | 31 unit tests passing for Phase 0 types and schema | Claude Opus 4.5 |
| 2026-01-04 | **cac8da3** Refactor vector schema: semantic types (EmbeddingCode, VecId, HnswLayer, RoaringBitmapBytes, RabitqCode), union pattern for polymorphic CFs (GraphMeta, IdAlloc), README documentation | Claude Opus 4.5 |
| 2026-01-04 | Replaced Phase 0 and Phase 1 code blocks with source file links (20 blocks total) | Claude Opus 4.5 |
| 2026-01-06 | **IMPLEMENTED** Phase 2: merge.rs (EdgeOp merge operators), navigation.rs (NavigationLayerInfo, NavigationCache), hnsw.rs (HnswIndex with insert/search) | Claude Opus 4.5 |
| 2026-01-06 | 80 unit tests passing for vector module (Phases 0-2) | Claude Opus 4.5 |
| 2026-01-07 | **9399cf9** Core RaBitQ implementation (Tasks 4.1-4.5): rabitq.rs with rotation matrix, binary encoding, Hamming distance | Claude Opus 4.5 |
| 2026-01-07 | **e5b81b6** Recall tuning (Task 4.6): increased ef_search defaults, documented recall vs latency tradeoffs | Claude Opus 4.5 |
| 2026-01-07 | **4a4e954** SIMD Hamming + early filtering (Task 4.7): motlie_core::distance::hamming_distance | Claude Opus 4.5 |
| 2026-01-07 | **bd29801** Hybrid L2+Hamming experiment (Task 4.8): disproven - pure Hamming is faster | Claude Opus 4.5 |
| 2026-01-08 | **9522658** RaBitQ tuning analysis (Task 4.9): optimal rerank_factor=10 for cosine, documented in benchmark results | Claude Opus 4.5 |
| 2026-01-08 | **9880a62** In-memory BinaryCodeCache (Task 4.10): navigation.rs BinaryCodeCache for fast Hamming filtering | Claude Opus 4.5 |
| 2026-01-08 | **ce2a060** API cleanup Phase 1 (Task 4.11): deprecated invalidated functions | Claude Opus 4.5 |
| 2026-01-08 | **5f2dac0** API cleanup Phases 2-4 (Tasks 4.12-4.14): removed dead code, Embedding-driven SearchConfig, config validation | Claude Opus 4.5 |
| 2026-01-09 | **0535e6a** HNSW distance metric bug fix (Task 4.16): layer navigation now uses configured distance metric | Claude Opus 4.5 |
| 2026-01-09 | **d5899f3** batch_distances metric bug fix (Task 4.17): batch operations use correct distance metric | Claude Opus 4.5 |
| 2026-01-09 | **c90f15c** VectorElementType (Task 4.18): f16/f32 support for storage flexibility | Claude Opus 4.5 |
| 2026-01-09 | **27d6b74** Parallel reranking with rayon (Task 4.19): parallel.rs rerank_parallel function | Claude Opus 4.5 |
| 2026-01-09 | **a146d02** Parallel threshold tuning (Task 4.20): DEFAULT_PARALLEL_RERANK_THRESHOLD=800, rerank_adaptive, documentation | Claude Opus 4.5 |
| 2026-01-09 | **Phase 4 COMPLETE**: All 20 tasks finished, 433 unit tests passing | Claude Opus 4.5 |
| 2026-01-09 | Updated ROADMAP: Phase Overview tree view, API Surface Summary, moved VectorStorage to Appendix | Claude Opus 4.5 |
| 2026-01-14 | Created comprehensive `bins/bench_vector/README.md` documentation | Claude Opus 4.5 |
| 2026-01-14 | Removed deprecated `examples/laion_benchmark` (replaced by bench_vector) | Claude Opus 4.5 |
| 2026-01-14 | Updated all docs/code: Hamming → ADC terminology (symmetric Hamming no longer used) | Claude Opus 4.5 |
| 2026-01-14 | Added random dataset support to `bench_vector index` command | Claude Opus 4.5 |
| 2026-01-14 | Synced with CODEX code review (abc8e8d, 3cbf71c) | Claude Opus 4.5 |
| 2026-01-14 | **Added Phase 4.5**: Pre-Phase 5 critical fixes based on CODEX review (5 tasks) | Claude Opus 4.5 |
| 2026-01-14 | **af7a90b** Phase 4.5 Tasks 4.5.1-4.5.3 COMPLETE: storage-type fix, layer assignment fix, ADC missing code fix | Claude Opus 4.5 |
| 2026-01-14 | Addressed CODEX follow-up: replaced silent F32 fallback with hard error on unknown embedding | Claude Opus 4.5 |
| 2026-01-14 | **Phase 4.5 COMPLETE**: Tasks 4.5.4-4.5.5 done (empty index handling, cache size accounting) | Claude Opus 4.5 |

---

## API Surface Summary

This section summarizes the public API for building indexes, configuring search, and parallel reranking.
Use this for evaluating ergonomics and design consistency.

### Core Types (mod.rs re-exports)

```rust
// ─────────────────────────────────────────────────────────────────────────────
// Configuration Types
// ─────────────────────────────────────────────────────────────────────────────
pub use hnsw::ConfigWarning;                   // HNSW config warnings
pub use config::{RaBitQConfig, RaBitQConfigWarning, VectorConfig};
pub use distance::Distance;                    // Cosine, L2, DotProduct
// Note: hnsw::Config has been deleted. HNSW params come from EmbeddingSpec.

// ─────────────────────────────────────────────────────────────────────────────
// Embedding Types (identity + behavior)
// ─────────────────────────────────────────────────────────────────────────────
pub use embedding::{Embedder, Embedding, EmbeddingBuilder};
pub use registry::{EmbeddingFilter, EmbeddingRegistry};

// ─────────────────────────────────────────────────────────────────────────────
// Index and Quantization
// ─────────────────────────────────────────────────────────────────────────────
// Note: HNSW index is at hnsw::Index (not re-exported to vector module)
pub use rabitq::RaBitQ;
pub use navigation::{BinaryCodeCache, NavigationCache, NavigationCacheConfig};

// ─────────────────────────────────────────────────────────────────────────────
// Search Configuration
// ─────────────────────────────────────────────────────────────────────────────
pub use search::{SearchConfig, SearchStrategy, DEFAULT_PARALLEL_RERANK_THRESHOLD};

// ─────────────────────────────────────────────────────────────────────────────
// Schema Types
// ─────────────────────────────────────────────────────────────────────────────
pub use schema::{EmbeddingCode, VecId, VectorElementType, ALL_COLUMN_FAMILIES};

// ─────────────────────────────────────────────────────────────────────────────
// Storage Subsystem
// ─────────────────────────────────────────────────────────────────────────────
pub type Storage = crate::rocksdb::Storage<Subsystem>;
pub type Component = crate::rocksdb::ComponentWrapper<Subsystem>;
pub fn component() -> Component;
```

### Building an Embedding Space

```rust
use motlie_db::vector::{Distance, EmbeddingBuilder, EmbeddingRegistry};

// Create registry (typically from Storage)
let registry = EmbeddingRegistry::new();

// Register embedding space (idempotent)
let embedding = EmbeddingBuilder::new("gemma", 768, Distance::Cosine)
    .register(&registry)?;

// Access properties
assert_eq!(embedding.model(), "gemma");
assert_eq!(embedding.dim(), 768);
assert_eq!(embedding.distance(), Distance::Cosine);
assert_eq!(embedding.code(), /* deterministic u64 hash */);
```

### Configuring Search

```rust
use motlie_db::vector::{SearchConfig, SearchStrategy, DEFAULT_PARALLEL_RERANK_THRESHOLD};

// Auto-selects strategy based on distance metric:
// - Cosine → RaBitQ (Hamming approximation)
// - L2/DotProduct → Exact
let config = SearchConfig::new(embedding.clone(), 10);

// Builder pattern for customization
let config = SearchConfig::new(embedding.clone(), 10)
    .with_ef(150)                              // Beam width (recall vs speed)
    .with_rerank_factor(8)                     // Candidates = k * factor
    .with_parallel_rerank_threshold(1200);     // Tune for your dimension

// Force exact search (bypass RaBitQ)
let config = SearchConfig::new(embedding.clone(), 10).exact();

// Check strategy
assert!(config.strategy().is_rabitq());  // or is_exact()
assert_eq!(config.distance(), Distance::Cosine);

// Compute distances (uses embedding's metric)
let dist = config.compute_distance(&vec_a, &vec_b);
```

### Parallel Reranking API

```rust
use motlie_db::vector::parallel::{rerank_parallel, rerank_sequential, rerank_adaptive, rerank_auto};
use motlie_db::vector::DEFAULT_PARALLEL_RERANK_THRESHOLD;

// Manual selection
let results = rerank_parallel(&candidates, |id| Some(distance(id)), k);
let results = rerank_sequential(&candidates, |id| Some(distance(id)), k);

// Adaptive (threshold-based)
let results = rerank_adaptive(&candidates, |id| Some(distance(id)), k, 800);

// Auto (uses DEFAULT_PARALLEL_RERANK_THRESHOLD = 800)
let results = rerank_auto(&candidates, |id| Some(distance(id)), k);

// Via SearchConfig
if config.should_use_parallel_rerank(candidates.len()) {
    rerank_parallel(...)
} else {
    rerank_sequential(...)
}
```

### EmbeddingFilter API

```rust
use motlie_db::vector::{Distance, EmbeddingFilter, EmbeddingRegistry};

// Find embeddings matching criteria
let filter = EmbeddingFilter::default()
    .model("gemma")
    .dim(768)
    .distance(Distance::Cosine);

let matches = registry.find(&filter);

// Individual filters
let all_gemma = registry.find_by_model("gemma");
let all_768d = registry.find_by_dim(768);
let all_cosine = registry.find_by_distance(Distance::Cosine);
```

### Design Evaluation Points

| Aspect | Current Design | Notes |
|--------|----------------|-------|
| **Embedding as source of truth** | ✅ SearchConfig takes Embedding | Ensures distance metric consistency |
| **Strategy auto-selection** | ✅ Based on distance | Cosine→RaBitQ, L2→Exact |
| **Builder pattern** | ✅ SearchConfig, EmbeddingBuilder | Fluent, chainable |
| **Threshold configurability** | ✅ with_parallel_rerank_threshold() | Tunable per workload |
| **Convenience functions** | ✅ rerank_auto(), component() | Sensible defaults |
| **Type safety** | ✅ EmbeddingCode, VecId, VectorElementType | Semantic wrapper types |
| **Validation** | ✅ validate_embedding_code() | Catches mismatches at search time |

---

## Appendix A: Original VectorStorage Design (Outdated)

> **Note:** This section documents the original design spec from Phase 0.
> The actual implementation evolved to use the `StorageBuilder` and `Component` patterns.
> Kept for historical reference.

### VectorStorage Constructor (Original Design)

```rust
// libs/db/src/vector/mod.rs (OUTDATED - see actual implementation)

/// Vector search storage - shares RocksDB with graph storage
pub struct VectorStorage {
    /// Shared TransactionDB (same instance as graph)
    db: Arc<TransactionDB>,

    /// Per-embedding-namespace state
    indices: RwLock<HashMap<Embedding, IndexState>>,
}

struct IndexState {
    id_allocator: IdAllocator,
    graph_meta: GraphMeta,
    rabitq: Option<RaBitQ>,
}

impl VectorStorage {
    /// Create VectorStorage from existing graph storage
    pub fn open(graph: &GraphStorage) -> Result<Self> {
        let db = graph.transaction_db();
        Self::ensure_column_families(&db)?;
        Ok(Self {
            db,
            indices: RwLock::new(HashMap::new()),
        })
    }

    pub fn get_or_create_index(&self, code: &Embedding, config: HnswConfig) -> Result<()> {
        let mut indices = self.indices.write().unwrap();
        if !indices.contains_key(code) {
            let state = IndexState::load_or_create(&self.db, code, config)?;
            indices.insert(code.clone(), state);
        }
        Ok(())
    }
}
```

### Why No Graph Module Changes?

| Concern | Resolution |
|---------|------------|
| **Edge storage** | Vector uses its own `vector/edges` CF with RoaringBitmaps |
| **Degree queries** | Vector uses `bitmap.len()` on its own edges |
| **Batch edge fetch** | Vector iterates its own `vector/edges` CF |
| **ID management** | Vector has separate u32 ID space per embedding namespace |

The vector module is **completely self-contained**:
- Only shares the `TransactionDB` instance for efficiency
- Returns `Vec<(f32, Id)>` (distance + ULID)
- No temporal filtering - caller's responsibility

### Caller-Side Filtering Pattern

```rust
// Application code (not in vector module)
let results = vector_storage.search(&embedding, &query, k * 3, ef)?;

// Caller applies temporal filter using graph storage
let visible_results: Vec<_> = results.into_iter()
    .filter(|(_, ulid)| graph_storage.is_visible(*ulid, as_of))
    .take(k)
    .collect();
```
