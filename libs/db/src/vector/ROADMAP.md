# Vector Search Implementation Roadmap

**Author:** David Chung + Claude
**Date:** January 2, 2026
**Scope:** `libs/db/src/vector` - Vector Search Module
**Status:** Implementation Planning

---

## Executive Summary

This document outlines the implementation roadmap for production-grade vector search in motlie_db.
The plan builds on completed POC work and core database optimizations to deliver 125x insert
throughput improvement and 10x search QPS improvement.

### Current State

| Component | Status | Reference |
|-----------|--------|-----------|
| Core DB: Name Interning | ✅ Complete | `libs/db/2025-12-FOLLOW-UP-ROADMAP.md` Phase 1 |
| Core DB: Blob Separation | ✅ Complete | `libs/db/2025-12-FOLLOW-UP-ROADMAP.md` Phase 2 |
| Core DB: rkyv Zero-Copy | ✅ Complete | `libs/db/2025-12-FOLLOW-UP-ROADMAP.md` Phase 3 |
| Vector POC: HNSW/Vamana | ✅ Complete | `examples/vector/POC.md` |
| Vector POC: SIMD Distance | ✅ Complete | `motlie_core::distance` (AVX-512/AVX2/NEON) |
| Vector: HNSW2 Optimization | 📋 Designed | `examples/vector/HNSW2.md` |
| Vector: RaBitQ Compression | 📋 Designed | `examples/vector/HNSW2.md` |
| Vector: HYBRID Architecture | 📋 Designed | `examples/vector/HYBRID.md` |

### Target Metrics

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Insert throughput | 40/s | 5,000/s | **125x** |
| Search QPS at 1M | 47 | 500+ | **10x** |
| Recall@10 at 1M | 95.3% | > 95% | ✅ Achieved |
| Memory at 1B | N/A | < 64 GB | Projected |

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
│  ├── 0.1 Module structure (mod.rs, schema.rs) ✓                             │
│  ├── 0.2 VectorConfig and VectorStorage types ✓                            │
│  ├── 0.3 Integration with existing graph Storage ✓                         │
│  ├── 0.4 ColumnFamilyProvider trait + StorageBuilder ✓                     │
│  ├── 0.5 Distance enum with compute behavior [ARCH-17] ✓                   │
│  ├── 0.6 Embedder trait for document-to-vector compute [ARCH-18] ✓         │
│  ├── 0.7 Rich Embedding struct (code, model, dim, distance, embedder) ✓    │
│  ├── 0.8 EmbeddingBuilder for fluent registration ✓                        │
│  ├── 0.9 EmbeddingRegistry with query/discovery API ✓                      │
│  ├── 0.10 EmbeddingFilter for multi-field queries ✓                        │
│  └── 0.11 IndexProvider trait + fulltext::Schema ✓                         │
│                                                                              │
│  Phase 1: ID Management [ARCH-4, ARCH-5, ARCH-6]                            │
│  ├── 1.1 u32 ID allocator with RoaringBitmap free list                     │
│  ├── 1.2 Forward mapping CF (ULID -> u32)                                  │
│  ├── 1.3 Reverse mapping (u32 -> ULID) - dense array or mmap              │
│  └── 1.4 ID allocation persistence and recovery                            │
│                                                                              │
│  Phase 2: HNSW2 Core + Navigation Layer [THR-1]                             │
│  ├── 2.1 RoaringBitmap edge storage                                        │
│  ├── 2.2 RocksDB merge operators for edge updates                          │
│  ├── 2.3 Vector storage CF (raw f32 bytes)                                 │
│  ├── 2.4 Node metadata CF (layer, flags)                                   │
│  ├── 2.5 Graph metadata CF (entry point, stats)                            │
│  ├── 2.6 HNSW insert algorithm with bitmap edges                           │
│  ├── 2.7 Navigation layer structure (entry points per layer)               │
│  ├── 2.8 Layer descent search algorithm                                    │
│  └── 2.9 Memory-cached top layers (NavigationCache)                        │
│                                                                              │
│  Phase 3: Batch Operations [THR-1, THR-3]                                   │
│  ├── 3.1 O(1) degree queries via RoaringBitmap.len()                       │
│  ├── 3.2 Batch neighbor fetch for beam search (MultiGet)                   │
│  ├── 3.3 Batch vector retrieval for re-ranking (MultiGet)                  │
│  └── 3.4 Batch ULID resolution                                             │
│                                                                              │
│  Phase 4: RaBitQ Compression [STOR-4, THR-3]                                │
│  ├── 4.1 Random rotation matrix generation                                 │
│  ├── 4.2 Binary code encoder                                               │
│  ├── 4.3 SIMD Hamming distance                                             │
│  ├── 4.4 Binary codes CF                                                   │
│  └── 4.5 Search with approximate + re-rank                                 │
│                                                                              │
│  Phase 5: Async Graph Updater (Online Updates) [LAT-4, LAT-5]               │
│  ├── 5.1 Pending queue CF (vector/pending)                                 │
│  ├── 5.2 Async updater configuration                                       │
│  ├── 5.3 Worker thread implementation (batch processing)                   │
│  ├── 5.4 Delete handling (mark + lazy cleanup)                             │
│  └── 5.5 Crash recovery (drain pending on startup)                         │
│                                                                              │
│  Phase 6: Production Hardening                                               │
│  ├── 6.1 Delete support [FUNC-3]                                           │
│  ├── 6.2 Concurrent access [CON-3]                                         │
│  └── 6.3 1B scale validation                                               │
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

```rust
/// Distance metric for vector similarity
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Distance {
    Cosine,
    L2,
    DotProduct,
}

impl Distance {
    /// Compute distance between two vectors
    pub fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Distance::Cosine => motlie_core::distance::cosine(a, b),
            Distance::L2 => motlie_core::distance::l2(a, b),
            Distance::DotProduct => motlie_core::distance::dot(a, b),
        }
    }

    /// Whether lower values mean more similar
    pub fn is_lower_better(&self) -> bool {
        match self {
            Distance::Cosine | Distance::L2 => true,
            Distance::DotProduct => false,  // higher = more similar
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Distance::Cosine => "cosine",
            Distance::L2 => "l2",
            Distance::DotProduct => "dot",
        }
    }
}
```

#### Embedder Trait (Compute Behavior)

```rust
/// Trait for computing embeddings from documents
/// Implementations can wrap external services (Ollama, OpenAI) or local models
pub trait Embedder: Send + Sync {
    /// Embed a single document
    fn embed(&self, document: &str) -> Result<Vec<f32>>;

    /// Batch embed for efficiency (default: sequential)
    fn embed_batch(&self, documents: &[&str]) -> Result<Vec<Vec<f32>>> {
        documents.iter().map(|d| self.embed(d)).collect()
    }

    /// Output dimensionality
    fn dim(&self) -> u32;

    /// Model identifier
    fn model(&self) -> &str;
}
```

#### Embedding Struct

```rust
/// Complete embedding space specification with optional compute behavior
///
/// Design rationale:
/// - `code`: u64 for fast key serialization (ARCH-14)
/// - `model`, `dim`, `distance`: Direct access without registry lookup (ARCH-17)
/// - `embedder`: Optional compute capability (ARCH-18)
#[derive(Clone)]
pub struct Embedding {
    /// Unique namespace code for storage keys (allocated by registry)
    code: u64,
    /// Model identifier (e.g., "gemma", "qwen3", "openai-ada-002")
    model: Arc<str>,
    /// Vector dimensionality
    dim: u32,
    /// Distance metric for similarity computation
    distance: Distance,
    /// Optional embedder for computing vectors from documents
    embedder: Option<Arc<dyn Embedder>>,
}

impl Embedding {
    // ─────────────────────────────────────────────────────────────
    // Accessors
    // ─────────────────────────────────────────────────────────────

    /// Namespace code for storage keys
    pub fn code(&self) -> u64 { self.code }

    /// Code as big-endian bytes for key construction
    pub fn code_bytes(&self) -> [u8; 8] { self.code.to_be_bytes() }

    pub fn model(&self) -> &str { &self.model }

    pub fn dim(&self) -> u32 { self.dim }

    pub fn distance(&self) -> Distance { self.distance }

    /// Check if this embedding has compute capability
    pub fn has_embedder(&self) -> bool { self.embedder.is_some() }

    // ─────────────────────────────────────────────────────────────
    // Behavior (delegated to Embedder trait)
    // ─────────────────────────────────────────────────────────────

    /// Compute embedding for a document (requires embedder)
    pub fn embed(&self, document: &str) -> Result<Vec<f32>> {
        self.embedder
            .as_ref()
            .ok_or(Error::NoEmbedder(self.model.to_string()))?
            .embed(document)
    }

    /// Batch embed documents
    pub fn embed_batch(&self, documents: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embedder
            .as_ref()
            .ok_or(Error::NoEmbedder(self.model.to_string()))?
            .embed_batch(documents)
    }

    /// Compute distance between two vectors using this space's metric
    pub fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        self.distance.compute(a, b)
    }

    // ─────────────────────────────────────────────────────────────
    // Validation
    // ─────────────────────────────────────────────────────────────

    /// Validate vector dimension matches this embedding space
    pub fn validate_vector(&self, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dim as usize {
            return Err(Error::DimensionMismatch {
                expected: self.dim,
                got: vector.len() as u32,
            });
        }
        Ok(())
    }
}

// Equality and Hash based on code only (for HashMap keys)
impl PartialEq for Embedding {
    fn eq(&self, other: &Self) -> bool { self.code == other.code }
}
impl Eq for Embedding {}
impl Hash for Embedding {
    fn hash<H: Hasher>(&self, state: &mut H) { self.code.hash(state); }
}
```

#### Embedding Registry

```rust
/// Builder for registering embedding spaces
pub struct EmbeddingBuilder {
    model: String,
    dim: u32,
    distance: Distance,
    embedder: Option<Arc<dyn Embedder>>,
}

impl EmbeddingBuilder {
    pub fn new(model: impl Into<String>, dim: u32, distance: Distance) -> Self {
        Self { model: model.into(), dim, distance, embedder: None }
    }

    /// Attach an embedder for compute capability
    pub fn with_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Register with the given registry
    pub fn register(self, registry: &EmbeddingRegistry) -> Result<Embedding> {
        registry.register(self)
    }
}

/// Registry for embedding spaces - manages specs and generates handles
/// Stored in vector/embedding_registry CF
pub struct EmbeddingRegistry {
    /// (model, dim, distance) -> Embedding mapping
    specs: RwLock<HashMap<(String, u32, Distance), Embedding>>,
    /// code -> Embedding mapping (for lookup by code)
    by_code: RwLock<HashMap<u64, Embedding>>,
    /// Next code to allocate
    next_code: AtomicU64,
    /// RocksDB handle for persistence
    db: Arc<TransactionDB>,
}

impl EmbeddingRegistry {
    // ─────────────────────────────────────────────────────────────
    // Registration
    // ─────────────────────────────────────────────────────────────

    /// Register a new embedding space. Idempotent: returns existing if spec matches.
    pub fn register(&self, builder: EmbeddingBuilder) -> Result<Embedding>;

    // ─────────────────────────────────────────────────────────────
    // Lookup
    // ─────────────────────────────────────────────────────────────

    /// Get embedding by exact spec (None if not registered)
    pub fn get(&self, model: &str, dim: u32, distance: Distance) -> Option<Embedding>;

    /// Get embedding by code (for deserialization from storage)
    pub fn get_by_code(&self, code: u64) -> Option<Embedding>;

    /// Attach/update embedder for existing space (runtime configuration)
    pub fn set_embedder(&self, code: u64, embedder: Arc<dyn Embedder>) -> Result<()>;

    // ─────────────────────────────────────────────────────────────
    // Query / Discovery
    // ─────────────────────────────────────────────────────────────

    /// List all registered embedding spaces
    pub fn list_all(&self) -> Vec<Embedding>;

    /// Find all spaces using a specific model
    pub fn find_by_model(&self, model: &str) -> Vec<Embedding>;

    /// Find all spaces using a specific distance metric
    pub fn find_by_distance(&self, distance: Distance) -> Vec<Embedding>;

    /// Find all spaces with a specific dimension
    pub fn find_by_dim(&self, dim: u32) -> Vec<Embedding>;

    /// Find with multiple filters (AND logic)
    pub fn find(&self, filter: &EmbeddingFilter) -> Vec<Embedding>;
}

/// Filter for querying embedding spaces
#[derive(Default)]
pub struct EmbeddingFilter {
    pub model: Option<String>,
    pub dim: Option<u32>,
    pub distance: Option<Distance>,
}

impl EmbeddingFilter {
    pub fn model(mut self, model: &str) -> Self { self.model = Some(model.into()); self }
    pub fn dim(mut self, dim: u32) -> Self { self.dim = Some(dim); self }
    pub fn distance(mut self, d: Distance) -> Self { self.distance = Some(d); self }
}
```

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
│  │ Value: Varies by field (entry_point, max_level, count, config)         │ │
│  │ Access: Point lookup (rarely changes)                                  │ │
│  │ Options: No compression                                                │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ CF: vector/id_forward                                                   │ │
│  │ Key:   [embedding: u64] + [ulid: 16] = 24 bytes                        │ │
│  │ Value: [vec_id: u32] = 4 bytes                                        │ │
│  │ Purpose: (embedding, ULID) -> internal u32 mapping (insert path)       │ │
│  │ Note: Embedding prefix required per FUNC-7 (multi-embedding support)   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ CF: vector/id_reverse                                                   │ │
│  │ Key:   [embedding: u64] + [vec_id: u32] = 12 bytes                    │ │
│  │ Value: [ulid: 16] = 16 bytes                                           │ │
│  │ Purpose: (embedding, vec_id) -> ULID mapping (search path, hot)       │ │
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
| id_forward | 48 bytes | 24 bytes | 24 bytes | 24 GB |
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
| id_forward | `[emb: u64] + [ulid: 16]` | 24 | vec_id (u32) | ULID → vec_id |
| id_reverse | `[emb: u64] + [vec_id: u32]` | 12 | ulid (16) | vec_id → ULID |
| id_alloc | `[emb: u64] + [field: u8]` | 9 | u32 / bitmap | ID allocator state |
| pending | `[emb: u64] + [ts: u64] + [vec_id: u32]` | 20 | empty | Async insert queue |

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Embedding ID | u64 (8 bytes) | Register-aligned, sufficient for <1000 models (ARCH-14) |
| Vec ID | u32 (4 bytes) | RoaringBitmap constraint, 4B vectors/space (ARCH-15) |
| ULID | 16 bytes | External doc ID, globally unique |
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

Per FUNC-7 (multi-embedding support), the same document (ULID) can exist in multiple embedding spaces:

```
Document ULID_abc:
  - In "qwen3" space: vec_id = 5
  - In "gemma" space: vec_id = 1000
```

Without embedding prefix, we couldn't distinguish which vec_id belongs to which space. The id_forward lookup is: "Given this ULID **in this embedding space**, what's its vec_id?"

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

```rust
// libs/db/src/vector/schema.rs

pub mod cf {
    pub const VECTORS: &str = "vector/vectors";
    pub const EDGES: &str = "vector/edges";
    pub const BINARY_CODES: &str = "vector/binary_codes";
    pub const NODE_META: &str = "vector/vec_meta";
    pub const GRAPH_META: &str = "vector/graph_meta";
    pub const ID_FORWARD: &str = "vector/id_forward";
    pub const ID_REVERSE: &str = "vector/id_reverse";
    pub const ID_ALLOC: &str = "vector/id_alloc";
    pub const PENDING: &str = "vector/pending";

    pub const ALL: &[&str] = &[
        VECTORS, EDGES, BINARY_CODES, NODE_META, GRAPH_META,
        ID_FORWARD, ID_REVERSE, ID_ALLOC, PENDING,
    ];
}
```

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

### VectorStorage Constructor

```rust
// libs/db/src/vector/mod.rs

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
    ///
    /// This shares the same RocksDB instance, adding vector-specific CFs.
    /// The vector module is completely independent - it only shares the DB.
    pub fn open(graph: &GraphStorage) -> Result<Self> {
        // Get the underlying TransactionDB from graph storage
        let db = graph.transaction_db();

        // Ensure vector CFs exist (idempotent)
        Self::ensure_column_families(&db)?;

        Ok(Self {
            db,
            indices: RwLock::new(HashMap::new()),
        })
    }

    /// Create or get an embedding namespace
    pub fn get_or_create_index(&self, code: &Embedding, config: HnswConfig) -> Result<()> {
        let mut indices = self.indices.write().unwrap();
        if !indices.contains_key(code) {
            let state = IndexState::load_or_create(&self.db, code, config)?;
            indices.insert(code.clone(), state);
        }
        Ok(())
    }

    fn ensure_column_families(db: &TransactionDB) -> Result<()> {
        for cf_name in cf::ALL {
            if db.cf_handle(cf_name).is_none() {
                db.create_cf(cf_name, &Self::cf_options(cf_name))?;
            }
        }
        Ok(())
    }
}
```

### Why No Graph Module Changes?

| Concern | Resolution |
|---------|------------|
| **Edge storage** | Vector uses its own `vector/edges` CF with RoaringBitmaps; graph edges untouched |
| **Degree queries** | Vector uses `bitmap.len()` on its own edges; no need for graph degree API |
| **Batch edge fetch** | Vector iterates its own `vector/edges` CF; graph CF not involved |
| **ID management** | Vector has separate u32 ID space per embedding namespace |

The vector module is **completely self-contained**:
- Only shares the `TransactionDB` instance for efficiency
- Returns `Vec<(f32, Id)>` (distance + ULID) - caller handles any filtering
- No temporal filtering logic - that's the caller's responsibility via `motlie_db::graph`

### Caller-Side Filtering Pattern

If the caller needs temporal or other filtering, they handle it after search:

```rust
// Application code (not in vector module)
let results = vector_storage.search(&embedding, &query, k * 3, ef)?;

// Caller applies temporal filter using graph storage
let visible_results: Vec<_> = results.into_iter()
    .filter(|(_, ulid)| graph_storage.is_visible(*ulid, as_of))
    .take(k)
    .collect();
```

This keeps the vector module simple and fast.

---

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

```rust
// libs/db/src/vector/config.rs

/// HNSW algorithm parameters
///
/// References:
/// - Original paper: https://arxiv.org/abs/1603.09320
/// - motlie design: examples/vector/HNSW2.md
#[derive(Clone, Debug)]
pub struct HnswConfig {
    /// Vector dimensionality (must match embedding model)
    /// Common values: 128 (SIFT), 768 (BERT), 1536 (OpenAI ada-002)
    pub dim: usize,

    /// Number of bidirectional links per node at layer 0
    /// Higher M = better recall, more memory, slower insert
    /// Recommended: 16-64, Default: 16
    pub m: usize,

    /// Maximum links per node at layers > 0 (typically 2*M)
    /// Default: 2 * m
    pub m_max: usize,

    /// Maximum links per node at layer 0 (typically M or 2*M)
    /// Layer 0 is denser, so higher limit improves recall
    /// Default: 2 * m
    pub m_max_0: usize,

    /// Search beam width during index construction
    /// Higher ef_construction = better graph quality, slower build
    /// Recommended: 100-500, Default: 200
    pub ef_construction: usize,

    /// Probability multiplier for layer assignment
    /// P(layer = L) = exp(-L * m_l)
    /// Default: 1.0 / ln(m)
    pub m_l: f32,

    /// Maximum search layer (auto-determined based on N)
    /// Default: None (auto-calculate as floor(ln(N) * m_l))
    pub max_level: Option<u8>,
}

impl Default for HnswConfig {
    fn default() -> Self {
        let m = 16;
        Self {
            dim: 128,
            m,
            m_max: 2 * m,
            m_max_0: 2 * m,
            ef_construction: 200,
            m_l: 1.0 / (m as f32).ln(),
            max_level: None,
        }
    }
}

impl HnswConfig {
    /// Create config for specific dimension
    pub fn for_dim(dim: usize) -> Self {
        Self { dim, ..Default::default() }
    }

    /// High-recall configuration (slower, more memory)
    pub fn high_recall(dim: usize) -> Self {
        Self {
            dim,
            m: 32,
            m_max: 64,
            m_max_0: 64,
            ef_construction: 400,
            m_l: 1.0 / 32f32.ln(),
            max_level: None,
        }
    }

    /// Memory-optimized configuration (faster, less recall)
    pub fn compact(dim: usize) -> Self {
        Self {
            dim,
            m: 8,
            m_max: 16,
            m_max_0: 16,
            ef_construction: 100,
            m_l: 1.0 / 8f32.ln(),
            max_level: None,
        }
    }
}

/// RaBitQ binary quantization parameters
#[derive(Clone, Debug)]
pub struct RaBitQConfig {
    /// Bits per dimension (1, 2, 4, or 8)
    /// 1-bit: 32x compression, ~70% recall without rerank
    /// 2-bit: 16x compression, ~85% recall without rerank
    /// 4-bit: 8x compression, ~92% recall without rerank
    pub bits_per_dim: u8,

    /// Seed for rotation matrix generation (deterministic)
    pub rotation_seed: u64,

    /// Enable RaBitQ (can be disabled for full-precision search)
    pub enabled: bool,
}

impl Default for RaBitQConfig {
    fn default() -> Self {
        Self {
            bits_per_dim: 1,
            rotation_seed: 42,
            enabled: true,
        }
    }
}

/// Complete vector storage configuration
#[derive(Clone, Debug, Default)]
pub struct VectorConfig {
    pub hnsw: HnswConfig,
    pub rabitq: RaBitQConfig,
    pub async_updater: AsyncUpdaterConfig,
    pub navigation_cache: NavigationCacheConfig,
}

impl VectorConfig {
    /// Configuration for 128-dimensional embeddings (e.g., SIFT)
    pub fn dim_128() -> Self {
        Self {
            hnsw: HnswConfig::for_dim(128),
            ..Default::default()
        }
    }

    /// Configuration for 768-dimensional embeddings (e.g., BERT)
    pub fn dim_768() -> Self {
        Self {
            hnsw: HnswConfig::for_dim(768),
            ..Default::default()
        }
    }

    /// Configuration for 1536-dimensional embeddings (e.g., OpenAI ada-002)
    pub fn dim_1536() -> Self {
        Self {
            hnsw: HnswConfig::for_dim(1536),
            ..Default::default()
        }
    }
}
```

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

Pre-warm EmbeddingRegistry during `ready()`, following the NameCache pattern in `graph/name_hash.rs`:

```rust
// libs/db/src/vector/embedding.rs

use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Thread-safe bidirectional embedding name <-> ID cache
/// Pattern: Similar to graph::NameCache (libs/db/src/graph/name_hash.rs:123-249)
pub struct EmbeddingRegistry {
    /// ID -> name mapping
    id_to_name: DashMap<u64, Arc<String>>,
    /// Name -> ID mapping
    name_to_id: DashMap<Arc<String>, u64>,
    /// Next available ID (monotonic)
    next_id: AtomicU64,
}

impl EmbeddingRegistry {
    pub fn new() -> Self {
        Self {
            id_to_name: DashMap::new(),
            name_to_id: DashMap::new(),
            next_id: AtomicU64::new(0),
        }
    }

    /// Pre-warm by loading ALL entries from embedding_registry CF
    /// Unlike NameCache (which limits to N entries), we load everything
    /// since we expect <1000 embeddings total
    pub fn prewarm(&self, db: &TransactionDB) -> Result<usize> {
        let cf = db.cf_handle(cf::EMBEDDING_REGISTRY)?;
        let iter = db.iterator_cf(cf, IteratorMode::Start);

        let mut count = 0u64;
        let mut max_id = 0u64;

        for item in iter {
            let (key, value) = item?;
            let id = u64::from_be_bytes(key[..8].try_into()?);
            let name: String = String::from_utf8(value.to_vec())?;

            self.id_to_name.insert(id, Arc::new(name.clone()));
            self.name_to_id.insert(Arc::new(name), id);

            max_id = max_id.max(id);
            count += 1;
        }

        // Set next_id to max + 1
        self.next_id.store(max_id + 1, Ordering::SeqCst);

        Ok(count as usize)
    }

    /// Get existing or create new embedding ID
    pub fn get_or_create(&self, name: &str, db: &TransactionDB) -> Result<Embedding> {
        // Fast path: already cached
        if let Some(id) = self.name_to_id.get(name) {
            return Ok(Embedding::from_id(*id));
        }

        // Slow path: allocate new ID and persist
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let name_arc = Arc::new(name.to_string());

        // Persist to RocksDB
        let cf = db.cf_handle(cf::EMBEDDING_REGISTRY)?;
        db.put_cf(cf, &id.to_be_bytes(), name.as_bytes())?;

        // Update cache
        self.id_to_name.insert(id, name_arc.clone());
        self.name_to_id.insert(name_arc, id);

        Ok(Embedding::from_id(id))
    }

    /// Lookup name by ID (for debugging/logging)
    pub fn get_name(&self, id: u64) -> Option<Arc<String>> {
        self.id_to_name.get(&id).map(|r| r.clone())
    }
}
```

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
- `GraphMeta`: `GraphMetaField` enum with `EntryPoint`, `MaxLevel`, `Count`, `Config` variants
- `IdAlloc`: `IdAllocField` enum with `NextId`, `FreeBitmap` variants
- Single enum for both key discrimination and value storage
- Key helpers: `GraphMetaCfKey::entry_point(code)`, etc.
- `value_from_bytes(key, bytes)` uses key's variant for type info

See `src/vector/README.md` Appendix A for detailed documentation.

#### StorageSubsystem Trait (Generic Storage)

The `StorageSubsystem` trait enables type-safe, zero-cost generic storage:

```rust
pub trait StorageSubsystem: Send + Sync + 'static {
    const NAME: &'static str;
    const COLUMN_FAMILIES: &'static [&'static str];

    type PrewarmConfig: Default + Clone + Send + Sync;
    type Cache: Send + Sync;

    fn create_cache() -> Arc<Self::Cache>;
    fn cf_descriptors(block_cache: &Cache, config: &BlockCacheConfig) -> Vec<ColumnFamilyDescriptor>;
    fn prewarm(db: &dyn DbAccess, cache: &Self::Cache, config: &Self::PrewarmConfig) -> Result<usize>;
}
```

- `graph::Subsystem` - Cache type is `NameCache`, pre-warms from Names CF
- `vector::Subsystem` - Cache type is `EmbeddingRegistry`, pre-warms from EmbeddingSpecs CF

#### ComponentWrapper (StorageBuilder Adapter)

`ComponentWrapper<S>` adapts any `StorageSubsystem` to implement `ColumnFamilyProvider` for use with `StorageBuilder`:

```rust
// Type aliases in each module
pub type Component = rocksdb::ComponentWrapper<Subsystem>;

// Convenience constructors
pub fn component() -> Component { Component::new() }
```

**Tantivy Providers (`IndexProvider` trait):**
```rust
pub trait IndexProvider: Send + Sync {
    fn name(&self) -> &'static str;
    fn schema(&self) -> tantivy::schema::Schema;
    fn on_ready(&self, index: &Index) -> Result<()>;
    fn writer_heap_size(&self) -> usize;
}
```

- `fulltext::Schema` - Provides Tantivy schema and initialization

#### StorageBuilder Design

```rust
// Create components and get cache references before boxing
let graph_component = graph::component();
let name_cache = graph_component.cache().clone();

let vector_component = vector::component();
let registry = vector_component.cache().clone();

// Unified initialization of RocksDB + Tantivy
let storage = StorageBuilder::new(&base_path)
    .with_component(Box::new(graph_component))
    .with_component(Box::new(vector_component))
    .with_index_provider(Box::new(fulltext::Schema::new()))
    .with_cache_size(512 * 1024 * 1024)
    .build()?;

// Access backends
let db = storage.db();       // RocksDB TransactionDB
let index = storage.index(); // Tantivy Index

// Caches remain accessible via cloned Arcs
assert_eq!(name_cache.len(), 0);
```

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
| `vector/id_forward` | (code, emb_id) | vec_id | ULID → u32 mapping |
| `vector/id_reverse` | (code, vec_id) | emb_id | u32 → ULID mapping |
| `vector/id_alloc` | code (u64) | (next_id, free_bitmap) | ID allocation state |
| `vector/pending` | (code, emb_id, ts) | operation | Async update queue |

#### Test Coverage

- 364 tests passing (285 unit + 79 integration)
- Includes `test_storage_builder.rs` with 10 integration tests for graph+vector+fulltext composition

#### Deferred Work

- **Secondary mode support for StorageBuilder**: The existing `graph::Storage::secondary()` API works for RocksDB secondary instances. Adding this to `StorageBuilder` is deferred as it requires different RocksDB APIs (`DB::open_as_secondary()` vs `TransactionDB`).

---

## Phase 1: ID Management

**Goal:** Implement u32 internal ID system per [ARCH-4, ARCH-5, ARCH-6].

**Design Reference:** `examples/vector/HNSW2.md` - Item ID Design

### Task 1.1: ID Allocator

```rust
// libs/db/src/vector/id.rs

use roaring::RoaringBitmap;
use std::sync::atomic::{AtomicU32, Ordering};

/// Lock-free ID allocator with free list
pub struct IdAllocator {
    /// Next ID to allocate (monotonically increasing)
    next_id: AtomicU32,

    /// Free IDs from deletions (protected by mutex for bitmap ops)
    free_ids: Mutex<RoaringBitmap>,

    /// RocksDB CF for persistence
    cf_name: &'static str,
}

impl IdAllocator {
    /// Allocate a new u32 ID
    pub fn allocate(&self) -> u32 {
        // First try to reuse a freed ID
        let mut free = self.free_ids.lock().unwrap();
        if let Some(id) = free.iter().next() {
            free.remove(id);
            return id;
        }
        drop(free);

        // Otherwise allocate fresh
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Return an ID to the free list
    pub fn free(&self, id: u32) {
        self.free_ids.lock().unwrap().insert(id);
    }

    /// Persist allocator state to RocksDB
    pub fn persist(&self, db: &TransactionDB) -> Result<()>;

    /// Recover state from RocksDB
    pub fn recover(db: &TransactionDB) -> Result<Self>;
}
```

**Effort:** 1-2 days

**Acceptance Criteria:**
- [ ] Allocator survives process restart
- [ ] Free IDs are reused before allocating new ones
- [ ] Thread-safe concurrent allocation

### Task 1.2: Forward Mapping (ULID -> u32)

```rust
// libs/db/src/vector/schema.rs

/// Forward mapping: ULID -> internal u32
/// Used during insert path (less frequent)
pub struct ForwardIdMapping;

impl ColumnFamilyRecord for ForwardIdMapping {
    const CF_NAME: &'static str = cf::ID_FORWARD;  // "vector/id_forward"
    type Key = Id;              // 16-byte ULID
    type Value = u32;           // 4-byte internal ID
}
```

**Effort:** 0.5 day

**Acceptance Criteria:**
- [ ] Insert stores ULID -> u32 mapping
- [ ] Lookup returns Option<u32>

### Task 1.3: Reverse Mapping (u32 -> ULID)

This is the hot path during search. Options:

| Approach | Memory at 1B | Lookup | Implementation |
|----------|--------------|--------|----------------|
| RocksDB CF | On-demand | ~100µs | Simple |
| Dense array | 16 GB | O(1) ~10ns | Complex |
| Mmap file | On-demand | O(1) ~100ns | Medium |

**Recommended:** Start with RocksDB CF, optimize to mmap if needed.

```rust
/// Reverse mapping: internal u32 -> ULID
/// Hot path during search - consider mmap optimization
pub struct ReverseIdMapping;

impl ColumnFamilyRecord for ReverseIdMapping {
    const CF_NAME: &'static str = cf::ID_REVERSE;  // "vector/id_reverse"
    type Key = u32;             // 4-byte internal ID
    type Value = Id;            // 16-byte ULID
}

impl ReverseIdMapping {
    /// Batch lookup for search results
    pub fn multi_get(db: &DB, ids: &[u32]) -> Result<Vec<Option<Id>>> {
        // Use RocksDB MultiGet for efficiency
        let keys: Vec<_> = ids.iter().map(|id| id.to_le_bytes()).collect();
        db.multi_get_cf(cf, keys)
    }
}
```

**Effort:** 0.5 day (RocksDB), +1 day (mmap optimization)

**Acceptance Criteria:**
- [ ] Search results correctly map back to ULIDs
- [ ] MultiGet batch performance validated

### Task 1.4: ID Allocation Persistence

Persist allocator state for crash recovery:

```rust
/// ID allocator metadata
pub struct IdAllocatorMeta;

impl ColumnFamilyRecord for IdAllocatorMeta {
    const CF_NAME: &'static str = cf::ID_ALLOC;  // "vector/id_alloc"
    type Key = &'static str;    // "next_id" | "free_bitmap"
    type Value = Vec<u8>;       // Serialized value
}
```

**Effort:** 0.5 day

**Acceptance Criteria:**
- [ ] `next_id` persists across restarts
- [ ] Free bitmap persists (for ID reuse after restart)

### Phase 1 Validation & Tests

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

## Phase 2: HNSW2 Core + Navigation Layer

**Goal:** Implement optimized HNSW with roaring bitmap edges per `examples/vector/HNSW2.md`,
including the hierarchical navigation layer for O(log N) search complexity.

**Target:** Achieve [THR-1] 5,000+ inserts/sec (125x improvement from 40/s).

**Design Reference:** `examples/vector/HYBRID.md` - Navigation Layer (Phase 4)

### Task 2.1: RoaringBitmap Edge Storage

Replace explicit edge lists with compressed bitmaps:

```rust
// libs/db/src/vector/schema.rs

/// HNSW graph edges using RoaringBitmap
/// Key: [embedding: u64] + [vec_id: u32] + [layer: u8] = 13 bytes
/// Value: RoaringBitmap serialized (~50-200 bytes)
pub struct GraphEdges;

impl ColumnFamilyRecord for GraphEdges {
    const CF_NAME: &'static str = cf::EDGES;  // "vector/edges"

    fn key_to_bytes(embedding: &Embedding, vec_id: u32, layer: u8) -> [u8; 13] {
        let mut key = [0u8; 13];
        key[0..8].copy_from_slice(&embedding.to_be_bytes());
        key[8..12].copy_from_slice(&vec_id.to_be_bytes());
        key[12] = layer;
        key
    }
}

// Storage comparison:
// - Old: 32 edges × 16 bytes (ULID) = 512 bytes
// - New: RoaringBitmap = ~50-200 bytes (4-10x smaller)
```

**Effort:** 1-2 days

**Acceptance Criteria:**
- [ ] Edge storage 4-10x smaller than current
- [ ] Membership test O(1): `bitmap.contains(neighbor_id)`
- [ ] Iteration works: `bitmap.iter()`

### Task 2.2: RocksDB Merge Operators

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
- [ ] `db.merge_cf()` adds neighbors without read-modify-write
- [ ] Concurrent writers don't conflict
- [ ] Crash-safe (logged in WAL)

### Task 2.3: Vector Storage CF

Store raw vectors for exact distance computation:

```rust
/// Full vector storage (raw f32 bytes)
/// Key: [embedding: u64] + [vec_id: u32] = 12 bytes
/// Value: [f32; D] as raw bytes (e.g., 512 bytes for 128D)
pub struct Vectors;

impl ColumnFamilyRecord for Vectors {
    const CF_NAME: &'static str = cf::VECTORS;  // "vector/vectors"

    fn column_family_options() -> Options {
        let mut opts = Options::default();
        opts.set_compression_type(DBCompressionType::Lz4);
        opts.set_block_size(16 * 1024);  // 16KB for vectors
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

**Effort:** 0.5 day

### Task 2.4: Node Metadata CF

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
- [ ] Insert throughput > 1,000/s (25x improvement, towards 5,000/s target)
- [ ] Recall@10 maintains > 95%
- [ ] Crash recovery works

### Task 2.7: Navigation Layer Structure

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
- [ ] Entry points tracked per layer
- [ ] Random layer assignment follows HNSW distribution
- [ ] Entry point updates correctly when higher-layer nodes inserted

### Task 2.8: Layer Descent Search Algorithm

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
- [ ] Layer descent correctly narrows search region
- [ ] Search uses O(log N) layers before layer-0 expansion
- [ ] ef parameter controls layer-0 beam width
- [ ] Search latency scales as O(log N) with dataset size

### Task 2.9: Memory-Cached Top Layers (NavigationCache)

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
- [ ] Layers 2+ fully cached in memory (~50MB at 1B)
- [ ] Layer 0-1 use LRU cache for hot nodes
- [ ] Cache invalidation on edge updates
- [ ] Search latency reduced by 3-5x for layer descent phase
- [ ] Memory usage stays within configured bounds

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
- [ ] RoaringBitmap edge storage and compression
- [ ] Merge operator: add, batch add, remove
- [ ] Single vector insert and search
- [ ] Multiple vector insert
- [ ] Bidirectional edge creation
- [ ] Layer assignment distribution (matches HNSW theory)
- [ ] Entry point tracking
- [ ] Multi-layer descent search
- [ ] Recall@10 > 95% on random data
- [ ] Recall@10 > 90% on SIFT (if available)
- [ ] Navigation cache loading
- [ ] Cache invalidation
- [ ] Crash recovery

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

## Phase 3: Batch Operations

**Goal:** Implement batch operations within the vector module to reduce I/O overhead.

**Note:** All operations are self-contained in `motlie_db::vector`. No changes to `motlie_db::graph`.

### Task 3.1: O(1) Degree Queries via RoaringBitmap

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

### Task 3.2: Batch Neighbor Fetch for Beam Search

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

### Task 3.3: Batch Vector Retrieval for Re-ranking

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

### Task 3.4: Batch ULID Resolution

Batch resolve internal IDs to ULIDs for search results:

```rust
// libs/db/src/vector/id.rs

impl VectorStorage {
    /// Batch resolve internal u32 IDs to ULIDs
    pub fn resolve_ulids_batch(
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

**Goal:** Implement training-free binary quantization per [DATA-1] constraint.

**Design Reference:** `examples/vector/HNSW2.md` - RaBitQ section, `examples/vector/ALTERNATIVES.md`

### Task 4.1: Random Rotation Matrix

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

### Task 4.2: Binary Code Encoder

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

### Task 4.3: SIMD Hamming Distance

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

### Task 4.4: Binary Codes CF

```rust
/// Binary codes storage
/// Key: [embedding: u64] + [vec_id: u32] = 12 bytes
/// Value: [u8; code_size] (e.g., 16 bytes for 128-dim 1-bit)
pub struct BinaryCodes;

impl ColumnFamilyRecord for BinaryCodes {
    const CF_NAME: &'static str = cf::BINARY_CODES;  // "vector/binary_codes"

    fn column_family_options() -> Options {
        let mut opts = Options::default();
        // No compression - already minimal
        opts.set_compression_type(DBCompressionType::None);
        opts.set_block_size(4 * 1024);  // 4KB blocks
        opts
    }
}
```

**Effort:** 0.25 day

### Task 4.5: Search with Approximate + Re-rank

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

## Phase 5: Async Graph Updater (Online Updates)

**Goal:** Decouple insert latency from graph quality per [LAT-4, LAT-5].

**Design Reference:** `examples/vector/HYBRID.md` - Async Graph Updater

### Overview: Two-Phase Insert Pattern

Online updates use a **two-phase pattern** to achieve low insert latency while
maintaining graph quality:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Two-Phase Insert Pattern                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Phase 1: Sync Write (< 5ms)                                                │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 1. Allocate internal u32 ID                                            │ │
│  │ 2. Store ID mappings (ULID <-> u32)                                    │ │
│  │ 3. Store vector in vector/vectors CF                                   │ │
│  │ 4. Store binary code in vector/binary_codes CF (if RaBitQ enabled)     │ │
│  │ 5. Add to pending queue (vector/pending CF)                            │ │
│  │ 6. Return success immediately                                          │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                              │                                               │
│                              ▼                                               │
│  Phase 2: Async Graph Update (background, batched)                          │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ 1. Batch pending vectors (e.g., every 100 vectors or 100ms)            │ │
│  │ 2. For each vector: greedy search to find neighbors                    │ │
│  │ 3. Add bidirectional edges (merge operator)                            │ │
│  │ 4. Prune over-connected nodes if needed                                │ │
│  │ 5. Remove from pending queue                                           │ │
│  │ 6. Update graph metadata (entry point if needed)                       │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Searchability Before Graph Edges

**Key Insight:** Vectors are searchable immediately after Phase 1, even before
graph edges are added. Here's how:

1. **Brute-force fallback for new vectors**: If a vector has no edges yet,
   it can still be found via the entry point's neighborhood expansion

2. **Binary code availability**: RaBitQ binary codes are stored in Phase 1,
   enabling approximate distance computation for candidate filtering

3. **Eventually consistent**: Graph edges are added within milliseconds to seconds,
   so the window of reduced connectivity is brief

```rust
impl VectorStorage {
    /// Insert with async graph update (low latency)
    pub fn insert_async(
        &self,
        code: &Embedding,
        ulid: Id,
        vector: &[f32],
    ) -> Result<()> {
        // Phase 1: Sync writes (fast)
        let internal_id = self.allocate_id(code)?;
        self.store_id_mappings(code, ulid, internal_id)?;
        self.store_vector(code, internal_id, vector)?;

        if let Some(rabitq) = self.get_rabitq(code) {
            let binary_code = rabitq.encode(vector);
            self.store_binary_code(code, internal_id, &binary_code)?;
        }

        // Add to pending queue for async graph update
        self.add_to_pending(code, internal_id)?;

        Ok(())  // Return immediately
    }

    /// Insert with sync graph update (higher latency, immediate connectivity)
    pub fn insert_sync(
        &self,
        code: &Embedding,
        ulid: Id,
        vector: &[f32],
    ) -> Result<()> {
        // Phase 1: Same as async
        let internal_id = self.allocate_id(code)?;
        self.store_id_mappings(code, ulid, internal_id)?;
        self.store_vector(code, internal_id, vector)?;

        if let Some(rabitq) = self.get_rabitq(code) {
            let binary_code = rabitq.encode(vector);
            self.store_binary_code(code, internal_id, &binary_code)?;
        }

        // Phase 2: Sync graph update (blocking)
        self.build_graph_connections(code, internal_id, vector)?;

        Ok(())
    }
}
```

### Task 5.1: Pending Queue CF

The pending queue tracks vectors awaiting graph integration:

```rust
// libs/db/src/vector/schema.rs

/// Pending inserts queue
/// Key: [embedding: u64] + [timestamp: u64] + [vec_id: u32] = 20 bytes
/// Value: empty (vector already stored in vector/vectors)
///
/// Key structure ensures:
/// - Namespace isolation via embedding prefix
/// - FIFO ordering via timestamp
/// - Efficient batch collection via prefix scan
pub struct PendingInserts;

impl ColumnFamilyRecord for PendingInserts {
    const CF_NAME: &'static str = cf::PENDING;  // "vector/pending"
}

impl PendingInserts {
    pub fn make_key(code: &Embedding, timestamp: u64, vec_id: u32) -> Vec<u8> {
        let mut key = Vec::with_capacity(20);  // u64 + u64 + u32
        key.extend_from_slice(&code.to_be_bytes());       // 8 bytes
        key.extend_from_slice(&timestamp.to_be_bytes());  // 8 bytes (BE for ordering)
        key.extend_from_slice(&vec_id.to_be_bytes());    // 4 bytes
        key
    }

    pub fn parse_key(key: &[u8]) -> (Embedding, u64, u32) {
        let code = Embedding::from_id(u64::from_be_bytes(key[0..8].try_into().unwrap()));
        let timestamp = u64::from_be_bytes(key[8..16].try_into().unwrap());
        let vec_id = u32::from_be_bytes(key[16..20].try_into().unwrap());
        (code, timestamp, vec_id)
    }
}
```

### Task 5.2: Async Updater Configuration

```rust
// libs/db/src/vector/async_updater.rs

/// Configuration for the async graph updater
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

### Task 5.3: Async Updater Implementation

```rust
// libs/db/src/vector/async_updater.rs

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Background graph updater
pub struct AsyncGraphUpdater {
    storage: Arc<VectorStorage>,
    config: AsyncUpdaterConfig,
    shutdown: Arc<AtomicBool>,
    workers: Vec<JoinHandle<()>>,
}

impl AsyncGraphUpdater {
    /// Start the async updater with worker threads
    pub fn start(storage: Arc<VectorStorage>, config: AsyncUpdaterConfig) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let num_workers = config.num_workers;

        // Drain pending queue on startup if configured
        if config.process_on_startup {
            Self::drain_pending(&storage, &config);
        }

        // Spawn worker threads
        let workers: Vec<_> = (0..num_workers)
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

    /// Graceful shutdown
    pub fn shutdown(self) {
        self.shutdown.store(true, Ordering::SeqCst);
        for worker in self.workers {
            let _ = worker.join();
        }
    }

    fn worker_loop(
        worker_id: usize,
        storage: Arc<VectorStorage>,
        config: AsyncUpdaterConfig,
        shutdown: Arc<AtomicBool>,
    ) {
        log::info!("Async updater worker {} started", worker_id);

        loop {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            // Collect a batch of pending inserts
            let batch = Self::collect_batch(&storage, &config);

            if batch.is_empty() {
                // No work, sleep briefly
                thread::sleep(Duration::from_millis(10));
                continue;
            }

            log::debug!("Worker {} processing batch of {} vectors", worker_id, batch.len());

            // Process each vector in the batch
            for (code, vec_id) in &batch {
                if let Err(e) = Self::process_insert(&storage, &config, code, *vec_id) {
                    log::error!("Failed to process insert for node {}: {}", vec_id, e);
                    // Continue with next - don't fail entire batch
                }
            }

            // Remove processed items from pending queue
            Self::clear_processed(&storage, &batch);
        }

        log::info!("Async updater worker {} stopped", worker_id);
    }

    fn collect_batch(
        storage: &VectorStorage,
        config: &AsyncUpdaterConfig,
    ) -> Vec<(Embedding, u32)> {
        let cf = storage.db.cf_handle(cf::PENDING).unwrap();
        let mut batch = Vec::with_capacity(config.batch_size);
        let deadline = Instant::now() + config.batch_timeout;

        // Scan pending queue (ordered by timestamp)
        let iter = storage.db.iterator_cf(cf, rocksdb::IteratorMode::Start);

        for item in iter {
            if batch.len() >= config.batch_size {
                break;
            }
            if Instant::now() > deadline {
                break;
            }

            if let Ok((key, _)) = item {
                let (code, _timestamp, vec_id) = PendingInserts::parse_key(&key);
                batch.push((code, vec_id));
            }
        }

        batch
    }

    fn process_insert(
        storage: &VectorStorage,
        config: &AsyncUpdaterConfig,
        code: &Embedding,
        vec_id: u32,
    ) -> Result<()> {
        // 1. Load the vector
        let vector = storage.get_vector(code, vec_id)?
            .ok_or_else(|| anyhow!("Vector not found for pending node {}", vec_id))?;

        // 2. Determine layer for this node
        let level = storage.get_node_level(code, vec_id)?;

        // 3. Get entry point
        let entry_point = storage.get_entry_point(code)?;

        // 4. Greedy search from entry point to find neighbors at each layer
        for layer in (0..=level).rev() {
            let neighbors = storage.greedy_search_layer(
                code,
                &vector,
                entry_point,
                layer,
                config.ef_construction,
            )?;

            // 5. Select M nearest neighbors
            let m = storage.get_m_for_layer(code, layer);
            let selected: Vec<u32> = neighbors.into_iter()
                .take(m)
                .map(|(_, id)| id)
                .collect();

            // 6. Add bidirectional edges (using merge operator)
            storage.add_edges_batch(code, vec_id, &selected, layer)?;

            // 7. Prune neighbors if they exceed M_max
            for &neighbor in &selected {
                storage.prune_if_needed(code, neighbor, layer)?;
            }
        }

        // 8. Update entry point if this node has higher level
        storage.maybe_update_entry_point(code, vec_id, level)?;

        Ok(())
    }

    fn clear_processed(storage: &VectorStorage, batch: &[(Embedding, u32)]) {
        let cf = storage.db.cf_handle(cf::PENDING).unwrap();

        // Delete processed entries
        // Note: We need to reconstruct the full key with timestamp
        // In practice, store the full key in the batch
        for (code, vec_id) in batch {
            // Scan for this node's pending entry and delete
            let prefix = code.to_be_bytes();
            let iter = storage.db.prefix_iterator_cf(cf, &prefix);

            for item in iter {
                if let Ok((key, _)) = item {
                    let (_, _, key_vec_id) = PendingInserts::parse_key(&key);
                    if key_vec_id == *vec_id {
                        let _ = storage.db.delete_cf(cf, &key);
                        break;
                    }
                }
            }
        }
    }

    fn drain_pending(storage: &VectorStorage, config: &AsyncUpdaterConfig) {
        log::info!("Draining pending queue on startup...");
        let start = Instant::now();
        let mut count = 0;

        loop {
            let batch = Self::collect_batch(storage, config);
            if batch.is_empty() {
                break;
            }

            for (code, vec_id) in &batch {
                if let Err(e) = Self::process_insert(storage, config, code, *vec_id) {
                    log::error!("Failed to drain pending node {}: {}", vec_id, e);
                }
                count += 1;
            }

            Self::clear_processed(storage, &batch);
        }

        log::info!("Drained {} pending vectors in {:?}", count, start.elapsed());
    }
}
```

### Task 5.4: Delete Handling

Deletes also use the two-phase pattern:

```rust
impl VectorStorage {
    /// Delete with async cleanup
    pub fn delete_async(&self, code: &Embedding, ulid: Id) -> Result<()> {
        let internal_id = self.get_internal_id(code, ulid)?
            .ok_or_else(|| anyhow!("Vector not found"))?;

        // Phase 1: Mark as deleted (fast)
        self.mark_deleted(code, internal_id)?;

        // Remove from pending queue if present
        self.remove_from_pending(code, internal_id)?;

        // Return ID to free list
        self.free_id(code, internal_id);

        // Phase 2: Async edge cleanup (background)
        // Edges pointing to deleted nodes are cleaned up lazily:
        // - During search: skip deleted nodes
        // - During compaction: merge operator removes stale edges
        // - Optionally: background cleanup thread

        Ok(())
    }

    /// Check if a node is deleted (for search filtering)
    fn is_deleted(&self, code: &Embedding, vec_id: u32) -> Result<bool> {
        let meta = self.get_vec_meta(code, vec_id)?;
        Ok(meta.map_or(true, |m| m.flags & FLAG_DELETED != 0))
    }
}

// Node flags
const FLAG_DELETED: u8 = 0x01;
const FLAG_PENDING: u8 = 0x02;  // Still in pending queue
```

### Consistency Guarantees

| Guarantee | Description |
|-----------|-------------|
| **Durability** | Vector is durable after `insert_async` returns (stored in RocksDB) |
| **Searchability** | Vector is searchable immediately (via brute-force expansion) |
| **Eventual Connectivity** | Graph edges added within `batch_timeout` (default 100ms) |
| **Crash Recovery** | Pending queue persisted; drained on restart |
| **Delete Visibility** | Deleted vectors excluded from search immediately |

### Effort Breakdown

| Task | Description | Effort |
|------|-------------|--------|
| 5.1 | Pending queue CF and key encoding | 0.5 day |
| 5.2 | Async updater configuration | 0.25 day |
| 5.3 | Worker thread implementation | 1.5 days |
| 5.4 | Delete handling | 0.5 day |
| 5.5 | Testing and crash recovery | 1 day |
| **Total** | | **3-4 days** |

**Acceptance Criteria:**
- [ ] `insert_async` latency < 5ms P99
- [ ] `insert_sync` latency < 50ms P99
- [ ] Pending queue drained within `batch_timeout`
- [ ] Crash recovery: pending queue survives restart
- [ ] Deleted nodes excluded from search results

### Phase 5 Validation & Tests

```rust
// libs/db/src/vector/tests/async_updater_tests.rs

#[cfg(test)]
mod async_updater_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    // =========================================================================
    // Two-Phase Insert Tests (Task 5.1)
    // =========================================================================

    #[test]
    fn test_insert_async_immediate_searchability() {
        let storage = setup_vector_storage_with_async_updater();
        let code = Embedding::new("test");

        let ulid = Id::new();
        let vector = vec![1.0; 128];

        // Async insert
        storage.insert_async(&code, ulid, &vector).unwrap();

        // Should be searchable IMMEDIATELY (brute force fallback)
        let results = storage.search(&code, &vector, 1, 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, ulid);
    }

    #[test]
    fn test_insert_async_low_latency() {
        let storage = setup_vector_storage_with_async_updater();
        let code = Embedding::new("test");

        let mut latencies = Vec::new();
        for _ in 0..100 {
            let ulid = Id::new();
            let vector: Vec<f32> = (0..128).map(|i| i as f32).collect();

            let start = Instant::now();
            storage.insert_async(&code, ulid, &vector).unwrap();
            latencies.push(start.elapsed());
        }

        latencies.sort();
        let p99 = latencies[98];
        assert!(p99 < Duration::from_millis(5),
            "insert_async P99 should be < 5ms, got {:?}", p99);
    }

    #[test]
    fn test_insert_sync_includes_graph_edges() {
        let storage = setup_vector_storage_temp();
        let code = Embedding::new("test");

        // Insert some vectors synchronously
        for i in 0..10 {
            let ulid = Id::new();
            let vector = vec![i as f32; 128];
            storage.insert_sync(&code, ulid, &vector).unwrap();
        }

        // All should have graph edges
        for id in 0u32..10 {
            let neighbors = storage.get_neighbors(&code, id, 0).unwrap();
            // After 10 inserts, each node should have some neighbors
            if id > 0 {
                assert!(neighbors.len() > 0, "Node {} should have neighbors", id);
            }
        }
    }

    // =========================================================================
    // Pending Queue Tests (Task 5.1)
    // =========================================================================

    #[test]
    fn test_pending_queue_ordering() {
        let storage = setup_vector_storage_with_async_updater();
        let code = Embedding::new("test");

        // Insert in order
        let ulids: Vec<Id> = (0..100).map(|_| Id::new()).collect();
        for (i, ulid) in ulids.iter().enumerate() {
            storage.insert_async(&code, *ulid, &[i as f32; 128]).unwrap();
        }

        // Read pending queue
        let pending = storage.get_pending_ids(&code).unwrap();

        // Should be in insertion order (FIFO)
        assert_eq!(pending.len(), 100);
    }

    #[test]
    fn test_pending_queue_per_embedding() {
        let storage = setup_vector_storage_with_async_updater();
        let qwen = Embedding::new("qwen3");
        let gemma = Embedding::new("gemma");

        // Insert into different spaces
        for i in 0..50 {
            storage.insert_async(&qwen, Id::new(), &[i as f32; 128]).unwrap();
        }
        for i in 0..30 {
            storage.insert_async(&gemma, Id::new(), &[i as f32; 128]).unwrap();
        }

        // Pending counts should be independent
        let qwen_pending = storage.get_pending_count(&qwen).unwrap();
        let gemma_pending = storage.get_pending_count(&gemma).unwrap();

        assert_eq!(qwen_pending, 50);
        assert_eq!(gemma_pending, 30);
    }

    // =========================================================================
    // Async Updater Worker Tests (Task 5.3)
    // =========================================================================

    #[test]
    fn test_async_updater_drains_queue() {
        let config = AsyncUpdaterConfig {
            batch_size: 10,
            batch_timeout: Duration::from_millis(100),
            num_workers: 1,
        };
        let storage = setup_vector_storage_with_config(config);
        let code = Embedding::new("test");

        // Insert 50 vectors async
        for i in 0..50 {
            storage.insert_async(&code, Id::new(), &[i as f32; 128]).unwrap();
        }

        // Wait for queue to drain
        std::thread::sleep(Duration::from_millis(1000));

        // Pending queue should be empty
        let pending = storage.get_pending_count(&code).unwrap();
        assert_eq!(pending, 0, "Queue should be drained");

        // All should have graph edges now
        for id in 0u32..50 {
            let neighbors = storage.get_neighbors(&code, id, 0).unwrap();
            assert!(neighbors.len() > 0, "Node {} should have neighbors after drain", id);
        }
    }

    #[test]
    fn test_async_updater_batch_processing() {
        let batch_count = Arc::new(AtomicUsize::new(0));
        let batch_count_clone = Arc::clone(&batch_count);

        let config = AsyncUpdaterConfig {
            batch_size: 10,
            batch_timeout: Duration::from_millis(50),
            num_workers: 1,
        };
        let storage = setup_vector_storage_with_hooks(config, move |batch| {
            batch_count_clone.fetch_add(1, Ordering::SeqCst);
        });
        let code = Embedding::new("test");

        // Insert 45 vectors (should be 5 batches: 4 full + 1 partial)
        for i in 0..45 {
            storage.insert_async(&code, Id::new(), &[i as f32; 128]).unwrap();
        }

        std::thread::sleep(Duration::from_millis(500));

        let batches = batch_count.load(Ordering::SeqCst);
        assert!(batches >= 4 && batches <= 5, "Expected 4-5 batches, got {}", batches);
    }

    #[test]
    fn test_async_updater_timeout_triggers_batch() {
        let config = AsyncUpdaterConfig {
            batch_size: 100,  // Large batch size
            batch_timeout: Duration::from_millis(50),  // Short timeout
            num_workers: 1,
        };
        let storage = setup_vector_storage_with_config(config);
        let code = Embedding::new("test");

        // Insert just 5 vectors (less than batch_size)
        for i in 0..5 {
            storage.insert_async(&code, Id::new(), &[i as f32; 128]).unwrap();
        }

        // Wait for timeout to trigger
        std::thread::sleep(Duration::from_millis(200));

        // Should be processed despite not reaching batch_size
        let pending = storage.get_pending_count(&code).unwrap();
        assert_eq!(pending, 0, "Timeout should trigger processing");
    }

    // =========================================================================
    // Delete Handling Tests (Task 5.4)
    // =========================================================================

    #[test]
    fn test_delete_removes_from_search() {
        let storage = setup_vector_storage_with_async_updater();
        let code = Embedding::new("test");

        let ulid = Id::new();
        storage.insert_sync(&code, ulid, &[1.0; 128]).unwrap();

        // Verify searchable
        let results = storage.search(&code, &[1.0; 128], 1, 10).unwrap();
        assert_eq!(results.len(), 1);

        // Delete
        storage.delete(&code, ulid).unwrap();

        // Should not appear in search
        let results = storage.search(&code, &[1.0; 128], 1, 10).unwrap();
        assert!(results.iter().all(|(_, id)| *id != ulid));
    }

    #[test]
    fn test_delete_pending_vector() {
        let storage = setup_vector_storage_with_async_updater();
        let code = Embedding::new("test");

        // Insert async (still pending)
        let ulid = Id::new();
        storage.insert_async(&code, ulid, &[1.0; 128]).unwrap();

        // Delete before graph update
        storage.delete(&code, ulid).unwrap();

        // Wait for async updater
        std::thread::sleep(Duration::from_millis(500));

        // Should not appear in search
        let results = storage.search(&code, &[1.0; 128], 10, 50).unwrap();
        assert!(results.iter().all(|(_, id)| *id != ulid));
    }

    // =========================================================================
    // Crash Recovery Tests (Task 5.5)
    // =========================================================================

    #[test]
    fn test_crash_recovery_pending_queue() {
        let tmp = TempDir::new().unwrap();

        // First session: insert async, don't wait for drain
        {
            let storage = setup_vector_storage_at(tmp.path());
            let code = Embedding::new("test");

            for i in 0..50 {
                storage.insert_async(&code, Id::new(), &[i as f32; 128]).unwrap();
            }

            // Don't wait - simulate crash
            drop(storage);
        }

        // Second session: recovery
        {
            let storage = setup_vector_storage_at(tmp.path());
            let code = Embedding::new("test");

            // Should recover pending queue
            let pending = storage.get_pending_count(&code).unwrap();
            assert!(pending > 0, "Should have pending items after recovery");

            // Wait for drain
            std::thread::sleep(Duration::from_secs(2));

            // All should be processed
            let final_pending = storage.get_pending_count(&code).unwrap();
            assert_eq!(final_pending, 0);
        }
    }

    #[test]
    fn test_recovery_maintains_searchability() {
        let tmp = TempDir::new().unwrap();
        let ulids: Vec<Id>;

        // First session
        {
            let storage = setup_vector_storage_at(tmp.path());
            let code = Embedding::new("test");

            ulids = (0..100).map(|i| {
                let ulid = Id::new();
                storage.insert_async(&code, ulid, &[i as f32; 128]).unwrap();
                ulid
            }).collect();

            // Partial drain
            std::thread::sleep(Duration::from_millis(200));
        }

        // Second session: all should be searchable
        {
            let storage = setup_vector_storage_at(tmp.path());
            let code = Embedding::new("test");

            // Wait for full drain
            std::thread::sleep(Duration::from_secs(2));

            // All should be searchable
            for ulid in &ulids {
                let internal_id = storage.get_internal_id(&code, *ulid).unwrap();
                assert!(internal_id.is_some(), "ULID {:?} should be searchable", ulid);
            }
        }
    }

    // =========================================================================
    // Concurrency Tests
    // =========================================================================

    #[test]
    fn test_concurrent_insert_async() {
        let storage = Arc::new(setup_vector_storage_with_async_updater());
        let code = Embedding::new("test");

        let mut handles = Vec::new();
        for t in 0..10 {
            let s = Arc::clone(&storage);
            let c = code.clone();
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    let ulid = Id::new();
                    s.insert_async(&c, ulid, &[(t * 100 + i) as f32; 128]).unwrap();
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // Wait for drain
        std::thread::sleep(Duration::from_secs(3));

        // All 1000 should be indexed
        assert_eq!(storage.count(&code).unwrap(), 1000);
    }
}
```

**Test Coverage Checklist:**
- [ ] Async insert provides immediate searchability
- [ ] Async insert latency < 5ms P99
- [ ] Sync insert creates graph edges
- [ ] Pending queue maintains insertion order
- [ ] Per-embedding-space pending queues
- [ ] Async updater drains queue
- [ ] Batch processing respects batch_size
- [ ] Timeout triggers partial batch processing
- [ ] Delete removes from search immediately
- [ ] Delete pending vectors correctly
- [ ] Crash recovery preserves pending queue
- [ ] Recovery maintains searchability
- [ ] Concurrent async inserts

**Benchmarks:**
```rust
#[bench]
fn bench_insert_async_latency(b: &mut Bencher) {
    let storage = setup_vector_storage_with_async_updater();
    let code = Embedding::new("bench");

    b.iter(|| {
        let ulid = Id::new();
        storage.insert_async(&code, ulid, &random_vector(128)).unwrap();
    });
}
// Target: < 1ms median, < 5ms P99

#[bench]
fn bench_insert_sync_latency(b: &mut Bencher) {
    let storage = setup_vector_storage_temp();
    let code = Embedding::new("bench");

    b.iter(|| {
        let ulid = Id::new();
        storage.insert_sync(&code, ulid, &random_vector(128)).unwrap();
    });
}
// Target: < 20ms median, < 50ms P99

#[bench]
fn bench_async_updater_throughput(b: &mut Bencher) {
    // Measure sustained insert throughput with async updater
    let storage = setup_vector_storage_with_async_updater();
    let code = Embedding::new("bench");

    b.iter(|| {
        for _ in 0..1000 {
            storage.insert_async(&code, Id::new(), &random_vector(128)).unwrap();
        }
        // Wait for drain
        while storage.get_pending_count(&code).unwrap() > 0 {
            std::thread::sleep(Duration::from_millis(10));
        }
    });
}
// Target: > 5,000 inserts/sec sustained
```

---

## Phase 6: Production Hardening

**Goal:** Production-ready vector search with full feature set.

### Task 6.1: Delete Support [FUNC-3]

```rust
impl VectorStorage {
    pub fn delete(&self, ulid: Id) -> Result<()> {
        // 1. Lookup internal ID
        let internal_id = self.get_internal_id(ulid)?
            .ok_or_else(|| anyhow!("Vector not found"))?;

        // 2. Mark as deleted in metadata
        self.mark_deleted(internal_id)?;

        // 3. Remove edges (optional - can be lazy)
        self.remove_all_edges(internal_id)?;

        // 4. Return ID to free list
        self.id_alloc.free(internal_id);

        // 5. Optionally remove from RocksDB (or leave for compaction)
        self.remove_vector(internal_id)?;
        self.remove_id_mappings(ulid, internal_id)?;

        Ok(())
    }
}
```

**Effort:** 1-2 days

### Task 6.2: Concurrent Access [CON-3]

- Use RocksDB snapshot isolation for reads
- Write serialization via transaction API

**Effort:** 2-3 days

### Task 6.3: 1B Scale Validation

- Benchmark at 10M, 100M, 1B scales
- Validate memory constraints
- Performance profiling

**Effort:** 1-2 weeks (including hardware procurement if needed)

### Phase 6 Validation & Tests

```rust
// libs/db/src/vector/tests/production_tests.rs

#[cfg(test)]
mod production_tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    // =========================================================================
    // Delete Support Tests (Task 6.1)
    // =========================================================================

    #[test]
    fn test_delete_basic() {
        let storage = setup_vector_storage_temp();
        let code = Embedding::new("test");

        let ulid = Id::new();
        storage.insert(&code, ulid, &[1.0; 128]).unwrap();

        assert!(storage.exists(&code, ulid).unwrap());

        storage.delete(&code, ulid).unwrap();

        assert!(!storage.exists(&code, ulid).unwrap());
    }

    #[test]
    fn test_delete_frees_internal_id() {
        let storage = setup_vector_storage_temp();
        let code = Embedding::new("test");

        // Insert and delete
        let ulid1 = Id::new();
        storage.insert(&code, ulid1, &[1.0; 128]).unwrap();
        let internal_id1 = storage.get_internal_id(&code, ulid1).unwrap().unwrap();

        storage.delete(&code, ulid1).unwrap();

        // New insert should reuse the freed ID
        let ulid2 = Id::new();
        storage.insert(&code, ulid2, &[2.0; 128]).unwrap();
        let internal_id2 = storage.get_internal_id(&code, ulid2).unwrap().unwrap();

        assert_eq!(internal_id1, internal_id2, "Freed ID should be reused");
    }

    #[test]
    fn test_delete_removes_from_neighbors() {
        let storage = setup_vector_storage_temp();
        let code = Embedding::new("test");

        // Insert several vectors
        let ulids: Vec<Id> = (0..10).map(|i| {
            let ulid = Id::new();
            storage.insert(&code, ulid, &[i as f32; 128]).unwrap();
            ulid
        }).collect();

        // Get neighbors of node 0
        let initial_neighbors = storage.get_neighbors(&code, 0, 0).unwrap();
        let has_node_5 = initial_neighbors.contains(5);

        if has_node_5 {
            // Delete node 5
            storage.delete(&code, ulids[5]).unwrap();

            // Node 5 should be removed from all neighbor lists (eventually)
            // Note: lazy cleanup may not be immediate
            storage.compact(&code).unwrap();

            let updated_neighbors = storage.get_neighbors(&code, 0, 0).unwrap();
            assert!(!updated_neighbors.contains(5), "Deleted node should be removed from neighbors");
        }
    }

    #[test]
    fn test_delete_not_in_search_results() {
        let storage = setup_vector_storage_temp();
        let code = Embedding::new("test");

        let ulids: Vec<Id> = (0..100).map(|i| {
            let ulid = Id::new();
            storage.insert(&code, ulid, &[i as f32; 128]).unwrap();
            ulid
        }).collect();

        // Delete some
        for i in (0..100).step_by(2) {
            storage.delete(&code, ulids[i]).unwrap();
        }

        // Search should not return deleted ULIDs
        let results = storage.search(&code, &[50.0; 128], 20, 100).unwrap();

        for (_, ulid) in &results {
            let idx = ulids.iter().position(|u| u == ulid).unwrap();
            assert!(idx % 2 == 1, "Deleted ULID {} (idx {}) appeared in results", ulid, idx);
        }
    }

    // =========================================================================
    // Concurrent Access Tests (Task 6.2)
    // =========================================================================

    #[test]
    fn test_concurrent_reads() {
        let storage = Arc::new(setup_vector_storage_with_vectors(10_000));
        let code = Embedding::new("test");

        let mut handles = Vec::new();
        for _ in 0..10 {
            let s = Arc::clone(&storage);
            let c = code.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    let query = random_vector(128);
                    s.search(&c, &query, 10, 50).unwrap();
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_concurrent_read_write() {
        let storage = Arc::new(setup_vector_storage_temp());
        let code = Embedding::new("test");

        // Seed with some vectors
        for i in 0..100 {
            storage.insert(&code, Id::new(), &[i as f32; 128]).unwrap();
        }

        let mut handles = Vec::new();

        // Writers
        for t in 0..4 {
            let s = Arc::clone(&storage);
            let c = code.clone();
            handles.push(thread::spawn(move || {
                for i in 0..50 {
                    s.insert(&c, Id::new(), &[(t * 50 + i) as f32; 128]).unwrap();
                }
            }));
        }

        // Readers
        for _ in 0..4 {
            let s = Arc::clone(&storage);
            let c = code.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    let query = random_vector(128);
                    let _ = s.search(&c, &query, 10, 50);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // Verify count
        assert_eq!(storage.count(&code).unwrap(), 300); // 100 + 4*50
    }

    #[test]
    fn test_snapshot_isolation() {
        let storage = Arc::new(setup_vector_storage_temp());
        let code = Embedding::new("test");

        // Insert initial vectors
        for i in 0..100 {
            storage.insert(&code, Id::new(), &[i as f32; 128]).unwrap();
        }

        // Take snapshot
        let snapshot = storage.snapshot(&code).unwrap();

        // Insert more while holding snapshot
        for i in 100..200 {
            storage.insert(&code, Id::new(), &[i as f32; 128]).unwrap();
        }

        // Snapshot should only see original 100
        let snapshot_count = snapshot.count().unwrap();
        assert_eq!(snapshot_count, 100);

        // Current should see 200
        assert_eq!(storage.count(&code).unwrap(), 200);
    }

    // =========================================================================
    // Scale Tests (Task 6.3) - Run with --release --ignored
    // =========================================================================

    #[test]
    #[ignore]  // Run explicitly: cargo test --release scale -- --ignored
    fn test_scale_10k() {
        let storage = setup_vector_storage_temp();
        let code = Embedding::new("test");

        let start = Instant::now();
        for _ in 0..10_000 {
            storage.insert(&code, Id::new(), &random_vector(128)).unwrap();
        }
        let insert_time = start.elapsed();

        println!("10K insert: {:?} ({:.0} vec/sec)",
            insert_time, 10_000.0 / insert_time.as_secs_f64());

        // Search benchmark
        let start = Instant::now();
        for _ in 0..1000 {
            storage.search(&code, &random_vector(128), 10, 100).unwrap();
        }
        let search_time = start.elapsed();

        println!("1K searches at 10K: {:?} ({:.1} QPS)",
            search_time, 1000.0 / search_time.as_secs_f64());

        // Recall check
        let recall = measure_recall(&storage, &code, 100);
        println!("Recall@10 at 10K: {:.1}%", recall * 100.0);

        assert!(recall > 0.90, "Recall should be > 90%");
    }

    #[test]
    #[ignore]
    fn test_scale_100k() {
        let storage = setup_vector_storage_temp();
        let code = Embedding::new("test");

        let start = Instant::now();
        for _ in 0..100_000 {
            storage.insert(&code, Id::new(), &random_vector(128)).unwrap();
        }
        let insert_time = start.elapsed();

        println!("100K insert: {:?} ({:.0} vec/sec)",
            insert_time, 100_000.0 / insert_time.as_secs_f64());

        // Memory check
        let stats = storage.stats(&code).unwrap();
        println!("Memory at 100K: {} MB", stats.memory_bytes / 1_000_000);

        // Search benchmark
        let start = Instant::now();
        for _ in 0..100 {
            storage.search(&code, &random_vector(128), 10, 100).unwrap();
        }
        let search_time = start.elapsed();

        println!("100 searches at 100K: {:?} ({:.1} QPS)",
            search_time, 100.0 / search_time.as_secs_f64());

        // Recall check
        let recall = measure_recall(&storage, &code, 50);
        println!("Recall@10 at 100K: {:.1}%", recall * 100.0);

        assert!(recall > 0.90);
    }

    #[test]
    #[ignore]
    fn test_scale_1m() {
        let storage = setup_vector_storage_temp();
        let code = Embedding::new("test");

        println!("Building 1M vector index...");
        let start = Instant::now();
        for i in 0..1_000_000 {
            if i % 100_000 == 0 {
                println!("  {} vectors inserted...", i);
            }
            storage.insert(&code, Id::new(), &random_vector(128)).unwrap();
        }
        let insert_time = start.elapsed();

        println!("1M insert: {:?} ({:.0} vec/sec)",
            insert_time, 1_000_000.0 / insert_time.as_secs_f64());

        // Memory check
        let stats = storage.stats(&code).unwrap();
        println!("Memory at 1M: {} GB", stats.memory_bytes / 1_000_000_000);
        assert!(stats.memory_bytes < 5_000_000_000, "Memory should be < 5GB at 1M");

        // Search benchmark
        let start = Instant::now();
        for _ in 0..100 {
            storage.search(&code, &random_vector(128), 10, 100).unwrap();
        }
        let search_time = start.elapsed();

        let qps = 100.0 / search_time.as_secs_f64();
        println!("100 searches at 1M: {:?} ({:.1} QPS, {:.1}ms avg)",
            search_time, qps, 1000.0 / qps);

        // Recall check
        let recall = measure_recall(&storage, &code, 20);
        println!("Recall@10 at 1M: {:.1}%", recall * 100.0);

        // Targets
        assert!(qps > 100.0, "QPS at 1M should be > 100");
        assert!(recall > 0.90, "Recall at 1M should be > 90%");
    }

    // =========================================================================
    // End-to-End Integration Test
    // =========================================================================

    #[test]
    fn test_e2e_workflow() {
        let storage = setup_vector_storage_temp();

        // Multiple embedding spaces
        let qwen = Embedding::new("qwen3");
        let gemma = Embedding::new("gemma");

        // Insert into both spaces
        for i in 0..100 {
            let ulid = Id::new();
            storage.insert(&qwen, ulid, &[i as f32; 128]).unwrap();
            storage.insert(&gemma, ulid, &[(i + 100) as f32; 128]).unwrap();
        }

        // Search in both spaces
        let qwen_results = storage.search(&qwen, &[50.0; 128], 10, 50).unwrap();
        let gemma_results = storage.search(&gemma, &[150.0; 128], 10, 50).unwrap();

        assert_eq!(qwen_results.len(), 10);
        assert_eq!(gemma_results.len(), 10);

        // Delete from one space shouldn't affect other
        let ulid_to_delete = qwen_results[0].1;
        storage.delete(&qwen, ulid_to_delete).unwrap();

        // Should still exist in gemma
        assert!(storage.exists(&gemma, ulid_to_delete).unwrap());

        // Restart simulation
        let tmp = TempDir::new().unwrap();
        {
            let storage = setup_vector_storage_at(tmp.path());
            for i in 0..50 {
                storage.insert(&qwen, Id::new(), &[i as f32; 128]).unwrap();
            }
        }
        {
            let storage = setup_vector_storage_at(tmp.path());
            assert_eq!(storage.count(&qwen).unwrap(), 50);
        }
    }

    // Helper function
    fn measure_recall(storage: &VectorStorage, code: &Embedding, num_queries: usize) -> f64 {
        let mut total_recall = 0.0;
        // ... brute force comparison logic
        total_recall / num_queries as f64
    }
}
```

**Test Coverage Checklist:**
- [ ] Basic delete functionality
- [ ] Delete frees internal ID for reuse
- [ ] Delete removes node from neighbor lists
- [ ] Deleted nodes excluded from search
- [ ] Concurrent reads don't conflict
- [ ] Concurrent read/write correctness
- [ ] Snapshot isolation
- [ ] Scale test: 10K vectors
- [ ] Scale test: 100K vectors
- [ ] Scale test: 1M vectors
- [ ] End-to-end workflow test

**Benchmarks:**
```rust
// Run with: cargo bench --release

#[bench]
fn bench_full_workflow(b: &mut Bencher) {
    b.iter(|| {
        let storage = setup_vector_storage_temp();
        let code = Embedding::new("bench");

        // Insert
        for _ in 0..1000 {
            storage.insert(&code, Id::new(), &random_vector(128)).unwrap();
        }

        // Search
        for _ in 0..100 {
            storage.search(&code, &random_vector(128), 10, 50).unwrap();
        }

        // Delete
        for id in storage.all_ulids(&code).unwrap().take(100) {
            storage.delete(&code, id).unwrap();
        }
    });
}
```

**Scale Validation Script:**
```bash
#!/bin/bash
# scripts/scale_validation.sh

echo "=== Vector Index Scale Validation ==="

# 10K
cargo test --release test_scale_10k -- --ignored --nocapture

# 100K
cargo test --release test_scale_100k -- --ignored --nocapture

# 1M (requires ~8GB RAM)
cargo test --release test_scale_1m -- --ignored --nocapture

# Generate report
echo "=== Summary ==="
# ... parse output and generate report
```

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
Phase 5: Async Updater ───────────────────────────────┘
    │
    ▼
Phase 6: Production Hardening
```

**Navigation Layer Integration:** Tasks 2.7-2.9 are placed in Phase 2 (rather than a
separate phase) because the navigation layer is fundamental to HNSW algorithm correctness.
The memory caching (2.9) can be deferred if timeline is tight, but 2.7-2.8 are required.

---

## Effort Summary

| Phase | Tasks | Effort | Cumulative |
|-------|-------|--------|------------|
| Phase 0: Foundation | 0.1-0.3 | 1.25 days | 1.25 days |
| Phase 1: ID Management | 1.1-1.4 | 3-4 days | 4-5 days |
| Phase 2: HNSW2 Core + Navigation | 2.1-2.9 | 11-17 days | 15-22 days |
| Phase 3: Batch APIs | 3.1-3.4 | 2.5-3 days | 17-25 days |
| Phase 4: RaBitQ | 4.1-4.5 | 5-7 days | 22-32 days |
| Phase 5: Async Updater | 5.1-5.5 | 3-4 days | 25-36 days |
| Phase 6: Production | 6.1-6.3 | 4-7 days | 29-43 days |
| **Total** | | **~6-9 weeks** | |

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
3. **SIMD Distance Functions**: `motlie_core::distance` module ([mod.rs](../../../core/src/distance/mod.rs))

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

Session 6: Phase 5 (Async Updater)
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
