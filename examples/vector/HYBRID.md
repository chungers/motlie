# Hybrid Vector Search Architecture

**Phase 4 - Billion-Scale Production Architecture**

A comprehensive design for production-grade vector search at billion scale with online inserts, low latency, and high recall.

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
IVFPQ.md            ← Phase 3: GPU-accelerated search (optional)
    ↓
HYBRID.md (this)    ← Phase 4: Billion-scale production (you are here)
```

**Links**: [REQUIREMENTS.md](./REQUIREMENTS.md) → [POC.md](./POC.md) → [HNSW2.md](./HNSW2.md) → [IVFPQ.md](./IVFPQ.md) → **HYBRID.md**

## Target Requirements

This phase is the culmination of all prior work, achieving:

| Requirement | Target | Addressed By |
|-------------|--------|--------------|
| **SCALE-1** | 1 billion vectors | PQ compression (8 bytes/vector) |
| **SCALE-3** | < 64 GB RAM at 1B | In-memory HNSW nav (~50MB) + PQ (8GB) |
| **LAT-4** | < 10ms P99 insert | Async graph updater |
| **LAT-3** | < 100ms P99 search at 1B | HNSW navigation + PQ re-rank |
| **REC-2** | > 95% recall at 1B | Exact distance re-ranking |
| **THR-1** | > 5,000 inserts/s | Batched async updates |
| **THR-4** | > 100 QPS at 1B | SIMD PQ distance |
| **STOR-4** | PQ compression | 512x compression |
| **STOR-5** | SIMD distance | AVX2/AVX-512 |

See [REQUIREMENTS.md](./REQUIREMENTS.md) for full requirement definitions.

---

## Executive Summary

This document proposes a hybrid architecture combining:
- **HNSW2** for navigation layer (small, fast, memory-resident)
- **RaBitQ** for training-free binary compression (32x compression, DATA-1 compliant)
- **RocksDB** for durability and efficient disk access
- **Async graph updater** for decoupling insert latency from graph quality

> **DATA-1 Constraint**: The original design proposed Product Quantization (PQ), which requires
> training on representative data. Per [REQUIREMENTS.md Section 5.4](./REQUIREMENTS.md), motlie_db
> has no pre-training data available. This document has been updated to use **RaBitQ** instead,
> which provides training-free compression with theoretical guarantees.

**Target specifications:**
| Metric | Target |
|--------|--------|
| Scale | 1 billion vectors |
| Insert latency | < 10ms p99 |
| Search latency | < 50ms p99 |
| Recall@10 | > 95% |
| Memory footprint | < 64GB |
| Disk usage | ~50GB |

---

## Table of Contents

1. [Problem Analysis](#problem-analysis)
2. [Architecture Overview](#architecture-overview)
3. [RocksDB Schema Design](#rocksdb-schema-design)
4. [Module Architecture](#module-architecture)
5. [API Design](#api-design)
6. [Resource Projections](#resource-projections)
7. [Performance Projections](#performance-projections)
8. [Implementation Phases](#implementation-phases)

---

## Problem Analysis

### Current Limitations

Based on benchmarks documented in [PERF.md](./PERF.md):

| Algorithm | Scale | Build Time | Recall@10 | QPS | Bottleneck |
|-----------|-------|------------|-----------|-----|------------|
| HNSW | 1M | 7.0 hours | 95.3% | 47 | Per-insert flush, graph traversal |
| Vamana | 1M | 14.9 hours | 81.9% | 74 | 3-pass construction, DB reads |

**Key bottlenecks identified:**
1. **Synchronous flush per insert** - 20-25ms overhead per vector
2. **Graph traversal I/O** - Each greedy search reads O(log N × M) edges from disk
3. **No batching** - Single-threaded construction with no pipelining
4. **Full vector storage** - 512 bytes per 128D float32 vector

### Requirements Analysis

| Requirement | Current State | Target |
|-------------|---------------|--------|
| Online inserts | Supported but slow (40 vec/s) | 5,000-10,000 vec/s |
| Search latency | 5-12ms at 100K | < 50ms at 1B |
| Recall@10 | 95.3% (HNSW 1M) | > 95% at 1B |
| Memory at 1B | 512GB (raw vectors) | < 64GB |
| GPU/SIMD | Not implemented | AVX2/AVX-512 PQ distance |

### Why Hybrid Architecture?

A single algorithm cannot meet all requirements:

| Approach | Pros | Cons |
|----------|------|------|
| HNSW only | High recall, fast search | Memory-bound, slow inserts |
| Vamana only | Disk-efficient | Lower recall, slow construction |
| IVF-PQ only | Fast, compressed | Lower recall on cold data |
| **Hybrid HNSW + PQ** | Fast search, low memory, online inserts | Complexity |

The hybrid approach:
1. Uses HNSW for fast navigation to approximate region
2. Uses PQ for 512x compressed storage
3. Re-ranks with exact distances from disk on final candidates

---

## Architecture Overview

```
                    ┌─────────────────────────────────────────────┐
                    │              Hybrid Vector Index            │
                    └─────────────────────────────────────────────┘
                                         │
         ┌───────────────────────────────┼───────────────────────────────┐
         ▼                               ▼                               ▼
┌─────────────────┐            ┌─────────────────┐            ┌─────────────────┐
│   Navigation    │            │   Compressed    │            │   Full Vector   │
│   Layer (HNSW)  │            │   Storage (PQ)  │            │   Storage       │
├─────────────────┤            ├─────────────────┤            ├─────────────────┤
│ • Entry points  │            │ • PQ codes      │            │ • Raw vectors   │
│ • Top-k layers  │            │ • Codebooks     │            │ • On-demand     │
│ • Memory-only   │            │ • 8 bytes/vec   │            │ • Re-ranking    │
│ • ~50MB at 1B   │            │ • 8GB at 1B     │            │ • 512GB at 1B   │
└────────┬────────┘            └────────┬────────┘            └────────┬────────┘
         │                               │                               │
         └───────────────────────────────┼───────────────────────────────┘
                                         │
                              ┌──────────┴──────────┐
                              │     RocksDB         │
                              │  (7 Column Families)│
                              └─────────────────────┘
```

### Component Responsibilities

| Component | Purpose | Storage | Memory |
|-----------|---------|---------|--------|
| **Navigation Layer** | Fast approximate neighbor finding | In-memory only | ~50MB |
| **PQ Codes** | Compressed vector representation | RocksDB CF | 8GB at 1B |
| **Codebooks** | PQ quantization tables | RocksDB CF + cache | ~1MB |
| **Graph Edges** | HNSW neighbor lists | RocksDB CF | On-demand |
| **Full Vectors** | Exact distance computation | RocksDB CF | On-demand |
| **Async Updater** | Background graph maintenance | Thread + queue | Minimal |

### Data Flow

#### Insert Path
```
Insert Request
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. Encode vector with PQ                                                │
│ 2. Store PQ code in RocksDB (pq_codes CF)                              │
│ 3. Store full vector in RocksDB (vectors CF)                           │
│ 4. Quick insert into pending queue                                      │
│ 5. Return success immediately (async graph update)                      │
└─────────────────────────────────────────────────────────────────────────┘
     │
     ▼ (background)
┌─────────────────────────────────────────────────────────────────────────┐
│ Async Graph Updater:                                                    │
│ 1. Batch pending vectors (e.g., every 100 or 100ms)                    │
│ 2. Greedy search from entry point using PQ distances                   │
│ 3. Connect to nearest neighbors                                         │
│ 4. Prune connections using RNG (merge operator)                        │
│ 5. Update navigation layer entry points                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Search Path
```
Search Query
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Navigation (HNSW layers)                                       │
│ • Start from entry point at top layer                                  │
│ • Greedy descent through layers                                        │
│ • Uses in-memory navigation graph                                      │
│ • O(log N) complexity                                                  │
└─────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 2: Candidate expansion (PQ distances)                            │
│ • Beam search at layer 0 using PQ codes                               │
│ • SIMD-accelerated PQ distance computation                            │
│ • Expand to ef candidates                                              │
│ • O(ef × M) PQ distance computations                                  │
└─────────────────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 3: Re-ranking (exact distances)                                  │
│ • Load top 2×K full vectors from disk                                 │
│ • Compute exact L2 distances                                          │
│ • Return top K results                                                │
│ • O(2K × D) exact distance computations                               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## RocksDB Schema Design

### Column Families

The schema uses 7 column families optimized for vector search workloads:

```rust
pub const ALL_VECTOR_COLUMN_FAMILIES: &[&str] = &[
    "vectors",       // Full vectors for re-ranking
    "pq_codes",      // PQ-compressed vectors (8 bytes each)
    "pq_codebooks",  // PQ codebooks (quantization tables)
    "graph_edges",   // HNSW neighbor lists (RoaringBitmap)
    "nav_layer",     // Navigation layer entry points
    "metadata",      // Index metadata and statistics
    "pending",       // Pending vectors for async processing
];
```

### Column Family Details

#### 1. `vectors` - Full Vector Storage
```rust
// Key: [16-byte ID]
// Value: MessagePack([f32; D]) compressed with LZ4

pub struct VectorsCF;
impl ColumnFamilyRecord for VectorsCF {
    const CF_NAME: &'static str = "vectors";
    type Key = Id;                    // 16 bytes
    type Value = Vec<f32>;            // D floats (128 for SIFT)

    fn column_family_options() -> Options {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        // Optimize for large sequential reads during re-ranking
        opts.set_optimize_filters_for_hits(true);
        opts
    }
}
```

#### 2. `pq_codes` - Compressed Vectors
```rust
// Key: [16-byte ID]
// Value: [u8; M] where M = num_subquantizers (typically 8-16)

pub struct PqCodesCF;
impl ColumnFamilyRecord for PqCodesCF {
    const CF_NAME: &'static str = "pq_codes";
    type Key = Id;                    // 16 bytes
    type Value = [u8; 8];             // 8 subquantizers × 1 byte = 8 bytes

    fn column_family_options() -> Options {
        let mut opts = Options::default();
        // No compression - already minimal
        opts.set_compression_type(rocksdb::DBCompressionType::None);
        // Optimize for random reads during search
        opts.set_block_size(4 * 1024);  // 4KB blocks
        opts.set_bloom_filter(10, true);
        opts
    }
}
```

#### 3. `pq_codebooks` - Quantization Tables
```rust
// Key: [subquantizer_id: u8]
// Value: MessagePack([[f32; D/M]; 256]) - 256 centroids per subquantizer

pub struct PqCodebooksCF;
impl ColumnFamilyRecord for PqCodebooksCF {
    const CF_NAME: &'static str = "pq_codebooks";
    type Key = u8;                              // Subquantizer index (0-7)
    type Value = Vec<[f32; 16]>;                // 256 centroids × (D/M) dims

    // Cached in memory for SIMD distance computation
    fn column_family_options() -> Options {
        let mut opts = Options::default();
        opts.set_compression_type(rocksdb::DBCompressionType::None);
        opts
    }
}
```

#### 4. `graph_edges` - HNSW Neighbor Lists
```rust
// Key: [16-byte ID | layer: u8]
// Value: RoaringBitmap of neighbor IDs (compressed)

pub struct GraphEdgesCF;
impl ColumnFamilyRecord for GraphEdgesCF {
    const CF_NAME: &'static str = "graph_edges";
    type Key = (Id, u8);                        // (node_id, layer)
    type Value = RoaringBitmap;                 // Neighbor set

    fn column_family_options() -> Options {
        let mut opts = Options::default();
        opts.set_prefix_extractor(rocksdb::SliceTransform::create_fixed_prefix(16));
        opts.set_bloom_filter(10, true);
        // Enable merge operator for atomic neighbor updates
        opts
    }
}
```

#### 5. `nav_layer` - Navigation Layer
```rust
// Key: [layer: u8]
// Value: Vec<Id> - entry points for each layer

pub struct NavLayerCF;
impl ColumnFamilyRecord for NavLayerCF {
    const CF_NAME: &'static str = "nav_layer";
    type Key = u8;                              // Layer number
    type Value = Vec<Id>;                       // Entry point IDs
}
```

#### 6. `metadata` - Index Statistics
```rust
// Key: "stats" | "config" | "checkpoint"
// Value: MessagePack(struct)

pub struct MetadataCF;
impl ColumnFamilyRecord for MetadataCF {
    const CF_NAME: &'static str = "metadata";
    type Key = String;
    type Value = IndexMetadata;
}

#[derive(Serialize, Deserialize)]
pub struct IndexMetadata {
    pub total_vectors: u64,
    pub dimension: u32,
    pub m: u32,                // HNSW connections per node
    pub ef_construction: u32,
    pub num_subquantizers: u8,
    pub last_checkpoint: u64,
}
```

#### 7. `pending` - Async Insert Queue
```rust
// Key: [timestamp: u64 | id: 16 bytes]
// Value: Vec<f32> - full vector

pub struct PendingCF;
impl ColumnFamilyRecord for PendingCF {
    const CF_NAME: &'static str = "pending";
    type Key = (u64, Id);                       // (insert_time, id)
    type Value = Vec<f32>;                      // Full vector
}
```

### Merge Operators for Atomic Updates

RocksDB merge operators enable lock-free concurrent graph updates:

```rust
pub struct NeighborMergeOperator;

impl MergeOperator for NeighborMergeOperator {
    fn full_merge(
        &self,
        key: &[u8],
        existing: Option<&[u8]>,
        operands: &MergeOperands,
    ) -> Option<Vec<u8>> {
        // Decode existing neighbor set
        let mut neighbors = existing
            .map(|bytes| RoaringBitmap::deserialize_from(bytes).unwrap())
            .unwrap_or_default();

        // Apply each operand (add/remove operations)
        for operand in operands {
            match operand[0] {
                0 => { // Add neighbor
                    let id = u32::from_be_bytes(operand[1..5].try_into().unwrap());
                    neighbors.insert(id);
                }
                1 => { // Remove neighbor
                    let id = u32::from_be_bytes(operand[1..5].try_into().unwrap());
                    neighbors.remove(id);
                }
                _ => {}
            }

            // Enforce M_max limit via RNG pruning
            if neighbors.len() > M_MAX as u64 {
                // Apply robust neighbor pruning inline
                self.prune_neighbors(&mut neighbors, key);
            }
        }

        let mut buffer = Vec::new();
        neighbors.serialize_into(&mut buffer).unwrap();
        Some(buffer)
    }
}
```

---

## Module Architecture

Following motlie_db design patterns, the vector search module will be organized as:

### Directory Structure

```
libs/db/src/
├── lib.rs                    # Re-exports vector module
├── graph/                    # Existing graph module
├── fulltext/                 # Existing fulltext module
└── vector/                   # NEW: Vector search module
    ├── mod.rs               # Module exports, VectorStorage, VectorIndex
    ├── schema.rs            # Column family definitions
    ├── mutation.rs          # InsertVector, DeleteVector mutations
    ├── query.rs             # SearchKNN, GetVector queries
    ├── reader.rs            # Query consumer infrastructure
    ├── writer.rs            # Mutation consumer infrastructure
    ├── pq.rs                # Product Quantization codec
    ├── hnsw.rs              # HNSW navigation layer
    ├── async_updater.rs     # Background graph updater thread
    └── simd.rs              # SIMD distance computation
```

### Module Definitions

#### `vector/mod.rs` - Core Types and Storage

```rust
//! Vector module - Hybrid HNSW + PQ vector search.
//!
//! ## Module Structure
//!
//! - `mod.rs` - VectorStorage, VectorIndex, module exports
//! - `schema.rs` - RocksDB column family definitions
//! - `mutation.rs` - InsertVector, DeleteVector mutations
//! - `query.rs` - SearchKNN, GetVector queries
//! - `pq.rs` - Product Quantization codec
//! - `hnsw.rs` - HNSW navigation layer
//! - `async_updater.rs` - Background graph maintenance
//! - `simd.rs` - SIMD-accelerated distance computation

use std::path::Path;
use std::sync::Arc;

pub mod schema;
pub mod mutation;
pub mod query;
pub mod reader;
pub mod writer;
pub mod pq;
pub mod hnsw;
pub mod async_updater;
pub mod simd;

// Re-exports
pub use mutation::{InsertVector, DeleteVector, Mutation};
pub use query::{SearchKNN, GetVector, Query};
pub use pq::{ProductQuantizer, PQConfig};
pub use hnsw::{HnswConfig, NavigationLayer};

/// Vector index configuration
#[derive(Debug, Clone)]
pub struct VectorConfig {
    /// Vector dimension
    pub dimension: u32,
    /// HNSW M parameter (connections per node)
    pub m: u32,
    /// HNSW ef_construction parameter
    pub ef_construction: u32,
    /// Number of PQ subquantizers (dimension / subquantizer_dim)
    pub num_subquantizers: u8,
    /// Enable SIMD distance computation
    pub enable_simd: bool,
}

impl Default for VectorConfig {
    fn default() -> Self {
        Self {
            dimension: 128,
            m: 16,
            ef_construction: 200,
            num_subquantizers: 8,
            enable_simd: true,
        }
    }
}

/// Vector storage (RocksDB backend)
pub struct VectorStorage {
    db_path: PathBuf,
    db: Option<TransactionDB>,
    config: VectorConfig,
}

impl VectorStorage {
    pub fn readwrite(path: &Path, config: VectorConfig) -> Self {
        Self {
            db_path: path.to_path_buf(),
            db: None,
            config,
        }
    }

    pub fn ready(&mut self) -> Result<()> {
        // Initialize RocksDB with column families
        // Load/create codebooks
        // Initialize navigation layer cache
    }
}

/// Vector index (combines storage + algorithms)
pub struct VectorIndex {
    storage: Arc<VectorStorage>,
    pq: ProductQuantizer,
    nav_layer: NavigationLayer,
    async_updater: AsyncUpdaterHandle,
}
```

#### `vector/pq.rs` - Product Quantization

```rust
//! Product Quantization for 512x vector compression.

use std::arch::x86_64::*;

/// Product Quantizer configuration
#[derive(Debug, Clone)]
pub struct PQConfig {
    /// Number of subquantizers (M)
    pub num_subquantizers: u8,
    /// Number of centroids per subquantizer (K, typically 256)
    pub num_centroids: u16,
    /// Vector dimension
    pub dimension: u32,
}

/// Product Quantizer codec
pub struct ProductQuantizer {
    config: PQConfig,
    /// Codebooks: [M][K][D/M] centroids
    codebooks: Vec<Vec<Vec<f32>>>,
    /// Precomputed distance tables for SIMD
    distance_tables: Option<Vec<Vec<f32>>>,
}

impl ProductQuantizer {
    /// Encode a vector to PQ codes
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let mut codes = Vec::with_capacity(self.config.num_subquantizers as usize);
        let subvec_dim = vector.len() / self.config.num_subquantizers as usize;

        for (m, codebook) in self.codebooks.iter().enumerate() {
            let subvec = &vector[m * subvec_dim..(m + 1) * subvec_dim];
            let mut best_code = 0u8;
            let mut best_dist = f32::MAX;

            for (k, centroid) in codebook.iter().enumerate() {
                let dist = euclidean_distance(subvec, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_code = k as u8;
                }
            }
            codes.push(best_code);
        }
        codes
    }

    /// Decode PQ codes back to approximate vector
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let subvec_dim = self.config.dimension as usize / codes.len();
        let mut vector = Vec::with_capacity(self.config.dimension as usize);

        for (m, &code) in codes.iter().enumerate() {
            vector.extend_from_slice(&self.codebooks[m][code as usize]);
        }
        vector
    }

    /// Compute asymmetric distance using precomputed tables (SIMD)
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    pub unsafe fn asymmetric_distance_simd(
        &self,
        query: &[f32],
        codes: &[u8],
    ) -> f32 {
        // Precompute distance table for query
        let tables = self.precompute_distance_table(query);

        // Sum distances using SIMD
        let mut sum = _mm256_setzero_ps();

        for chunk in codes.chunks(8) {
            // Gather 8 distances from tables
            let indices = _mm256_set_epi32(
                tables[7][chunk[7] as usize] as i32,
                tables[6][chunk[6] as usize] as i32,
                tables[5][chunk[5] as usize] as i32,
                tables[4][chunk[4] as usize] as i32,
                tables[3][chunk[3] as usize] as i32,
                tables[2][chunk[2] as usize] as i32,
                tables[1][chunk[1] as usize] as i32,
                tables[0][chunk[0] as usize] as i32,
            );

            // ... SIMD accumulation
        }

        // Horizontal sum
        // ...
    }
}
```

#### `vector/async_updater.rs` - Background Graph Maintenance

```rust
//! Async graph updater for decoupling insert latency from graph quality.

use tokio::sync::mpsc;
use std::time::Duration;

/// Configuration for async updater
#[derive(Debug, Clone)]
pub struct AsyncUpdaterConfig {
    /// Batch size before triggering update
    pub batch_size: usize,
    /// Maximum wait time before flushing batch
    pub batch_timeout: Duration,
    /// Number of concurrent update workers
    pub num_workers: usize,
}

impl Default for AsyncUpdaterConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            batch_timeout: Duration::from_millis(100),
            num_workers: 4,
        }
    }
}

/// Update operation for background processing
pub enum UpdateOp {
    Insert { id: Id, vector: Vec<f32> },
    Delete { id: Id },
    RebuildLayer { layer: u8 },
}

/// Handle to the async updater
pub struct AsyncUpdaterHandle {
    tx: mpsc::Sender<UpdateOp>,
}

impl AsyncUpdaterHandle {
    /// Queue a vector for background graph insertion
    pub async fn queue_insert(&self, id: Id, vector: Vec<f32>) -> Result<()> {
        self.tx.send(UpdateOp::Insert { id, vector }).await?;
        Ok(())
    }
}

/// Async updater worker
pub struct AsyncUpdater {
    rx: mpsc::Receiver<UpdateOp>,
    storage: Arc<VectorStorage>,
    nav_layer: Arc<RwLock<NavigationLayer>>,
    config: AsyncUpdaterConfig,
}

impl AsyncUpdater {
    /// Main update loop
    pub async fn run(mut self) {
        let mut batch = Vec::with_capacity(self.config.batch_size);
        let mut deadline = tokio::time::Instant::now() + self.config.batch_timeout;

        loop {
            tokio::select! {
                op = self.rx.recv() => {
                    match op {
                        Some(op) => {
                            batch.push(op);
                            if batch.len() >= self.config.batch_size {
                                self.process_batch(&mut batch).await;
                                deadline = tokio::time::Instant::now() + self.config.batch_timeout;
                            }
                        }
                        None => break, // Channel closed
                    }
                }
                _ = tokio::time::sleep_until(deadline) => {
                    if !batch.is_empty() {
                        self.process_batch(&mut batch).await;
                    }
                    deadline = tokio::time::Instant::now() + self.config.batch_timeout;
                }
            }
        }
    }

    async fn process_batch(&self, batch: &mut Vec<UpdateOp>) {
        // 1. For each insert: greedy search to find neighbors
        // 2. Connect new nodes to neighbors (merge operator)
        // 3. Update navigation layer entry points if needed
        // 4. Clear batch

        for op in batch.drain(..) {
            match op {
                UpdateOp::Insert { id, vector } => {
                    self.insert_to_graph(id, &vector).await;
                }
                UpdateOp::Delete { id } => {
                    self.remove_from_graph(id).await;
                }
                UpdateOp::RebuildLayer { layer } => {
                    self.rebuild_layer(layer).await;
                }
            }
        }
    }
}
```

#### `vector/simd.rs` - SIMD Distance Computation

```rust
//! SIMD-accelerated distance computations.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Check for AVX2 support at runtime
pub fn has_avx2() -> bool {
    is_x86_feature_detected!("avx2")
}

/// L2 distance using AVX2 (8 floats per instruction)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn l2_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let mut sum = _mm256_setzero_ps();
    let chunks = a.len() / 8;

    for i in 0..chunks {
        let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    // Horizontal sum
    let sum128 = _mm_add_ps(
        _mm256_extractf128_ps(sum, 0),
        _mm256_extractf128_ps(sum, 1),
    );
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));

    let mut result: f32 = 0.0;
    _mm_store_ss(&mut result, sum32);

    // Handle remainder
    for i in (chunks * 8)..a.len() {
        let diff = a[i] - b[i];
        result += diff * diff;
    }

    result.sqrt()
}

/// Inner product using AVX2
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn inner_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let mut sum = _mm256_setzero_ps();
    let chunks = a.len() / 8;

    for i in 0..chunks {
        let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    // Horizontal sum (same as above)
    // ...
}
```

---

## API Design

### Changes to Existing API

The vector module integrates with motlie_db's existing patterns:

#### 1. Storage Integration

```rust
// Current unified Storage in storage.rs
pub struct Storage<Mode> {
    path: PathBuf,
    _mode: PhantomData<Mode>,
}

// NEW: Extended to include vector storage
impl Storage<ReadWrite> {
    pub fn ready(self, config: StorageConfig) -> Result<ReadWriteHandles> {
        // ... existing graph/fulltext initialization ...

        // NEW: Initialize vector storage if configured
        if let Some(vector_config) = config.vector {
            let vector_path = self.path.join("vector");
            let mut vector_storage = vector::VectorStorage::readwrite(&vector_path, vector_config);
            vector_storage.ready()?;
            // ... spawn vector workers ...
        }
    }
}
```

#### 2. New StorageConfig Fields

```rust
#[derive(Debug, Clone)]
pub struct StorageConfig {
    pub reader: reader::ReaderConfig,
    pub writer: writer::WriterConfig,
    pub num_query_workers: usize,

    // NEW: Optional vector configuration
    pub vector: Option<vector::VectorConfig>,
}
```

#### 3. New Mutations

```rust
// In mutation.rs (unified)
pub enum Mutation {
    // Existing
    Graph(graph::Mutation),

    // NEW
    Vector(vector::Mutation),
}

// In vector/mutation.rs
#[derive(Debug, Clone)]
pub enum Mutation {
    Insert(InsertVector),
    Delete(DeleteVector),
    BatchInsert(Vec<InsertVector>),
}

#[derive(Debug, Clone)]
pub struct InsertVector {
    pub id: Id,
    pub vector: Vec<f32>,
    pub metadata: Option<HashMap<String, Value>>,
}
```

#### 4. New Queries

```rust
// In query.rs (unified)
pub enum Query {
    // Existing
    Graph(graph::Query),
    Fulltext(fulltext::Query),

    // NEW
    Vector(vector::Query),
}

// In vector/query.rs
#[derive(Debug, Clone)]
pub enum Query {
    SearchKNN(SearchKNN),
    GetVector(GetVector),
    GetPQCodes(GetPQCodes),
}

#[derive(Debug, Clone)]
pub struct SearchKNN {
    pub query: Vec<f32>,
    pub k: usize,
    pub ef: usize,  // Search beam width
    pub filter: Option<Filter>,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: Id,
    pub distance: f32,
    pub metadata: Option<HashMap<String, Value>>,
}
```

### Usage Example

```rust
use motlie_db::{Storage, StorageConfig};
use motlie_db::vector::{VectorConfig, InsertVector, SearchKNN};
use motlie_db::mutation::Runnable;
use motlie_db::query::Runnable as QueryRunnable;

// Configure with vector support
let config = StorageConfig::default()
    .with_vector(VectorConfig {
        dimension: 128,
        m: 16,
        ef_construction: 200,
        num_subquantizers: 8,
        enable_simd: true,
    });

// Initialize storage
let storage = Storage::readwrite(db_path);
let handles = storage.ready(config)?;

// Insert vectors
for (id, vector) in vectors {
    InsertVector { id, vector, metadata: None }
        .run(handles.writer())
        .await?;
}

// Search
let results = SearchKNN {
    query: query_vector,
    k: 10,
    ef: 100,
    filter: None,
}
.run(handles.reader(), Duration::from_secs(5))
.await?;

// Results include distance and ID
for result in results {
    println!("{}: {}", result.id, result.distance);
}
```

---

## Resource Projections

### Memory Requirements at 1B Vectors

| Component | Calculation | Size |
|-----------|-------------|------|
| **Navigation Layer** | 1M entry points × 16 bytes | 16 MB |
| **PQ Codebooks** | 8 subq × 256 centroids × 16 dims × 4 bytes | 512 KB |
| **Distance Tables** | 8 subq × 256 × 4 bytes (per query) | 8 KB |
| **Block Cache** | Configurable, recommended | 8-16 GB |
| **Write Buffer** | RocksDB memtables | 512 MB |
| **OS Page Cache** | Linux will use available RAM | Variable |
| **Total Minimum** | | **~10 GB** |
| **Recommended** | | **32-64 GB** |

### Disk Requirements at 1B Vectors

| Column Family | Calculation | Size |
|---------------|-------------|------|
| **vectors** | 1B × 128 dims × 4 bytes (with compression) | ~300 GB |
| **pq_codes** | 1B × 8 bytes | 8 GB |
| **graph_edges** | 1B × 32 neighbors × ~4 bytes (RoaringBitmap) | ~64 GB |
| **pq_codebooks** | 8 × 256 × 64 bytes | ~128 KB |
| **metadata** | Negligible | ~1 KB |
| **pending** | Transient | ~100 MB |
| **Total** | | **~375 GB** |

### CPU Requirements

| Operation | CPU Usage | Notes |
|-----------|-----------|-------|
| Insert (sync) | 1 core | PQ encoding + disk write |
| Insert (async graph) | 2-4 cores | Background graph updates |
| Search | 1-2 cores | SIMD PQ distance + re-ranking |
| Training (PQ) | 8+ cores | K-means clustering (one-time) |

**Recommended:** 16-32 cores for production at 1B scale

### Throughput Projections

| Operation | Current (100K) | Projected (1B) | Notes |
|-----------|----------------|----------------|-------|
| **Insert** | 40 vec/s | 5,000-10,000 vec/s | Async + batching |
| **Search QPS** | 88-200 | 500-2,000 | SIMD + caching |
| **Re-ranking** | N/A | 10,000+ vec/s | Sequential I/O |

---

## Performance Projections

### Search Latency Breakdown at 1B Scale

| Phase | Time | Notes |
|-------|------|-------|
| **Navigation** | 1-2 ms | In-memory HNSW descent |
| **PQ Expansion** | 5-10 ms | Beam search with SIMD PQ distance |
| **Re-ranking** | 10-20 ms | Load 20-50 full vectors from disk |
| **Total** | **16-32 ms** | P50 estimate |
| **P99** | **40-60 ms** | With disk cache misses |

### Recall Projections

| Configuration | Recall@10 | Latency |
|---------------|-----------|---------|
| ef=50, rerank=20 | ~90% | 15 ms |
| ef=100, rerank=50 | ~95% | 25 ms |
| ef=200, rerank=100 | ~98% | 45 ms |
| ef=500, rerank=200 | ~99% | 100 ms |

### Build Time Projections

| Scale | Current Time | Projected Time | Improvement |
|-------|--------------|----------------|-------------|
| 1M | 7 hours | 10-20 min | 20-40x |
| 10M | ~70 hours | 2-4 hours | 20-30x |
| 100M | ~700 hours | 1-2 days | 15-20x |
| 1B | ~7000 hours | 1-2 weeks | 15-20x |

Key improvements:
1. Async graph updates (no blocking on flush)
2. Batch processing (amortize overhead)
3. SIMD PQ distance (8x speedup)
4. Merge operators (lock-free updates)

---

## DATA-1 Modification: RaBitQ Instead of PQ

### The Problem with PQ

The original HYBRID design used Product Quantization (PQ) for compression:

```
Original Design (PQ):
  Training Phase:
    1. Collect N representative vectors
    2. Split each vector into M subvectors
    3. Run k-means on each subspace → codebooks
    4. Store codebooks for encoding

  Encoding Phase:
    For each vector v:
      code = [nearest_centroid(subvector_i) for i in 1..M]
```

**Problem**: Step 1-3 require representative training data, violating DATA-1.

### The RaBitQ Solution

RaBitQ uses randomization instead of learning:

```
RaBitQ Design (Training-Free):
  Initialization (once, at index creation):
    1. Generate random D×D orthonormal matrix R
       (via QR decomposition of random Gaussian matrix)
    2. Store R in metadata

  Encoding Phase:
    For each vector v:
      rotated = R × v
      binary_code = sign(rotated)  // D bits
```

**Key Insight**: The random rotation "spreads" vector information uniformly across dimensions. The sign quantization then captures the most significant bit of each rotated component.

### Mathematical Guarantee

From [RaBitQ (SIGMOD 2024)](https://arxiv.org/abs/2405.12497):

> Distance estimation error is O(1/√D), which is asymptotically optimal.

| Dimension | Error Bound | Practical Error |
|-----------|-------------|-----------------|
| 128 | ~8.8% | 5-7% |
| 256 | ~6.3% | 4-5% |
| 512 | ~4.4% | 3-4% |
| 1024 | ~3.1% | 2-3% |

**Higher dimensions = better compression "for free".**

### Architecture Comparison

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Original HYBRID (PQ)                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  Navigation Layer (HNSW)    →  PQ Codes (8 bytes)   →  Full Vectors     │
│  ~50 MB                         8 GB at 1B              512 GB at 1B    │
│                                 ❌ Requires training                     │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                    Modified HYBRID (RaBitQ)                              │
├─────────────────────────────────────────────────────────────────────────┤
│  Navigation Layer (HNSW)    →  RaBitQ Binary        →  Full Vectors     │
│  ~50 MB                         16 GB at 1B             512 GB at 1B    │
│                                 ✅ No training needed                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Memory Comparison at 1B Scale

| Component | PQ (Original) | RaBitQ (Modified) | Notes |
|-----------|---------------|-------------------|-------|
| Navigation Layer | 50 MB | 50 MB | Same |
| Compressed Codes | 8 GB | 16 GB | 2x larger |
| Full Vectors | 512 GB (disk) | 512 GB (disk) | Same |
| **Total RAM** | **~10 GB** | **~18 GB** | +8 GB |
| Training Required | Yes | **No** | Critical difference |

**Trade-off**: RaBitQ uses 2x memory for compressed codes, but requires zero training. For DATA-1 compliance, this is acceptable.

### Search Path Modification

```
Original Search (PQ):
  1. HNSW navigation → entry region
  2. Beam search with PQ distance (8-byte codes)
  3. Re-rank top-100 with exact distance

Modified Search (RaBitQ):
  1. HNSW navigation → entry region
  2. Beam search with Hamming distance (D-bit codes, SIMD popcount)
  3. Re-rank top-100 with exact distance
```

**Performance Note**: Hamming distance via SIMD popcount is actually faster than PQ table lookups, potentially improving QPS despite larger codes.

### RocksDB Schema Changes

```rust
// Original: pq_codes CF
// Key: [16-byte ID]
// Value: [u8; 8]  // 8 subquantizers × 1 byte = 8 bytes

// Modified: binary_codes CF
// Key: [4-byte u32 ID]  // Using HNSW2's compact IDs
// Value: [u8; D/8]      // D bits = D/8 bytes (16 bytes for 128D)

pub const VECTOR_COLUMN_FAMILIES: &[&str] = &[
    "vectors",        // Full vectors for re-ranking
    "binary_codes",   // RaBitQ binary codes (replaces pq_codes)
    "rotation_matrix",// D×D orthonormal matrix (stored once)
    "graph_edges",    // HNSW neighbor lists (RoaringBitmap)
    "nav_layer",      // Navigation layer entry points
    "metadata",       // Index metadata and statistics
    "pending",        // Pending vectors for async processing
];
```

### RaBitQ Implementation

```rust
/// RaBitQ encoder (training-free)
pub struct RaBitQEncoder {
    dimension: usize,
    /// Random orthonormal rotation matrix (D × D)
    rotation: Vec<Vec<f32>>,
}

impl RaBitQEncoder {
    /// Create encoder with random rotation matrix
    pub fn new(dimension: usize, seed: u64) -> Self {
        // Generate random orthonormal matrix via QR decomposition
        let rotation = generate_random_orthonormal(dimension, seed);
        Self { dimension, rotation }
    }

    /// Load encoder from stored rotation matrix
    pub fn from_stored(rotation: Vec<Vec<f32>>) -> Self {
        let dimension = rotation.len();
        Self { dimension, rotation }
    }

    /// Encode vector to binary (D bits = D/8 bytes)
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        debug_assert_eq!(vector.len(), self.dimension);

        // Rotate vector
        let mut rotated = vec![0.0f32; self.dimension];
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                rotated[i] += self.rotation[i][j] * vector[j];
            }
        }

        // Quantize to binary (sign of each component)
        let mut binary = vec![0u8; self.dimension / 8];
        for i in 0..self.dimension {
            if rotated[i] >= 0.0 {
                binary[i / 8] |= 1 << (i % 8);
            }
        }
        binary
    }

    /// Approximate distance using Hamming distance
    /// Returns scaled distance estimate
    pub fn approximate_distance(code_a: &[u8], code_b: &[u8]) -> f32 {
        let hamming = hamming_distance_simd(code_a, code_b);
        // Scale factor from RaBitQ paper
        let dimension = code_a.len() * 8;
        let scale = (std::f32::consts::PI / 2.0) / (dimension as f32);
        (hamming as f32) * scale
    }
}

/// SIMD Hamming distance using popcount
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "popcnt")]
pub unsafe fn hamming_distance_simd(a: &[u8], b: &[u8]) -> u32 {
    debug_assert_eq!(a.len(), b.len());
    let mut count = 0u32;

    // Process 8 bytes at a time
    for i in (0..a.len()).step_by(8) {
        let chunk_a = u64::from_le_bytes(a[i..i+8].try_into().unwrap_or([0; 8]));
        let chunk_b = u64::from_le_bytes(b[i..i+8].try_into().unwrap_or([0; 8]));
        count += (chunk_a ^ chunk_b).count_ones();
    }
    count
}
```

### Performance Projections (Modified)

| Metric | PQ (Original) | RaBitQ (Modified) | Notes |
|--------|---------------|-------------------|-------|
| Encode time | ~10 μs | ~5 μs | Matrix multiply vs table lookup |
| Distance time | ~200 ns (table) | ~50 ns (popcount) | SIMD popcount faster |
| Memory at 1B | 8 GB | 16 GB | 2x for codes |
| QPS at 1B | 500-2,000 | **1,000-5,000** | Faster distance |
| Training time | Hours | **0** | Critical for DATA-1 |

### Recall Considerations

RaBitQ alone achieves ~70% recall. Combined with HNSW navigation and re-ranking:

| Configuration | Recall@10 | Latency |
|---------------|-----------|---------|
| HNSW only | ~95% | 20-30 ms |
| HNSW + RaBitQ filter | ~95% | 15-25 ms |
| HNSW + RaBitQ + rerank@100 | **~98%** | 30-50 ms |

The HNSW navigation provides the high recall, RaBitQ accelerates candidate scoring.

### Extended-RaBitQ Option

For scenarios where slightly more memory is acceptable for higher recall:

[Extended-RaBitQ (SIGMOD 2025)](https://arxiv.org/html/2409.09913v1) supports 2-6 bit quantization:

| Bits/Dim | Bytes (128D) | Recall (no rerank) | Training |
|----------|--------------|---------------------|----------|
| 1 bit | 16 bytes | ~70% | None |
| 2 bits | 32 bytes | ~85% | None |
| 4 bits | 64 bytes | ~92% | None |

Extended-RaBitQ also uses random rotations, maintaining DATA-1 compliance.

---

## Implementation Phases

### Phase 1: Core Infrastructure (Foundation)

**Goal:** Basic PQ + synchronous HNSW working with RocksDB

**Tasks:**
1. Create `vector/` module structure
2. Implement `schema.rs` with column families
3. Implement `pq.rs` with encode/decode
4. Port HNSW to use new schema
5. Implement basic `InsertVector` and `SearchKNN`

**Deliverables:**
- Working vector search at 100K scale
- PQ compression functional
- Tests passing

### Phase 2: SIMD Optimization

**Goal:** 5-8x speedup in distance computation

**Tasks:**
1. Implement `simd.rs` with AVX2 L2 distance
2. Implement SIMD PQ asymmetric distance
3. Add runtime feature detection
4. Benchmark and validate

**Deliverables:**
- SIMD distance functions
- 5-8x speedup in search
- Fallback for non-AVX2 systems

### Phase 3: Async Graph Updater

**Goal:** Decouple insert latency from graph quality

**Tasks:**
1. Implement `async_updater.rs`
2. Add pending queue column family
3. Implement batch processing
4. Add merge operators for graph edges
5. Implement background pruning

**Deliverables:**
- Async insert path
- < 10ms insert latency
- Background graph maintenance

### Phase 4: Navigation Layer

**Goal:** Optimize HNSW multi-layer structure

**Tasks:**
1. Implement NavigationLayer with memory-resident top layers
2. Add entry point management
3. Optimize layer descent
4. Implement layer rebuild logic

**Deliverables:**
- O(log N) navigation
- Memory-efficient top layers
- Fast entry point lookup

### Phase 5: Scaling & Production

**Goal:** Validate at 1B scale

**Tasks:**
1. Benchmark at 10M, 100M, 1B scales
2. Tune RocksDB options (block cache, compression)
3. Add monitoring and metrics
4. Implement periodic compaction
5. Add checkpoint/recovery

**Deliverables:**
- 1B scale validation
- Production-ready configuration
- Operational documentation

---

## References

1. **HNSW Paper:** Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs.

2. **DiskANN Paper:** Subramanya, S. J., et al. (2019). DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node.

3. **Product Quantization:** Jégou, H., Douze, M., & Schmid, C. (2010). Product quantization for nearest neighbor search.

4. **RocksDB Merge Operators:** https://github.com/facebook/rocksdb/wiki/Merge-Operator

5. **SIMD Intrinsics:** Intel Intrinsics Guide - https://www.intel.com/content/www/us/en/docs/intrinsics-guide/

---

## Appendix: Current Benchmark Results

From [PERF.md](./PERF.md):

### HNSW (Current Implementation)
| Scale | Build Time | Recall@10 | QPS |
|-------|------------|-----------|-----|
| 1K | 25s | 52.6% | - |
| 10K | 338s | 80.7% | - |
| 100K | 2,492s | 81.7% | - |
| 1M | 25,119s | 95.3% | 47 |

### Vamana (Current Implementation)
| Scale | Build Time | Recall@10 | QPS |
|-------|------------|-----------|-----|
| 1K | 26s | 73.1% | 174 |
| 10K | 255s | 83.7% | 201 |
| 100K (L=100) | 963s | 70.7% | 163 |
| 100K (L=200) | 1,786s | 76.0% | 88 |
| 1M (L=200) | 53,633s | 81.9% | 74 |

### Reference: hnswlib
| Scale | Recall@10 | QPS |
|-------|-----------|-----|
| 1M | 98.5% | 16,108 |

The hybrid architecture targets closing the gap with hnswlib while maintaining disk-based storage and online insert capability.
