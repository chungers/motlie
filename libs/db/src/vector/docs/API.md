# Vector Module API Reference

**Module:** `motlie_db::vector`
**Purpose:** HNSW-based vector similarity search with RaBitQ binary quantization

---

## Table of Contents

1. [Overview](#overview)
2. [Part 1: Index Building](#part-1-index-building)
   - [Distance Metrics](#distance-metrics)
   - [Embedding Registration](#embedding-registration)
   - [HNSW Configuration](#hnsw-configuration)
   - [Building the Index](#building-the-index)
3. [Part 2: Search](#part-2-search)
   - [Exact Search (All Distance Metrics)](#exact-search-all-distance-metrics)
   - [RaBitQ Search (Cosine Only)](#rabitq-search-cosine-only)
   - [Search Configuration](#search-configuration)
4. [Part 3: Performance Tuning](#part-3-performance-tuning)
   - [HNSW Parameters](#hnsw-parameters)
   - [Cache Configuration](#cache-configuration)
   - [Parallel Reranking](#parallel-reranking)
5. [Part 4: Delete Lifecycle and Garbage Collection](#part-4-delete-lifecycle-and-garbage-collection)
   - [Delete State Machine](#delete-state-machine)
   - [Soft Delete Operations](#soft-delete-operations)
   - [Garbage Collector](#garbage-collector)
   - [Search-time Safety](#search-time-safety)
   - [Best Practices for Deletion](#best-practices-for-deletion)
6. [Part 5: Concurrent Access Guarantees](#part-5-concurrent-access-guarantees)
   - [Thread Safety](#thread-safety)
   - [Transaction Isolation](#transaction-isolation)
   - [Reader-Writer Concurrency](#reader-writer-concurrency)
   - [Error Handling Under Contention](#error-handling-under-contention)
   - [Performance Characteristics](#performance-characteristics)
   - [Failure Recovery](#failure-recovery)
   - [Best Practices for Concurrency](#best-practices-for-concurrency)
7. [Part 6: Subsystem Lifecycle Management](#part-6-subsystem-lifecycle-management)
   - [Subsystem Overview](#subsystem-overview)
   - [Starting the Subsystem](#starting-the-subsystem)
   - [Shutdown Order](#shutdown-order)
   - [Best Practices](#best-practices-for-lifecycle)
8. [Complete Usage Flow](#complete-usage-flow)
9. [Public API Reference](#public-api-reference)

---

## Overview

The vector module provides approximate nearest neighbor (ANN) search using the HNSW algorithm with optional RaBitQ binary quantization. Key features:

- **HNSW Graph Index**: Hierarchical navigable small world graph for O(log n) search
- **Multiple Distance Metrics**: Cosine, L2 (Euclidean), DotProduct
- **RaBitQ Quantization**: Training-free binary codes with ADC for fast approximate search
- **Caching**: NavigationCache for layer traversal, BinaryCodeCache for RaBitQ codes
- **Parallel Reranking**: Adaptive rayon parallelism for large candidate sets

### Search Mode Summary

| Search Mode | Distance Metrics | Use Case |
|-------------|------------------|----------|
| **Exact HNSW** | Cosine, L2, DotProduct | High recall, any embedding type |
| **RaBitQ** | Cosine only | Speed optimization for normalized vectors |

---

## Part 1: Index Building

### Distance Metrics

The distance metric is fixed per embedding space and determines how similarity is computed:

```rust
use motlie_db::vector::Distance;

// Available metrics
Distance::Cosine      // 1 - cos(a,b), range [0,2], for normalized embeddings
Distance::L2          // ||a-b||, range [0,∞), for spatial/image features
Distance::DotProduct  // -a·b (negated), for unnormalized embeddings
```

**Choosing a Distance Metric:**

| Embedding Type | Recommended Distance | Notes |
|----------------|---------------------|-------|
| OpenAI, Cohere, CLIP | Cosine | Pre-normalized, unit vectors |
| SIFT, image features | L2 | Spatial relationships matter |
| Custom unnormalized | DotProduct | Magnitude carries meaning |

**Important:** The HNSW graph structure is optimized for the metric used during construction. You cannot change metrics at search time.

### Embedding Registration

An `Embedding` represents a registered embedding space with a unique code, model name, dimensionality, and distance metric. The `EmbeddingRegistry` manages all registered embeddings and owns a storage reference to prevent cross-DB misuse.

```rust
use motlie_db::vector::{
    EmbeddingBuilder, EmbeddingRegistry, Distance, Storage,
};
use std::path::Path;
use std::sync::Arc;

// Initialize storage
let mut storage = Storage::readwrite(Path::new("./vector_db"));
storage.ready()?;
let storage = Arc::new(storage);

// Create registry (storage is captured once inside the registry)
let registry = EmbeddingRegistry::new(storage.clone());

// Pre-warm from existing DB (loads previously registered embeddings)
let count = registry.prewarm()?;
println!("Loaded {} existing embeddings", count);

// Register a new embedding space
let builder = EmbeddingBuilder::new("openai-ada-002", 1536, Distance::Cosine);
let embedding = registry.register(builder)?;

println!("Registered embedding: code={}, dim={}",
         embedding.code(), embedding.dim());

// Query registered embeddings
let all = registry.list_all();
let cosine_embeddings = registry.find_by_distance(Distance::Cosine);
let by_model = registry.find_by_model("openai");
```

**Subsystem note:** When storage isn’t available at construction (e.g., Subsystem pattern),
use `EmbeddingRegistry::new_without_storage()` and call `set_storage(storage)` once before
`prewarm()` or `register()`.

**Lookup + insert using registry filters:**

```rust
use motlie_db::vector::{Distance, EmbeddingFilter, EmbeddingRegistry, ExternalKey, Processor, Storage};
use motlie_db::Id;
use std::path::Path;
use std::sync::Arc;

// Open storage + registry
let mut storage = Storage::readwrite(Path::new("./vector_db"));
storage.ready()?;
let storage = Arc::new(storage);
let registry = Arc::new(EmbeddingRegistry::new(storage.clone()));
registry.prewarm()?;

// Find an existing embedding by model/dim/distance
let filter = EmbeddingFilter::default()
    .model("clip-vit-b32")
    .dim(512)
    .distance(Distance::Cosine);
let embedding = registry
    .find(&filter)
    .into_iter()
    .next()
    .ok_or_else(|| anyhow::anyhow!("Embedding not found"))?;

// Insert vectors using Processor (it loads EmbeddingSpec from storage internally)
let processor = Processor::new(storage.clone(), registry.clone());
let external_key = ExternalKey::NodeId(Id::new());
let _vec_id = processor.insert_vector(&embedding, external_key, &vector, true)?;
```

**Insert-time validation (Processor/ops):**

- **Embedding exists**: lookup by code via registry
- **Dimension match**: `vector.len() == spec.dim()`
- **External key uniqueness**: IdForward is checked for duplicates
- **SpecHash consistency**: on first insert, the spec hash is stored; later inserts
  must match the stored hash or they fail with “EmbeddingSpec changed…”
- **Storage type**: vectors are encoded using the persisted `EmbeddingSpec.storage_type`
  (F32/F16). Input is always `&[f32]`; conversion happens at write time.

**EmbeddingBuilder Fields:**

| Field | Description |
|-------|-------------|
| `model` | Model identifier (e.g., "openai-ada-002", "clip-vit-b32") |
| `dim` | Vector dimensionality (128, 512, 768, 1536, etc.) |
| `distance` | Distance metric (Cosine, L2, DotProduct) |

**Build Parameters (optional):**

These parameters are persisted with the embedding spec and control how the index is built:

```rust
let builder = EmbeddingBuilder::new("gemma", 768, Distance::Cosine)
    .with_hnsw_m(32)              // HNSW connections per node (default: 16, min: 2)
    .with_hnsw_ef_construction(400) // Build-time beam width (default: 200)
    .with_rabitq_bits(2)          // Binary quantization bits (default: 1, valid: 1, 2, 4)
    .with_rabitq_seed(12345);     // RNG seed for rotation matrix (default: 42)

let embedding = registry.register(builder)?;
```

| Parameter | Default | Valid Range | Description |
|-----------|---------|-------------|-------------|
| `hnsw_m` | 16 | >= 2 | HNSW connections per node. Higher = better recall, more memory |
| `hnsw_ef_construction` | 200 | > 0 | Build-time beam width. Higher = better graph quality |
| `rabitq_bits` | 1 | 1, 2, 4 | Bits per dimension. Higher = better recall, larger codes |
| `rabitq_seed` | 42 | any u64 | Seed for rotation matrix. Same seed = reproducible codes |

**⚠️ Important: SpecHash Timing**

Build parameters are **immutable after the first vector insert**. When the first vector is inserted into an embedding space, a `SpecHash` is computed and stored. All subsequent inserts and searches validate against this hash.

- **Changing parameters after data exists requires rebuilding the index**
- Parameters should be set at registration time, before inserting any vectors
- This ensures consistency: all vectors in an index use the same build parameters

```rust
// ✅ CORRECT: Set parameters before inserting vectors
let builder = EmbeddingBuilder::new("gemma", 768, Distance::Cosine)
    .with_hnsw_m(32);
let embedding = registry.register(builder)?;
processor.insert_vector(...)?;  // SpecHash set here

// ❌ ERROR: Cannot change parameters after data exists
// Attempting to register with different params and insert will fail:
// "EmbeddingSpec changed since index build. Rebuild required."
```

### HNSW Configuration

HNSW (Hierarchical Navigable Small World) is a graph-based ANN algorithm. HNSW parameters
are configured at registration time via `EmbeddingBuilder` and persisted in the registry
as an `EmbeddingSpec`. `EmbeddingSpec` is the single source of truth used to construct
HNSW indexes after restart; you should not manually instantiate it in normal usage.

```rust
use motlie_db::vector::{EmbeddingBuilder, EmbeddingRegistry, Distance, Processor, Storage};
use std::path::Path;
use std::sync::Arc;

// Register embedding with HNSW parameters via builder
let mut storage = Storage::readwrite(Path::new("./vector_db"));
storage.ready()?;
let storage = Arc::new(storage);
let registry = EmbeddingRegistry::new(storage.clone());
registry.prewarm()?;

let embedding = registry.register(
    EmbeddingBuilder::new("openai-ada", 512, Distance::Cosine)
        .with_hnsw_m(16)                // Connections per node
        .with_hnsw_ef_construction(100) // Build-time beam width
        .with_rabitq_bits(1)
        .with_rabitq_seed(42),
)?;

// In normal usage, let Processor construct HNSW indices from the persisted spec
let _processor = Processor::new(storage.clone(), Arc::new(registry));

// Derived values are computed from EmbeddingSpec at index construction:
// - m_max = 2 * hnsw_m (always)
// - m_max_0 = 2 * hnsw_m (always)
// - m_l = 1.0 / ln(hnsw_m)
```

> **Note**: `hnsw::Config` has been deleted. All HNSW parameters come from
> the persisted `EmbeddingSpec` created during registration.

**Parameter Impact:**

| Parameter | Higher Value | Lower Value |
|-----------|--------------|-------------|
| `M` | Better recall, more memory, slower insert | Lower recall, less memory, faster insert |
| `ef_construction` | Better graph quality, slower build | Faster build, potentially lower recall |
| `m_max` / `m_max_0` | More connections, better recall | Fewer connections, less memory |

**Recommended Values by Dimension:**

| Dimension | M | ef_construction | Notes |
|-----------|---|-----------------|-------|
| 128 (SIFT) | 16 | 100 | Standard ANN benchmark |
| 512 (CLIP) | 16-32 | 100-200 | Higher dim needs more connections |
| 768+ (LLM) | 32 | 200 | High-dimensional embeddings |

### Building the Index

```rust
use motlie_db::vector::{EmbeddingBuilder, EmbeddingRegistry, ExternalKey, Processor, Storage, Distance};
use motlie_db::Id;
use std::sync::Arc;
use std::path::Path;

// Initialize storage and registry
let mut storage = Storage::readwrite(Path::new("./vector_db"));
storage.ready()?;
let storage = Arc::new(storage);
let registry = Arc::new(EmbeddingRegistry::new(storage.clone()));
registry.prewarm()?;

// Register embedding space (build parameters persisted in EmbeddingSpec)
let embedding = registry.register(
    EmbeddingBuilder::new("clip", 512, Distance::Cosine)
        .with_hnsw_m(16)
        .with_hnsw_ef_construction(100)
        .with_rabitq_bits(1)
        .with_rabitq_seed(42),
)?;

// Create processor (manages transactions, HNSW, caches)
let processor = Processor::new(storage.clone(), registry.clone());

// Insert vectors (atomic insert + optional HNSW build)
for (id, vector) in vectors.iter().enumerate() {
    let external_key = ExternalKey::NodeId(Id::new());
    let _vec_id = processor.insert_vector(&embedding, external_key, vector, true)?;
}
```

**VectorElementType:**

| Type | Size | Precision | Use Case |
|------|------|-----------|----------|
| `F32` | 4 bytes/dim | Full | Default, maximum precision |
| `F16` | 2 bytes/dim | ~0.1% loss | Normalized vectors, memory constrained |

---

## Part 2: Search

### Exact Search (All Distance Metrics)

Exact HNSW search computes actual distances during beam search. Works with all distance metrics.

```rust
// Search parameters
let query: Vec<f32> = /* your query vector */;
let ef_search = 100;  // Beam width (higher = better recall, slower)
let k = 10;           // Number of results

// Perform search
let results = processor.search(&embedding, &query, k, ef_search)?;

// Results are SearchResult structs, sorted by distance ascending
for result in results {
    println!(
        "vec_id={}, distance={:.4}",
        result.vec_id,
        result.distance
    );
}
```

**ef_search Impact:**

| ef_search | Recall@10 (typical) | Latency |
|-----------|---------------------|---------|
| 50 | 85-90% | Fast |
| 100 | 90-95% | Moderate |
| 200 | 95-98% | Slower |

### RaBitQ Search (Cosine Only)

RaBitQ uses binary quantization with ADC (Asymmetric Distance Computation) for fast approximate search, then reranks top candidates with exact distance. **Only works with Cosine distance** because:

1. Sign quantization captures **direction**, not magnitude
2. ADC computes weighted dot products that approximate **angular distance**
3. L2 on unnormalized data would lose magnitude information

```rust
use motlie_db::vector::SearchConfig;

// Two-phase search: ADC navigation → exact reranking
let config = SearchConfig::new(embedding.clone(), k)
    .rabitq()?  // Err if distance != Cosine
    .with_ef(ef_search)
    .with_rerank_factor(rerank_factor);

let results = processor.search_with_config(&config, &query)?;
```

**RaBitQ Parameters:**

| Parameter | Value | Effect |
|-----------|-------|--------|
| `bits_per_dim` | 1 | 32x compression, ~70% recall with ADC |
| `bits_per_dim` | 2 | 16x compression, ~85% recall with ADC |
| `bits_per_dim` | 4 | 8x compression, **recommended** - ~92-99% recall with ADC |
| `rerank_factor` | 4-20 | Higher = better recall, more I/O |

> **Note:** The implementation uses ADC (Asymmetric Distance Computation), NOT symmetric
> Hamming distance. ADC keeps the query as float32 and computes weighted dot products
> with quantized codes, preserving numeric ordering. This achieves 91-99% recall with
> 4-bit quantization vs only ~24% with symmetric Hamming. See RABITQ.md for details.

### Search Configuration

`SearchConfig` provides a builder pattern for configuring searches:

```rust
use motlie_db::vector::{SearchConfig, SearchStrategy};

// Exact search configuration
let config = SearchConfig::new(embedding.clone(), 10)
    .exact()
    .with_ef(100);

// RaBitQ configuration (returns Result - fails if not Cosine)
let config = SearchConfig::new(embedding.clone(), 10)
    .rabitq()?  // Err if distance != Cosine
    .with_ef(100)
    .with_rerank_factor(10)
    .with_parallel_rerank_threshold(3200);

// Check strategy
match config.strategy() {
    SearchStrategy::Exact => println!("Using exact search"),
    SearchStrategy::RaBitQ { use_cache } => {
        println!("Using RaBitQ, cache={}", use_cache);
    }
}
```

---

## Part 3: Performance Tuning

### HNSW Parameters

**Build-time vs Search-time Trade-offs:**

```
                    RECALL
                      ↑
                      │     ┌─────────────────┐
                      │     │ High M, High ef │
                      │     │ (Best recall)   │
                      │     └────────┬────────┘
                      │              │
                      │     ┌────────┴────────┐
                      │     │ Balanced        │
                      │     │ (M=16, ef=100)  │
                      │     └────────┬────────┘
                      │              │
                      │     ┌────────┴────────┐
                      │     │ Low M, Low ef   │
                      │     │ (Fast, compact) │
                      │     └─────────────────┘
                      │
                      └──────────────────────────→ SPEED/MEMORY
```

**Dimension-specific Recommendations:**

| Dimension | M | ef_construction | ef_search | Notes |
|-----------|---|-----------------|-----------|-------|
| 128 | 16 | 100 | 50 | SIFT-like features |
| 512 | 16-32 | 100-200 | 100 | CLIP embeddings |
| 768 | 32 | 200 | 100-200 | Transformer embeddings |
| 1536 | 32 | 200 | 200 | OpenAI ada-002 |

### Cache Configuration

**NavigationCache** caches HNSW layer information and edges:

```rust
use motlie_db::vector::{NavigationCache, NavigationCacheConfig};

// Default configuration
let nav_cache = NavigationCache::new();

// Custom configuration
let config = NavigationCacheConfig {
    cache_upper_layers: true,  // Cache layers > 0
    max_cached_layer: 4,       // Only cache layers 0-4
};
let nav_cache = NavigationCache::with_config(config);

// Check cache stats
let (entries, bytes) = nav_cache.edge_cache_stats();
println!("Edge cache: {} entries, {} bytes", entries, bytes);
```

**BinaryCodeCache** stores RaBitQ binary codes in memory:

```rust
use motlie_db::vector::BinaryCodeCache;

let cache = BinaryCodeCache::new();

// Memory usage estimation:
// 1-bit codes: 16 bytes per vector (128D) to 64 bytes (512D)
// 100K vectors at 512D: ~6.4 MB

let (count, bytes) = cache.stats();
println!("Binary cache: {} codes, {} bytes", count, bytes);
```

**Memory Budget Guidelines:**

| Scale | NavigationCache | BinaryCodeCache (512D, 1-bit) | Total |
|-------|-----------------|-------------------------------|-------|
| 10K | ~10 MB | ~0.6 MB | ~11 MB |
| 100K | ~100 MB | ~6.4 MB | ~107 MB |
| 1M | ~1 GB | ~64 MB | ~1.1 GB |

### Parallel Reranking

RaBitQ search uses parallel reranking for large candidate sets. The threshold determines when to use rayon:

```rust
use motlie_db::vector::search::{
    rerank_adaptive, rerank_auto, DEFAULT_PARALLEL_RERANK_THRESHOLD,
};

// DEFAULT_PARALLEL_RERANK_THRESHOLD = 3200

// Adaptive reranking (auto-selects based on candidate count)
let results = rerank_adaptive(
    &candidates,
    |vec_id| compute_distance(vec_id),
    k,
    3200,  // threshold
);

// Convenience wrapper using default threshold
let results = rerank_auto(&candidates, |vec_id| compute_distance(vec_id), k);
```

**Parallel Threshold Tuning:**

| Candidates | Sequential | Parallel | Speedup |
|------------|------------|----------|---------|
| 800 | 106ms | 139ms | 0.76x (overhead) |
| 1600 | 188ms | 206ms | 0.91x (near equal) |
| **3200** | 414ms | 319ms | **1.30x** (crossover) |
| 6400 | 827ms | 478ms | 1.73x |

*Benchmarked on 512D LAION-CLIP, aarch64 NEON, January 2026*

---

## Part 4: Delete Lifecycle and Garbage Collection

### Overview

Vector deletion follows a **two-phase** approach for safety:

1. **Soft Delete** (immediate): Mark vector as deleted, remove ID mappings
2. **Hard Delete** (background GC): Remove edges, data, recycle VecId

This ensures HNSW graph consistency: edges pointing to deleted vectors are cleaned up asynchronously rather than during the delete operation.

### Delete State Machine

```
                    ┌─────────────┐
                    │   Insert    │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
           ┌────────│   Pending   │  (immediate_index=false)
           │        └──────┬──────┘
           │               │ Async worker processes
           │               ▼
           │        ┌─────────────┐
           │        │   Indexed   │
           │        └──────┬──────┘
           │               │ delete_vector()
           ▼               ▼
    ┌──────────────┐  ┌─────────────┐
    │PendingDeleted│  │   Deleted   │
    └──────┬───────┘  └──────┬──────┘
           │                 │
           └────────┬────────┘
                    │ GC cleanup
                    ▼
              ┌───────────┐
              │  Removed  │  (data + VecMeta deleted)
              └───────────┘
```

### Soft Delete Operations

When `DeleteVector` is executed:

1. **IdForward removed**: External ID → VecId mapping deleted
2. **IdReverse removed**: VecId → External ID mapping deleted
3. **VecMeta updated**: Lifecycle set to `Deleted` or `PendingDeleted`
4. **Data preserved**: Vector data, binary codes, edges kept (for GC)

```rust
use motlie_db::vector::{DeleteVector, MutationRunnable};

// Delete via mutation channel
DeleteVector::new(&embedding, ExternalKey::NodeId(external_id))
    .run(&writer)
    .await?;
writer.flush().await?;

// After soft delete:
// - Search won't return this vector (IdReverse missing)
// - Vector data still in storage (pending GC)
```

### Garbage Collector

The `GarbageCollector` runs in the background to clean up soft-deleted vectors:

```rust
use motlie_db::vector::{GarbageCollector, GcConfig};

// Configure GC
let config = GcConfig::new()
    .with_interval(Duration::from_secs(60))  // Run every 60s
    .with_batch_size(100)                    // Process 100 vectors per cycle
    .with_edge_scan_limit(10000)             // Max edges to scan
    .with_id_recycling(false);               // Don't recycle VecIds (safe default)

// Start GC (spawns background thread)
let gc = GarbageCollector::start(storage.clone(), registry.clone(), config);

// ... application runs ...

// Graceful shutdown
gc.shutdown();
```

**GC Cleanup Steps (per deleted vector):**

1. Scan VecMeta for `is_deleted() == true`
2. Prune edges: Remove vec_id from all neighbor bitmaps
3. Delete vector data from Vectors CF
4. Delete binary codes from BinaryCodes CF
5. Delete VecMeta entry
6. Optionally recycle VecId (if `enable_id_recycling=true` AND edge scan completed fully)

> **Note:** ID recycling is automatically skipped when `edge_scan_limit` is hit, ensuring
> no vec_id is reused while edges may still reference it. The `recycling_skipped_incomplete`
> metric tracks when this safety guard activates.

### GC Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `interval` | 60s | Time between GC cycles |
| `batch_size` | 100 | Max vectors per cycle |
| `process_on_startup` | true | Run cleanup on GC start |
| `edge_scan_limit` | 10000 | Max edges to scan per vector |
| `enable_id_recycling` | false | Return VecIds to free list |

**ID Recycling:**

When enabled, freed VecIds are added to a bitmap and can be reused for new vectors. This is disabled by default because:
- It's only beneficial at very large scale (billions of vectors)
- In-memory allocator may be out of sync with persisted state
- Most deployments don't need ID reuse

### Search-time Safety

Search results are filtered using two mechanisms:

1. **Primary**: IdReverse lookup - deleted vectors have no reverse mapping
2. **Defense-in-depth**: VecMeta lifecycle check - catches edge cases

```rust
// Search automatically filters deleted vectors
let results = processor.search_with_config(&config, &query)?;
// results will never contain deleted vectors
```

### Monitoring GC

```rust
// Check GC metrics
let metrics = gc.metrics();
println!("Vectors cleaned: {}", metrics.vectors_cleaned.load(Ordering::Relaxed));
println!("Edges pruned: {}", metrics.edges_pruned.load(Ordering::Relaxed));
println!("IDs recycled: {}", metrics.ids_recycled.load(Ordering::Relaxed));
println!("Cycles completed: {}", metrics.cycles_completed.load(Ordering::Relaxed));

// Manual GC cycle (for testing)
let cleaned = gc.run_cycle()?;
println!("Cleaned {} vectors in this cycle", cleaned);
```

### Best Practices for Deletion

1. **Start GC early**: Start the GC when your application starts, even if no deletes expected
2. **Tune interval**: Frequent deletes → shorter interval; rare deletes → longer interval
3. **Monitor edge pruning**: High `edges_pruned` indicates well-connected HNSW graph
4. **Avoid ID recycling** unless at billion-scale with ID exhaustion concerns
5. **Graceful shutdown**: Always call `gc.shutdown()` to complete in-flight cleanup

### Delete + Async Indexing

When a vector is deleted while pending async indexing:

1. The vector's VecMeta transitions to `PendingDeleted`
2. The async worker skips `PendingDeleted` vectors
3. GC cleans up the vector data

This ensures no race conditions between async indexing and deletion.

---

## Part 5: Concurrent Access Guarantees

### Overview

The vector module is designed for concurrent multi-threaded access. This section documents the thread safety guarantees, transaction isolation behavior, and best practices for concurrent workloads.

### Thread Safety

**Thread-safe components:**

| Component | Thread Safety | Notes |
|-----------|--------------|-------|
| `Storage` | Thread-safe | Wraps RocksDB TransactionDB with Arc |
| `hnsw::Index` | Thread-safe | Immutable config, Arc-wrapped caches |
| `NavigationCache` | Thread-safe | Internal DashMap, atomic counters |
| `BinaryCodeCache` | Thread-safe | Internal DashMap |
| `EmbeddingRegistry` | Thread-safe | Internal DashMap + OnceLock |

**Usage pattern:**

```rust
use motlie_db::vector::{EmbeddingRegistry, ExternalKey, Processor, Storage};
use motlie_db::Id;
use std::path::Path;
use std::sync::Arc;
use std::thread;

let mut storage = Storage::readwrite(Path::new("./db"))?;
storage.ready()?;
let storage = Arc::new(storage);
let registry = Arc::new(EmbeddingRegistry::new(storage.clone()));
registry.prewarm()?;
let processor = Arc::new(Processor::new(storage.clone(), registry.clone()));

// Spawn multiple reader threads
let handles: Vec<_> = (0..4).map(|_| {
    let processor = processor.clone();
    let embedding = embedding.clone();
    thread::spawn(move || {
        processor.search(&embedding, &query, 10, 100)
    })
}).collect();

// Writer thread (separate)
let processor_w = processor.clone();
thread::spawn(move || {
    let _ = processor_w.insert_vector(&embedding, ExternalKey::NodeId(Id::new()), &vector, true);
});
```

### Transaction Isolation

RocksDB TransactionDB provides **snapshot isolation** for reads and **pessimistic locking** for writes.

**Snapshot Isolation (Readers):**

The vector APIs use TransactionDB internally, which provides snapshot isolation.
`Processor::search()` and the query channel create their own transactions and
operate on a consistent snapshot per call. Custom snapshot APIs are not currently
exposed on the public vector API.

**Key guarantees:**

1. **Read consistency**: Snapshots provide repeatable reads within a transaction
2. **No dirty reads**: Uncommitted writes are invisible to other transactions
3. **No phantom reads**: New inserts don't appear in existing snapshots

**Pessimistic Locking (Writers):**

```rust
let txn_db = storage.transaction_db()?;

// Thread 1: Acquires lock on vec_id=42
let txn1 = txn_db.transaction();
txn1.put_cf(&cf, key_42, value)?;
// Lock held until commit/rollback

// Thread 2: Blocks waiting for lock
let txn2 = txn_db.transaction();
txn2.put_cf(&cf, key_42, value)?;  // Blocks here
// Proceeds after txn1 commits

txn1.commit()?;  // Releases lock
// txn2 now proceeds
```

**Lock timeout behavior:**

| Scenario | Behavior |
|----------|----------|
| Lock available | Immediate acquisition |
| Lock held, timeout not reached | Block and wait |
| Lock held, timeout reached | `Operation timed out` error |

Lock timeout is determined by RocksDB `TransactionDBOptions` (defaults unless overridden). For high-contention workloads, consider:
- Batching operations to reduce lock frequency
- Using shorter transactions
- Partitioning data across embedding spaces

### Reader-Writer Concurrency

The system supports concurrent readers with concurrent writers:

```
  ┌─────────────────────────────────────────────────┐
  │                   Time →                        │
  ├─────────────────────────────────────────────────┤
  │ Reader 1: ──[snapshot]────[search]────►         │
  │ Reader 2:     ──[snapshot]────[search]────►     │
  │ Writer 1: ────[insert]──[commit]──►             │
  │ Writer 2:       ────[insert]──[commit]──►       │
  │                                                 │
  │ ✓ Readers see consistent snapshots              │
  │ ✓ Writers serialize on conflicting keys         │
  │ ✓ New inserts visible after commit              │
  └─────────────────────────────────────────────────┘
```

**Concurrent search during insert:**

- Active searches continue using their snapshot
- New vectors appear in searches started after commit
- HNSW graph updates are atomic (edge bitmap writes)

**Concurrent search during delete:**

- Soft-deleted vectors filtered by IdReverse lookup
- Snapshot-based searches may still see pre-delete state
- This is correct behavior (snapshot isolation)

### Error Handling Under Contention

High contention may produce these errors:

| Error | Cause | Mitigation |
|-------|-------|------------|
| `Operation timed out` | Lock wait exceeded | Increase timeout or batch operations |
| `Resource busy` | Too many concurrent transactions | Add backpressure |
| `Write conflict` | Optimistic write failed | Retry with exponential backoff |

**Recommended error handling:**

```rust
fn insert_with_retry(
    processor: &Processor,
    embedding: &Embedding,
    external_key: ExternalKey,
    vector: &[f32],
    max_retries: usize,
) -> Result<VecId> {
    for attempt in 0..max_retries {
        match processor.insert_vector(embedding, external_key.clone(), vector, true) {
            Ok(vec_id) => return Ok(vec_id),
            Err(_) if attempt + 1 < max_retries => {
                // Exponential backoff for transient conflicts
                std::thread::sleep(Duration::from_millis(10 << attempt));
                continue;
            }
            Err(e) => return Err(e),
        }
    }
    Err(anyhow!("Max retries exceeded"))
}
```

### Performance Characteristics

**Throughput under concurrency:**

| Threads | Read QPS | Write QPS | Notes |
|---------|----------|-----------|-------|
| 1 | Baseline | Baseline | Single-threaded |
| 4 | ~3.5x | ~2x | Good scaling |
| 8 | ~6x | ~2.5x | Readers scale better |
| 16 | ~8x | ~2.5x | Write contention limits scaling |

*Benchmarked with 512D vectors, M=16, 10K vectors in index*

**Best practices for throughput:**

1. **Separate read and write threads**: Readers don't block each other
2. **Batch writes**: Commit multiple vectors per transaction
3. **Use BinaryCodeCache**: Reduces storage reads during RaBitQ search
4. **Pre-warm NavigationCache**: Load frequently accessed layers at startup

### Failure Recovery

The system handles failures gracefully:

**Writer crash mid-transaction:**

- Uncommitted transactions automatically roll back
- No partial writes visible to readers
- HNSW graph remains consistent

**Reader crash:**

- Snapshot automatically released
- No impact on other readers or writers

**Process crash:**

- RocksDB WAL ensures durability
- Restart recovers to last committed state
- In-flight transactions are rolled back

### Best Practices for Concurrency

1. **Use Processor inserts**: `Processor::insert_vector()` and `Processor::insert_batch()` handle transactions and cache updates
2. **Keep transactions short**: Long-held locks reduce concurrency
3. **Batch operations**: Use `Processor::insert_batch()` or `InsertVectorBatch` mutations for bulk inserts
4. **Monitor contention**: Track `errors` in metrics for timeout/conflict rates
5. **Size NavigationCache**: Larger cache reduces storage reads under load
6. **Test under load**: Run concurrent stress tests before production

---

## Part 6: Subsystem Lifecycle Management

### Subsystem Overview

The `Subsystem` struct manages the complete lifecycle of the vector storage system, including:
- Storage and column family initialization
- Embedding registry pre-warming
- Writer and reader consumers
- Async graph updater (optional)
- Garbage collector (optional)

### Starting the Subsystem

Use `start()` for simple usage or `start_with_async()` for full control:

```rust
use motlie_db::vector::{
    Subsystem, WriterConfig, ReaderConfig, AsyncUpdaterConfig, GcConfig,
};
use std::sync::Arc;
use std::time::Duration;

// Create and initialize subsystem
let subsystem = Arc::new(Subsystem::new());
let storage = StorageBuilder::new(path)
    .with_rocksdb(subsystem.clone())
    .build()?;

// Simple start (no async updater, no GC)
let (writer, search_reader) = subsystem.start(
    storage.vector_storage().clone(),
    WriterConfig::default(),
    ReaderConfig::default(),
    4,  // query workers
);

// Or: Full control with async updater and GC
let async_config = AsyncUpdaterConfig::default()
    .with_num_workers(4)
    .with_batch_size(200);

let gc_config = GcConfig::default()
    .with_interval(Duration::from_secs(60))
    .with_batch_size(100);

let (writer, search_reader) = subsystem.start_with_async(
    storage.vector_storage().clone(),
    WriterConfig::default(),
    ReaderConfig::default(),
    4,
    Some(async_config),  // Enable async graph updates
    Some(gc_config),     // Enable garbage collection
);
```

### Shutdown Order

When `storage.shutdown()` is called, the subsystem shuts down components in this order:

```
on_shutdown():
  1. Writer.flush()          - Flush pending mutations (may add to pending queue)
  2. AsyncUpdater.shutdown() - Drain pending queue, build remaining edges
  3. Join consumer tasks     - Wait for consumers to exit
  4. GC.shutdown()           - Stop background cleanup (last)
  5. (storage closes)        - RocksDB cleanup
```

> **Note:** Writer must flush BEFORE AsyncUpdater shuts down. Otherwise, mutations
> using the async path (`build_index=false`) would be flushed after AsyncUpdater
> is already shut down, leaving vectors without HNSW edges.
>
> GC shuts down last because it has no dependencies on Writer/AsyncUpdater and can
> continue cleaning deleted vectors while other components shut down.

```
  ┌─────────────────────────────────────────────────────────┐
  │                    Shutdown Timeline                     │
  ├─────────────────────────────────────────────────────────┤
  │                                                          │
  │  shutdown()                                              │
  │      │                                                   │
  │      ▼                                                   │
  │  ┌────────────────┐                                      │
  │  │ Writer.flush() │  Drain + close mutation channel      │
  │  └───────┬────────┘  (may add to async pending queue)    │
  │          │                                               │
  │          ▼                                               │
  │  ┌────────────────────────┐                              │
  │  │ AsyncUpdater.shutdown()│  Drain pending, build edges  │
  │  └───────┬────────────────┘                              │
  │          │                                               │
  │          ▼                                               │
  │  ┌──────────────────────┐                                │
  │  │ Join consumer tasks  │  Consumers exit, join handles  │
  │  └───────┬──────────────┘                                │
  │          │                                               │
  │          ▼                                               │
  │  ┌────────────────┐                                      │
  │  │ GC.shutdown()  │  Stop background cleanup (last)      │
  │  └───────┬────────┘                                      │
  │          │                                               │
  │          ▼                                               │
  │     (RocksDB closes)                                     │
  │                                                          │
  └─────────────────────────────────────────────────────────┘
```

**Why this order matters:**

1. **Writer flush first**: Ensures all async-path mutations have been enqueued to the pending queue before the updater stops
2. **AsyncUpdater second**: Drains pending queue and completes graph builds
3. **Join consumers**: Cooperative shutdown after channels close
4. **GC last**: Can continue cleanup while other components drain

### Best Practices for Lifecycle

1. **Always use `Subsystem.start()` or `start_with_async()`**: These methods manage consumer handles and ensure proper shutdown

2. **Enable GC for long-running applications**:
   ```rust
   let gc_config = GcConfig::default()
       .with_interval(Duration::from_secs(60));
   ```

3. **Call `storage.shutdown()` on application exit**: This triggers the proper shutdown sequence

4. **Don't manually manage consumers**: Let `Subsystem` handle spawning and joining

5. **Monitor GC metrics** for operational visibility:
   ```rust
   // Note: GC is managed by Subsystem, metrics available via storage telemetry
   ```

---

## Complete Usage Flow

### Flow 1: Exact Search with L2 Distance

```rust
use motlie_db::vector::{
    Storage, EmbeddingRegistry, EmbeddingBuilder, Distance, Processor, ExternalKey,
};
use motlie_db::Id;
use std::sync::Arc;
use std::path::Path;

// 1. Initialize storage
let mut storage = Storage::readwrite(Path::new("./vector_db"));
storage.ready()?;
let storage = Arc::new(storage);

// 2. Register embedding space
let registry = Arc::new(EmbeddingRegistry::new(storage.clone()));
registry.prewarm()?;

let embedding = registry.register(
    EmbeddingBuilder::new("sift", 128, Distance::L2),
)?;

// 3. Create Processor (handles index + cache)
let processor = Processor::new(storage.clone(), registry.clone());

// 4. Insert vectors
for (i, vector) in base_vectors.iter().enumerate() {
    let external_key = ExternalKey::NodeId(Id::new());
    let _vec_id = processor.insert_vector(&embedding, external_key, vector, true)?;
}

// 5. Search
let results = processor.search(&embedding, &query, 10, 100)?;
for result in results {
    println!("Key: {:?}, L2 Distance: {:.4}", result.external_key, result.distance);
}
```

### Flow 2: RaBitQ Search with Cosine Distance

```rust
use motlie_db::vector::{
    Storage, EmbeddingRegistry, EmbeddingBuilder, Distance,
    Processor, ExternalKey, SearchConfig,
};
use motlie_db::Id;
use std::sync::Arc;
use std::path::Path;

// 1. Initialize storage
let mut storage = Storage::readwrite(Path::new("./vector_db"));
storage.ready()?;
let storage = Arc::new(storage);

// 2. Register embedding (MUST be Cosine for RaBitQ)
let registry = Arc::new(EmbeddingRegistry::new(storage.clone()));
registry.prewarm()?;
let embedding = registry.register(
    EmbeddingBuilder::new("clip-vit-b32", 512, Distance::Cosine),
)?;

// 3. Create Processor (manages HNSW + RaBitQ cache)
let processor = Processor::new(storage.clone(), registry.clone());

// 4. Insert vectors (binary codes cached on insert)
for (i, vector) in base_vectors.iter().enumerate() {
    let external_key = ExternalKey::NodeId(Id::new());
    let _vec_id = processor.insert_vector(&embedding, external_key, vector, true)?;
}

// 5. Two-phase search: ADC pre-filter → exact rerank
let config = SearchConfig::new(embedding.clone(), 10)
    .rabitq()?
    .with_ef(100)
    .with_rerank_factor(10);
let results = processor.search_with_config(&config, &query)?;

for result in results {
    println!("Key: {:?}, Cosine Distance: {:.4}", result.external_key, result.distance);
}
```

### Flow 3: Using SearchConfig Builder

```rust
use motlie_db::vector::{SearchConfig, SearchStrategy};

// For exact search (any distance metric)
let config = SearchConfig::new(embedding.clone(), 10)
    .exact()
    .with_ef(100);

// For RaBitQ (Cosine only - returns Result)
let config = SearchConfig::new(embedding.clone(), 10)
    .rabitq()?
    .with_ef(100)
    .with_rerank_factor(10)
    .with_parallel_rerank_threshold(3200);

// Inspect configuration
println!("Strategy: {:?}", config.strategy());
println!("k: {}", config.k());
println!("ef: {}", config.ef());
println!("Parallel threshold: {}", config.parallel_rerank_threshold());
```

---

## Public API Reference

### Distance Metric

```rust
pub enum Distance {
    Cosine,      // 1 - cos(a,b), range [0,2]
    L2,          // ||a-b||, range [0,∞)
    DotProduct,  // -a·b (negated for min-heap)
}

impl Distance {
    pub fn compute(&self, a: &[f32], b: &[f32]) -> f32;
    pub fn is_lower_better(&self) -> bool;  // Always true
    pub fn as_str(&self) -> &'static str;
}
```

### Embedding Types

```rust
// Immutable embedding specification
pub struct Embedding { /* private */ }
impl Embedding {
    pub fn code(&self) -> u64;
    pub fn code_bytes(&self) -> [u8; 8];
    pub fn model(&self) -> &str;
    pub fn dim(&self) -> u32;
    pub fn distance(&self) -> Distance;
    pub fn storage_type(&self) -> VectorElementType;
    pub fn has_embedder(&self) -> bool;
    pub fn embed(&self, document: &str) -> Result<Vec<f32>>;
    pub fn embed_batch(&self, documents: &[&str]) -> Result<Vec<Vec<f32>>>;
    pub fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32;
    pub fn validate_vector(&self, vector: &[f32]) -> Result<()>;
}

// Builder for creating embeddings
pub struct EmbeddingBuilder { /* private */ }
impl EmbeddingBuilder {
    pub fn new(model: impl Into<String>, dim: u32, distance: Distance) -> Self;
    pub fn with_embedder(self, embedder: Arc<dyn Embedder>) -> Self;
    pub fn with_hnsw_m(self, m: u16) -> Self;
    pub fn with_hnsw_ef_construction(self, ef_construction: u16) -> Self;
    pub fn with_rabitq_bits(self, bits: u8) -> Self;
    pub fn with_rabitq_seed(self, seed: u64) -> Self;
    pub fn model(&self) -> &str;
    pub fn dim(&self) -> u32;
    pub fn distance(&self) -> Distance;
}

// Registry for managing embedding spaces
pub struct EmbeddingRegistry { /* private */ }
impl EmbeddingRegistry {
    pub fn new(storage: Arc<Storage>) -> Self;
    pub fn new_without_storage() -> Self;
    pub fn set_storage(&self, storage: Arc<Storage>) -> Result<()>;
    pub fn prewarm(&self) -> Result<usize>;
    pub fn register(&self, builder: EmbeddingBuilder) -> Result<Embedding>;
    pub fn register_in_txn(
        &self,
        builder: EmbeddingBuilder,
        txn: &Transaction<'_, TransactionDB>,
    ) -> Result<Embedding>;
    pub fn get(&self, model: &str, dim: u32, distance: Distance) -> Option<Embedding>;
    pub fn get_by_code(&self, code: u64) -> Option<Embedding>;
    pub fn set_embedder(&self, code: u64, embedder: Arc<dyn Embedder>) -> Result<()>;
    pub fn list_all(&self) -> Vec<Embedding>;
    pub fn find_by_model(&self, model: &str) -> Vec<Embedding>;
    pub fn find_by_distance(&self, distance: Distance) -> Vec<Embedding>;
    pub fn find_by_dim(&self, dim: u32) -> Vec<Embedding>;
    pub fn find(&self, filter: &EmbeddingFilter) -> Vec<Embedding>;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
}

// Filter for querying embeddings
pub struct EmbeddingFilter { /* private */ }
impl EmbeddingFilter {
    pub fn default() -> Self;
    pub fn model(self, model: impl Into<String>) -> Self;
    pub fn dim(self, dim: u32) -> Self;
    pub fn distance(self, distance: Distance) -> Self;
}
```

### ExternalKey (Polymorphic External ID)

```rust
pub enum ExternalKey {
    NodeId(Id),
    NodeFragment(Id, TimestampMilli),
    Edge(Id, Id, NameHash),
    EdgeFragment(Id, Id, NameHash, TimestampMilli),
    NodeSummary(SummaryHash),
    EdgeSummary(SummaryHash),
}

impl ExternalKey {
    pub fn to_bytes(&self) -> Vec<u8>;
    pub fn from_bytes(bytes: &[u8]) -> Result<Self>;
    pub fn variant_name(&self) -> &'static str;
    pub fn node_id(&self) -> Option<Id>;
}
```

### Configuration Types

```rust
// EmbeddingSpec is the single source of truth for HNSW parameters
pub struct EmbeddingSpec {
    pub model: String,
    pub dim: u32,
    pub distance: Distance,
    pub storage_type: VectorElementType,
    pub hnsw_m: u16,                 // HNSW M parameter
    pub hnsw_ef_construction: u16,   // HNSW build beam width
    pub rabitq_bits: u8,
    pub rabitq_seed: u64,
}
impl EmbeddingSpec {
    // Derived HNSW values (always computed, never stored)
    pub fn m(&self) -> usize;              // hnsw_m as usize
    pub fn m_max(&self) -> usize;          // 2 * m()
    pub fn m_max_0(&self) -> usize;        // 2 * m()
    pub fn m_l(&self) -> f32;              // 1.0 / ln(m())
    pub fn ef_construction(&self) -> usize; // hnsw_ef_construction as usize
}

// HNSW ConfigWarning for validation
pub mod hnsw {
    pub enum ConfigWarning {
        LowM(usize),
        HighM(usize),
        LowEfConstruction { ef_construction: usize, recommended: usize },
        HighEfConstruction(usize),
        LowDimension(usize),
        HighDimension(usize),
    }
}

// RaBitQ configuration
pub struct RaBitQConfig {
    pub bits_per_dim: u8,
    pub rotation_seed: u64,
    pub enabled: bool,
    pub use_simd_dot: bool,
}
impl RaBitQConfig {
    pub fn code_size(&self, dim: usize) -> usize;
    pub fn validate(&self) -> Vec<RaBitQConfigWarning>;
    pub fn is_valid(&self) -> bool;
}

// VectorConfig only contains RaBitQ config (HNSW params in EmbeddingSpec)
pub struct VectorConfig {
    pub rabitq: RaBitQConfig,
}

// Vector storage precision
pub enum VectorElementType {
    F32,  // 4 bytes/dim, full precision
    F16,  // 2 bytes/dim, ~0.1% loss for normalized vectors
}
```

### HNSW Index

```rust
// Access via hnsw::Index
pub struct Index { /* private */ }
impl Index {
    // Construction (the only constructor)
    pub fn from_spec(
        embedding: EmbeddingCode,
        spec: &EmbeddingSpec,
        batch_threshold: usize,  // Runtime knob for neighbor fetching
        nav_cache: Arc<NavigationCache>,
    ) -> Self;

    // Accessors
    pub fn embedding(&self) -> EmbeddingCode;
    pub fn storage_type(&self) -> VectorElementType;
    pub fn distance_metric(&self) -> Distance;
    pub fn nav_cache(&self) -> &Arc<NavigationCache>;
    pub fn dim(&self) -> usize;
    pub fn m(&self) -> usize;
    pub fn ef_construction(&self) -> usize;
    pub fn m_l(&self) -> f32;
    pub fn batch_threshold(&self) -> usize;

    // Search - Exact (all distance metrics)
    pub fn search(
        &self,
        storage: &Storage,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<Vec<(f32, VecId)>>;

    // Search - RaBitQ (Cosine only)
    pub fn search_with_rabitq_cached(
        &self,
        storage: &Storage,
        query: &[f32],
        encoder: &RaBitQ,
        code_cache: &BinaryCodeCache,
        k: usize,
        ef: usize,
        rerank_factor: usize,
    ) -> Result<Vec<(f32, VecId)>>;

    // Batch operations
    pub fn get_vectors_batch(&self, storage: &Storage, vec_ids: &[VecId]) -> Result<Vec<Option<Vec<f32>>>>;
    pub fn batch_distances(&self, storage: &Storage, query: &[f32], vec_ids: &[VecId]) -> Result<Vec<(VecId, f32)>>;
    pub fn get_neighbors_batch(
        &self,
        storage: &Storage,
        vec_ids: &[VecId],
        layer: u8,
    ) -> Result<Vec<(VecId, roaring::RoaringBitmap)>>;
    pub fn get_neighbor_count(&self, storage: &Storage, vec_id: VecId, layer: u8) -> Result<u64>;
    pub fn get_binary_code(&self, storage: &Storage, vec_id: VecId) -> Result<Option<Vec<u8>>>;
    pub fn get_binary_codes_batch(&self, storage: &Storage, vec_ids: &[VecId]) -> Result<Vec<Option<Vec<u8>>>>;
}

// Transaction-aware insert API (free functions in hnsw module)
pub struct CacheUpdate { /* public fields */ }
impl CacheUpdate {
    pub fn apply(self, nav_cache: &NavigationCache);
}

pub fn insert(
    index: &Index,
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    storage: &Storage,
    vec_id: VecId,
    vector: &[f32],
) -> Result<CacheUpdate>;
```

### Processor (Internal API)

`Processor` is the internal execution engine used by mutation/query consumers.
It is exposed for internal integrations and test helpers.

**Note:** HNSW structural parameters (m, ef_construction, etc.) are derived from
`EmbeddingSpec` which is persisted in RocksDB and protected by SpecHash. Only
`batch_threshold` is a runtime knob that can vary per process without affecting
index integrity. See `docs/CONFIG.md` for the design rationale.

```rust
pub struct Processor { /* private */ }
impl Processor {
    // Preferred constructors (HNSW params derived from EmbeddingSpec)
    pub fn new(storage: Arc<Storage>, registry: Arc<EmbeddingRegistry>) -> Self;
    pub fn with_batch_threshold(
        storage: Arc<Storage>,
        registry: Arc<EmbeddingRegistry>,
        batch_threshold: usize,
    ) -> Self;
    pub fn new_with_nav_cache(
        storage: Arc<Storage>,
        registry: Arc<EmbeddingRegistry>,
        nav_cache: Arc<NavigationCache>,
    ) -> Self;
    pub fn with_rabitq_config(
        storage: Arc<Storage>,
        registry: Arc<EmbeddingRegistry>,
        rabitq_config: RaBitQConfig,
    ) -> Self;
    pub fn with_rabitq_config_and_batch_threshold(
        storage: Arc<Storage>,
        registry: Arc<EmbeddingRegistry>,
        rabitq_config: RaBitQConfig,
        batch_threshold: usize,
        nav_cache: Arc<NavigationCache>,
    ) -> Self;

    // Observability / control
    pub fn set_async_backpressure_threshold(&self, threshold: usize);
    pub fn async_backpressure_threshold(&self) -> usize;
    pub fn pending_queue_size(&self) -> usize;

    // Storage/registry accessors
    pub fn storage(&self) -> &Storage;
    pub fn storage_arc(&self) -> &Arc<Storage>;
    pub fn registry(&self) -> &EmbeddingRegistry;
    pub fn registry_arc(&self) -> &Arc<EmbeddingRegistry>;

    // Vector ops
    pub fn insert_vector(
        &self,
        embedding: &Embedding,
        external_key: ExternalKey,
        vector: &[f32],
        build_index: bool,
    ) -> Result<VecId>;

    pub fn insert_batch(
        &self,
        embedding: &Embedding,
        vectors: &[(ExternalKey, Vec<f32>)],
        build_index: bool,
    ) -> Result<Vec<VecId>>;

    pub fn search(
        &self,
        embedding: &Embedding,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<SearchResult>>;

    pub fn search_with_config(
        &self,
        config: &SearchConfig,
        query: &[f32],
    ) -> Result<Vec<SearchResult>>;
}
```

### Mutation Types

The `Mutation` enum represents all write operations. Each variant has a corresponding
struct with builder methods for ergonomic construction.

```rust
/// All write operations dispatched through the Writer channel.
pub enum Mutation {
    AddEmbeddingSpec(AddEmbeddingSpec),  // Register new embedding space
    InsertVector(InsertVector),           // Insert single vector
    DeleteVector(DeleteVector),           // Delete vector by external key
    InsertVectorBatch(InsertVectorBatch), // Batch insert (optimization)
    Flush(FlushMarker),                   // Internal: sync marker
}
```

#### InsertVector

Insert a single vector into an embedding space.

```rust
pub struct InsertVector {
    /// Embedding space code (from Embedding::code())
    pub embedding: EmbeddingCode,
    /// External key identifying the source entity (node, edge, fragment, summary)
    pub external_key: ExternalKey,
    /// Vector data (must match embedding dimension)
    pub vector: Vec<f32>,
    /// If true, builds HNSW index synchronously; if false, queues for async processing
    pub immediate_index: bool,
}

impl InsertVector {
    /// Create with async indexing (default, recommended for throughput)
    pub fn new(embedding: &Embedding, external_key: ExternalKey, vector: Vec<f32>) -> Self;
    /// Enable synchronous HNSW indexing (use for latency-sensitive single inserts)
    pub fn immediate(self) -> Self;
}
```

**Example:**
```rust
let insert = InsertVector::new(&embedding, ExternalKey::NodeId(node_id), vec![0.1; 512])
    .immediate();  // Optional: force sync indexing
insert.run(&writer).await?;
```

#### DeleteVector

Delete a vector by its external key.

```rust
pub struct DeleteVector {
    /// Embedding space code
    pub embedding: EmbeddingCode,
    /// External key of the vector to delete
    pub external_key: ExternalKey,
}

impl DeleteVector {
    pub fn new(embedding: &Embedding, external_key: ExternalKey) -> Self;
}
```

**Behavior:** Removes ID mappings, vector data, metadata, and marks HNSW edges for
lazy cleanup by the garbage collector.

#### InsertVectorBatch

Efficiently insert multiple vectors in a single operation.

```rust
pub struct InsertVectorBatch {
    /// Embedding space code
    pub embedding: EmbeddingCode,
    /// Batch of (external_key, vector) pairs
    pub vectors: Vec<(ExternalKey, Vec<f32>)>,
    /// If true, builds HNSW index synchronously for all vectors
    pub immediate_index: bool,
}

impl InsertVectorBatch {
    pub fn new(embedding: &Embedding, vectors: Vec<(ExternalKey, Vec<f32>)>) -> Self;
    pub fn immediate(self) -> Self;
}
```

**Performance:** More efficient than individual InsertVector for bulk loading
(amortizes transaction overhead, enables batch HNSW construction).

#### AddEmbeddingSpec

Register a new embedding space (typically done via EmbeddingRegistry::register).

```rust
pub struct AddEmbeddingSpec {
    pub code: EmbeddingCode,           // Allocated code (primary key)
    pub model: String,                 // Model name (e.g., "gemma", "ada-002")
    pub dim: u32,                      // Vector dimensionality
    pub distance: Distance,            // Similarity metric
    pub storage_type: VectorElementType, // F32 or F16
    pub hnsw_m: u16,                   // HNSW M parameter (default: 16)
    pub hnsw_ef_construction: u16,     // HNSW build beam width (default: 200)
    pub rabitq_bits: u8,               // RaBitQ bits per dim (default: 1)
    pub rabitq_seed: u64,              // RaBitQ rotation seed (default: 42)
}
```

### Query Types

The `Query` enum represents all read operations. Each variant has a corresponding
struct that implements `QueryExecutor` for direct execution or can be dispatched
through the Reader channel.

```rust
pub enum Query {
    // Point Lookups
    GetVector(GetVectorDispatch),       // Get raw vector by external ID
    GetInternalId(GetInternalIdDispatch), // External key → internal vec_id
    GetExternalId(GetExternalIdDispatch), // Internal vec_id → external key
    ResolveIds(ResolveIdsDispatch),     // Batch resolve vec_ids → external keys

    // Registry Queries
    ListEmbeddings(ListEmbeddingsDispatch),  // List all embedding spaces
    FindEmbeddings(FindEmbeddingsDispatch),  // Filter by model/dim/distance

    // Search Operations
    SearchKNN(SearchKNNDispatch),       // K-nearest neighbor search
}
```

#### SearchKNN

K-nearest neighbor search using HNSW + optional RaBitQ acceleration.

```rust
pub struct SearchKNN {
    /// Embedding space to search
    pub embedding: Embedding,
    /// Query vector (must match embedding dimension)
    pub query: Vec<f32>,
    /// Number of results to return
    pub k: usize,
    /// Search expansion factor (higher = better recall, slower). Default: 100
    pub ef: usize,
    /// Force exact distance computation (bypass RaBitQ). Default: false
    pub exact: bool,
    /// Re-rank factor for RaBitQ mode (k * rerank_factor candidates). Default: 10
    pub rerank_factor: usize,
}

impl SearchKNN {
    pub fn new(embedding: &Embedding, query: Vec<f32>, k: usize) -> Self;
    pub fn with_ef(self, ef: usize) -> Self;
    pub fn exact(self) -> Self;
    pub fn with_rerank_factor(self, factor: usize) -> Self;
}
```

**Example:**
```rust
let results = SearchKNN::new(&embedding, query_vec, 10)
    .with_ef(200)           // Higher ef for better recall
    .with_rerank_factor(20) // More candidates for re-ranking
    .run(&search_reader, Duration::from_secs(5))
    .await?;
```

**Returns:** `Vec<SearchResult>` where:
```rust
pub struct SearchResult {
    pub external_key: ExternalKey,  // Typed external key
    pub vec_id: VecId,              // Internal vector ID (u32)
    pub distance: f32,              // Distance from query (lower = more similar)
}
```

#### GetVector

Retrieve raw vector data by external ID.

```rust
pub struct GetVector {
    pub embedding: EmbeddingCode,
    pub id: Id,  // External document ID (ULID)
}
// Returns: Option<Vec<f32>>
```

#### GetInternalId / GetExternalId

Resolve between external keys and internal vector IDs.

```rust
pub struct GetInternalId {
    pub embedding: EmbeddingCode,
    pub external_key: ExternalKey,
}
// Returns: Option<VecId>

pub struct GetExternalId {
    pub embedding: EmbeddingCode,
    pub vec_id: VecId,
}
// Returns: Option<ExternalKey>
```

#### ResolveIds

Batch resolve internal vec_ids to external keys (used after SearchKNN).

```rust
pub struct ResolveIds {
    pub embedding: EmbeddingCode,
    pub vec_ids: Vec<VecId>,
}
// Returns: Vec<Option<ExternalKey>>
```

### Runnable Traits

Two `Runnable` traits provide ergonomic execution patterns:

#### MutationRunnable (for mutations)

```rust
// Re-exported as: pub use mutation::Runnable as MutationRunnable;
#[async_trait]
pub trait MutationRunnable {
    /// Execute this mutation against the writer (fire-and-forget).
    /// Use Writer::flush() after for confirmation.
    async fn run(self, writer: &Writer) -> Result<()>;
}

// Implemented for:
impl MutationRunnable for InsertVector { ... }
impl MutationRunnable for InsertVectorBatch { ... }
impl MutationRunnable for DeleteVector { ... }
impl MutationRunnable for AddEmbeddingSpec { ... }
impl MutationRunnable for MutationBatch { ... }
```

**Example:**
```rust
use motlie_db::vector::{InsertVector, MutationRunnable};

InsertVector::new(&embedding, key, vector).run(&writer).await?;
writer.flush().await?;  // Wait for commit confirmation
```

#### Runnable<R> (for queries)

```rust
// Generic over reader type (Reader or SearchReader)
#[async_trait]
pub trait Runnable<R> {
    type Output: Send + 'static;
    async fn run(self, reader: &R, timeout: Duration) -> Result<Self::Output>;
}

// SearchKNN implements Runnable for both Reader and SearchReader
impl Runnable<Reader> for SearchKNN { type Output = Vec<SearchResult>; ... }
impl Runnable<SearchReader> for SearchKNN { type Output = Vec<SearchResult>; ... }
```

**Example:**
```rust
use motlie_db::vector::{SearchKNN, Runnable};

let results: Vec<SearchResult> = SearchKNN::new(&embedding, query, 10)
    .with_ef(100)
    .run(&search_reader, Duration::from_secs(5))
    .await?;
```

### MutationBatch Helper

Batch multiple mutations into a single channel send for efficiency.

```rust
pub struct MutationBatch(pub Vec<Mutation>);

impl MutationBatch {
    pub fn new() -> Self;
    pub fn push<M: Into<Mutation>>(mut self, mutation: M) -> Self;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
}

impl MutationRunnable for MutationBatch {
    async fn run(self, writer: &Writer) -> Result<()>;
}
```

**Example:**
```rust
MutationBatch::new()
    .push(InsertVector::new(&embedding, key1, vec1))
    .push(InsertVector::new(&embedding, key2, vec2))
    .push(DeleteVector::new(&embedding, old_key))
    .run(&writer)
    .await?;
```

### MPSC / Channel API (Public)

The async API uses channels for mutations (MPSC) and queries (MPMC).

```rust
// Mutations (Writer - MPSC)
pub struct Writer { /* mpsc::Sender<Vec<Mutation>> */ }
impl Writer {
    pub async fn send(&self, mutations: Vec<Mutation>) -> Result<()>;
    pub async fn send_sync(&self, mutations: Vec<Mutation>) -> Result<()>;
    pub async fn flush(&self) -> Result<()>;
    pub fn is_closed(&self) -> bool;
}

pub struct WriterConfig { pub channel_buffer_size: usize }
pub struct MutationConsumer { /* receiver, config, processor */ }
pub fn create_writer(config: WriterConfig) -> (Writer, mpsc::Receiver<Vec<Mutation>>);
pub fn spawn_mutation_consumer(consumer: MutationConsumer) -> tokio::task::JoinHandle<Result<()>>;
pub fn spawn_mutation_consumer_with_storage(
    receiver: mpsc::Receiver<Vec<Mutation>>,
    config: WriterConfig,
    storage: Arc<Storage>,
    registry: Arc<EmbeddingRegistry>,
) -> tokio::task::JoinHandle<Result<()>>;
pub fn spawn_mutation_consumer_with_storage_autoreg(
    receiver: mpsc::Receiver<Vec<Mutation>>,
    config: WriterConfig,
    storage: Arc<Storage>,
) -> tokio::task::JoinHandle<Result<()>>;

// Queries (Reader - MPMC / flume)
pub struct Reader { /* flume::Sender<Query> */ }
impl Reader {
    pub async fn send_query(&self, query: Query) -> Result<()>;
    pub fn is_closed(&self) -> bool;
}

pub struct ReaderConfig { pub channel_buffer_size: usize }
pub struct QueryConsumer { /* receiver, config, storage */ }
pub struct ProcessorQueryConsumer { /* receiver, config, processor */ }
pub fn create_reader(config: ReaderConfig) -> (Reader, flume::Receiver<Query>);
pub fn spawn_query_consumer(consumer: QueryConsumer) -> tokio::task::JoinHandle<Result<()>>;
pub fn spawn_query_consumer_with_processor(
    consumer: ProcessorQueryConsumer,
) -> tokio::task::JoinHandle<Result<()>>;
pub fn spawn_query_consumers(
    receiver: flume::Receiver<Query>,
    config: ReaderConfig,
    storage: Arc<Storage>,
    count: usize,
) -> Vec<tokio::task::JoinHandle<Result<()>>>;
pub fn spawn_query_consumers_with_processor(
    receiver: flume::Receiver<Query>,
    config: ReaderConfig,
    processor: Arc<Processor>,
    count: usize,
) -> Vec<tokio::task::JoinHandle<Result<()>>>;
pub fn spawn_query_consumers_with_storage(
    receiver: flume::Receiver<Query>,
    config: ReaderConfig,
    storage: Arc<Storage>,
    registry: Arc<EmbeddingRegistry>,
    count: usize,
) -> Vec<tokio::task::JoinHandle<Result<()>>>;
pub fn spawn_query_consumers_with_storage_autoreg(
    receiver: flume::Receiver<Query>,
    config: ReaderConfig,
    storage: Arc<Storage>,
    count: usize,
) -> Vec<tokio::task::JoinHandle<Result<()>>>;

// SearchReader (bundles Reader + Processor)
pub struct SearchReader { /* reader + processor */ }
impl SearchReader {
    pub async fn search_knn(
        &self,
        embedding: &Embedding,
        query: Vec<f32>,
        k: usize,
        ef: usize,
        timeout: Duration,
    ) -> Result<Vec<SearchResult>>;

    pub async fn search_knn_exact(
        &self,
        embedding: &Embedding,
        query: Vec<f32>,
        k: usize,
        ef: usize,
        timeout: Duration,
    ) -> Result<Vec<SearchResult>>;

    pub fn reader(&self) -> &Reader;
    pub fn processor(&self) -> &Arc<Processor>;
    pub fn is_closed(&self) -> bool;
}

pub fn create_search_reader(
    config: ReaderConfig,
    processor: Arc<Processor>,
) -> (SearchReader, flume::Receiver<Query>);

pub fn create_search_reader_with_storage(
    config: ReaderConfig,
    storage: Arc<Storage>,
) -> (SearchReader, flume::Receiver<Query>);
```

**Runnable helpers:**

```rust
// Mutations: use MutationRunnable (alias from mutation module)
InsertVector::new(&embedding, ExternalKey::NodeId(id), vec![...])
    .run(&writer)
    .await?;

// Queries: use Runnable<Reader> or Runnable<SearchReader>
let result = GetVector::new(embedding.code(), id)
    .run(&reader, Duration::from_secs(5))
    .await?;

let results = SearchKNN::new(&embedding, query, 10)
    .run(&search_reader, Duration::from_secs(5))
    .await?;
```

### Channel API with Runnable Helpers

The channel APIs are designed to be used through `Runnable` helpers for both
mutations and queries. These helpers build the correct enum variants and handle
timeouts ergonomically.

#### Flow 1: Mutations via `Writer` (Insert + Delete)

```rust
use motlie_db::vector::{
    create_writer, spawn_mutation_consumer_with_storage, WriterConfig,
    InsertVector, DeleteVector, MutationBatch, EmbeddingBuilder, EmbeddingRegistry, Distance, ExternalKey, Storage,
};
use motlie_db::Id;
use std::path::Path;
use std::sync::Arc;

let mut storage = Storage::readwrite(Path::new("./vector_db"));
storage.ready()?;
let storage = Arc::new(storage);
let registry = Arc::new(EmbeddingRegistry::new(storage.clone()));
registry.prewarm()?;

let embedding = registry.register(
    EmbeddingBuilder::new("clip-vit-b32", 512, Distance::Cosine),
)?;

// Create writer + consumer
let (writer, receiver) = create_writer(WriterConfig::default());
let _handle = spawn_mutation_consumer_with_storage(
    receiver,
    WriterConfig::default(),
    storage.clone(),
    registry.clone(),
);

// Build mutations
let insert = InsertVector::new(&embedding, ExternalKey::NodeId(Id::new()), vec![0.1; 512])
    .immediate(); // build index synchronously
let delete = DeleteVector::new(&embedding, ExternalKey::NodeId(Id::new()));

// Runnable helpers
insert.run(&writer).await?;
delete.run(&writer).await?;
writer.flush().await?;

// Batch helper (sends Vec<Mutation> under the hood)
MutationBatch::new()
    .push(insert)
    .push(delete)
    .run(&writer)
    .await?;
```

#### Flow 2: Queries via `Reader` (SearchKNN)

```rust
use motlie_db::vector::{
    create_reader, spawn_query_consumers_with_processor, ReaderConfig,
    SearchKNN, Processor,
};
use std::sync::Arc;
use std::time::Duration;

// Assume storage/registry/embedding are already created (see Flow 1 above)
// Create processor explicitly (needed for SearchKNNDispatch)
let processor = Arc::new(Processor::new(storage.clone(), registry.clone()));

// Create reader + consumer
let (reader, receiver) = create_reader(ReaderConfig::default());
spawn_query_consumers_with_processor(receiver, ReaderConfig::default(), processor.clone(), 2);

// Runnable helper (SearchKNN implements Runnable for Reader/SearchReader)
let results = SearchKNN::new(&embedding, query_vec, 10)
    .with_ef(100)
    .run(&reader, Duration::from_secs(5))
    .await?;

// Assume embedding_cosine uses Distance::Cosine, embedding_l2 uses Distance::L2

// Example: Cosine embedding (auto-selects RaBitQ) with higher rerank factor
let results = SearchKNN::new(&embedding_cosine, query_vec, 10)
    .with_ef(200)
    .with_rerank_factor(20) // k * 20 candidates re-ranked
    .run(&reader, Duration::from_secs(5))
    .await?;

// Example: Exact L2 search (forces exact even if Cosine is available)
let results = SearchKNN::new(&embedding_l2, query_vec, 10)
    .with_ef(100)
    .exact()
    .run(&reader, Duration::from_secs(5))
    .await?;

// Inspect ExternalKey variants in SearchResult
for result in results {
    match result.external_key {
        ExternalKey::NodeId(id) => println!("NodeId: {id}"),
        ExternalKey::NodeFragment(id, ts) => println!("NodeFragment: {id} @ {ts:?}"),
        ExternalKey::Edge(src, dst, name) => println!("Edge: {src} -> {dst} ({name:?})"),
        ExternalKey::EdgeFragment(src, dst, name, ts) => {
            println!("EdgeFragment: {src} -> {dst} ({name:?}) @ {ts:?}")
        }
        ExternalKey::NodeSummary(hash) => println!("NodeSummary: {hash:?}"),
        ExternalKey::EdgeSummary(hash) => println!("EdgeSummary: {hash:?}"),
    }
}
```

**Notes:**
- The `Query` enum is an internal dispatch mechanism; the public, ergonomic path
  is `SearchReader` or `Reader` + `Runnable::run(...)`.

### Search Configuration

```rust
pub enum SearchStrategy {
    Exact,
    RaBitQ { use_cache: bool },
}
impl SearchStrategy {
    pub fn is_rabitq(&self) -> bool;
    pub fn is_exact(&self) -> bool;
}

pub const DEFAULT_PARALLEL_RERANK_THRESHOLD: usize = 3200;
pub const DEFAULT_PENDING_SCAN_LIMIT: usize = 1000;

pub struct SearchConfig { /* private */ }
impl SearchConfig {
    // Construction
    pub fn new(embedding: Embedding, k: usize) -> Self;
    pub fn new_exact(embedding: Embedding, k: usize) -> Self;

    // Builder methods
    pub fn exact(self) -> Self;                           // All metrics
    pub fn rabitq(self) -> Result<Self>;                  // Cosine only
    pub fn rabitq_uncached(self) -> Result<Self>;         // Cosine only
    pub fn with_ef(self, ef: usize) -> Self;
    pub fn with_rerank_factor(self, factor: usize) -> Self;
    pub fn with_k(self, k: usize) -> Self;
    pub fn with_parallel_rerank_threshold(self, threshold: usize) -> Self;
    pub fn with_pending_scan_limit(self, limit: usize) -> Self;
    pub fn no_pending_fallback(self) -> Self;

    // Accessors
    pub fn embedding(&self) -> &Embedding;
    pub fn strategy(&self) -> SearchStrategy;
    pub fn k(&self) -> usize;
    pub fn ef(&self) -> usize;
    pub fn rerank_factor(&self) -> usize;
    pub fn parallel_rerank_threshold(&self) -> usize;
    pub fn pending_scan_limit(&self) -> usize;
    pub fn should_use_parallel_rerank(&self, candidate_count: usize) -> bool;
    pub fn distance(&self) -> Distance;
    pub fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32;
    pub fn validate_embedding_code(&self, expected_code: u64) -> Result<()>;
}
```

### Caching

```rust
// Navigation cache for HNSW layer traversal
pub struct NavigationCache { /* private */ }
impl NavigationCache {
    pub fn new() -> Self;
    pub fn with_config(config: NavigationCacheConfig) -> Self;
    pub fn get(&self, embedding: EmbeddingCode) -> Option<NavigationLayerInfo>;
    pub fn put(&self, embedding: EmbeddingCode, info: NavigationLayerInfo);
    pub fn update<F>(&self, embedding: EmbeddingCode, m: usize, f: F);
    pub fn remove(&self, embedding: EmbeddingCode) -> Option<NavigationLayerInfo>;
    pub fn config(&self) -> &NavigationCacheConfig;
    pub fn should_cache_layer(&self, layer: HnswLayer) -> bool;
    pub fn get_neighbors_cached<F, E>(
        &self,
        embedding: EmbeddingCode,
        vec_id: VecId,
        layer: HnswLayer,
        fetch_fn: F,
    ) -> Result<roaring::RoaringBitmap, E>;
    pub fn invalidate_edges(&self, embedding: EmbeddingCode, vec_id: VecId, layer: HnswLayer);
    pub fn edge_cache_stats(&self) -> (usize, usize);
    pub fn hot_cache_stats(&self) -> usize;
}

pub struct NavigationLayerInfo { /* private */ }
impl NavigationLayerInfo {
    pub fn new(m: usize) -> Self;
    pub fn random_layer<R: Rng>(&self, rng: &mut R) -> HnswLayer;
    pub fn entry_point(&self) -> Option<VecId>;
    pub fn entry_point_for_layer(&self, layer: HnswLayer) -> Option<VecId>;
    pub fn is_empty(&self) -> bool;
    pub fn maybe_update_entry(&mut self, vec_id: VecId, node_layer: HnswLayer) -> bool;
    pub fn total_nodes(&self) -> u64;
}

// Binary code cache for RaBitQ
pub struct BinaryCodeCache { /* private */ }
pub struct BinaryCodeEntry {
    pub code: Vec<u8>,
    pub correction: AdcCorrection,
}
impl BinaryCodeCache {
    pub fn new() -> Self;
    pub fn put(&self, embedding: EmbeddingCode, vec_id: VecId, code: Vec<u8>, correction: AdcCorrection);
    pub fn get(&self, embedding: EmbeddingCode, vec_id: VecId) -> Option<Arc<BinaryCodeEntry>>;
    pub fn get_batch(&self, embedding: EmbeddingCode, vec_ids: &[VecId]) -> Vec<Option<Arc<BinaryCodeEntry>>>;
    pub fn contains(&self, embedding: EmbeddingCode, vec_id: VecId) -> bool;
    pub fn stats(&self) -> (usize, usize);  // (count, bytes)
    pub fn clear(&self);
    pub fn count_for_embedding(&self, embedding: EmbeddingCode) -> usize;
}
```

### RaBitQ Quantization

```rust
pub struct RaBitQ { /* private */ }
impl RaBitQ {
    // Construction
    pub fn new(dim: usize, bits_per_dim: u8, seed: u64) -> Self;
    pub fn from_config(dim: usize, config: &RaBitQConfig) -> Self;

    // Accessors
    pub fn dim(&self) -> usize;
    pub fn bits_per_dim(&self) -> u8;
    pub fn code_size(&self) -> usize;

    // Encoding
    pub fn encode(&self, vector: &[f32]) -> Vec<u8>;
    pub fn encode_with_correction(&self, vector: &[f32]) -> (Vec<u8>, AdcCorrection);

    // ADC Distance computation (recommended)
    pub fn rotate_query(&self, query: &[f32]) -> Vec<f32>;
    pub fn adc_distance(&self, query_rotated: &[f32], query_norm: f32,
                        code: &[u8], correction: &AdcCorrection) -> f32;

    // Hamming distance (legacy - not used in search, kept for diagnostics)
    pub fn hamming_distance(a: &[u8], b: &[u8]) -> u32;
}
```

### Parallel Reranking

```rust
// Adaptive reranking (auto-selects sequential vs parallel)
pub fn rerank_adaptive<F>(
    candidates: &[VecId],
    distance_fn: F,
    k: usize,
    threshold: usize,
) -> Vec<(f32, VecId)>
where F: Fn(VecId) -> Option<f32> + Sync;

// Convenience wrapper using DEFAULT_PARALLEL_RERANK_THRESHOLD (3200)
pub fn rerank_auto<F>(
    candidates: &[VecId],
    distance_fn: F,
    k: usize,
) -> Vec<(f32, VecId)>
where F: Fn(VecId) -> Option<f32> + Sync;

// Explicit parallel/sequential
pub fn rerank_parallel<F>(candidates: &[VecId], distance_fn: F, k: usize) -> Vec<(f32, VecId)>;
pub fn rerank_sequential<F>(candidates: &[VecId], distance_fn: F, k: usize) -> Vec<(f32, VecId)>;

// Batch distance computation
pub fn batch_distances_parallel<F>(candidates: &[VecId], distance_fn: F) -> Vec<(VecId, f32)>;
pub fn distances_from_vectors_parallel<F>(query: &[f32], vectors: &[(VecId, Vec<f32>)], distance_fn: F) -> Vec<(f32, VecId)>;
```

### Type Aliases

```rust
pub type EmbeddingCode = u64;
pub type VecId = u32;
pub type HnswLayer = u8;
pub type Storage = crate::rocksdb::Storage<Subsystem>;
```
