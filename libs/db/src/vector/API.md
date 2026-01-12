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
5. [Complete Usage Flow](#complete-usage-flow)
6. [Public API Reference](#public-api-reference)

---

## Overview

The vector module provides approximate nearest neighbor (ANN) search using the HNSW algorithm with optional RaBitQ binary quantization. Key features:

- **HNSW Graph Index**: Hierarchical navigable small world graph for O(log n) search
- **Multiple Distance Metrics**: Cosine, L2 (Euclidean), DotProduct
- **RaBitQ Quantization**: Training-free binary codes for fast Hamming pre-filtering
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

An `Embedding` represents a registered embedding space with a unique code, model name, dimensionality, and distance metric. The `EmbeddingRegistry` manages all registered embeddings.

```rust
use motlie_db::vector::{
    EmbeddingBuilder, EmbeddingRegistry, Distance, Storage,
};
use std::sync::Arc;

// Initialize storage
let mut storage = Storage::readwrite(&db_path);
storage.ready()?;

// Create registry
let registry = EmbeddingRegistry::new();

// Pre-warm from existing DB (loads previously registered embeddings)
let count = registry.prewarm(storage.transaction_db()?)?;
println!("Loaded {} existing embeddings", count);

// Register a new embedding space
let builder = EmbeddingBuilder::new("openai-ada-002", 1536, Distance::Cosine);
let embedding = registry.register(builder, storage.transaction_db()?)?;

println!("Registered embedding: code={}, dim={}",
         embedding.code(), embedding.dim());

// Query registered embeddings
let all = registry.list_all();
let cosine_embeddings = registry.find_by_distance(Distance::Cosine);
let by_model = registry.find_by_model("openai");
```

**EmbeddingBuilder Fields:**

| Field | Description |
|-------|-------------|
| `model` | Model identifier (e.g., "openai-ada-002", "clip-vit-b32") |
| `dim` | Vector dimensionality (128, 512, 768, 1536, etc.) |
| `distance` | Distance metric (Cosine, L2, DotProduct) |

### HNSW Configuration

HNSW (Hierarchical Navigable Small World) is a graph-based ANN algorithm. Key parameters:

```rust
use motlie_db::vector::hnsw;

// Default configuration (dim=128, M=16)
let config = hnsw::Config::default();

// Auto-tuned for specific dimension
let config = hnsw::Config::for_dim(512);

// Presets
let high_recall = hnsw::Config::high_recall(768);  // M=32, ef_construction=200
let compact = hnsw::Config::compact(768);          // M=8, ef_construction=50

// Custom configuration
let config = hnsw::Config {
    dim: 512,
    m: 16,              // Connections per node (layer > 0)
    m_max: 32,          // Max connections (layer > 0)
    m_max_0: 64,        // Max connections (layer 0)
    ef_construction: 100, // Build-time beam width
    ..Default::default()
};

// Validate configuration
let warnings = config.validate();
if !warnings.is_empty() {
    for w in warnings {
        println!("Warning: {:?}", w);
    }
}
```

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
use motlie_db::vector::{
    hnsw, NavigationCache, Distance, VectorElementType,
};
use std::sync::Arc;

// Create navigation cache (required for HNSW traversal)
let nav_cache = Arc::new(NavigationCache::new());

// Create HNSW index
let config = hnsw::Config::for_dim(512);
let index = hnsw::Index::new(
    embedding.code(),      // From registered embedding
    Distance::Cosine,
    config,
    nav_cache.clone(),
);

// Or with explicit storage type (F16 saves 50% memory)
let index = hnsw::Index::with_storage_type(
    embedding.code(),
    Distance::Cosine,
    VectorElementType::F16,  // or F32 (default)
    config,
    nav_cache.clone(),
);

// Insert vectors
for (id, vector) in vectors.iter().enumerate() {
    let vec_id = id as u32;
    index.insert(&storage, vec_id, vector)?;
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
let results = index.search(&storage, &query, ef_search, k)?;

// Results are (distance, vec_id) pairs, sorted by distance ascending
for (distance, vec_id) in results {
    println!("vec_id={}, distance={:.4}", vec_id, distance);
}
```

**ef_search Impact:**

| ef_search | Recall@10 (typical) | Latency |
|-----------|---------------------|---------|
| 50 | 85-90% | Fast |
| 100 | 90-95% | Moderate |
| 200 | 95-98% | Slower |

### RaBitQ Search (Cosine Only)

RaBitQ uses binary quantization for fast Hamming distance pre-filtering, then reranks top candidates with exact distance. **Only works with Cosine distance** because:

1. Sign quantization captures **direction**, not magnitude
2. Hamming distance approximates **angular distance** for unit vectors
3. L2 on unnormalized data would lose magnitude information

```rust
use motlie_db::vector::{RaBitQ, BinaryCodeCache};

// Initialize RaBitQ encoder
let rabitq = RaBitQ::new(
    512,  // dimension
    1,    // bits per dimension (1, 2, or 4)
    42,   // seed for rotation matrix
);

// Build binary code cache (in-memory for speed)
let binary_cache = BinaryCodeCache::new();

// Encode all vectors and cache
for (id, vector) in vectors.iter().enumerate() {
    let vec_id = id as u32;
    let code = rabitq.encode(vector);
    binary_cache.put(embedding.code(), vec_id, code);
}

// Two-phase search: Hamming filtering → exact reranking
let results = index.search_with_rabitq_cached(
    &storage,
    &query,
    &rabitq,
    &binary_cache,
    k,              // top-k results
    ef_search,      // beam width for Hamming search
    rerank_factor,  // candidates to rerank = k * rerank_factor
)?;
```

**RaBitQ Parameters:**

| Parameter | Value | Effect |
|-----------|-------|--------|
| `bits_per_dim` | 1 | 32x compression, ~50% recall (no rerank), **recommended** |
| `bits_per_dim` | 2 | 16x compression, ⚠️ **broken** - see note below |
| `bits_per_dim` | 4 | 8x compression, ⚠️ **broken** - see note below |
| `rerank_factor` | 4-20 | Higher = better recall, more I/O |

> ⚠️ **2-bit and 4-bit modes are currently broken** (Issue #43). Binary encoding + Hamming
> distance is incompatible with multi-bit quantization: adjacent quantization levels
> (e.g., level 1=01, level 2=10) have maximum Hamming distance instead of minimum.
> Benchmarks show 2-bit achieves 45.4% recall and 4-bit achieves 37.6% - **worse than 1-bit**.
> Fix requires Gray code encoding (00→01→11→10). Use 1-bit mode until this is resolved.

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

## Complete Usage Flow

### Flow 1: Exact Search with L2 Distance

```rust
use motlie_db::vector::{
    Storage, EmbeddingRegistry, EmbeddingBuilder, Distance,
    hnsw, NavigationCache, VectorElementType,
};
use std::sync::Arc;

// 1. Initialize storage
let mut storage = Storage::readwrite("./vector_db");
storage.ready()?;

// 2. Register embedding space
let registry = EmbeddingRegistry::new();
registry.prewarm(storage.transaction_db()?)?;

let embedding = registry.register(
    EmbeddingBuilder::new("sift", 128, Distance::L2),
    storage.transaction_db()?,
)?;

// 3. Create HNSW index
let nav_cache = Arc::new(NavigationCache::new());
let config = hnsw::Config {
    dim: 128,
    m: 16,
    ef_construction: 100,
    ..Default::default()
};

let index = hnsw::Index::new(
    embedding.code(),
    Distance::L2,
    config,
    nav_cache,
);

// 4. Insert vectors
for (i, vector) in base_vectors.iter().enumerate() {
    index.insert(&storage, i as u32, vector)?;
}

// 5. Search
let results = index.search(&storage, &query, 100, 10)?;
for (dist, id) in results {
    println!("ID: {}, L2 Distance: {:.4}", id, dist);
}
```

### Flow 2: RaBitQ Search with Cosine Distance

```rust
use motlie_db::vector::{
    Storage, EmbeddingRegistry, EmbeddingBuilder, Distance,
    hnsw, NavigationCache, VectorElementType,
    RaBitQ, BinaryCodeCache,
};
use std::sync::Arc;

// 1. Initialize storage
let mut storage = Storage::readwrite("./vector_db");
storage.ready()?;

// 2. Register embedding (MUST be Cosine for RaBitQ)
let registry = EmbeddingRegistry::new();
let embedding = registry.register(
    EmbeddingBuilder::new("clip-vit-b32", 512, Distance::Cosine),
    storage.transaction_db()?,
)?;

// 3. Create HNSW index with F16 storage (50% smaller)
let nav_cache = Arc::new(NavigationCache::new());
let config = hnsw::Config::high_recall(512);

let index = hnsw::Index::with_storage_type(
    embedding.code(),
    Distance::Cosine,
    VectorElementType::F16,
    config,
    nav_cache,
);

// 4. Initialize RaBitQ encoder and cache
let rabitq = RaBitQ::new(512, 1, 42);  // 1-bit quantization
let binary_cache = BinaryCodeCache::new();

// 5. Insert vectors and build binary cache
for (i, vector) in base_vectors.iter().enumerate() {
    let vec_id = i as u32;

    // Insert into HNSW
    index.insert(&storage, vec_id, vector)?;

    // Encode and cache binary code
    let code = rabitq.encode(vector);
    binary_cache.put(embedding.code(), vec_id, code);
}

// 6. Two-phase search: Hamming pre-filter → exact rerank
let results = index.search_with_rabitq_cached(
    &storage,
    &query,
    &rabitq,
    &binary_cache,
    10,     // k
    100,    // ef_search
    10,     // rerank_factor (rerank top 100 candidates)
)?;

for (dist, id) in results {
    println!("ID: {}, Cosine Distance: {:.4}", id, dist);
}

// 7. Check cache stats
let (count, bytes) = binary_cache.stats();
println!("Binary cache: {} codes, {} bytes", count, bytes);
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
    pub fn model(&self) -> &str;
    pub fn dim(&self) -> u32;
    pub fn distance(&self) -> Distance;
}

// Registry for managing embedding spaces
pub struct EmbeddingRegistry { /* private */ }
impl EmbeddingRegistry {
    pub fn new() -> Self;
    pub fn prewarm(&self, db: &TransactionDB) -> Result<usize>;
    pub fn register(&self, builder: EmbeddingBuilder, db: &TransactionDB) -> Result<Embedding>;
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

### Configuration Types

```rust
// HNSW index configuration: vector::hnsw::Config
pub mod hnsw {
    pub struct Config {
        pub dim: usize,
        pub m: usize,
        pub m_max: usize,
        pub m_max_0: usize,
        pub ef_construction: usize,
        pub batch_threshold: usize,
    }
    impl Config {
        pub fn default() -> Self;            // dim=128, m=16
        pub fn for_dim(dim: usize) -> Self;  // Auto-tuned
        pub fn high_recall(dim: usize) -> Self;  // m=32, ef_construction=200
        pub fn compact(dim: usize) -> Self;      // m=8, ef_construction=50
        pub fn validate(&self) -> Vec<ConfigWarning>;
        pub fn is_valid(&self) -> bool;
    }
    pub enum ConfigWarning { LowM, HighM, LowEfConstruction, ... }
}

// RaBitQ configuration
pub struct RaBitQConfig {
    pub bits_per_dim: u8,
    pub seed: u64,
}
impl RaBitQConfig {
    pub fn code_size(&self, dim: usize) -> usize;
    pub fn validate(&self) -> Vec<RaBitQConfigWarning>;
    pub fn is_valid(&self) -> bool;
}

// Combined configuration
pub struct VectorConfig {
    pub hnsw: hnsw::Config,
    pub rabitq: Option<RaBitQConfig>,
}
impl VectorConfig {
    pub fn dim_128() -> Self;
    pub fn dim_768() -> Self;
    pub fn dim_1024() -> Self;
    pub fn dim_1536() -> Self;
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
    // Construction
    pub fn new(
        embedding: EmbeddingCode,
        distance: Distance,
        config: hnsw::Config,
        nav_cache: Arc<NavigationCache>,
    ) -> Self;

    pub fn with_storage_type(
        embedding: EmbeddingCode,
        distance: Distance,
        storage_type: VectorElementType,
        config: hnsw::Config,
        nav_cache: Arc<NavigationCache>,
    ) -> Self;

    // Accessors
    pub fn embedding(&self) -> EmbeddingCode;
    pub fn storage_type(&self) -> VectorElementType;
    pub fn config(&self) -> &hnsw::Config;
    pub fn distance_metric(&self) -> Distance;
    pub fn nav_cache(&self) -> &Arc<NavigationCache>;

    // Insert
    pub fn insert(&self, storage: &Storage, vec_id: VecId, vector: &[f32]) -> Result<()>;

    // Search - Exact (all distance metrics)
    pub fn search(
        &self,
        storage: &Storage,
        query: &[f32],
        ef: usize,
        k: usize,
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
    pub fn get_neighbors_batch(&self, storage: &Storage, vec_id: VecId, layers: &[u8]) -> Result<Vec<Vec<VecId>>>;
    pub fn get_neighbor_count(&self, storage: &Storage, vec_id: VecId, layer: u8) -> Result<u64>;
    pub fn get_binary_code(&self, storage: &Storage, vec_id: VecId) -> Result<Option<Vec<u8>>>;
    pub fn get_binary_codes_batch(&self, storage: &Storage, vec_ids: &[VecId]) -> Result<Vec<Option<Vec<u8>>>>;
}
```

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

    // Accessors
    pub fn embedding(&self) -> &Embedding;
    pub fn strategy(&self) -> SearchStrategy;
    pub fn k(&self) -> usize;
    pub fn ef(&self) -> usize;
    pub fn rerank_factor(&self) -> usize;
    pub fn parallel_rerank_threshold(&self) -> usize;
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
    pub fn get_neighbors_cached<F, E>(&self, embedding: EmbeddingCode, vec_id: VecId, layer: HnswLayer, fetch_fn: F) -> Result<Vec<VecId>, E>;
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
impl BinaryCodeCache {
    pub fn new() -> Self;
    pub fn put(&self, embedding: EmbeddingCode, vec_id: VecId, code: Vec<u8>);
    pub fn get(&self, embedding: EmbeddingCode, vec_id: VecId) -> Option<Vec<u8>>;
    pub fn get_batch(&self, embedding: EmbeddingCode, vec_ids: &[VecId]) -> Vec<Option<Vec<u8>>>;
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

    // Distance computation
    pub fn hamming_distance(a: &[u8], b: &[u8]) -> u32;
    pub fn hamming_distance_fast(a: &[u8], b: &[u8]) -> u32;
    pub fn batch_hamming_distances(query_code: &[u8], candidate_codes: &[&[u8]]) -> Vec<u32>;
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
pub type Storage = rocksdb::Storage<Subsystem>;
```
