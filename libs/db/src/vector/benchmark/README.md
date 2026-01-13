# Vector Benchmark Infrastructure

## Overview

This module (`libs/db/src/vector/benchmark`) provides benchmarking infrastructure for HNSW vector search with RaBitQ quantization. It supports multiple datasets, parameter sweeps, and standardized metrics collection.

## Current State

### Existing Components

| File | Purpose | Status |
|------|---------|--------|
| `mod.rs` | Module exports | Complete |
| `dataset.rs` | LAION-CLIP + GIST dataset loaders | Complete |
| `sift.rs` | SIFT dataset loader (128D, fvecs) | Complete |
| `runner.rs` | Experiment runner (HNSW + RaBitQ) | Complete |
| `metrics.rs` | Recall, latency, Pareto, rotation stats | Complete |
| `metadata.rs` | Checkpoint/resume for incremental builds | Complete |

### Supported Datasets

| Dataset | Dimensions | Distance | Format | Feature | Status |
|---------|------------|----------|--------|---------|--------|
| LAION-CLIP | 512 | Cosine | NPY (f16) | default | ✅ Supported |
| SIFT-1M | 128 | L2 | fvecs | default | ✅ Supported |
| GIST-960 | 960 | L2 | fvecs | default | ✅ Supported |
| Cohere Wikipedia | 768 | Cosine | Parquet | `parquet` | ✅ Supported |
| GloVe-100 | 100 | Angular | HDF5 | `hdf5` | ✅ Supported |

### Implementation Status

| Phase | Description | Status |
|-------|-------------|--------|
| A.1 | ExperimentConfig RaBitQ fields | ✅ Complete |
| A.2 | RabitqExperimentResult struct | ✅ Complete |
| A.3 | run_rabitq_experiments() function | ✅ Complete |
| A.4 | Pareto frontier computation | ✅ Complete |
| A.5 | Rotated variance metric | ✅ Complete |
| B.1 | GistDataset loader | ✅ Complete |
| B.2 | Shared fvecs/ivecs loaders | ✅ Complete (in sift.rs) |
| C.1 | Parquet loader | ✅ Complete (`--features parquet`) |
| C.2 | CohereWikipediaDataset | ✅ Complete (768D, Cosine) |
| D.1 | HDF5 loader | ✅ Complete (`--features hdf5`*) |
| D.2 | GloveDataset | ✅ Complete (100D, angular) |
| E | CLI tool (bins/bench_vector) | ✅ Complete (all commands: download, index, query, sweep, datasets) |
| F | Example migration | ✅ Complete (deprecation notices added) |

*Note: HDF5 feature requires system library installation (`libhdf5-dev` on Ubuntu).

### Optional Features

```toml
# Enable Parquet support (Cohere Wikipedia dataset)
cargo build --features parquet

# Enable HDF5 support (requires system libhdf5)
cargo build --features hdf5

# Enable all benchmark formats
cargo build --features benchmark-all
```

---

## Infrastructure Improvement Plan

This plan addresses the improvements outlined in [GEMINI-BENCHMARK.md](./GEMINI-BENCHMARK.md).

### Phase A: RaBitQ Runner Integration

**GEMINI-BENCHMARK Reference:** Sections 2.1, 2.2, 2.4

#### A.1: ExperimentConfig Expansion (Section 2.1)

**File:** `runner.rs`

**Current:**
```rust
pub struct ExperimentConfig {
    pub scales: Vec<usize>,
    pub ef_search_values: Vec<usize>,
    pub k_values: Vec<usize>,
    pub num_queries: usize,
    pub m: usize,
    pub ef_construction: usize,
    pub dim: usize,
    pub distance: Distance,
    pub storage_type: VectorElementType,
    pub data_dir: PathBuf,
    pub results_dir: PathBuf,
    pub verbose: bool,
}
```

**Add:**
```rust
pub struct ExperimentConfig {
    // ... existing fields ...

    // RaBitQ parameters (Section 2.1)
    pub use_rabitq: bool,
    pub rabitq_bits: Vec<u8>,         // e.g., [1, 2, 4]
    pub rerank_factors: Vec<usize>,   // e.g., [1, 4, 10, 20]
    pub use_binary_cache: bool,       // Test with/without cache
}
```

**Builder methods:**
```rust
impl ExperimentConfig {
    pub fn with_rabitq(mut self, bits: Vec<u8>, rerank: Vec<usize>) -> Self {
        self.use_rabitq = true;
        self.rabitq_bits = bits;
        self.rerank_factors = rerank;
        self
    }

    pub fn with_binary_cache(mut self, enabled: bool) -> Self {
        self.use_binary_cache = enabled;
        self
    }
}
```

#### A.2: Runner Logic Update (Section 2.2)

**File:** `runner.rs`

**Add `run_rabitq_experiment()` function:**

```rust
/// Run RaBitQ experiment with parameter sweep.
///
/// Iterates over (bits_per_dim, ef_search, rerank_factor) combinations.
pub fn run_rabitq_experiment(
    config: &ExperimentConfig,
    dataset: &impl BenchmarkDataset,
) -> Result<Vec<RabitqExperimentResult>> {
    let mut results = Vec::new();

    for &bits in &config.rabitq_bits {
        // Create encoder
        let encoder = RaBitQ::new(config.dim, bits, 42);

        // Create and populate binary code cache
        let cache = BinaryCodeCache::new();
        for (vec_id, vector) in dataset.vectors().iter().enumerate() {
            let (code, correction) = encoder.encode_with_correction(vector);
            cache.put(embedding_code, vec_id as VecId, code, correction);
        }

        for &ef in &config.ef_search_values {
            for &rerank in &config.rerank_factors {
                // Run queries
                let mut latencies = Vec::new();
                let mut recalls = Vec::new();

                for (qi, query) in dataset.queries().iter().enumerate() {
                    let start = Instant::now();
                    let results = index.search_with_rabitq_cached(
                        &storage, query, &encoder, &cache,
                        config.k, ef, rerank,
                    )?;
                    latencies.push(start.elapsed());

                    let recall = compute_recall(&results, dataset.ground_truth(qi), config.k);
                    recalls.push(recall);
                }

                results.push(RabitqExperimentResult {
                    bits_per_dim: bits,
                    ef_search: ef,
                    rerank_factor: rerank,
                    recall_mean: mean(&recalls),
                    latency_stats: LatencyStats::from(&latencies),
                    qps: 1000.0 / mean_latency_ms,
                });
            }
        }
    }

    Ok(results)
}
```

**Add `RabitqExperimentResult` struct:**

```rust
#[derive(Debug, Clone, Serialize)]
pub struct RabitqExperimentResult {
    pub bits_per_dim: u8,
    pub ef_search: usize,
    pub rerank_factor: usize,
    pub recall_mean: f32,
    pub recall_std: f32,
    pub latency_stats: LatencyStats,
    pub qps: f32,
}
```

#### A.3: Metrics Enhancements (Section 2.4)

**File:** `metrics.rs`

**Add Pareto frontier computation:**

```rust
/// Point on the Recall vs QPS Pareto frontier.
#[derive(Debug, Clone, Serialize)]
pub struct ParetoPoint {
    pub bits_per_dim: u8,
    pub ef_search: usize,
    pub rerank_factor: usize,
    pub recall: f32,
    pub qps: f32,
}

/// Compute Pareto-optimal points from experiment results.
///
/// A point is Pareto-optimal if no other point has both higher recall AND higher QPS.
pub fn compute_pareto_frontier(results: &[RabitqExperimentResult]) -> Vec<ParetoPoint> {
    let mut frontier = Vec::new();

    for r in results {
        let dominated = results.iter().any(|other| {
            other.recall_mean > r.recall_mean && other.qps > r.qps
        });

        if !dominated {
            frontier.push(ParetoPoint {
                bits_per_dim: r.bits_per_dim,
                ef_search: r.ef_search,
                rerank_factor: r.rerank_factor,
                recall: r.recall_mean,
                qps: r.qps,
            });
        }
    }

    // Sort by recall ascending
    frontier.sort_by(|a, b| a.recall.partial_cmp(&b.recall).unwrap());
    frontier
}

/// Export Pareto frontier to CSV.
pub fn export_pareto_csv(points: &[ParetoPoint], path: &Path) -> Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "bits,ef_search,rerank_factor,recall,qps")?;
    for p in points {
        writeln!(file, "{},{},{},{:.4},{:.1}",
            p.bits_per_dim, p.ef_search, p.rerank_factor, p.recall, p.qps)?;
    }
    Ok(())
}
```

**Add distribution metrics (verify √D scaling):**

```rust
/// Compute variance of rotated vector components.
///
/// For correctly scaled rotation matrix (√D scaling), unit vectors
/// should have component variance ≈ 1.0 after rotation.
/// Without scaling, variance would be ≈ 1/D.
pub fn compute_rotated_variance(encoder: &RaBitQ, vectors: &[Vec<f32>]) -> RotationStats {
    let mut all_components = Vec::new();

    for vector in vectors.iter().take(1000) {  // Sample for efficiency
        let rotated = encoder.rotate_query(vector);
        all_components.extend(rotated);
    }

    let n = all_components.len() as f32;
    let mean: f32 = all_components.iter().sum::<f32>() / n;
    let variance: f32 = all_components.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f32>() / n;

    RotationStats {
        component_mean: mean,
        component_variance: variance,
        expected_variance: 1.0,  // With √D scaling
        scaling_valid: (0.8..1.2).contains(&variance),
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RotationStats {
    pub component_mean: f32,
    pub component_variance: f32,
    pub expected_variance: f32,
    pub scaling_valid: bool,
}
```

---

### Phase B: GIST-960 Dataset

**GEMINI-BENCHMARK Reference:** Section 2.3

#### B.1: Add GIST Dataset Support

**File:** `dataset.rs`

**Justification:** High-dimensional (960D) unnormalized L2 dataset. Tests RaBitQ's limits on non-unit vectors.

```rust
/// GIST-960 dataset configuration.
pub const GIST_DIM: usize = 960;
pub const GIST_BASE_VECTORS: usize = 1_000_000;
pub const GIST_QUERIES: usize = 1_000;

/// GIST dataset URLs (texmex corpus).
const GIST_BASE_URL: &str = "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz";

/// GIST dataset with high-dimensional descriptors.
#[derive(Debug, Clone)]
pub struct GistDataset {
    pub base_vectors: Vec<Vec<f32>>,
    pub queries: Vec<Vec<f32>>,
    pub ground_truth: Vec<Vec<usize>>,
    pub dim: usize,
}

impl GistDataset {
    /// Load GIST dataset from directory.
    ///
    /// Expected files:
    /// - gist_base.fvecs (1M × 960D)
    /// - gist_query.fvecs (1K × 960D)
    /// - gist_groundtruth.ivecs (1K × 100)
    pub fn load(data_dir: &Path, max_vectors: usize) -> Result<Self> {
        let base_path = data_dir.join("gist_base.fvecs");
        let query_path = data_dir.join("gist_query.fvecs");
        let gt_path = data_dir.join("gist_groundtruth.ivecs");

        let base_vectors = load_fvecs(&base_path, max_vectors)?;
        let queries = load_fvecs(&query_path, GIST_QUERIES)?;
        let ground_truth = load_ivecs(&gt_path, GIST_QUERIES)?;

        Ok(Self {
            base_vectors,
            queries,
            ground_truth,
            dim: GIST_DIM,
        })
    }

    /// Download GIST dataset if not present.
    pub fn download(data_dir: &Path) -> Result<()> {
        // Download and extract gist.tar.gz
        // ...
    }
}

impl BenchmarkDataset for GistDataset {
    fn vectors(&self) -> &[Vec<f32>] { &self.base_vectors }
    fn queries(&self) -> &[Vec<f32>] { &self.queries }
    fn ground_truth(&self, query_idx: usize) -> &[usize] { &self.ground_truth[query_idx] }
    fn dim(&self) -> usize { self.dim }
    fn distance(&self) -> Distance { Distance::L2 }
}
```

#### B.2: Shared fvecs/ivecs Loaders

**File:** `dataset.rs`

```rust
/// Load vectors from fvecs format.
///
/// Format: [dim: i32][v0: f32]...[v_{dim-1}: f32] repeated
pub fn load_fvecs(path: &Path, max_vectors: usize) -> Result<Vec<Vec<f32>>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut vectors = Vec::new();

    while vectors.len() < max_vectors {
        let dim = match reader.read_i32::<LittleEndian>() {
            Ok(d) => d as usize,
            Err(_) => break,  // EOF
        };

        let mut vec = vec![0.0f32; dim];
        reader.read_f32_into::<LittleEndian>(&mut vec)?;
        vectors.push(vec);
    }

    Ok(vectors)
}

/// Load indices from ivecs format.
///
/// Format: [dim: i32][i0: i32]...[i_{dim-1}: i32] repeated
pub fn load_ivecs(path: &Path, max_vectors: usize) -> Result<Vec<Vec<usize>>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut indices = Vec::new();

    while indices.len() < max_vectors {
        let dim = match reader.read_i32::<LittleEndian>() {
            Ok(d) => d as usize,
            Err(_) => break,
        };

        let mut vec = vec![0i32; dim];
        reader.read_i32_into::<LittleEndian>(&mut vec)?;
        indices.push(vec.into_iter().map(|i| i as usize).collect());
    }

    Ok(indices)
}
```

---

### Phase C: Parquet Loader + Cohere Wikipedia

**GEMINI-BENCHMARK Reference:** Section 2.3 (Cohere/OpenAI datasets)

#### C.1: Add Parquet Dependency

**File:** `libs/db/Cargo.toml`

```toml
[features]
default = []
parquet = ["dep:parquet", "dep:arrow"]

[dependencies]
parquet = { version = "53", optional = true }
arrow = { version = "53", optional = true }
```

#### C.2: Parquet Loader

**File:** `dataset.rs`

```rust
#[cfg(feature = "parquet")]
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

/// Load embeddings from Parquet file.
///
/// Expects a column containing fixed-size float arrays.
#[cfg(feature = "parquet")]
pub fn load_parquet_embeddings(
    path: &Path,
    embedding_column: &str,
    max_vectors: usize,
) -> Result<Vec<Vec<f32>>> {
    use arrow::array::{Array, FixedSizeListArray, Float32Array};

    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    let mut vectors = Vec::new();

    for batch in reader {
        let batch = batch?;
        let column = batch
            .column_by_name(embedding_column)
            .ok_or_else(|| anyhow::anyhow!("Column '{}' not found", embedding_column))?;

        let list_array = column
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| anyhow::anyhow!("Expected FixedSizeListArray"))?;

        for i in 0..list_array.len() {
            if vectors.len() >= max_vectors {
                break;
            }

            let values = list_array.value(i);
            let float_array = values
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| anyhow::anyhow!("Expected Float32Array"))?;

            let vec: Vec<f32> = float_array.values().to_vec();
            vectors.push(vec);
        }

        if vectors.len() >= max_vectors {
            break;
        }
    }

    Ok(vectors)
}
```

#### C.3: Cohere Wikipedia Dataset

**File:** `dataset.rs`

```rust
/// Cohere Wikipedia dataset configuration.
pub const COHERE_WIKI_DIM: usize = 768;
pub const COHERE_WIKI_VECTORS: usize = 485_000;

/// HuggingFace dataset: Cohere/wikipedia-22-12-simple-embeddings
const COHERE_WIKI_URL: &str =
    "https://huggingface.co/datasets/Cohere/wikipedia-22-12-simple-embeddings/resolve/main/data/train-00000-of-00001.parquet";

/// Cohere Wikipedia embeddings (768D, embed-english-v3.0).
///
/// Real production embeddings with natural clustering from semantic similarity.
#[cfg(feature = "parquet")]
#[derive(Debug, Clone)]
pub struct CohereWikipediaDataset {
    pub embeddings: Vec<Vec<f32>>,
    pub queries: Vec<Vec<f32>>,
    pub ground_truth: Vec<Vec<usize>>,
    pub dim: usize,
}

#[cfg(feature = "parquet")]
impl CohereWikipediaDataset {
    /// Load Cohere Wikipedia dataset.
    ///
    /// Uses last N vectors as queries, computes ground truth via brute force.
    pub fn load(data_dir: &Path, num_db: usize, num_queries: usize) -> Result<Self> {
        let parquet_path = data_dir.join("cohere_wikipedia.parquet");

        let total = num_db + num_queries;
        let all_embeddings = load_parquet_embeddings(&parquet_path, "emb", total)?;

        let (db_vectors, query_vectors) = all_embeddings.split_at(num_db);

        // Compute ground truth (brute force cosine)
        let ground_truth = compute_ground_truth_cosine(
            db_vectors,
            query_vectors,
            100,  // top-100 ground truth
        );

        Ok(Self {
            embeddings: db_vectors.to_vec(),
            queries: query_vectors.to_vec(),
            ground_truth,
            dim: COHERE_WIKI_DIM,
        })
    }

    /// Download dataset from HuggingFace.
    pub fn download(data_dir: &Path) -> Result<()> {
        let dest = data_dir.join("cohere_wikipedia.parquet");
        if !dest.exists() {
            println!("Downloading Cohere Wikipedia embeddings (~1.5GB)...");
            download_file(COHERE_WIKI_URL, &dest)?;
        }
        Ok(())
    }
}

#[cfg(feature = "parquet")]
impl BenchmarkDataset for CohereWikipediaDataset {
    fn vectors(&self) -> &[Vec<f32>] { &self.embeddings }
    fn queries(&self) -> &[Vec<f32>] { &self.queries }
    fn ground_truth(&self, query_idx: usize) -> &[usize] { &self.ground_truth[query_idx] }
    fn dim(&self) -> usize { self.dim }
    fn distance(&self) -> Distance { Distance::Cosine }
}
```

---

### Phase D: HDF5 Loader + ann-benchmarks Datasets

**GEMINI-BENCHMARK Reference:** Section 2.3, Step 2

#### D.1: Add HDF5 Dependency

**File:** `libs/db/Cargo.toml`

```toml
[features]
default = []
parquet = ["dep:parquet", "dep:arrow"]
hdf5 = ["dep:hdf5"]

[dependencies]
hdf5 = { version = "0.8", optional = true }
```

#### D.2: HDF5 Loader (ann-benchmarks format)

**File:** `dataset.rs`

```rust
/// ann-benchmarks HDF5 dataset format.
///
/// Standard format used by https://ann-benchmarks.com/
/// Contains: train, test, neighbors, distances datasets.
#[cfg(feature = "hdf5")]
#[derive(Debug, Clone)]
pub struct AnnBenchmarkDataset {
    pub train: Vec<Vec<f32>>,      // Database vectors
    pub test: Vec<Vec<f32>>,       // Query vectors
    pub neighbors: Vec<Vec<usize>>, // Ground truth indices
    pub distances: Vec<Vec<f32>>,  // Ground truth distances
    pub dim: usize,
    pub distance: Distance,
}

#[cfg(feature = "hdf5")]
impl AnnBenchmarkDataset {
    /// Load ann-benchmarks HDF5 file.
    pub fn load(path: &Path, distance: Distance) -> Result<Self> {
        let file = hdf5::File::open(path)?;

        let train = file.dataset("train")?.read_2d::<f32>()?;
        let test = file.dataset("test")?.read_2d::<f32>()?;
        let neighbors = file.dataset("neighbors")?.read_2d::<i32>()?;
        let distances = file.dataset("distances")?.read_2d::<f32>()?;

        let dim = train.shape()[1];

        Ok(Self {
            train: train.rows().into_iter().map(|r| r.to_vec()).collect(),
            test: test.rows().into_iter().map(|r| r.to_vec()).collect(),
            neighbors: neighbors.rows().into_iter()
                .map(|r| r.iter().map(|&i| i as usize).collect())
                .collect(),
            distances: distances.rows().into_iter().map(|r| r.to_vec()).collect(),
            dim,
            distance,
        })
    }
}

#[cfg(feature = "hdf5")]
impl BenchmarkDataset for AnnBenchmarkDataset {
    fn vectors(&self) -> &[Vec<f32>] { &self.train }
    fn queries(&self) -> &[Vec<f32>] { &self.test }
    fn ground_truth(&self, query_idx: usize) -> &[usize] { &self.neighbors[query_idx] }
    fn dim(&self) -> usize { self.dim }
    fn distance(&self) -> Distance { self.distance }
}
```

#### D.3: OpenAI DBpedia Dataset (1536D)

```rust
/// OpenAI DBpedia dataset (1536D, ada-002 embeddings).
pub const OPENAI_DBPEDIA_DIM: usize = 1536;

/// Download URL from ann-benchmarks
const OPENAI_DBPEDIA_URL: &str =
    "https://ann-benchmarks.com/dbpedia-entities-openai-1M-1536-angular.hdf5";

#[cfg(feature = "hdf5")]
pub fn load_openai_dbpedia(data_dir: &Path, max_vectors: usize) -> Result<AnnBenchmarkDataset> {
    let path = data_dir.join("dbpedia-openai-1536.hdf5");
    if !path.exists() {
        download_file(OPENAI_DBPEDIA_URL, &path)?;
    }
    AnnBenchmarkDataset::load(&path, Distance::Cosine)
}
```

#### D.4: GloVe-100 Dataset (100D)

```rust
/// GloVe-100 dataset (100D word embeddings).
pub const GLOVE_100_DIM: usize = 100;

const GLOVE_100_URL: &str =
    "https://ann-benchmarks.com/glove-100-angular.hdf5";

#[cfg(feature = "hdf5")]
pub fn load_glove_100(data_dir: &Path) -> Result<AnnBenchmarkDataset> {
    let path = data_dir.join("glove-100-angular.hdf5");
    if !path.exists() {
        download_file(GLOVE_100_URL, &path)?;
    }
    AnnBenchmarkDataset::load(&path, Distance::Cosine)
}
```

---

### Phase E: CLI Tool Skeleton

**GEMINI-BENCHMARK Reference:** Section 3, Step 4

#### E.1: Create CLI Binary

**File:** `bins/bench_vector/main.rs`

```rust
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "bench_vector")]
#[command(about = "Vector search benchmark tool")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build index incrementally
    Index(IndexArgs),
    /// Run queries on existing index
    Query(QueryArgs),
    /// Parameter sweep (grid search)
    Sweep(SweepArgs),
    /// Benchmark parallel re-ranking
    BenchRerank(BenchRerankArgs),
}

#[derive(Parser)]
struct IndexArgs {
    /// Dataset: laion, sift, gist, cohere, openai, glove, random
    #[arg(long)]
    dataset: String,

    /// Number of vectors to index
    #[arg(long)]
    num_vectors: usize,

    /// Database path
    #[arg(long)]
    db_path: PathBuf,

    /// Enable incremental build
    #[arg(long)]
    incremental: bool,

    /// Checkpoint interval
    #[arg(long, default_value = "10000")]
    checkpoint_interval: usize,
}

#[derive(Parser)]
struct SweepArgs {
    /// Dataset
    #[arg(long)]
    dataset: String,

    /// RaBitQ bits per dimension (comma-separated)
    #[arg(long, default_value = "1,2,4")]
    bits: String,

    /// Rerank factors (comma-separated)
    #[arg(long, default_value = "1,4,10,20")]
    rerank: String,

    /// ef_search values (comma-separated)
    #[arg(long, default_value = "50,100,200")]
    ef: String,

    /// Output file (JSON or CSV)
    #[arg(long)]
    output: Option<PathBuf>,
}
```

#### E.2: CLI Usage Examples

```bash
# Index GIST dataset with incremental checkpointing
bench_vector index \
    --dataset gist \
    --num-vectors 100000 \
    --db-path /data/gist_bench \
    --incremental \
    --checkpoint-interval 10000

# Run RaBitQ parameter sweep on Cohere embeddings
bench_vector sweep \
    --dataset cohere \
    --bits 1,2,4 \
    --rerank 1,4,10,20,50 \
    --ef 50,100,200 \
    --output results/cohere_sweep.json

# Query-only benchmark
bench_vector query \
    --db-path /data/gist_bench \
    --num-queries 1000 \
    --ef 100 \
    --k 10

# Parallel re-ranking microbenchmark
bench_vector bench-rerank \
    --dataset laion \
    --rerank-sizes 50,100,200,500,1000
```

---

### Phase F: Example Migration

**Goal:** Consolidate `examples/vector2` and `examples/laion_benchmark` into `bins/bench_vector`.

#### F.1: Migration Mapping

| Source | Feature | Destination |
|--------|---------|-------------|
| `examples/vector2/main.rs` | Incremental build | `bench_vector index --incremental` |
| `examples/vector2/main.rs` | Checkpoint/resume | `libs/db/.../benchmark/metadata.rs` (done) |
| `examples/vector2/main.rs` | ADC brute-force | `bench_vector sweep --mode adc-bruteforce` |
| `examples/vector2/main.rs` | Query-only mode | `bench_vector query` |
| `examples/vector2/benchmark.rs` | Dataset loaders | `libs/db/.../benchmark/dataset.rs` |
| `examples/vector2/benchmark.rs` | Ground truth | `libs/db/.../benchmark/ground_truth.rs` |
| `examples/laion_benchmark/main.rs` | LAION runner | `bench_vector sweep --dataset laion` |
| `examples/laion_benchmark/rabitq_bench.rs` | Rerank benchmark | `bench_vector bench-rerank` |

#### F.2: Files to Create/Move

```
libs/db/src/vector/benchmark/
├── mod.rs              # Update exports
├── dataset.rs          # Merge all dataset loaders
├── ground_truth.rs     # NEW: Move from examples/vector2
├── metrics.rs          # Add Pareto frontier
├── runner.rs           # Add RaBitQ support
├── metadata.rs         # Already exists
└── sift.rs             # Already exists

bins/bench_vector/
├── Cargo.toml          # NEW
├── main.rs             # NEW: CLI entry point
└── commands/
    ├── mod.rs          # NEW
    ├── index.rs        # NEW: Incremental indexing
    ├── query.rs        # NEW: Query benchmark
    ├── sweep.rs        # NEW: Parameter sweep
    └── rerank.rs       # NEW: Rerank microbenchmark
```

#### F.3: Deprecation Timeline

1. **After Phase E:** Add deprecation warnings to examples
2. **After Phase F complete:** Examples print "DEPRECATED: use bench_vector"
3. **Next release:** Delete `examples/vector2/` and `examples/laion_benchmark/`

---

## Proposed Benchmark Suite

**GEMINI-BENCHMARK Reference:** Section 4

After implementation, run this validation suite:

### 1. RaBitQ Bits vs Recall

**Dataset:** LAION-CLIP (512D)
**Sweep:** bits {1, 2, 4}, rerank {1..50}
**Goal:** Determine optimal bits/rerank tradeoff

```bash
bench_vector sweep --dataset laion --bits 1,2,4 --rerank 1,2,4,10,20,50 --ef 100
```

### 2. High-Dimensional Validation

**Dataset:** GIST-960 (960D, L2)
**Sweep:** bits {4}, rerank {10..100}
**Goal:** Confirm recall >90% achievable on high-dim L2

```bash
bench_vector sweep --dataset gist --bits 4 --rerank 10,20,50,100 --ef 100,200
```

### 3. Production Embeddings

**Dataset:** Cohere Wikipedia (768D, Cosine)
**Sweep:** bits {2, 4}, rerank {4..20}
**Goal:** Validate on real clustered embeddings

```bash
bench_vector sweep --dataset cohere --bits 2,4 --rerank 4,10,20 --ef 100
```

### 4. Maximum Dimension

**Dataset:** OpenAI DBpedia (1536D, Cosine)
**Sweep:** bits {4}, rerank {10..50}
**Goal:** Stress test highest dimension

```bash
bench_vector sweep --dataset openai --bits 4 --rerank 10,20,50 --ef 100,200
```

### 5. Gray Code Verification (Issue #43)

**Dataset:** Random-1024D (Unit)
**Test:** Compare bits=1 vs bits=2 vs bits=4
**Success:** Recall MUST increase with bits (was broken before Gray code fix)

```bash
bench_vector sweep --dataset random --dim 1024 --bits 1,2,4 --rerank 10 --ef 100
```

### 6. ADC vs Hamming Comparison (Task 4.24)

**Dataset:** Any
**Test:** Compare `--mode hamming` vs `--mode adc`
**Success:** ADC 4-bit ~92% recall vs Hamming 4-bit ~24%

```bash
# This validates Task 4.24 implementation
bench_vector sweep --dataset laion --bits 4 --mode adc --rerank 10 --ef 100
```

---

## Dataset Summary

| Dataset | Dim | Distance | Format | Feature Flag | Phase |
|---------|-----|----------|--------|--------------|-------|
| LAION-CLIP | 512 | Cosine | NPY | - | Exists |
| SIFT-1M | 128 | L2 | fvecs | - | Exists |
| **GIST-960** | 960 | L2 | fvecs | - | B |
| **Cohere Wikipedia** | 768 | Cosine | Parquet | `parquet` | C |
| **OpenAI DBpedia** | 1536 | Cosine | HDF5 | `hdf5` | D |
| **GloVe-100** | 100 | Angular | HDF5 | `hdf5` | D |
| **Random** | Any | Any | Generate | - | E |

---

## Implementation Priority

| Phase | Description | Effort | Dependencies | Priority |
|-------|-------------|--------|--------------|----------|
| **A** | RaBitQ runner + metrics | Medium | None | HIGH |
| **B** | GIST-960 dataset | Small | None | HIGH |
| **C** | Parquet + Cohere | Medium | Phase A | HIGH |
| **E** | CLI skeleton | Medium | Phase A | MEDIUM |
| **F** | Example migration | Large | Phases A-E | MEDIUM |
| **D** | HDF5 + ann-benchmarks | Medium | Phase A | LOW |

**Recommended order:** A → B → C → E → F → D

---

## References

- [GEMINI-BENCHMARK.md](./GEMINI-BENCHMARK.md) - Original improvement proposal
- [RABITQ.md](../RABITQ.md) - RaBitQ implementation details and ADC analysis
- [ROADMAP.md](../ROADMAP.md) - Overall project roadmap
- [ann-benchmarks](https://ann-benchmarks.com/) - Standard benchmark datasets
- [Cohere Wikipedia](https://huggingface.co/datasets/Cohere/wikipedia-22-12-simple-embeddings) - HuggingFace dataset
