# Benchmark Datasets

The vector benchmark module supports 6 datasets across 4 file formats. This document
captures the current API surface, format readers, overlap with the
[VectorDBBench](https://github.com/zilliztech/VectorDBBench) suite, and a proposed
`Dataset` trait unification.

---

## 1. Supported Datasets

| Dataset | Dim | Distance | Format | Source | Feature | Max Size | Pre-computed GT |
|---------|-----|----------|--------|--------|---------|----------|-----------------|
| LAION-CLIP | 512 | Cosine | NPY (f16) | deploy.laion.ai | `benchmark` | 400M | No |
| SIFT-1M | 128 | L2 | fvecs/ivecs | HuggingFace `qbo-odp/sift1m` | `benchmark` | 1M + 10K queries | Yes |
| GIST-960 | 960 | L2 | fvecs/ivecs | corpus-texmex.irisa.fr | `benchmark` | 1M + 1K queries | Yes |
| Cohere Wiki | 768 | Cosine | Parquet | HuggingFace `Cohere/wikipedia-22-12` | `benchmark` | 485K | Yes |
| GloVe-100 | 100 | Cosine | HDF5 | ann-benchmarks.com | `benchmark` | 1.18M + 10K queries | Yes |
| Random | configurable | Cosine/L2 | in-memory | ChaCha8Rng (seeded) | `benchmark` | unlimited (streaming) | No |

### Dimension constants

```rust
pub const LAION_EMBEDDING_DIM: usize = 512;
pub const SIFT_EMBEDDING_DIM: usize = 128;
pub const GIST_EMBEDDING_DIM: usize = 960;
pub const GIST_BASE_VECTORS: usize = 1_000_000;
pub const GIST_QUERIES: usize = 1_000;
pub const COHERE_WIKI_DIM: usize = 768;
pub const COHERE_WIKI_VECTORS: usize = 485_000;
pub const GLOVE_DIM: usize = 100;
pub const GLOVE_VECTORS: usize = 1_183_514;
pub const GLOVE_QUERIES: usize = 10_000;
```

---

## 2. Format Readers

### `NpyLoader`

```rust
pub struct NpyLoader { verbose: bool }

impl NpyLoader {
    pub fn new() -> Self;
    pub fn with_verbose(self, verbose: bool) -> Self;
    pub fn load(&self, path: &Path, max_count: usize, expected_dim: Option<usize>)
        -> Result<Vec<Vec<f32>>>;
}
```

Parses the NPY header for shape and dtype. Supports `<f2` (f16 -> f32 via the `half`
crate) and `<f4` (native f32). Used by **LAION**.

### `read_fvecs` / `read_ivecs`

```rust
pub fn read_fvecs(path: &Path) -> Result<Vec<Vec<f32>>>;
pub fn read_fvecs_limited(path: &Path, limit: usize) -> Result<Vec<Vec<f32>>>;
pub fn read_ivecs(path: &Path) -> Result<Vec<Vec<i32>>>;
pub fn read_ivecs_limited(path: &Path, limit: usize) -> Result<Vec<Vec<i32>>>;
```

Binary format: 4-byte little-endian i32 (dimension), then `dim * 4` bytes of
little-endian f32 (or i32 for ivecs), repeated per vector. The `_limited` variants
stop after reading `limit` vectors. Used by **SIFT** and **GIST** (base, query, and
ground-truth files).

### `load_parquet_embeddings`

```rust
pub fn load_parquet_embeddings(
    path: &Path, embedding_column: &str, max_vectors: usize,
) -> Result<Vec<Vec<f32>>>;
```

Reads a `FixedSizeListArray<Float32>` column from an Apache Parquet file. Used by
**Cohere Wikipedia**.

### `load_hdf5_embeddings` / `load_hdf5_ground_truth`

```rust
pub fn load_hdf5_embeddings(path: &Path, dataset_name: &str, max_vectors: usize)
    -> Result<Vec<Vec<f32>>>;
pub fn load_hdf5_ground_truth(path: &Path, max_queries: usize)
    -> Result<Vec<Vec<usize>>>;
```

Reads `ndarray::Array2<f32>` from HDF5 datasets (`"train"`, `"test"`, `"neighbors"`).
Used by **GloVe**.

### `StreamingVectorGenerator`

```rust
pub struct StreamingVectorGenerator { /* ChaCha8Rng, dim, total, normalize */ }

impl StreamingVectorGenerator {
    pub fn new(dim: usize, total: usize, seed: u64) -> Self;
    pub fn new_with_distance(dim: usize, total: usize, seed: u64, distance: Distance) -> Self;
    pub fn next_batch(&mut self, batch_size: usize) -> Option<Vec<Vec<f32>>>;
    pub fn generate_query(&mut self) -> Vec<f32>;
    pub fn generated(&self) -> usize;
    pub fn total(&self) -> usize;
}
```

Generates unit-normalized random vectors from a seeded ChaCha8Rng. When `distance` is
Cosine, vectors are L2-normalized. Used by **Random**.

---

## 3. Per-Dataset API Pattern

Every dataset follows the same convention (concrete structs with a shared `Dataset` trait):

1. **Dataset struct** (`LaionDataset`, `SiftDataset`, `GistDataset`,
   `CohereWikipediaDataset`, `GloveDataset`, `RandomDataset`):
   - `download(data_dir: &Path) -> Result<()>` — fetch from upstream
   - `load(data_dir: &Path, ...) -> Result<Self>` — parse local files
   - `subset(num_vectors, num_queries) -> *Subset` — carve out a working set
   - `len()`, `is_empty()`

2. **Subset struct** (`LaionSubset`, `SiftSubset`, `GistSubset`,
   `CohereWikipediaSubset`, `GloveSubset`):
   - `db_vectors: Vec<Vec<f32>>`, `queries: Vec<Vec<f32>>`, `dim: usize`
   - `compute_ground_truth_topk(k: usize) -> Vec<Vec<usize>>`
   - `compute_ground_truth_topk_with_distance(k, distance) -> Vec<Vec<usize>>`
   - `num_vectors()`, `num_queries()`

3. **`DatasetConfig`** builder (used by `load_with_config`):
   ```rust
   pub struct DatasetConfig { pub max_vectors: usize, pub dim: usize, pub verbose: bool }

   impl DatasetConfig {
       pub fn with_dim(self, dim: usize) -> Self;
       pub fn with_max_vectors(self, max: usize) -> Self;
       pub fn with_verbose(self, verbose: bool) -> Self;
   }
   ```

4. **Ground truth**: pre-computed when the dataset ships neighbors (SIFT, GIST, Cohere,
   GloVe). Otherwise computed at runtime via:
   ```rust
   pub fn compute_ground_truth_parallel(
       db_vectors: &[Vec<f32>], queries: &[Vec<f32>], k: usize, distance: Distance,
   ) -> Vec<Vec<usize>>;
   ```
   This is a rayon-parallelized brute-force search used by LAION and Random.

---

## 4. VectorDBBench Overlap Analysis

### VectorDBBench dataset catalog

| Dataset | Dim | Metric | Sizes | Has GT | Has Scalar Labels | Source |
|---------|-----|--------|-------|--------|-------------------|--------|
| LAION | 768 | L2 | 100M | Yes | No | LAION-2B CLIP |
| SIFT | 128 | L2 | 500K, 5M | No | No | corpus-texmex |
| GIST | 960 | L2 | 100K, 1M | No | No | corpus-texmex |
| Cohere | 768 | Cosine | 100K, 1M, 10M | Yes | Yes | Wikipedia-22-12/en |
| Bioasq | 1024 | Cosine | 1M, 10M | Yes | Yes | BioASQ biomedical QA |
| GloVe | 200 | Cosine | 1M | No | No | GloVe word embeddings |
| OpenAI | 1536 | Cosine | 50K, 500K, 5M | Yes | Yes | C4 corpus |

### VectorDBBench data hosting

- **Storage**: Parquet-only on S3 (`assets.zilliz.com/benchmark/`) with Aliyun OSS mirror
- **Access**: anonymous — `s3fs.S3FileSystem(anon=True)`
- **URL pattern**: `assets.zilliz.com/benchmark/{dataset}_{label}_{size}/`
- **File naming**: `train.parquet` (single) or `train-{NN}-of-{total}.parquet` (sharded, 10 files)
- **Shuffled variants**: `shuffle_train-{NN}-of-{total}.parquet`
- **Parquet schema**: `id` (int) + `emb` (array\<f32\>) + optional `neighbors_id` (list\<int\>)
- **Ground truth**: `neighbors.parquet` with filter variants (`neighbors_head_1p.parquet`, etc.)
- **Scalar labels**: `scalar_labels.parquet` (for filtered search evaluation)

### Side-by-side comparison

| Logical Dataset | motlie | VectorDBBench | Same data? |
|---|---|---|---|
| SIFT | 128D, fvecs, 1M | 128D, Parquet, 500K/5M | Same vectors, different format + sizes |
| GIST | 960D, fvecs, 1M | 960D, Parquet, 100K/1M | Same vectors, different format |
| Cohere | 768D, Parquet, 485K | 768D, Parquet, 100K/1M/10M | Same model, different subsets |
| LAION | 512D NPY (ViT-B-32) | 768D Parquet (LAION-2B) | **Different model & dim** |
| GloVe | 100D HDF5 | 200D Parquet | **Different dimensions** |
| Bioasq | — | 1024D, 1M/10M | motlie doesn't have this |
| OpenAI | — | 1536D, 50K/500K/5M | motlie doesn't have this |
| Random | in-memory, configurable | — | VectorDBBench doesn't have this |

**Key observation**: For SIFT, GIST, and Cohere the underlying vectors are the same —
only the file format, hosting, and subset sizes differ. For LAION and GloVe the data is
genuinely different (different embedding models / dimensions).

---

## 5. Integration — `Dataset` Trait + Pluggable Sources

### Problem

- Each dataset was a bespoke struct with convention-based method names but no shared contract
- `bench_vector` has per-dataset branching everywhere
- Adding a new dataset requires touching many files
- No way to test the HNSW/RaBitQ pipeline generically over "any dataset"
- No way to load the same logical dataset (e.g. SIFT-128D) from different sources

### Three-layer design

```
Layer 1: DataSource          (where bytes come from)
Layer 2: Dataset trait        (what the vectors represent)
Layer 3: Runner/Metrics       (what we do with them) — unchanged
```

### Layer 1: `VdbBenchSource`

```rust
/// Remote Parquet source compatible with VectorDBBench's S3 layout.
/// URL pattern: assets.zilliz.com/benchmark/{name}_{label}_{size}/
pub struct VdbBenchSource {
    name: &'static str,       // "cohere", "sift", "openai", ...
    label: &'static str,      // "small", "medium", "large"
    file_count: usize,        // 1 or 10 (sharded)
    has_ground_truth: bool,
    has_scalar_labels: bool,
}
```

Keep existing native loaders (fvecs is ~10x faster to parse than Parquet for SIFT/GIST).
VectorDBBench Parquet is an **alternative source** for overlapping datasets and the
**only source** for Bioasq and OpenAI.

### Layer 2: `Dataset` trait

```rust
pub trait Dataset {
    /// Human-readable name (e.g. "SIFT-1M", "Cohere-Wikipedia-768D")
    fn name(&self) -> &str;

    /// Vector dimensionality
    fn dim(&self) -> usize;

    /// Distance metric for this dataset
    fn distance(&self) -> Distance;

    /// Database vectors (the corpus to index)
    fn vectors(&self) -> &[Vec<f32>];

    /// Query vectors
    fn queries(&self) -> &[Vec<f32>];

    /// Pre-computed ground truth (if available).
    /// Returns None if GT must be computed at runtime.
    fn ground_truth(&self, k: usize) -> Option<Vec<Vec<usize>>>;
}
```

**Why a trait, not an enum:**
- Open for extension — users can implement `Dataset` for their own data
- The runner/metrics layer becomes generic:
  `fn run_experiment(dataset: &dyn Dataset, config: &ExperimentConfig)`
- `bench_vector` CLI dispatches once at the top and then works with
  `Box<dyn Dataset>` everywhere

Existing structs adapt in-place. `SiftSubset`, `LaionSubset`, `GistSubset`, etc. all
get `impl Dataset for ...` with zero API breakage — the old concrete methods remain,
the trait is additive.

### `DatasetId` enum for CLI dispatch

```rust
pub enum DatasetId {
    Sift,
    Gist,
    LaionClip,
    CohereWiki,
    Glove,
    Random,
    // New from VectorDBBench
    Bioasq,
    OpenAI,
    // VectorDBBench variants of existing datasets (larger sizes)
    VdbSift5M,
    VdbCohere10M,
    VdbLaion768,
}
```

Each variant knows its default dim, distance, and how to construct a loader. The CLI
uses this for `--dataset` parsing.

### Files changed

| File | Change | Status |
|---|---|---|
| `benchmark/mod.rs` | `pub trait Dataset` definition + `use Distance` | **Done** |
| `benchmark/dataset.rs` | `impl Dataset for LaionSubset/GistSubset/CohereSubset/GloveSubset/RandomDataset` | **Done** |
| `benchmark/sift.rs` | `impl Dataset for SiftSubset` | **Done** |
| `benchmark/runner.rs` | `run_single_experiment` et al. accept `&dyn Dataset` instead of concrete types | **Done** |
| `benchmark/metrics.rs` | No change (already generic over `&[Vec<f32>]`) | N/A |
| `benchmark/source.rs` | **New.** `VdbBenchSource` S3 Parquet downloader (anonymous `reqwest` GET) | Pending |
| `bins/bench_vector/commands.rs` | Replace per-dataset match arms with `DatasetId::load() -> Box<dyn Dataset>` | Pending |

### What NOT to do

1. **Don't rehost VectorDBBench data.** Download from `assets.zilliz.com` at runtime.
   Parquet reader already exists behind the `parquet` feature flag.
2. **Don't drop native format loaders.** fvecs is ~10x faster to parse than Parquet for
   SIFT/GIST. Keep native loaders as the preferred source when available; VectorDBBench
   Parquet as fallback or for sizes not available natively (SIFT-5M, Cohere-10M).
3. **Don't add OpenAI/Bioasq behind a default feature.** Gate behind
   `feature = "benchmark"` like today. The S3 downloads are large (multi-GB).
4. **Don't duplicate ground truth.** For datasets where VectorDBBench provides
   pre-computed `neighbors.parquet`, load it rather than brute-force recomputing. Keep
   `compute_ground_truth_parallel` as fallback for native-format datasets and Random.

### Migration path

| Step | Description | Status |
|------|-------------|--------|
| 1 | Add the `Dataset` trait and `impl` it on existing subset types (non-breaking, additive) | **Done** |
| 2 | Add `VdbBenchSource` behind `parquet` feature (S3 Parquet downloader) | Pending |
| 3 | Refactor `runner.rs` to be generic over `&dyn Dataset` (internal, no public API break) | **Done** |
| 4 | Add Bioasq and OpenAI dataset types that load exclusively from `VdbBenchSource` | Pending |
| 5 | Update `bench_vector` CLI to use `DatasetId` catalog | Pending |

---

## 6. CLI Reference (`bench_vector`)

```
download  --dataset {laion,sift,gist,cohere,glove} --data-dir ./data
index     --dataset {laion,sift,gist,random,cohere,glove} --num-vectors N --db-path ...
query     --db-path ... --dataset ... --k 10 --ef-search 100
sweep     --dataset ... --bits 1,2,4 --ef 50,100,200 --rerank 1,4,10
datasets  (list available datasets with dimensions and distances)
```

`--dataset` auto-maps to default dim + distance (e.g. `sift` -> 128D L2,
`cohere` -> 768D Cosine).

---

## Source files

| Module | Path |
|---|---|
| Public re-exports | `libs/db/src/vector/benchmark/mod.rs` |
| LAION, GIST, Cohere, GloVe, Random loaders | `libs/db/src/vector/benchmark/dataset.rs` |
| SIFT fvecs/ivecs loader | `libs/db/src/vector/benchmark/sift.rs` |
| ExperimentConfig, runner | `libs/db/src/vector/benchmark/runner.rs` |
| StreamingVectorGenerator, ScaleBenchmark | `libs/db/src/vector/benchmark/scale.rs` |
| Metrics (recall, Pareto, latency) | `libs/db/src/vector/benchmark/metrics.rs` |
| CLI dispatch | `bins/bench_vector/src/commands.rs` |

---

## Changelog

### 2026-02-12 — Dataset trait unification (steps 1 & 3)

**Author:** Claude (Opus 4.6)
**Branch:** `feature/vector-dataset`
**Status:** Complete

Implemented the `Dataset` trait and refactored `runner.rs` to accept `&dyn Dataset`
instead of hardcoded `&LaionSubset`. This is an additive, non-breaking change — all
existing concrete methods on subset structs remain.

**What was done:**

- Defined `pub trait Dataset` in `benchmark/mod.rs` with 6 methods: `name()`, `dim()`,
  `distance()`, `vectors()`, `queries()`, `ground_truth(k)`
- Implemented the trait on all 6 subset/dataset types:
  - `LaionSubset` — Cosine, no pre-computed GT
  - `SiftSubset` — L2, no pre-computed GT (subset indices may not match full-dataset GT)
  - `GistSubset` — L2, returns pre-computed GT when depth >= k
  - `CohereWikipediaSubset` — Cosine, returns pre-computed GT when depth >= k (`#[cfg(feature = "parquet")]`)
  - `GloveSubset` — Cosine, returns pre-computed GT when depth >= k (`#[cfg(feature = "hdf5")]`)
  - `RandomDataset` — Cosine, no pre-computed GT
- Refactored `runner.rs`: `run_single_experiment`, `run_flat_baseline`, and
  `run_rabitq_experiments` now accept `&dyn Dataset`; `run_all_experiments` uses
  `Dataset::ground_truth()` with `compute_ground_truth_parallel` fallback
- Added `test_random_dataset_trait` unit test
- Trait is object-safe (`&dyn Dataset` and `Box<dyn Dataset>` work)

**Verification:**

- `cargo check -p motlie-db` — clean
- `cargo check --bin bench_vector --features benchmark` — CLI builds
- `cargo test -p motlie-db --features benchmark --lib` — 658 tests pass

### 2026-02-12 — Consolidate feature flags into single `benchmark` flag

**Author:** Claude (Opus 4.6)
**Branch:** `feature/vector-dataset`
**Status:** Complete

Consolidated `parquet`, `hdf5`, and `benchmark-full` feature flags into a single
`benchmark` flag. All dataset formats (NPY, fvecs, Parquet, HDF5) are now available
whenever the `benchmark` feature is enabled. Requires `libhdf5-dev` system package.

**What was done:**

- `libs/db/Cargo.toml`: `benchmark` now includes `dep:parquet`, `dep:arrow`, `dep:hdf5`,
  `dep:byteorder`; removed `parquet`, `hdf5`, `benchmark-full` feature definitions
- Workspace `Cargo.toml`: removed `parquet` and `hdf5` workspace features
- `dataset.rs`: removed all 21 `#[cfg(feature = "parquet")]` and `#[cfg(feature = "hdf5")]`
  annotations (code is already inside the benchmark-gated module)
- `mod.rs`: removed feature gates from Cohere and GloVe re-exports
- `commands.rs`: removed feature gates from download and list_datasets commands
- `README.md`, `DATASET.md`: updated feature references from `parquet`/`hdf5` to `benchmark`
