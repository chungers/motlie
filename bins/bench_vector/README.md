# bench_vector - Vector Search Benchmark CLI

Benchmark and diagnostics CLI for Motlie vector search (HNSW + optional RaBitQ reranking).

## Scope

`bench_vector` is for:

- building benchmark vector indexes
- running ANN queries and recall/latency measurements
- sweeping parameters (`ef_search`, `bits`, `rerank`, `k`)
- validating RaBitQ rotation behavior (`check-distribution`)
- inspecting embedding registry and vector storage (`embeddings`, `admin`)

## Build Requirements

`bench_vector` is feature-gated and requires the `benchmark` feature.

```bash
# Recommended (native SIMD + benchmark datasets)
cargo build --release --bin bench_vector --features benchmark,simd-native
```

Because the binary is declared with `required-features = ["benchmark"]`, this will fail without `--features benchmark`.

### System dependency for benchmark datasets

- macOS: `brew install hdf5`
- Ubuntu/Debian: `sudo apt-get install libhdf5-dev`

## Quick Start

```bash
# Show available datasets and examples
cargo run --release --features benchmark --bin bench_vector -- datasets

# Build a random dataset index
cargo run --release --features benchmark --bin bench_vector -- index \
  --dataset random \
  --num-vectors 100000 \
  --dim 256 \
  --db-path /tmp/bench_db \
  --fresh

# List embedding specs and grab code
cargo run --release --features benchmark --bin bench_vector -- embeddings list \
  --db-path /tmp/bench_db

# Query by embedding code
cargo run --release --features benchmark --bin bench_vector -- query \
  --db-path /tmp/bench_db \
  --dataset random \
  --embedding-code <CODE> \
  --num-queries 1000 \
  --ef-search 100 \
  --k 10 \
  --skip-recall

# Sweep RaBitQ parameters
cargo run --release --features benchmark --bin bench_vector -- sweep \
  --dataset random \
  --dim 256 \
  --num-vectors 50000 \
  --rabitq \
  --bits 1,2,4 \
  --ef 50,100,200 \
  --rerank 1,4,10,20 \
  --show-pareto
```

## Datasets

| Dataset | Dim | Distance | Notes |
|---|---:|---|---|
| `laion` | 512 | cosine | downloadable |
| `sift` | 128 | l2 | downloadable |
| `gist` | 960 | l2 | manual download (TEXMEX) |
| `cohere` | 768 | cosine | downloadable |
| `glove` | 100 | angular/cosine-style workflow | downloadable |
| `random` | configurable | cosine | generated at runtime |

## Commands

| Command | Purpose |
|---|---|
| `download` | Download benchmark dataset assets |
| `index` | Build/extend vector index in RocksDB |
| `query` | Execute ANN queries and measure recall/latency |
| `sweep` | Grid search for recall/throughput trade-offs |
| `check-distribution` | Validate RaBitQ rotation variance (sqrt(D) scaling) |
| `embeddings` | List/inspect embedding specs and generate ground truth |
| `datasets` | Print available datasets and usage examples |
| `admin` | Storage diagnostics/validation tooling |

## Key Command Options

### `index`

```bash
bench_vector index --dataset <name> --num-vectors <N> --db-path <PATH>
```

Useful flags:

- `--m`, `--ef-construction`: HNSW build settings
- `--fresh`: remove existing DB path first
- `--cosine` / `--l2`: override auto distance selection
- `--dim`, `--seed`: random dataset generation
- `--stream`: stream random vectors instead of preloading
- `--batch-size`, `--async-workers`, `--drain-pending`: ingestion mode controls
- `--storage-type f32|f16`: vector precision/storage mode
- `--output <file.json>`: write build summary JSON

### `query`

```bash
bench_vector query --db-path <PATH> --dataset <name> [--embedding-code <CODE>]
```

Useful flags:

- `--k`, `--ef-search`, `--num-queries`
- `--embedding-code`: preferred to avoid ambiguous embedding resolution
- `--storage-type f32|f16`: enforce expected embedding storage type
- `--seed` / `--vector-seed` / `--query-seed`: random dataset reproducibility
- `--recall-sample-size`: enable sampled recall for `random`
- `--skip-recall`: skip exact ground-truth recall calculation
- `--stdin`: read one query vector as JSON array from stdin
- `--output <file.json>`: write query benchmark summary JSON

### `sweep`

```bash
bench_vector sweep --dataset <name> [--rabitq]
```

Useful flags:

- `--ef`, `--k`
- `--rabitq`, `--bits`, `--rerank`
- `--compare-simd`, `--simd-dot`
- `--results-dir`
- `--show-pareto`
- `--assert-recall <float>`: fail run if minimum observed recall is below threshold

### `embeddings`

```bash
bench_vector embeddings list --db-path <PATH>
bench_vector embeddings inspect --db-path <PATH> --code <CODE>
bench_vector embeddings groundtruth --db-path <PATH> --code <CODE> --output gt.json
```

### `admin`

```bash
bench_vector admin stats --db-path <PATH>
bench_vector admin validate --db-path <PATH> --strict
bench_vector admin rocksdb --db-path <PATH>
```

## Output Artifacts

- `index --output ...json`: index build metadata and throughput
- `query --output ...json`: recall/latency/qps metrics
- `sweep` in `--rabitq` mode writes CSV under `--results-dir`:
  - RaBitQ mode: `rabitq_sweep.csv`
  - SIMD comparison mode: `rabitq_sweep_simd.csv`, `rabitq_sweep_scalar.csv`

## Common Pitfalls

1. `bench_vector` binary does not build unless `--features benchmark` is enabled.
2. Querying without `--embedding-code` can fail if multiple specs match model/dim/distance.
3. `random` dataset recall is skipped unless you set `--recall-sample-size` (or disable with `--skip-recall`).
4. `gist` dataset must be downloaded and extracted manually.

## CI Usage Example

```bash
cargo run --release --features benchmark --bin bench_vector -- sweep \
  --dataset laion \
  --num-vectors 10000 \
  --rabitq \
  --bits 2,4 \
  --rerank 10 \
  --assert-recall 0.80
```

## Exit Behavior

- Successful run: exit code `0`
- Any command error (invalid args, missing dataset files, failed recall assertion, etc.): non-zero exit

## See Also

- [`libs/db/docs/getting-started.md`](../../libs/db/docs/getting-started.md) (API surface overview)
- [`libs/db/examples/vector_basic.rs`](../../libs/db/examples/vector_basic.rs) (minimal vector insert/search example)
- [`libs/db/src/vector/docs/REQUIREMENTS.md`](../../libs/db/src/vector/docs/REQUIREMENTS.md)
- [`libs/db/src/vector/docs/BENCHMARK.md`](../../libs/db/src/vector/docs/BENCHMARK.md)
- [`libs/db/src/vector/docs/ADMIN.md`](../../libs/db/src/vector/docs/ADMIN.md)
- [`examples/vector2/main.rs`](../../examples/vector2/main.rs)
