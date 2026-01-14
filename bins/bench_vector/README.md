# bench_vector - Vector Search Benchmark Tool

A comprehensive CLI for benchmarking HNSW + RaBitQ vector search performance. This tool enables systematic evaluation of recall, latency, and throughput across different datasets, parameters, and search strategies.

## Why This Tool Exists

Vector search systems face a fundamental trade-off between **recall** (result quality) and **throughput** (queries per second). Finding the optimal configuration requires systematic benchmarking across:

- **Scale**: Performance characteristics change significantly from 10K to 1M+ vectors
- **Parameters**: `ef_search`, `rerank_factor`, and `bits_per_dim` interact non-linearly
- **Datasets**: Real embeddings (CLIP, BERT) behave differently than random vectors

This tool provides reproducible benchmarks to:

1. **Validate RaBitQ correctness** - Ensure the training-free quantization meets DATA-1 requirements
2. **Find optimal parameters** - Discover the best recall-QPS trade-off for your use case
3. **Compare search strategies** - Standard HNSW vs RaBitQ two-phase search
4. **Diagnose issues** - Verify √D scaling and rotation quality on new datasets

## Installation

```bash
# Build from the workspace root
cargo build --release --bin bench_vector

# Or run directly
cargo run --release --bin bench_vector -- <command>
```

## Quick Start

```bash
# List available datasets and commands
bench_vector datasets

# Download LAION-CLIP dataset (~2GB)
bench_vector download --dataset laion --data-dir ./data

# Run a parameter sweep with RaBitQ
bench_vector sweep --dataset laion --num-vectors 50000 --rabitq --show-pareto

# Check rotation quality on a new dataset
bench_vector check-distribution --dataset random --dim 1024
```

## Commands

### `download` - Fetch Benchmark Datasets

Downloads standard benchmark datasets for reproducible evaluation.

```bash
bench_vector download --dataset <name> --data-dir <path>
```

| Dataset | Dimensions | Distance | Size | Use Case |
|---------|------------|----------|------|----------|
| `laion` | 512D | Cosine | ~2GB | Semantic search, RaBitQ validation |
| `sift` | 128D | L2 | ~500MB | Standard ANN benchmarks |
| `gist` | 960D | L2 | Manual | High-dimensional stress test |
| `random` | Configurable | Cosine | Generated | Worst-case ranking analysis |

**Note**: `gist` requires manual download from the TEXMEX corpus. `random` is generated on-the-fly (no download needed).

### `index` - Build HNSW Index

Creates a persistent HNSW index with incremental checkpoint support.

```bash
bench_vector index --dataset laion --num-vectors 100000 --db-path ./bench_db
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | Required | Dataset name |
| `--num-vectors` | Required | Number of vectors to index |
| `--db-path` | Required | RocksDB storage path |
| `--data-dir` | `./data` | Dataset files location |
| `--m` | 16 | HNSW connectivity (higher = better recall, more memory) |
| `--ef-construction` | 200 | Build beam width (higher = better recall, slower build) |
| `--fresh` | false | Delete existing index and start fresh |
| `--cosine` | auto | Force cosine distance |
| `--l2` | auto | Force L2/Euclidean distance |

**Why incremental?** Building large indices takes hours. The `index` command saves metadata checkpoints, allowing you to resume after interruption.

### `query` - Run Search Queries

Executes search queries against an existing index and reports recall/latency.

```bash
bench_vector query --db-path ./bench_db --dataset laion --k 10 --ef-search 100
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--db-path` | Required | Path to existing index |
| `--dataset` | Required | Dataset for query vectors |
| `--k` | 10 | Number of results per query |
| `--ef-search` | 100 | Search beam width |
| `--num-queries` | 1000 | Number of queries to run |

### `sweep` - Parameter Grid Search

The most powerful command - runs a systematic sweep over parameters to find optimal configurations.

```bash
bench_vector sweep --dataset laion --num-vectors 100000 --rabitq --show-pareto
```

**Core Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | Required | Dataset name (`laion`, `sift`, `gist`, `random`) |
| `--num-vectors` | 100000 | Database size |
| `--num-queries` | 1000 | Query count |
| `--ef` | 50,100,200 | ef_search values to sweep |
| `--k` | 1,10 | Recall@k values to measure |

**RaBitQ Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--rabitq` | false | Enable RaBitQ two-phase search |
| `--bits` | 1,2,4 | Bits per dimension to sweep |
| `--rerank` | 1,4,10,20 | Rerank factors to sweep |
| `--simd-dot` | true | Use SIMD-optimized dot products |
| `--compare-simd` | false | Run both SIMD and scalar for comparison |

**Random Dataset Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--dim` | 1024 | Vector dimension (random only) |
| `--seed` | 42 | RNG seed for reproducibility |

**Output Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--results-dir` | `./results` | CSV output directory |
| `--show-pareto` | false | Display Pareto-optimal configurations |

**Example - Full RaBitQ Sweep:**

```bash
bench_vector sweep \
  --dataset laion \
  --num-vectors 100000 \
  --rabitq \
  --bits 1,2,4 \
  --ef 50,100,200 \
  --rerank 4,10,20 \
  --show-pareto
```

**Output:**

```
=== RaBitQ Parameter Sweep (SIMD) ===

--- 4 bits/dim (SIMD) ---
Encoded 100000 vectors in 0.45s (25.60 MB)
ef       rerank   k        Recall     QPS        P50(ms)    P99(ms)
----------------------------------------------------------------
50       4        10       87.2%      523        1.82       4.21
100      10       10       92.1%      312        3.05       6.44
200      20       10       95.8%      187        5.12       9.87

Pareto Frontier (Recall@10 vs QPS)
----------------------------------------------------------------------
  bits   ef       rerank    k      recall        QPS
----------------------------------------------------------------------
     4       50        4   10      87.2%         523
     4      100       10   10      92.1%         312
     4      200       20   10      95.8%         187
----------------------------------------------------------------------
```

### `check-distribution` - Validate RaBitQ Rotation

Verifies that RaBitQ's random rotation matrix has correct √D scaling. This is critical for ensuring quantization quality on new datasets.

```bash
bench_vector check-distribution --dataset random --dim 1024 --sample-size 1000
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | Required | Dataset to analyze |
| `--sample-size` | 1000 | Vectors to sample |
| `--dim` | 1024 | Dimension (random only) |
| `--bits` | 4 | RaBitQ bits for rotation |
| `--seed` | 42 | Rotation matrix seed |

**Why this matters:**

RaBitQ uses a random orthonormal rotation matrix to decorrelate vector components before quantization. For this to work:

- **Component mean** should be ≈ 0 (centered)
- **Component variance** should be ≈ 1.0 (properly scaled)

Without √D scaling, variance would be ≈ 1/D, causing poor quantization. This command detects such issues before you waste time on a full benchmark.

**Example Output:**

```
=== RaBitQ Distribution Check ===
Dataset: random
Sample size: 1000
Bits per dim: 4
Loaded 1000 vectors, dim=1024

RaBitQ Rotation Stats (4-bit)
--------------------------------------------------
  Sample size: 1000 vectors
  Component mean: 0.000312
  Component variance: 0.998721 (expected: 1.0)
  Scaling valid: YES ✓ (variance in [0.8, 1.2])
--------------------------------------------------

Interpretation:
  ✓ Rotation matrix has correct √D scaling
  ✓ RaBitQ should work well on this dataset
```

### `datasets` - List Available Options

Shows all available datasets and usage examples.

```bash
bench_vector datasets
```

## Understanding the Output

### Recall@k

Measures result quality: what fraction of true nearest neighbors appear in the top-k results.

| Recall | Interpretation |
|--------|----------------|
| > 95% | Excellent - suitable for production RAG systems |
| 90-95% | Good - acceptable for most use cases |
| 80-90% | Moderate - may miss relevant results |
| < 80% | Poor - investigate parameters or data distribution |

### QPS (Queries Per Second)

Throughput metric. For disk-based systems like motlie_db:

| QPS | Interpretation |
|-----|----------------|
| > 500 | Excellent for disk-based |
| 200-500 | Good for production workloads |
| 50-200 | Acceptable for batch processing |
| < 50 | May need optimization |

### Latency (P50, P99)

Response time distribution:

- **P50**: Median latency (50% of queries faster)
- **P99**: Tail latency (99% of queries faster)

For interactive applications, P99 matters more than average.

### Pareto Frontier

The `--show-pareto` flag identifies configurations where you cannot improve recall without sacrificing QPS (or vice versa). These are the optimal trade-off points.

## Recommended Workflows

### 1. Validating a New Embedding Model

Before deploying a new embedding model (e.g., fine-tuned BERT):

```bash
# Step 1: Check rotation quality
bench_vector check-distribution --dataset your_data --sample-size 5000

# Step 2: Quick parameter sweep
bench_vector sweep --dataset your_data --num-vectors 10000 --rabitq --show-pareto

# Step 3: Scale up if looks good
bench_vector sweep --dataset your_data --num-vectors 100000 --rabitq --bits 4 --rerank 10,20
```

### 2. Finding Optimal Production Config

For a production deployment at 1M scale:

```bash
# Step 1: Sweep at representative scale
bench_vector sweep --dataset laion --num-vectors 100000 --rabitq \
  --bits 1,2,4 --ef 100,200,300 --rerank 4,10,20,40 --show-pareto

# Step 2: Test chosen config at full scale
bench_vector index --dataset laion --num-vectors 1000000 --db-path ./prod_test_db
bench_vector query --db-path ./prod_test_db --dataset laion --ef-search 200 --k 10
```

### 3. SIMD Performance Comparison

To verify SIMD provides expected speedup:

```bash
bench_vector sweep --dataset random --dim 512 --num-vectors 50000 \
  --rabitq --bits 1,2,4 --compare-simd
```

## Output Files

The `sweep` command generates CSV results for further analysis:

```
./results/
├── rabitq_sweep.csv      # Full parameter sweep data
```

**CSV Columns:**

| Column | Description |
|--------|-------------|
| `scale` | Number of vectors |
| `bits_per_dim` | RaBitQ bits |
| `ef_search` | Search beam width |
| `rerank_factor` | Candidates multiplier |
| `k` | Results per query |
| `recall_mean` | Average recall |
| `recall_std` | Recall standard deviation |
| `latency_avg_ms` | Mean latency |
| `latency_p50_ms` | Median latency |
| `latency_p95_ms` | 95th percentile latency |
| `latency_p99_ms` | 99th percentile latency |
| `qps` | Queries per second |

## Reference: Requirements Traceability

This tool validates requirements from `REQUIREMENTS.md`:

| Requirement | Target | How to Verify |
|-------------|--------|---------------|
| **REC-1** | Recall@10 > 95% at 1M | `sweep --num-vectors 1000000 --k 10` |
| **LAT-1** | P50 < 20ms at 1M | Check `latency_p50_ms` in results |
| **THR-3** | QPS > 500 at 1M | Check `qps` column in CSV |
| **DATA-1** | No training data required | Works with `--dataset random` |
| **STOR-4** | Training-free compression | RaBitQ uses random rotation |

## Troubleshooting

### Low Recall with RaBitQ

1. **Check rotation quality**: Run `check-distribution` first
2. **Increase rerank_factor**: Try `--rerank 20,40` instead of default
3. **Use more bits**: `--bits 4` captures more information than `--bits 1`
4. **Verify normalization**: RaBitQ works best with normalized (unit) vectors

### Slow Build Times

1. **Reduce ef_construction**: `--ef-construction 100` instead of 200
2. **Lower M**: `--m 12` trades recall for faster builds
3. **Use SSD storage**: HDD I/O dominates at scale

### Variance Outside [0.8, 1.2]

If `check-distribution` reports invalid variance:

- **Too low (< 0.1)**: Rotation matrix may lack √D scaling
- **Too high (> 2.0)**: Vectors may not be normalized

Both indicate potential issues that will degrade RaBitQ recall.

## See Also

- `libs/db/src/vector/REQUIREMENTS.md` - Ground truth requirements
- `libs/db/src/vector/BENCHMARK.md` - Detailed benchmark results
- `libs/db/src/vector/GEMINI-BENCHMARK.md` - Benchmark infrastructure design
- `examples/vector2/` - Additional benchmark examples
