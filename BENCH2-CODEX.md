# BENCH2-CODEX

## Refactor Summary

- Deprecate `bench_vector scale` in favor of `bench_vector index` + `bench_vector query`.
- Use public channel APIs (`Writer`, `SearchReader`) in `index` and `query`.
- Add streaming random dataset support to avoid loading vectors into memory.
- Support async indexing with `AsyncGraphUpdater` for large-scale runs.

## Command Capabilities

### Index (Streaming Random)

- `--dataset random --stream` enables streaming vector generation.
- `--batch-size` controls streaming insert batch size.
- `--progress-interval` prints throughput updates during inserts.
- `--async` + `--async-workers` enables deferred HNSW construction.
- `--fresh` required for streaming random (no incremental streaming).

### Query (Random)

- `--dataset random` runs queries against streaming-generated vectors.
- `--skip-recall` avoids expensive brute-force ground truth.
- `--recall-sample-size` enables sampled recall with brute-force.
- `--query-seed` / `--vector-seed` control reproducible random runs.

## Public API Usage (Index/Query)

- `create_writer` + `spawn_mutation_consumer_with_storage_autoreg`
- `InsertVectorBatch` + `Writer::send` / `Writer::flush`
- `create_search_reader_with_storage` + `spawn_query_consumers_with_storage_autoreg`
- `SearchKNN::run` for query execution

## Example Workflows

```bash
# Index (10K, random streaming)
./target/release/bench_vector index \
  --dataset random \
  --num-vectors 10000 \
  --dim 128 \
  --stream \
  --batch-size 500 \
  --db-path /tmp/bench_10k \
  --fresh

# Query (10K)
./target/release/bench_vector query \
  --dataset random \
  --db-path /tmp/bench_10k \
  --num-queries 100 \
  --skip-recall
```

## Deprecation Note

`bench_vector scale` remains available but hidden from help output. It prints
an explicit deprecation warning and is superseded by the streaming `index`/`query`
flow above.
