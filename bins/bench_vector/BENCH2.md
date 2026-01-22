# Deprecation Plan: `scale` → `index` + `query` Composition

**Status:** Phase 1-2 Complete, Phase 4 In Progress

## Current State

| Feature | `scale` | `index` | `query` |
|---------|---------|---------|---------|
| Synthetic random vectors | ✅ | ✅ (`--dataset random --stream`) | ✅ (`--dataset random`) |
| Real datasets (LAION, SIFT, GIST) | ❌ | ✅ | ✅ |
| Batch insert API | ✅ | ✅ (InsertVectorBatch via Writer) | N/A |
| Async insert mode | ✅ | ✅ (`--async-workers > 0`) | N/A |
| Progress reporting | ✅ | ✅ (ETA + throughput) | ✅ (query count) |
| Memory profiling (RSS) | ✅ | ✅ | ✅ |
| Recall computation | ✅ | ❌ | ✅ (sampled) |
| JSON output | ✅ | ✅ | ✅ |
| Reproducible seed | ✅ | ✅ | ✅ |
| Fresh start (`--fresh`) | ✅ | ✅ | N/A |
| Embedding discovery | ❌ | ✅ (`embeddings list/inspect`) | ✅ (reuses registry) |
| Ground truth export | ❌ | ✅ (`embeddings groundtruth`) | ✅ (`--stdin` for replay) |

## Consolidation Tasks

### Phase 1: Enhance `index` command ✅ COMPLETE
```
1.1 ✅ Add --batch-size for batched inserts
1.2 ✅ Add --async-workers for deferred HNSW construction
1.3 ✅ Add --output for JSON results (vectors inserted, duration, throughput, RSS)
1.4 ✅ Add --progress-interval and ETA reporting
1.5 ✅ Add memory profiling (peak RSS)
1.6 ✅ Add --drain-pending to process async backlog
```

### Phase 2: Enhance `query` command ✅ COMPLETE
```
2.1 ✅ Add --dataset random support (regenerate vectors from seed)
2.2 ✅ Add --seed convenience flag
2.3 ✅ Add --output for JSON results (recall, QPS, latencies, RSS)
2.4 ✅ Add --recall-sample-size for sampled ground truth
```

### Phase 3: Optional Enhancements (Deferred)
```
3.1 Stream dataset readers for LAION/SIFT/GIST
3.2 Optional "generate" command for offline random datasets
```

### Phase 4: Deprecate `scale` (In Progress)
```
4.1 ✅ Add deprecation warning to scale command
4.2 ✅ Document migration path (PHASE8, BASELINE, BENCH2-CODEX)
4.3 ⏳ Remove scale command and scale.rs module (future)
```

## Scripting Equivalence

### `scale` (legacy)
```bash
bench_vector scale \
  --num-vectors 1000000 --dim 128 --m 8 --ef-construction 100 \
  --ef-search 200 --k 10 --recall-sample-size 100 \
  --db-path /tmp/bench --output /tmp/results.json
```

### `index` + `query` (current)
```bash
# Build index
bench_vector index \
  --dataset random --num-vectors 1000000 --dim 128 --seed 42 \
  --m 8 --ef-construction 100 --fresh --stream --batch-size 5000 \
  --db-path /tmp/bench --output /tmp/index_results.json

# Query with recall
bench_vector query \
  --db-path /tmp/bench --dataset random --seed 42 \
  --num-queries 1000 --ef-search 200 --k 10 \
  --recall-sample-size 100 --output /tmp/query_results.json

# Combine results (shell)
jq -s '.[0] * .[1]' /tmp/index_results.json /tmp/query_results.json > /tmp/results.json
```

## Design Notes

1. **Seeded reproducibility**: `BenchmarkMetadata` stores vector/query seeds for regeneration.
2. **Async backlog**: `--drain-pending` replays pending HNSW updates.
3. **Public APIs**: `index` and `query` use channel-based `Writer` and `SearchReader`.

## Ground Truth Export + Query Replay

Generate brute-force ground truth using a stored embedding, then replay a query
through `bench_vector query --stdin`:

```bash
bench_vector embeddings groundtruth \
  --db-path /tmp/bench \
  --model bench-random \
  --dim 128 \
  --distance cosine \
  --count 1000 \
  --queries 5 \
  --k 10 \
  --output /tmp/gt.json

jq -c '.queries[0].vector' /tmp/gt.json | \
  bench_vector query --db-path /tmp/bench --dataset random --stdin \
    --k 10 --ef-search 100
```

Compare `jq '.queries[0].matches' /tmp/gt.json` to the printed results.
