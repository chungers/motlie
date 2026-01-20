# Deprecation Plan: `scale` → `index` + `query` Composition

**Status:** Phase 1-2 Complete, Phase 3 Optional, Phase 4 Pending

## Current State

| Feature | `scale` | `index` | `query` |
|---------|---------|---------|---------|
| Synthetic random vectors | ✅ (built-in) | ✅ (`--dataset random`) | ❌ |
| Real datasets (LAION, SIFT, etc.) | ❌ | ✅ | ✅ |
| Batch insert API | ✅ (`Processor.insert_batch`) | ❌ (single insert) | N/A |
| Async insert mode | ✅ | ❌ | N/A |
| Progress reporting | ✅ (with ETA) | ✅ (basic) | ❌ |
| Memory profiling (RSS) | ✅ | ❌ | ❌ |
| Recall computation | ✅ (brute-force GT) | ❌ | ✅ |
| JSON output | ✅ | ❌ | ❌ |
| Reproducible seed | ✅ | ✅ | ❌ |
| Fresh start (`--fresh`) | ✅ (always fresh) | ✅ | N/A |

## Gap Analysis

### Features in `scale` missing from `index`:
1. **Batch insert API** - `index` uses single-vector insert; `scale` uses `Processor.insert_batch`
2. **Async insert mode** - `index` has no `--async` support
3. **Memory profiling** - No RSS tracking in `index`
4. **JSON output** - No `--output` for machine-readable results
5. **Progress with ETA** - `index` reports basic progress, not throughput/ETA

### Features in `scale` missing from `query`:
1. **Random query generation** - `query` requires dataset files for queries
2. **Brute-force recall on random data** - `scale` regenerates vectors for ground truth
3. **JSON output** - No `--output` for machine-readable results
4. **Memory profiling** - No RSS tracking

## Consolidation Tasks

### Phase 1: Enhance `index` command ✅ COMPLETE
```
1.1 ✅ Add --batch-size parameter (use Processor.insert_batch)
1.2 ✅ Add --async and --async-workers parameters
1.3 ✅ Add --output for JSON results (vectors inserted, duration, throughput)
1.4 ✅ Add --progress-interval for configurable reporting
1.5 ✅ Improve progress to show throughput and ETA
1.6 ✅ Add memory profiling (peak RSS) to output
```

### Phase 2: Enhance `query` command ✅ COMPLETE
```
2.1 ✅ Add --dataset random support (generate queries from seed)
2.2 ✅ Add --seed parameter for reproducible random queries
2.3 ✅ Add --output for JSON results (recall, QPS, latencies)
2.4 ✅ Add --recall-sample-size to sample subset for ground truth
2.5 ✅ Compute ground truth on-the-fly from regenerated vectors
2.6 ✅ Add memory profiling to output
```

### Phase 3: Add `generate` command (optional, deferred)
```
3.1 New command to pre-generate random vectors to fvecs files
3.2 Enables: bench_vector generate --num-vectors 1M --dim 128 --seed 42 --output /tmp/random.fvecs
3.3 Then: bench_vector index --dataset /tmp/random.fvecs ...
3.4 Allows index/query to work uniformly with files
```

### Phase 4: Deprecate `scale` (in progress)
```
4.1 ✅ Add deprecation warning to scale command
4.2 ⏳ Document migration path in README
4.3 ⏳ Remove scale command and scale.rs module (future)
```

## Scripting Equivalence

### Current `scale` usage:
```bash
bench_vector scale \
    --num-vectors 1000000 --dim 128 --m 8 --ef-construction 100 \
    --ef-search 200 --k 10 --recall-sample-size 100 \
    --db-path /tmp/bench --output /tmp/results.json
```

### After consolidation:
```bash
# Build index
bench_vector index \
    --dataset random --num-vectors 1000000 --dim 128 --seed 42 \
    --m 8 --ef-construction 100 --fresh --batch-size 5000 \
    --db-path /tmp/bench --output /tmp/index_results.json

# Query with recall
bench_vector query \
    --db-path /tmp/bench --dataset random --seed 42 \
    --num-queries 1000 --ef-search 200 --k 10 \
    --recall-sample-size 100 --output /tmp/query_results.json

# Combine results (shell)
jq -s '.[0] * .[1]' /tmp/index_results.json /tmp/query_results.json > /tmp/results.json
```

## Key Design Decisions

1. **Ground truth for random data**: Store seed in metadata so `query` can regenerate vectors for brute-force comparison

2. **Batch insert API**: `index` should use `Processor.insert_batch` by default for performance parity

3. **Async mode**: Move async logic to `index` command, not a separate module

4. **Unified output format**: Both commands produce compatible JSON that can be merged

5. **Memory profiling**: Add `get_rss_bytes()` helper to shared utils, use in both commands
