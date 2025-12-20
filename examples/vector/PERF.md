# Vector Search Performance Results

Performance benchmarks for HNSW and Vamana (DiskANN) implementations on `motlie_db`.

## Test Environment

- **CPU**: AMD EPYC 7742 64-Core Processor (20 cores)
- **RAM**: 119GB
- **Disk**: 3.5TB available
- **OS**: Linux 6.14.0-1015-nvidia
- **Date**: 2025-12-20

## Test Parameters

- **Vector Dimensions**: 1024
- **K (neighbors)**: 10
- **Queries**: 100
- **Distance Metric**: Euclidean (L2)

## Results Summary

| Algorithm | Vectors | Index Time | Throughput | Disk Usage | Latency (avg) | QPS | Recall@10 |
|-----------|---------|------------|------------|------------|---------------|-----|-----------|
| HNSW | 1K | 24.18s | 41.35/s | 23MB | 9.23ms | 108.4 | 0.995 |
| HNSW | 10K | 329.74s | 30.33/s | 257MB | 21.97ms | 45.5 | 0.820 |
| Vamana | 1K | 14.67s | 68.18/s | 30MB | 4.98ms | 200.9 | 0.579 |
| Vamana | 10K | 225.51s | 44.34/s | 281MB | 12.21ms | 81.9 | 0.269 |

## Observations

### HNSW

- **Strengths**: High recall (99.5% at 1K, 82% at 10K), robust hierarchical structure
- **Weaknesses**: Slower indexing due to multi-layer construction and 10ms write visibility delay
- **Scaling**: Indexing throughput degrades from 41/s to 30/s as graph size increases
- **Recall Degradation**: Recall drops from 99.5% to 82% at 10K due to approximate graph traversal

### Vamana (DiskANN)

- **Strengths**: Faster indexing (1.65x faster than HNSW at 1K), lower search latency
- **Weaknesses**: Low recall on uniform random data (57.9% at 1K, 26.9% at 10K)
- **Scaling**: Better throughput but recall degrades significantly at scale
- **Note**: DiskANN is optimized for clustered real-world data, not uniform random vectors

### Key Findings

1. **10ms Sleep Bottleneck**: Both algorithms are limited by the 10ms read-after-write delay in `motlie_db`
   - Theoretical max: 100 inserts/sec without the sleep
   - HNSW achieves only 30-41/s due to additional graph construction overhead

2. **Disk Usage**: ~25-28KB per vector (1024 dimensions * 4 bytes = 4KB raw + graph edges + metadata)
   - Linear scaling observed: 23MB at 1K, 257MB at 10K (~10x)

3. **Search Performance**:
   - HNSW: 45-108 QPS (higher recall, slower search)
   - Vamana: 82-201 QPS (lower recall, faster search)

4. **Recall vs Throughput Tradeoff**:
   - HNSW prioritizes recall over speed
   - Vamana prioritizes speed over recall (on random data)

## Bottlenecks Identified

1. **Write Visibility Delay**: 10ms sleep between inserts limits throughput to ~100/sec max
2. **Sequential Vector Loading**: Each vector fetch requires a DB read during search
3. **Graph Complexity**: HNSW multi-layer construction is compute-intensive
4. **Random Data**: Vamana's RNG pruning is less effective on uniformly distributed vectors

## Extrapolated Estimates

Based on observed trends (assuming linear scaling):

| Scale | Est. HNSW Index Time | Est. Vamana Index Time | Est. Disk Usage |
|-------|----------------------|------------------------|-----------------|
| 100K | ~55 min | ~38 min | ~2.5GB |
| 1M | ~9 hours | ~6 hours | ~25GB |
| 10M | ~90 hours (~4 days) | ~60 hours (~2.5 days) | ~250GB |
| 100M | ~900 hours (~37 days) | ~600 hours (~25 days) | ~2.5TB |

*Note: These estimates assume linear scaling, which may not hold at larger scales due to graph complexity and memory pressure.*

## Recommendations

1. **For < 10K vectors**: HNSW preferred for higher recall
2. **For 10K-100K vectors**: Consider Vamana if lower recall is acceptable
3. **For > 100K vectors**: Requires optimization work:
   - Remove 10ms sleep delay (implement proper read-after-write consistency)
   - Batch vector loading for search
   - Consider memory-mapped vectors for large datasets

## Future Work

- [ ] Remove 10ms write visibility delay
- [ ] Implement batch vector retrieval
- [ ] Add memory tracking during index build
- [ ] Test with real-world clustered data (e.g., embeddings)
- [ ] Implement incremental/online updates
