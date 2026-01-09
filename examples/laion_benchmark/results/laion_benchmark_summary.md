# LAION-CLIP Benchmark Results

Reproducing experiments from "HNSW at Scale" article.

## Configuration

- Dataset: LAION-400M CLIP ViT-B-32 (512D)
- HNSW: M=16, ef_construction=100
- Distance: Cosine

## 50000 Vectors

| Strategy | ef_search | Recall@5 | Recall@10 | Latency (ms) | QPS |
|----------|-----------|----------|-----------|--------------|-----|
| HNSW-Cosine | 10 | 91.5% | 89.3% | 2.76 | 362.1 |
| HNSW-Cosine | 20 | 91.5% | 89.3% | 2.77 | 360.7 |
| HNSW-Cosine | 40 | 91.5% | 89.3% | 2.77 | 361.3 |
| HNSW-Cosine | 80 | 91.5% | 89.3% | 2.77 | 360.5 |
| HNSW-Cosine | 160 | 91.5% | 89.3% | 2.78 | 360.1 |
| Flat | N/A | 100.0% | 100.0% | 33.20 | 30.1 |

