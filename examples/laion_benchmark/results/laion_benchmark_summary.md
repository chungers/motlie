# LAION-CLIP Benchmark Results

Reproducing experiments from "HNSW at Scale" article.

## Configuration

- Dataset: LAION-400M CLIP ViT-B-32 (512D)
- HNSW: M=16, ef_construction=100
- Distance: Cosine

## 50000 Vectors

| Strategy | ef_search | Recall@5 | Recall@10 | Latency (ms) | QPS |
|----------|-----------|----------|-----------|--------------|-----|
| HNSW-Cosine | 10 | 91.4% | 89.4% | 5.32 | 188.1 |
| HNSW-Cosine | 20 | 91.4% | 89.4% | 5.31 | 188.3 |
| HNSW-Cosine | 40 | 91.4% | 89.4% | 5.26 | 190.0 |
| HNSW-Cosine | 80 | 91.4% | 89.4% | 5.28 | 189.5 |
| HNSW-Cosine | 160 | 91.4% | 89.4% | 5.23 | 191.3 |
| Flat | N/A | 100.0% | 100.0% | 33.30 | 30.0 |

