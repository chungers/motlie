# LAION-CLIP Benchmark Results

Reproducing experiments from "HNSW at Scale" article.

## Configuration

- Dataset: LAION-400M CLIP ViT-B-32 (512D)
- HNSW: M=16, ef_construction=100
- Distance: Cosine

## 50000 Vectors

| Strategy | ef_search | Recall@5 | Recall@10 | Latency (ms) | QPS |
|----------|-----------|----------|-----------|--------------|-----|
| HNSW-Cosine | 10 | 91.7% | 89.5% | 3.50 | 285.8 |
| HNSW-Cosine | 20 | 91.7% | 89.5% | 3.49 | 286.3 |
| HNSW-Cosine | 40 | 91.7% | 89.5% | 3.51 | 284.9 |
| HNSW-Cosine | 80 | 91.7% | 89.5% | 3.47 | 288.3 |
| HNSW-Cosine | 160 | 91.7% | 89.5% | 3.49 | 286.3 |
| Flat | N/A | 100.0% | 100.0% | 33.06 | 30.3 |

## 100000 Vectors

| Strategy | ef_search | Recall@5 | Recall@10 | Latency (ms) | QPS |
|----------|-----------|----------|-----------|--------------|-----|
| HNSW-Cosine | 10 | 87.9% | 86.8% | 4.03 | 248.0 |
| HNSW-Cosine | 20 | 87.9% | 86.8% | 4.07 | 245.6 |
| HNSW-Cosine | 40 | 87.9% | 86.8% | 4.05 | 246.9 |
| HNSW-Cosine | 80 | 87.9% | 86.8% | 4.01 | 249.1 |
| HNSW-Cosine | 160 | 87.9% | 86.8% | 3.97 | 252.0 |
| Flat | N/A | 100.0% | 100.0% | 67.13 | 14.9 |

## 150000 Vectors

| Strategy | ef_search | Recall@5 | Recall@10 | Latency (ms) | QPS |
|----------|-----------|----------|-----------|--------------|-----|
| HNSW-Cosine | 10 | 88.2% | 86.3% | 4.42 | 226.0 |
| HNSW-Cosine | 20 | 88.2% | 86.3% | 4.47 | 223.7 |
| HNSW-Cosine | 40 | 88.2% | 86.3% | 4.48 | 223.1 |
| HNSW-Cosine | 80 | 88.2% | 86.3% | 4.43 | 225.7 |
| HNSW-Cosine | 160 | 88.2% | 86.3% | 4.47 | 223.8 |
| Flat | N/A | 100.0% | 100.0% | 100.56 | 9.9 |

## 200000 Vectors

| Strategy | ef_search | Recall@5 | Recall@10 | Latency (ms) | QPS |
|----------|-----------|----------|-----------|--------------|-----|
| HNSW-Cosine | 10 | 87.1% | 84.9% | 5.14 | 194.4 |
| HNSW-Cosine | 20 | 87.1% | 84.9% | 5.25 | 190.4 |
| HNSW-Cosine | 40 | 87.1% | 84.9% | 4.98 | 201.0 |
| HNSW-Cosine | 80 | 87.1% | 84.9% | 5.02 | 199.3 |
| HNSW-Cosine | 160 | 87.1% | 84.9% | 5.02 | 199.1 |
| Flat | N/A | 100.0% | 100.0% | 135.93 | 7.4 |

