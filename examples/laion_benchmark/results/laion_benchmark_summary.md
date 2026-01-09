# LAION-CLIP Benchmark Results

Reproducing experiments from "HNSW at Scale" article.

## Configuration

- Dataset: LAION-400M CLIP ViT-B-32 (512D)
- HNSW: M=16, ef_construction=100
- Distance: Cosine

## 50000 Vectors

| Strategy | ef_search | Recall@5 | Recall@10 | Latency (ms) | QPS |
|----------|-----------|----------|-----------|--------------|-----|
| HNSW-Cosine | 10 | 91.8% | 89.6% | 2.89 | 346.3 |
| HNSW-Cosine | 20 | 91.8% | 89.6% | 2.90 | 344.9 |
| HNSW-Cosine | 40 | 91.8% | 89.6% | 2.90 | 345.2 |
| HNSW-Cosine | 80 | 91.8% | 89.6% | 2.92 | 342.8 |
| HNSW-Cosine | 160 | 91.8% | 89.6% | 2.98 | 335.4 |
| Flat | N/A | 100.0% | 100.0% | 33.36 | 30.0 |

## 100000 Vectors

| Strategy | ef_search | Recall@5 | Recall@10 | Latency (ms) | QPS |
|----------|-----------|----------|-----------|--------------|-----|
| HNSW-Cosine | 10 | 88.2% | 86.8% | 3.55 | 282.0 |
| HNSW-Cosine | 20 | 88.2% | 86.8% | 3.54 | 282.5 |
| HNSW-Cosine | 40 | 88.2% | 86.8% | 3.44 | 290.9 |
| HNSW-Cosine | 80 | 88.2% | 86.8% | 3.45 | 290.3 |
| HNSW-Cosine | 160 | 88.2% | 86.8% | 3.45 | 290.2 |
| Flat | N/A | 100.0% | 100.0% | 66.63 | 15.0 |

## 150000 Vectors

| Strategy | ef_search | Recall@5 | Recall@10 | Latency (ms) | QPS |
|----------|-----------|----------|-----------|--------------|-----|
| HNSW-Cosine | 10 | 87.5% | 85.7% | 3.94 | 253.8 |
| HNSW-Cosine | 20 | 87.5% | 85.7% | 3.83 | 260.9 |
| HNSW-Cosine | 40 | 87.5% | 85.7% | 3.84 | 260.6 |
| HNSW-Cosine | 80 | 87.5% | 85.7% | 3.81 | 262.6 |
| HNSW-Cosine | 160 | 87.5% | 85.7% | 3.81 | 262.7 |
| Flat | N/A | 100.0% | 100.0% | 100.17 | 10.0 |

## 200000 Vectors

| Strategy | ef_search | Recall@5 | Recall@10 | Latency (ms) | QPS |
|----------|-----------|----------|-----------|--------------|-----|
| HNSW-Cosine | 10 | 87.2% | 84.9% | 4.34 | 230.3 |
| HNSW-Cosine | 20 | 87.2% | 84.9% | 4.36 | 229.5 |
| HNSW-Cosine | 40 | 87.2% | 84.9% | 4.33 | 230.9 |
| HNSW-Cosine | 80 | 87.2% | 84.9% | 4.33 | 231.0 |
| HNSW-Cosine | 160 | 87.2% | 84.9% | 4.31 | 232.0 |
| Flat | N/A | 100.0% | 100.0% | 133.97 | 7.5 |

