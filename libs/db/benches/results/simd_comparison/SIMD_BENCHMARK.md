# SIMD vs Scalar Quantized Dot Product Benchmark

**Date:** 2026-01-13
**Platform:** aarch64 Linux (NEON)
**Dataset:** LAION CLIP (512D, cosine similarity)
**Scale:** 100,000 vectors, 500 queries

## Summary

| Quantization | SIMD Speedup | Best Config |
|--------------|--------------|-------------|
| 1-bit | **2.6x** | Algebraic optimization avoids branching |
| 2-bit | 1.2x | Lookup table still benefits from SIMD |
| 4-bit | 1.2x | Linear scale+offset with FMA |

## Detailed Results (100K vectors, ef=100, k=10)

### 1-bit Quantization (64 bytes/vector for 512D)

| rerank | Mode | QPS | P50 (ms) | P99 (ms) | Recall |
|--------|------|-----|----------|----------|--------|
| 4 | SIMD | 119.1 | 8.43 | 11.19 | 86.8% |
| 4 | scalar | 44.9 | 22.56 | 29.92 | 86.8% |
| 10 | SIMD | 59.8 | 16.75 | 20.37 | 93.0% |
| 10 | scalar | 22.4 | 45.15 | 56.16 | 93.0% |
| 20 | SIMD | 35.6 | 28.17 | 33.01 | 96.0% |
| 20 | scalar | 13.7 | 73.98 | 87.10 | 96.0% |

**Speedup: 2.6x**

### 2-bit Quantization (128 bytes/vector for 512D)

| rerank | Mode | QPS | P50 (ms) | P99 (ms) | Recall |
|--------|------|-----|----------|----------|--------|
| 4 | SIMD | 126.0 | 7.83 | 10.72 | 97.6% |
| 4 | scalar | 104.0 | 9.61 | 12.90 | 97.6% |
| 10 | SIMD | 65.1 | 15.41 | 18.53 | 99.5% |
| 10 | scalar | 53.2 | 18.90 | 22.98 | 99.5% |
| 20 | SIMD | 38.8 | 25.70 | 30.39 | 99.7% |
| 20 | scalar | 31.8 | 31.53 | 37.44 | 99.7% |

**Speedup: 1.2x**

### 4-bit Quantization (256 bytes/vector for 512D)

| rerank | Mode | QPS | P50 (ms) | P99 (ms) | Recall |
|--------|------|-----|----------|----------|--------|
| 4 | SIMD | 124.0 | 8.01 | 11.74 | 99.2% |
| 4 | scalar | 102.2 | 9.75 | 13.47 | 99.2% |
| 10 | SIMD | 64.0 | 15.64 | 19.80 | 99.8% |
| 10 | scalar | 52.9 | 19.08 | 24.38 | 99.8% |
| 20 | SIMD | 38.2 | 26.31 | 30.90 | 99.9% |
| 20 | scalar | 31.4 | 32.05 | 38.17 | 99.9% |

**Speedup: 1.2x**

## Analysis

### Why 1-bit shows 2.6x speedup

The 1-bit SIMD implementation uses the algebraic identity:
```
dot(q, b) = 2 * sum(q[i] where b[i]=1) - sum(q)
```

This avoids per-bit conditional branches. The NEON implementation:
1. Computes `sum(q)` with SIMD reduction
2. Uses bit masking to compute `positive_sum` without branching
3. Final result: `2 * positive_sum - sum_q`

### Why 2-bit/4-bit show only 1.2x speedup

These require per-element operations:
- **2-bit**: Gray code decode → lookup table (4 values)
- **4-bit**: Gray code decode → linear transform (`level * scale + offset`)

The decode and lookup operations are inherently scalar-like, limiting SIMD benefit.

## Recommended Configuration

For **512D embeddings** at **100K scale**:

| Use Case | Config | Recall | QPS |
|----------|--------|--------|-----|
| High speed | 2-bit, rerank=4 | 97.6% | 126 |
| Balanced | 2-bit, rerank=10 | 99.5% | 65 |
| High recall | 4-bit, rerank=10 | 99.8% | 64 |

## Files

- Implementation: `libs/core/src/distance/quantized/`
- Config flag: `RaBitQConfig.use_simd_dot` in `libs/db/src/vector/config.rs`
- CLI: `--simd-dot` flag in `vector2` example and `bench_vector` binary
