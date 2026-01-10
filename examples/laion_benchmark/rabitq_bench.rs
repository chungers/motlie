//! RaBitQ parallel re-ranking benchmark.
//!
//! Compares sequential vs parallel (rayon) re-ranking performance using
//! real LAION-CLIP embeddings.

use anyhow::Result;
use motlie_db::vector::benchmark::{LaionDataset, LAION_EMBEDDING_DIM};
use motlie_db::vector::search::rerank_parallel;
use motlie_db::vector::{Distance, RaBitQ, VecId};
use std::path::Path;
use std::time::Instant;

/// Run the RaBitQ parallel re-ranking benchmark.
pub fn run_rabitq_benchmark(
    data_dir: &Path,
    num_queries: usize,
    rerank_sizes: &[usize],
) -> Result<()> {
    println!("=== RaBitQ Parallel Re-ranking Benchmark ===\n");

    // Load LAION dataset
    let num_vectors = 50_000; // Use 50K for this benchmark
    println!("Loading LAION dataset ({} vectors)...", num_vectors);
    let dataset = LaionDataset::load(data_dir, num_vectors)?;

    let vectors: Vec<Vec<f32>> = dataset.image_embeddings;
    let queries: Vec<Vec<f32>> = dataset
        .text_embeddings
        .into_iter()
        .take(num_queries)
        .collect();

    println!(
        "  Loaded {} vectors, {} queries ({}D)\n",
        vectors.len(),
        queries.len(),
        LAION_EMBEDDING_DIM
    );

    // Create RaBitQ encoder
    println!("Creating RaBitQ encoder ({}D, 1 bit)...", LAION_EMBEDDING_DIM);
    let encoder = RaBitQ::new(LAION_EMBEDDING_DIM, 1, 42);
    println!("  Code size: {} bytes", encoder.code_size());

    // Encode all vectors
    println!("Encoding vectors to binary codes...");
    let start = Instant::now();
    let codes: Vec<Vec<u8>> = vectors.iter().map(|v| encoder.encode(v)).collect();
    let encode_time = start.elapsed();
    println!(
        "  Encoded {} vectors in {:.2}s ({:.0} vec/s)\n",
        vectors.len(),
        encode_time.as_secs_f64(),
        vectors.len() as f64 / encode_time.as_secs_f64()
    );

    // Encode queries
    let query_codes: Vec<Vec<u8>> = queries.iter().map(|v| encoder.encode(v)).collect();

    // Run benchmarks
    println!("{}", "=".repeat(80));
    println!("Re-ranking Benchmark: Sequential vs Parallel (Rayon)");
    println!("{}\n", "=".repeat(80));

    println!(
        "{:>10} | {:>12} {:>10} | {:>12} {:>10} | {:>8}",
        "Candidates", "Seq (ms)", "Seq QPS", "Par (ms)", "Par QPS", "Speedup"
    );
    println!("{}", "-".repeat(80));

    for &rerank_count in rerank_sizes {
        let (seq_ms, par_ms) = benchmark_reranking(
            &vectors,
            &codes,
            &queries,
            &query_codes,
            rerank_count,
            10, // k
        )?;

        let seq_qps = 1000.0 * queries.len() as f64 / seq_ms;
        let par_qps = 1000.0 * queries.len() as f64 / par_ms;
        let speedup = seq_ms / par_ms;

        println!(
            "{:>10} | {:>12.2} {:>10.0} | {:>12.2} {:>10.0} | {:>7.2}x",
            rerank_count, seq_ms, seq_qps, par_ms, par_qps, speedup
        );
    }

    println!("\n{}", "=".repeat(80));
    println!("Analysis");
    println!("{}\n", "=".repeat(80));

    println!("Key findings:");
    println!("- Rayon parallelism benefits larger candidate sets (100+ vectors)");
    println!("- Overhead dominates for small sets (<50 candidates)");
    println!("- Optimal speedup typically 2-4x on multi-core systems");
    println!("\nFor HNSW + RaBitQ two-phase search:");
    println!("- Phase 1: Hamming distance filtering (fast, select top candidates)");
    println!("- Phase 2: Exact re-ranking with parallel (this benchmark)");

    Ok(())
}

/// Benchmark sequential vs parallel re-ranking.
fn benchmark_reranking(
    vectors: &[Vec<f32>],
    codes: &[Vec<u8>],
    queries: &[Vec<f32>],
    query_codes: &[Vec<u8>],
    rerank_count: usize,
    k: usize,
) -> Result<(f64, f64)> {
    // Pre-select candidates for each query using Hamming distance
    let candidates_per_query: Vec<Vec<VecId>> = query_codes
        .iter()
        .map(|qc| select_candidates_hamming(qc, codes, rerank_count))
        .collect();

    // Warmup (5 iterations)
    for _ in 0..5 {
        for (qi, candidates) in candidates_per_query.iter().enumerate() {
            let _ = rerank_sequential(&queries[qi], vectors, candidates, k);
            let _ = rerank_parallel_wrapper(&queries[qi], vectors, candidates, k);
        }
    }

    // Benchmark sequential
    let start = Instant::now();
    for (qi, candidates) in candidates_per_query.iter().enumerate() {
        let _ = rerank_sequential(&queries[qi], vectors, candidates, k);
    }
    let seq_time = start.elapsed().as_secs_f64() * 1000.0;

    // Benchmark parallel
    let start = Instant::now();
    for (qi, candidates) in candidates_per_query.iter().enumerate() {
        let _ = rerank_parallel_wrapper(&queries[qi], vectors, candidates, k);
    }
    let par_time = start.elapsed().as_secs_f64() * 1000.0;

    Ok((seq_time, par_time))
}

/// Select top candidates by Hamming distance (simulates Phase 1).
fn select_candidates_hamming(query_code: &[u8], codes: &[Vec<u8>], count: usize) -> Vec<VecId> {
    let mut distances: Vec<(u32, VecId)> = codes
        .iter()
        .enumerate()
        .map(|(i, c)| (RaBitQ::hamming_distance(query_code, c), i as VecId))
        .collect();

    distances.sort_by_key(|&(d, _)| d);
    distances.truncate(count);
    distances.into_iter().map(|(_, id)| id).collect()
}

/// Sequential re-ranking (baseline).
fn rerank_sequential(
    query: &[f32],
    vectors: &[Vec<f32>],
    candidates: &[VecId],
    k: usize,
) -> Vec<(f32, VecId)> {
    let mut results: Vec<(f32, VecId)> = candidates
        .iter()
        .map(|&id| {
            let dist = Distance::L2.compute(query, &vectors[id as usize]);
            (dist, id)
        })
        .collect();

    results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    results.truncate(k);
    results
}

/// Parallel re-ranking using rayon (wrapper around vector::search::rerank_parallel).
fn rerank_parallel_wrapper(
    query: &[f32],
    vectors: &[Vec<f32>],
    candidates: &[VecId],
    k: usize,
) -> Vec<(f32, VecId)> {
    rerank_parallel(
        candidates,
        |id| Some(Distance::L2.compute(query, &vectors[id as usize])),
        k,
    )
}
