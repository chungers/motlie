//! Vector2: Benchmark for Phase 2 HNSW Implementation
//!
//! This example demonstrates and validates the new HNSW implementation in
//! `motlie_db::vector` using RocksDB storage with RoaringBitmap edges.
//!
//! # Features
//!
//! - Load SIFT benchmark datasets (SIFT10K, SIFT1M)
//! - Compare new RocksDB-backed HNSW vs old graph-based implementation
//! - Collect detailed performance metrics (build time, QPS, recall, latency)
//!
//! # Usage
//!
//! ```bash
//! # Run with synthetic data
//! cargo run --release --example vector2 -- --num-vectors 10000 --num-queries 100 --k 10
//!
//! # Run with SIFT10K dataset
//! cargo run --release --example vector2 -- --dataset sift10k --num-vectors 10000 --num-queries 100 --k 10
//!
//! # Compare implementations
//! cargo run --release --example vector2 -- --dataset sift10k --num-vectors 10000 --compare
//! ```

// Local benchmark utilities (standalone, no dependency on examples/vector)
mod benchmark;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use tempfile::TempDir;

use benchmark::{BenchmarkDataset, BenchmarkMetrics, DatasetConfig, DatasetName};
use motlie_db::rocksdb::ColumnFamily;
use motlie_db::vector::{
    EmbeddingCode, HnswConfig, HnswIndex, NavigationCache, Storage, VecId, VectorCfKey,
    VectorCfValue, Vectors,
};

// ============================================================================
// CLI Arguments
// ============================================================================

#[derive(Parser, Debug)]
#[command(name = "vector2")]
#[command(about = "Benchmark Phase 2 HNSW implementation with SIFT datasets")]
struct Args {
    /// Database path (uses temp dir if not specified)
    #[arg(long)]
    db_path: Option<PathBuf>,

    /// Dataset to use: sift1m, sift10k, random
    #[arg(long, default_value = "random")]
    dataset: String,

    /// Number of vectors to index
    #[arg(long, default_value = "10000")]
    num_vectors: usize,

    /// Number of queries to run
    #[arg(long, default_value = "100")]
    num_queries: usize,

    /// Number of nearest neighbors to find
    #[arg(long, default_value = "10")]
    k: usize,

    /// Search beam width (ef parameter)
    #[arg(long, default_value = "100")]
    ef: usize,

    /// HNSW M parameter (max connections per layer)
    #[arg(long, default_value = "16")]
    m: usize,

    /// ef_construction parameter
    #[arg(long, default_value = "200")]
    ef_construction: usize,

    /// Run comparison with old implementation
    #[arg(long)]
    compare: bool,

    /// Verbose output
    #[arg(long, short)]
    verbose: bool,
}

// ============================================================================
// Main Entry Point
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    motlie_core::telemetry::init_dev_subscriber_with_env_filter();

    let args = Args::parse();

    println!("=== Vector2: Phase 2 HNSW Benchmark ===");
    println!();

    // Parse dataset name
    let dataset_name = DatasetName::from_str(&args.dataset);
    let dataset_str = dataset_name
        .map(|d| d.to_string())
        .unwrap_or_else(|| "random".to_string());

    println!("Configuration:");
    println!("  Dataset:         {}", dataset_str);
    println!("  Vectors:         {}", args.num_vectors);
    println!("  Queries:         {}", args.num_queries);
    println!("  K:               {}", args.k);
    println!("  ef (search):     {}", args.ef);
    println!("  M:               {}", args.m);
    println!("  ef_construction: {}", args.ef_construction);
    println!();

    // Load benchmark dataset
    let benchmark_data = if let Some(ds_name) = dataset_name {
        println!("Loading {} dataset...", ds_name);
        let config = DatasetConfig {
            name: ds_name,
            num_base: Some(args.num_vectors),
            num_queries: Some(args.num_queries),
            ground_truth_k: 100,
            ..Default::default()
        };
        Some(benchmark::load_dataset(&config).await?)
    } else {
        None
    };

    // Run Phase 2 benchmark
    let metrics = run_phase2_benchmark(&args, benchmark_data.as_ref()).await?;
    metrics.print();

    // Optionally compare with old implementation
    if args.compare {
        println!("\n--- Comparison mode not yet implemented ---");
        println!("The old implementation uses graph edges, which requires different setup.");
        println!("For now, use the separate `hnsw` example for comparison.");
    }

    println!("\n=== Benchmark Complete ===");
    Ok(())
}

// ============================================================================
// Phase 2 HNSW Benchmark
// ============================================================================

/// Run benchmark using the new Phase 2 HNSW implementation
async fn run_phase2_benchmark(
    args: &Args,
    benchmark_data: Option<&BenchmarkDataset>,
) -> Result<BenchmarkMetrics> {
    let dataset_str = benchmark_data
        .map(|d| d.name.to_string())
        .unwrap_or_else(|| "random".to_string());

    let mut metrics = BenchmarkMetrics::new(&dataset_str, "HNSW2-RocksDB");

    // Create temp directory or use specified path
    let _temp_dir: Option<TempDir>;
    let db_path = if let Some(ref path) = args.db_path {
        if path.exists() {
            std::fs::remove_dir_all(path)?;
        }
        path.clone()
    } else {
        let temp = TempDir::new()?;
        let path = temp.path().to_path_buf();
        _temp_dir = Some(temp);
        path
    };

    println!("Database path: {}", db_path.display());

    // Initialize vector storage
    let mut storage = Storage::readwrite(&db_path);
    storage.ready()?;

    // Create HNSW configuration
    let hnsw_config = HnswConfig {
        dim: benchmark_data.map(|d| d.dimensions).unwrap_or(128),
        m: args.m,
        m_max: args.m * 2,
        ef_construction: args.ef_construction,
        ..Default::default()
    };

    println!("\nHNSW Config: {:?}", hnsw_config);

    // Create navigation cache and index
    let nav_cache = Arc::new(NavigationCache::new());
    let embedding_code: EmbeddingCode = 1; // Single embedding space for benchmark
    let index = HnswIndex::new(embedding_code, hnsw_config.clone(), nav_cache.clone());

    // Prepare random number generator
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Index vectors
    println!("\nIndexing {} vectors...", args.num_vectors);
    let start = Instant::now();

    // First, store all vectors in the Vectors CF
    let txn_db = storage.transaction_db()?;
    let vectors_cf = txn_db
        .cf_handle(Vectors::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("Vectors CF not found"))?;

    let mut vec_id_to_index: HashMap<VecId, usize> = HashMap::new();

    for i in 0..args.num_vectors {
        let vec_id = i as VecId;
        let vector = if let Some(data) = benchmark_data {
            data.base_vectors[i].clone()
        } else {
            generate_random_vector(128, &mut rng)
        };

        // Store vector in Vectors CF
        let key = VectorCfKey(embedding_code, vec_id);
        let value = VectorCfValue(vector.clone());
        txn_db.put_cf(
            &vectors_cf,
            Vectors::key_to_bytes(&key),
            Vectors::value_to_bytes(&value),
        )?;

        vec_id_to_index.insert(vec_id, i);

        // Insert into HNSW index
        index.insert(&storage, vec_id, &vector)?;

        if args.verbose && (i + 1) % 1000 == 0 {
            let elapsed = start.elapsed();
            let rate = (i + 1) as f64 / elapsed.as_secs_f64();
            println!("  Indexed {}/{} vectors ({:.1} vec/s)", i + 1, args.num_vectors, rate);
        }
    }

    let index_time = start.elapsed();
    metrics.num_vectors = args.num_vectors;
    metrics.build_time_secs = index_time.as_secs_f64();
    metrics.build_throughput = args.num_vectors as f64 / index_time.as_secs_f64();

    println!(
        "Indexing complete: {:.2}s ({:.1} vectors/sec)",
        index_time.as_secs_f64(),
        metrics.build_throughput
    );

    // Run queries
    let actual_queries = if let Some(data) = benchmark_data {
        data.num_queries().min(args.num_queries)
    } else {
        args.num_queries
    };

    println!("\nRunning {} queries (k={}, ef={})...", actual_queries, args.k, args.ef);

    let mut total_recall = 0.0;
    let mut search_latencies: Vec<f64> = Vec::with_capacity(actual_queries);

    for i in 0..actual_queries {
        let query = if let Some(data) = benchmark_data {
            data.query_vectors[i].clone()
        } else {
            generate_random_vector(128, &mut rng)
        };

        // HNSW search
        let search_start = Instant::now();
        let results = index.search(&storage, &query, args.k, args.ef)?;
        let search_time = search_start.elapsed();
        search_latencies.push(search_time.as_secs_f64() * 1000.0);

        // Compute recall
        let recall = if let Some(data) = benchmark_data {
            // Convert results to (distance, index) for benchmark validation
            let result_indices: Vec<(f32, usize)> = results
                .iter()
                .filter_map(|(dist, vec_id)| {
                    vec_id_to_index.get(vec_id).map(|idx| (*dist, *idx))
                })
                .collect();
            data.calculate_recall(i, &result_indices, args.k)
        } else {
            // For random data, compute brute-force ground truth
            let vectors: Vec<Vec<f32>> = (0..args.num_vectors)
                .map(|j| {
                    let key = VectorCfKey(embedding_code, j as VecId);
                    let bytes = txn_db
                        .get_cf(&vectors_cf, Vectors::key_to_bytes(&key))
                        .ok()
                        .flatten()
                        .unwrap_or_default();
                    Vectors::value_from_bytes(&bytes)
                        .map(|v| v.0)
                        .unwrap_or_default()
                })
                .collect();

            let gt = brute_force_knn(&query, &vectors, args.k);
            let gt_set: std::collections::HashSet<_> = gt.iter().map(|(_, idx)| *idx).collect();
            let result_set: std::collections::HashSet<_> = results
                .iter()
                .map(|(_, vec_id)| *vec_id as usize)
                .collect();
            let intersection = gt_set.intersection(&result_set).count();
            intersection as f32 / args.k as f32
        };

        total_recall += recall as f64;

        // Debug output for first query
        if args.verbose && i == 0 {
            println!("\n  Query 0 results (top 5):");
            for (j, (dist, vec_id)) in results.iter().take(5).enumerate() {
                let idx = vec_id_to_index.get(vec_id).map(|i| *i).unwrap_or(9999);
                println!("    {}: vec_id={}, index={}, dist={:.4}", j, vec_id, idx, dist);
            }
            if let Some(data) = benchmark_data {
                println!("  Ground truth (top 5):");
                for j in 0..5.min(data.ground_truth[0].len()) {
                    println!("    {}: index={}", j, data.ground_truth[0][j]);
                }
            }
        }

        if (i + 1) % 10 == 0 || i == 0 {
            println!(
                "  Query {}/{}: recall@{}={:.3}, latency={:.2}ms",
                i + 1,
                actual_queries,
                args.k,
                recall,
                search_time.as_secs_f64() * 1000.0
            );
        }
    }

    let avg_recall = total_recall / actual_queries as f64;

    // Finalize metrics
    metrics.num_queries = actual_queries;
    metrics.k = args.k;
    metrics.recall_at_k = avg_recall;
    metrics.set_latency_percentiles(search_latencies);

    // Calculate disk usage
    if let Ok(entries) = std::fs::read_dir(&db_path) {
        let mut total_size = 0u64;
        for entry in entries.flatten() {
            if let Ok(metadata) = entry.metadata() {
                total_size += metadata.len();
            }
        }
        metrics.disk_usage_bytes = total_size;
    }

    Ok(metrics)
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Generate a random vector
fn generate_random_vector(dim: usize, rng: &mut impl Rng) -> Vec<f32> {
    (0..dim).map(|_| rng.gen::<f32>()).collect()
}

/// Brute-force k-NN search
fn brute_force_knn(query: &[f32], vectors: &[Vec<f32>], k: usize) -> Vec<(f32, usize)> {
    let mut distances: Vec<(f32, usize)> = vectors
        .iter()
        .enumerate()
        .map(|(idx, v)| (l2_distance_squared(query, v), idx))
        .collect();

    distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    distances.truncate(k);
    distances
}

/// L2 distance squared
fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum()
}
