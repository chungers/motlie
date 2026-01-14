//! Vector2: Benchmark for Phase 2 HNSW Implementation
//!
//! **⚠️ DEPRECATION NOTICE**: This example will be removed in a future release.
//! For dataset downloads, use `bench_vector download --dataset <name>`.
//! For listing datasets, use `bench_vector datasets`.
//! Full benchmark functionality will be migrated to `bins/bench_vector`.
//!
//! This example demonstrates and validates the new HNSW implementation in
//! `motlie_db::vector` using RocksDB storage with RoaringBitmap edges.
//!
//! # Features
//!
//! - Load SIFT benchmark datasets (SIFT10K, SIFT1M)
//! - Compare new RocksDB-backed HNSW vs old graph-based implementation
//! - Collect detailed performance metrics (build time, QPS, recall, latency)
//! - **Incremental index builds** for large-scale testing
//!
//! # Usage
//!
//! ```bash
//! # Run with synthetic data (temp directory)
//! cargo run --release --example vector2 -- --num-vectors 10000 --num-queries 100 --k 10
//!
//! # Incremental build: start with 100K vectors
//! cargo run --release --example vector2 -- --db-path /data/bench --num-vectors 100000 --cosine --rabitq-cached --bits-per-dim 4
//!
//! # Incremental build: extend to 500K vectors (adds 400K to existing index)
//! cargo run --release --example vector2 -- --db-path /data/bench --num-vectors 500000 --cosine --rabitq-cached --bits-per-dim 4
//!
//! # Query-only mode (skip indexing, just run queries on existing index)
//! cargo run --release --example vector2 -- --db-path /data/bench --num-vectors 500000 --query-only --num-queries 1000
//!
//! # Fresh start (delete existing index)
//! cargo run --release --example vector2 -- --db-path /data/bench --num-vectors 100000 --fresh
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

use benchmark::{
    BenchmarkDataset, BenchmarkMetrics, DatasetConfig, DatasetName,
    compute_ground_truth_cosine, cosine_distance_f32, normalize_vector, normalize_vectors,
};
use motlie_db::vector::benchmark::{BenchmarkMetadata, GroundTruthCache};
use motlie_db::rocksdb::{BlockCacheConfig, ColumnFamily};
use motlie_db::vector::{
    hnsw, BinaryCodeCache, BinaryCodeCfKey, BinaryCodeCfValue, BinaryCodes,
    Distance, EmbeddingCode, NavigationCache, RaBitQ, Storage, VecId, VectorCfKey, VectorCfValue,
    Vectors,
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

    /// Block cache size in MB (default: 256MB)
    #[arg(long, default_value = "256")]
    cache_size_mb: usize,

    /// Enable RaBitQ with in-memory cached binary codes
    /// Uses Hamming filtering at layer 0 + exact L2 re-ranking
    #[arg(long)]
    rabitq_cached: bool,

    /// RaBitQ re-rank factor (how many candidates to fetch before re-ranking)
    #[arg(long, default_value = "4")]
    rerank_factor: usize,

    /// RaBitQ bits per dimension (1, 2, or 4)
    #[arg(long, default_value = "1")]
    bits_per_dim: u8,

    /// Use Cosine distance instead of L2
    /// Normalizes vectors and computes cosine ground truth
    /// Enables optimal RaBitQ performance (Hamming ≈ angular)
    #[arg(long)]
    cosine: bool,

    /// Use ADC (Asymmetric Distance Computation) instead of symmetric Hamming.
    /// Requires --rabitq-cached. Query stays float32, uses weighted dot product.
    /// This should fix multi-bit recall issues (see RABITQ.md).
    #[arg(long)]
    adc: bool,

    /// Use SIMD-optimized dot products (default: true).
    /// Pass --no-simd-dot to use scalar implementation for comparison.
    #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
    simd_dot: bool,

    /// Force fresh start (delete existing index and start over)
    /// Without this flag, incremental builds will add vectors to existing index
    #[arg(long)]
    fresh: bool,

    /// Query-only mode (skip indexing, just run queries on existing index)
    /// Requires --db-path with existing index
    #[arg(long)]
    query_only: bool,

    /// Checkpoint interval (save metadata every N vectors during indexing)
    #[arg(long, default_value = "10000")]
    checkpoint_interval: usize,

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

    // Validate ADC mode requirements
    if args.adc && !args.rabitq_cached {
        anyhow::bail!("--adc requires --rabitq-cached");
    }
    if args.adc && !args.cosine {
        anyhow::bail!("--adc requires --cosine (ADC is designed for cosine/angular distance)");
    }

    // Validate query-only mode
    if args.query_only && args.db_path.is_none() {
        anyhow::bail!("--query-only requires --db-path with existing index");
    }
    if args.query_only && args.fresh {
        anyhow::bail!("--query-only and --fresh are mutually exclusive");
    }

    println!("=== Vector2: Phase 2 HNSW Benchmark ===");
    println!();

    // Parse dataset name
    let dataset_name = DatasetName::from_str(&args.dataset);
    let dataset_str = dataset_name
        .map(|d| d.to_string())
        .unwrap_or_else(|| "random".to_string());

    let distance_str = if args.cosine { "Cosine" } else { "L2" };
    println!("Configuration:");
    println!("  Dataset:         {}", dataset_str);
    println!("  Vectors:         {}", args.num_vectors);
    println!("  Queries:         {}", args.num_queries);
    println!("  K:               {}", args.k);
    println!("  ef (search):     {}", args.ef);
    println!("  M:               {}", args.m);
    println!("  ef_construction: {}", args.ef_construction);
    println!("  Cache Size:      {} MB", args.cache_size_mb);
    println!("  Distance:        {}", distance_str);
    let search_mode_str = if args.rabitq_cached {
        if args.adc {
            "rabitq-ADC (asymmetric distance + Cosine rerank)"
        } else if args.cosine {
            "rabitq-cached (in-memory Hamming nav + Cosine rerank)"
        } else {
            "rabitq-cached (in-memory Hamming nav + L2 rerank)"
        }
    } else if args.cosine {
        "standard (Cosine)"
    } else {
        "standard (L2)"
    };
    println!("  Search Mode:     {}", search_mode_str);
    if args.rabitq_cached {
        println!("  Rerank Factor:   {}", args.rerank_factor);
        println!("  Bits/Dim:        {}", args.bits_per_dim);
        if args.adc {
            println!("  ADC Mode:        enabled (query stays float32)");
        }
        println!("  SIMD Dot:        {}", if args.simd_dot { "enabled" } else { "disabled (scalar)" });
    }
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

    let distance_mode = if args.cosine { "Cosine" } else { "L2" };
    let mut metrics = BenchmarkMetrics::new(&dataset_str, "HNSW2-RocksDB");
    metrics.search_mode = if args.rabitq_cached {
        format!("RaBitQ-cached({}+rerank={}x,bits={})", distance_mode, args.rerank_factor, args.bits_per_dim)
    } else {
        format!("standard({})", distance_mode)
    };

    // Prepare normalized vectors for cosine mode
    let (base_vectors, query_vectors, cosine_ground_truth) = if args.cosine {
        if let Some(data) = benchmark_data {
            println!("Normalizing vectors for Cosine distance...");
            let norm_base = normalize_vectors(&data.base_vectors);
            let norm_query = normalize_vectors(&data.query_vectors);
            let gt = compute_ground_truth_cosine(&norm_base, &norm_query, 100);
            (Some(norm_base), Some(norm_query), Some(gt))
        } else {
            (None, None, None)
        }
    } else {
        (None, None, None)
    };

    // Create temp directory or use specified path
    let _temp_dir: Option<TempDir>;
    let db_path = if let Some(ref path) = args.db_path {
        path.clone()
    } else {
        let temp = TempDir::new()?;
        let path = temp.path().to_path_buf();
        _temp_dir = Some(temp);
        path
    };

    // Determine distance string for metadata
    let distance_str = if args.cosine { "cosine" } else { "l2" };
    let dim = benchmark_data.map(|d| d.dimensions).unwrap_or(128);

    // Handle incremental builds via metadata
    let mut bench_metadata = if args.fresh {
        // Fresh start: delete existing database
        if db_path.exists() {
            println!("Deleting existing database for fresh start...");
            std::fs::remove_dir_all(&db_path)?;
        }
        BenchmarkMetadata::new(
            dim,
            distance_str,
            args.rabitq_cached,
            args.bits_per_dim,
            args.adc,
            args.m,
            args.ef_construction,
            &dataset_str,
        )
    } else if BenchmarkMetadata::exists(&db_path) {
        // Incremental build: load existing metadata
        let existing = BenchmarkMetadata::load(&db_path)?;
        println!(
            "Found existing index with {} vectors (created {})",
            existing.num_vectors, existing.last_updated
        );

        // Validate configuration matches
        existing.validate_config(
            dim,
            distance_str,
            args.rabitq_cached,
            args.bits_per_dim,
            args.adc,
            args.m,
            args.ef_construction,
            &dataset_str,
        )?;

        if args.query_only {
            println!("Query-only mode: skipping indexing");
        } else if existing.num_vectors >= args.num_vectors {
            println!(
                "Index already has {} vectors (requested {}), skipping indexing",
                existing.num_vectors, args.num_vectors
            );
        } else {
            println!(
                "Will add {} vectors (from {} to {})",
                args.num_vectors - existing.num_vectors,
                existing.num_vectors,
                args.num_vectors
            );
        }

        existing
    } else {
        // New index
        BenchmarkMetadata::new(
            dim,
            distance_str,
            args.rabitq_cached,
            args.bits_per_dim,
            args.adc,
            args.m,
            args.ef_construction,
            &dataset_str,
        )
    };

    let starting_vectors = bench_metadata.num_vectors;

    println!("Database path: {}", db_path.display());

    // Initialize vector storage with configurable cache size
    let cache_config = BlockCacheConfig::with_cache_size(args.cache_size_mb * 1024 * 1024);
    let mut storage = Storage::readwrite(&db_path).with_block_cache_config(cache_config);
    storage.ready()?;

    // Create HNSW configuration
    let hnsw_config = hnsw::Config {
        dim: bench_metadata.dim,
        m: args.m,
        m_max: args.m * 2,
        ef_construction: args.ef_construction,
        ..Default::default()
    };

    println!("\nHNSW Config: {:?}", hnsw_config);

    // Create navigation cache and index
    let nav_cache = Arc::new(NavigationCache::new());
    let embedding_code: EmbeddingCode = 1; // Single embedding space for benchmark
    let distance = if args.cosine { Distance::Cosine } else { Distance::L2 };
    let index = hnsw::Index::new(embedding_code, distance, hnsw_config.clone(), nav_cache.clone());

    // Prepare random number generator with seed from metadata (for reproducibility)
    let mut rng = ChaCha8Rng::seed_from_u64(bench_metadata.vector_seed);

    // Skip vectors that are already indexed (advance RNG state)
    if starting_vectors > 0 && benchmark_data.is_none() {
        println!("Skipping RNG state for {} already-indexed vectors...", starting_vectors);
        for _ in 0..starting_vectors {
            // Generate and discard vector to advance RNG state
            for _ in 0..bench_metadata.dim {
                let _: f32 = rng.gen();
            }
        }
    }

    // Create RaBitQ encoder for binary code generation (if rabitq_cached enabled)
    let build_encoder = if args.rabitq_cached {
        let dim = benchmark_data.map(|d| d.dimensions).unwrap_or(128);
        Some(RaBitQ::with_options(dim, args.bits_per_dim, 42, args.simd_dot))
    } else {
        None
    };

    // Create in-memory binary code cache (if rabitq_cached enabled)
    let code_cache = if args.rabitq_cached {
        Some(BinaryCodeCache::new())
    } else {
        None
    };

    // Get database handles
    let txn_db = storage.transaction_db()?;
    let vectors_cf = txn_db
        .cf_handle(Vectors::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("Vectors CF not found"))?;

    // Get BinaryCodes CF handle if RaBitQ-cached enabled
    let binary_codes_cf = if args.rabitq_cached {
        Some(
            txn_db
                .cf_handle(BinaryCodes::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!("BinaryCodes CF not found"))?,
        )
    } else {
        None
    };

    let mut vec_id_to_index: HashMap<VecId, usize> = HashMap::new();

    // Build vec_id_to_index mapping for already-indexed vectors
    for i in 0..starting_vectors {
        vec_id_to_index.insert(i as VecId, i);
    }

    // Load existing binary codes with ADC corrections into cache (for query phase)
    if args.rabitq_cached && starting_vectors > 0 {
        println!("Loading {} existing binary codes into cache...", starting_vectors);
        if let Some(ref cache) = code_cache {
            for i in 0..starting_vectors {
                let vec_id = i as VecId;
                let bc_key = BinaryCodeCfKey(embedding_code, vec_id);
                if let Ok(Some(bytes)) = txn_db.get_cf(binary_codes_cf.as_ref().unwrap(), BinaryCodes::key_to_bytes(&bc_key)) {
                    if let Ok(bc_value) = BinaryCodes::value_from_bytes(&bytes) {
                        // Cache now stores both code and correction
                        cache.put(embedding_code, vec_id, bc_value.code, bc_value.correction);
                    }
                }
            }
        }
    }

    // Determine what to index
    let vectors_to_index = if args.query_only || starting_vectors >= args.num_vectors {
        0
    } else {
        args.num_vectors - starting_vectors
    };

    // Index new vectors (incremental)
    let start = Instant::now();
    if vectors_to_index > 0 {
        println!(
            "\nIndexing {} vectors (from {} to {})...",
            vectors_to_index, starting_vectors, args.num_vectors
        );

        for i in starting_vectors..args.num_vectors {
            let vec_id = i as VecId;
            let vector = if args.cosine {
                // Use normalized vectors for cosine mode
                if let Some(ref norm_base) = base_vectors {
                    norm_base[i].clone()
                } else {
                    let v = generate_random_vector(bench_metadata.dim, &mut rng);
                    normalize_vector(&v)
                }
            } else if let Some(data) = benchmark_data {
                data.base_vectors[i].clone()
            } else {
                generate_random_vector(bench_metadata.dim, &mut rng)
            };

            // Store vector in Vectors CF
            let key = VectorCfKey(embedding_code, vec_id);
            let value = VectorCfValue(vector.clone());
            txn_db.put_cf(
                &vectors_cf,
                Vectors::key_to_bytes(&key),
                Vectors::value_to_bytes(&value),
            )?;

            // Store binary code with ADC correction if RaBitQ enabled
            if let (Some(ref encoder), Some(ref cf)) = (&build_encoder, &binary_codes_cf) {
                // Always use encode_with_correction for ADC support
                let (code, correction) = encoder.encode_with_correction(&vector);

                // Store in RocksDB
                let bc_key = BinaryCodeCfKey(embedding_code, vec_id);
                let bc_value = BinaryCodeCfValue { code: code.clone(), correction };
                txn_db.put_cf(
                    cf,
                    BinaryCodes::key_to_bytes(&bc_key),
                    BinaryCodes::value_to_bytes(&bc_value),
                )?;

                // Also populate in-memory cache if rabitq_cached mode
                if let Some(ref cache) = code_cache {
                    cache.put(embedding_code, vec_id, code, correction);
                }
            }

            vec_id_to_index.insert(vec_id, i);

            // Insert into HNSW index
            index.insert(&storage, vec_id, &vector)?;

            // Progress and checkpointing
            let vectors_indexed = i - starting_vectors + 1;
            if vectors_indexed % args.checkpoint_interval == 0 {
                bench_metadata.num_vectors = i + 1;
                bench_metadata.checkpoint(&db_path)?;

                let elapsed = start.elapsed();
                let rate = vectors_indexed as f64 / elapsed.as_secs_f64();
                println!(
                    "  Checkpoint: {}/{} vectors indexed ({:.1} vec/s)",
                    i + 1,
                    args.num_vectors,
                    rate
                );
            } else if args.verbose && vectors_indexed % 1000 == 0 {
                let elapsed = start.elapsed();
                let rate = vectors_indexed as f64 / elapsed.as_secs_f64();
                println!(
                    "  Indexed {}/{} vectors ({:.1} vec/s)",
                    i + 1,
                    args.num_vectors,
                    rate
                );
            }
        }

        // Final metadata update
        bench_metadata.num_vectors = args.num_vectors;
        bench_metadata.checkpoint(&db_path)?;
    } else if args.query_only {
        println!("\nQuery-only mode: skipping indexing");
    } else {
        println!("\nIndex already at target size, skipping indexing");
    }

    let index_time = start.elapsed();
    metrics.num_vectors = args.num_vectors;
    metrics.cache_size_mb = args.cache_size_mb;

    if vectors_to_index > 0 {
        metrics.build_time_secs = index_time.as_secs_f64();
        metrics.build_throughput = vectors_to_index as f64 / index_time.as_secs_f64();
        println!(
            "Indexing complete: {:.2}s ({:.1} vectors/sec)",
            index_time.as_secs_f64(),
            metrics.build_throughput
        );
    } else {
        metrics.build_time_secs = 0.0;
        metrics.build_throughput = 0.0;
    }

    // Run queries
    let actual_queries = if let Some(data) = benchmark_data {
        data.num_queries().min(args.num_queries)
    } else {
        args.num_queries
    };

    // Use the same encoder for search (if RaBitQ enabled)
    let rabitq_encoder = build_encoder;

    // Create separate RNG for query generation (independent of vector generation)
    let mut query_rng = ChaCha8Rng::seed_from_u64(bench_metadata.query_seed);

    let search_mode = if args.adc {
        format!("RaBitQ-ADC (asymmetric, rerank={}x, bits={})", args.rerank_factor, args.bits_per_dim)
    } else if args.rabitq_cached {
        format!("RaBitQ-cached (Hamming, rerank={}x, bits={})", args.rerank_factor, args.bits_per_dim)
    } else {
        "standard".to_string()
    };

    // Print cache stats if rabitq_cached enabled
    if let Some(ref cache) = code_cache {
        let (count, bytes) = cache.stats();
        println!("Binary Code Cache: {} entries, {} KB", count, bytes / 1024);
    }
    println!("\nRunning {} queries (k={}, ef={}, mode={})...", actual_queries, args.k, args.ef, search_mode);

    let mut total_recall = 0.0;
    let mut search_latencies: Vec<f64> = Vec::with_capacity(actual_queries);

    for i in 0..actual_queries {
        let query = if args.cosine {
            // Use normalized queries for cosine mode
            if let Some(ref norm_query) = query_vectors {
                norm_query[i].clone()
            } else {
                let v = generate_random_vector(bench_metadata.dim, &mut query_rng);
                normalize_vector(&v)
            }
        } else if let Some(data) = benchmark_data {
            data.query_vectors[i].clone()
        } else {
            generate_random_vector(bench_metadata.dim, &mut query_rng)
        };

        // HNSW search (standard, rabitq_cached Hamming, or ADC)
        let search_start = Instant::now();
        let results = if args.adc {
            // ADC brute-force search for benchmarking
            // (No HNSW navigation yet - this measures pure ADC distance quality)
            let encoder = rabitq_encoder.as_ref().expect("Encoder required for ADC mode");
            let cache = code_cache.as_ref().expect("Code cache required for ADC mode");

            // Rotate query once (NOT binarized - this is the key ADC difference)
            let query_rotated = encoder.rotate_query(&query);
            let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();

            // Compute ADC distance to all vectors
            let mut adc_candidates: Vec<(f32, VecId)> = Vec::with_capacity(args.num_vectors);
            for vec_id in 0..args.num_vectors as VecId {
                // Cache now stores (code, correction) tuples directly
                if let Some((code, correction)) = cache.get(embedding_code, vec_id) {
                    let adc_dist = encoder.adc_distance(&query_rotated, query_norm, &code, &correction);
                    adc_candidates.push((adc_dist, vec_id));
                }
            }

            // Sort by ADC distance (lower = more similar)
            adc_candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            // Take top k * rerank_factor candidates for re-ranking
            let rerank_count = args.k * args.rerank_factor;
            adc_candidates.truncate(rerank_count);

            // Re-rank with exact cosine distance
            let mut reranked: Vec<(f32, VecId)> = adc_candidates
                .iter()
                .filter_map(|(_, vec_id)| {
                    let key = VectorCfKey(embedding_code, *vec_id);
                    let bytes = txn_db
                        .get_cf(&vectors_cf, Vectors::key_to_bytes(&key))
                        .ok()
                        .flatten()?;
                    let vector = Vectors::value_from_bytes(&bytes).ok()?.0;
                    let exact_dist = cosine_distance_f32(&query, &vector);
                    Some((exact_dist, *vec_id))
                })
                .collect();

            reranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            reranked.truncate(args.k);
            reranked
        } else if args.rabitq_cached {
            let encoder = rabitq_encoder.as_ref().expect("Encoder required for rabitq_cached mode");
            let cache = code_cache.as_ref().expect("Code cache required for rabitq_cached mode");
            index.search_with_rabitq_cached(&storage, &query, encoder, cache, args.k, args.ef, args.rerank_factor)?
        } else {
            index.search(&storage, &query, args.k, args.ef)?
        };
        let search_time = search_start.elapsed();
        search_latencies.push(search_time.as_secs_f64() * 1000.0);

        // Compute recall
        let recall = if args.cosine && cosine_ground_truth.is_some() {
            // Use cosine ground truth for cosine mode
            let gt = cosine_ground_truth.as_ref().unwrap();
            let gt_k: std::collections::HashSet<_> = gt[i].iter().take(args.k).cloned().collect();
            let result_set: std::collections::HashSet<_> = results
                .iter()
                .filter_map(|(_, vec_id)| vec_id_to_index.get(vec_id).cloned())
                .collect();
            let intersection = gt_k.intersection(&result_set).count();
            intersection as f32 / args.k as f32
        } else if let Some(data) = benchmark_data {
            // Convert results to (distance, index) for benchmark validation (L2)
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

            let gt = if args.cosine {
                brute_force_knn_cosine(&query, &vectors, args.k)
            } else {
                brute_force_knn(&query, &vectors, args.k)
            };
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

/// Brute-force k-NN search using cosine distance
fn brute_force_knn_cosine(query: &[f32], vectors: &[Vec<f32>], k: usize) -> Vec<(f32, usize)> {
    let mut distances: Vec<(f32, usize)> = vectors
        .iter()
        .enumerate()
        .map(|(idx, v)| (cosine_distance_f32(query, v), idx))
        .collect();

    distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    distances.truncate(k);
    distances
}
