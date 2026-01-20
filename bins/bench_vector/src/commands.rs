//! CLI command implementations for vector benchmarks.
//!
//! Provides full benchmark functionality:
//! - Download: Fetch benchmark datasets
//! - Index: Build HNSW index (incremental support)
//! - Query: Run queries on existing index
//! - Sweep: Parameter sweep over ef_search, bits, rerank

use anyhow::{Context, Result};
use clap::Parser;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use motlie_db::vector::benchmark::{
    self, compute_recall, compute_rotated_variance, print_pareto_frontier, print_rotation_stats,
    BenchmarkMetadata, GistDataset, LaionDataset, LatencyStats, ParetoInput, RabitqExperimentResult,
    RandomDataset, SiftDataset, save_rabitq_results_csv, GIST_QUERIES,
};
use motlie_db::vector::{
    hnsw, BinaryCodeCache, Distance, EmbeddingCode, NavigationCache, RaBitQ,
    Storage, VecId, VectorCfKey, VectorElementType, Vectors,
};
use motlie_db::rocksdb::ColumnFamily;

// ============================================================================
// Download Command
// ============================================================================

#[derive(Parser)]
pub struct DownloadArgs {
    /// Dataset to download: laion, sift, gist, cohere, glove
    #[arg(long)]
    pub dataset: String,

    /// Directory to store downloaded data
    #[arg(long, default_value = "./data")]
    pub data_dir: PathBuf,
}

pub async fn download(args: DownloadArgs) -> Result<()> {
    std::fs::create_dir_all(&args.data_dir)
        .context("Failed to create data directory")?;

    match args.dataset.to_lowercase().as_str() {
        "laion" => {
            println!("Downloading LAION-CLIP embeddings...");
            benchmark::LaionDataset::download(&args.data_dir)?;
        }
        "sift" => {
            println!("Downloading SIFT-1M dataset...");
            benchmark::SiftDataset::download(&args.data_dir)?;
        }
        "gist" => {
            println!("GIST-960 dataset requires manual download.");
            println!("Download from: ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz");
            println!("Extract to: {:?}", args.data_dir);
            println!("\nExpected files:");
            println!("  - gist_base.fvecs (1M × 960D)");
            println!("  - gist_query.fvecs (1K × 960D)");
            println!("  - gist_groundtruth.ivecs");
            return Ok(());
        }
        #[cfg(feature = "parquet")]
        "cohere" | "cohere-wiki" => {
            println!("Downloading Cohere Wikipedia embeddings...");
            benchmark::CohereWikipediaDataset::download(&args.data_dir)?;
        }
        #[cfg(feature = "hdf5")]
        "glove" | "glove-100" => {
            println!("Downloading GloVe-100 angular dataset...");
            benchmark::GloveDataset::download(&args.data_dir)?;
        }
        _ => {
            anyhow::bail!(
                "Unknown dataset: {}. Run 'bench_vector datasets' for available options.",
                args.dataset
            );
        }
    }

    println!("Download complete: {:?}", args.data_dir);
    Ok(())
}

// ============================================================================
// Index Command
// ============================================================================

#[derive(Parser)]
pub struct IndexArgs {
    /// Dataset: laion, sift, gist, random
    #[arg(long)]
    pub dataset: String,

    /// Number of vectors to index
    #[arg(long)]
    pub num_vectors: usize,

    /// Database path for RocksDB storage
    #[arg(long)]
    pub db_path: PathBuf,

    /// Data directory (for dataset files, not used for 'random')
    #[arg(long, default_value = "./data")]
    pub data_dir: PathBuf,

    /// HNSW M parameter (graph connectivity)
    #[arg(long, default_value = "16")]
    pub m: usize,

    /// HNSW ef_construction (build quality)
    #[arg(long, default_value = "200")]
    pub ef_construction: usize,

    /// Delete existing index and start fresh
    #[arg(long)]
    pub fresh: bool,

    /// Use cosine distance (default: auto-detect from dataset)
    #[arg(long)]
    pub cosine: bool,

    /// Use L2/Euclidean distance
    #[arg(long)]
    pub l2: bool,

    /// Vector dimension (only for 'random' dataset)
    #[arg(long, default_value = "1024")]
    pub dim: usize,

    /// Random seed (only for 'random' dataset)
    #[arg(long, default_value = "42")]
    pub seed: u64,

    /// Batch size for inserts (vectors per transaction)
    #[arg(long, default_value = "1000")]
    pub batch_size: usize,

    /// Use async inserts (deferred HNSW graph construction)
    #[arg(long, default_value = "false")]
    pub r#async: bool,

    /// Number of async workers (only used with --async)
    #[arg(long, default_value = "1")]
    pub async_workers: usize,

    /// Output results to JSON file
    #[arg(long)]
    pub output: Option<PathBuf>,

    /// Progress reporting interval in seconds
    #[arg(long, default_value = "10")]
    pub progress_interval: u64,
}

pub async fn index(args: IndexArgs) -> Result<()> {
    use motlie_db::vector::benchmark::scale::get_rss_bytes;
    use motlie_db::vector::{AsyncGraphUpdater, AsyncUpdaterConfig, EmbeddingBuilder, Processor};
    use motlie_db::Id;
    use std::time::Duration;

    println!("=== Vector Index Build ===");
    println!("Dataset: {}", args.dataset);
    println!("Vectors: {}", args.num_vectors);
    println!("Database: {:?}", args.db_path);
    println!("HNSW: M={}, ef_construction={}", args.m, args.ef_construction);
    println!("Batch size: {}", args.batch_size);
    if args.r#async {
        println!("Insert mode: ASYNC ({} workers)", args.async_workers);
    } else {
        println!("Insert mode: SYNC (immediate indexing)");
    }

    // Determine distance metric
    let distance = if args.cosine {
        Distance::Cosine
    } else if args.l2 {
        Distance::L2
    } else {
        // Auto-detect from dataset
        match args.dataset.to_lowercase().as_str() {
            "laion" | "cohere" | "random" => Distance::Cosine,
            "sift" | "gist" => Distance::L2,
            _ => Distance::Cosine,
        }
    };
    println!("Distance: {:?}", distance);

    // Load dataset
    let (vectors, dim) = load_dataset_vectors_with_random(
        &args.dataset,
        &args.data_dir,
        args.num_vectors,
        args.dim,
        args.seed,
    )?;
    println!("Loaded {} vectors, dim={}", vectors.len(), dim);

    // Handle fresh start
    if args.fresh && args.db_path.exists() {
        println!("Removing existing database...");
        std::fs::remove_dir_all(&args.db_path)?;
    }

    // Check for existing metadata (incremental build)
    let existing_count = if BenchmarkMetadata::exists(&args.db_path) {
        let meta = BenchmarkMetadata::load(&args.db_path)?;
        println!("Found existing index with {} vectors", meta.num_vectors);
        meta.num_vectors
    } else {
        0
    };

    if existing_count >= args.num_vectors {
        println!("Index already has {} vectors, nothing to do.", existing_count);
        return Ok(());
    }

    // Create database directory
    std::fs::create_dir_all(&args.db_path)?;

    // Initialize storage with Processor API
    let mut storage = Storage::readwrite(&args.db_path);
    storage.ready()?;
    let storage = Arc::new(storage);

    // Register embedding
    let txn_db = storage.transaction_db().context("Failed to get transaction DB")?;
    let embedding = storage
        .cache()
        .register(
            EmbeddingBuilder::new("bench", dim as u32, distance)
                .with_hnsw_m(args.m as u16)
                .with_hnsw_ef_construction(args.ef_construction as u16),
            &txn_db,
        )
        .context("Failed to register embedding")?;

    // Create processor
    let registry = storage.cache().clone();
    let processor = Processor::new(storage.clone(), registry);

    // Start async updater if async mode enabled
    let async_updater: Option<Arc<AsyncGraphUpdater>> = if args.r#async {
        let async_config = AsyncUpdaterConfig::new()
            .with_num_workers(args.async_workers)
            .with_batch_size(100)
            .with_ef_construction(args.ef_construction)
            .with_process_on_startup(false);

        let registry = storage.cache().clone();
        let nav_cache = Arc::new(NavigationCache::new());

        Some(Arc::new(AsyncGraphUpdater::start(
            storage.clone(),
            registry,
            nav_cache,
            async_config,
        )))
    } else {
        None
    };

    // Insert vectors using batch API
    let start = Instant::now();
    let vectors_to_add = &vectors[existing_count..];
    let total = vectors_to_add.len();
    let mut inserted = 0usize;
    let mut errors = 0usize;
    let mut peak_rss: u64 = 0;
    let mut last_progress = Instant::now();
    let progress_interval = Duration::from_secs(args.progress_interval);

    println!("\nInserting {} vectors (starting from {})...", total, existing_count);
    println!();

    for batch_start in (0..total).step_by(args.batch_size) {
        let batch_end = (batch_start + args.batch_size).min(total);
        let batch: Vec<(Id, Vec<f32>)> = vectors_to_add[batch_start..batch_end]
            .iter()
            .map(|v| (Id::new(), v.clone()))
            .collect();
        let batch_len = batch.len();

        match processor.insert_batch(&embedding, &batch, !args.r#async) {
            Ok(_) => {
                inserted += batch_len;
            }
            Err(e) => {
                tracing::warn!("Batch insert failed: {}", e);
                errors += batch_len;
            }
        }

        // Progress reporting with ETA
        if last_progress.elapsed() >= progress_interval {
            let rss = get_rss_bytes();
            peak_rss = peak_rss.max(rss);
            let elapsed = start.elapsed().as_secs_f64();
            let rate = inserted as f64 / elapsed;
            let remaining = total - inserted;
            let eta_secs = if rate > 0.0 { remaining as f64 / rate } else { 0.0 };
            let eta_str = format_eta(eta_secs);

            // Async metrics if available
            if let Some(ref updater) = async_updater {
                let pending = updater.pending_queue_size();
                print!(
                    "\r[{:>6.2}%] {}/{} vectors | {:.1} vec/s | Pending: {} | Errors: {} | ETA: {}    ",
                    (inserted as f64 / total as f64) * 100.0,
                    inserted, total, rate, pending, errors, eta_str
                );
            } else {
                print!(
                    "\r[{:>6.2}%] {}/{} vectors | {:.1} vec/s | Errors: {} | ETA: {}    ",
                    (inserted as f64 / total as f64) * 100.0,
                    inserted, total, rate, errors, eta_str
                );
            }
            std::io::stdout().flush().ok();
            last_progress = Instant::now();
        }
    }

    // Wait for async workers if needed
    if let Some(updater) = async_updater {
        println!("\n\nWaiting for async workers to drain pending queue...");
        loop {
            let pending = updater.pending_queue_size();
            if pending == 0 {
                break;
            }
            println!(
                "  Pending: {} vectors, processed: {} items",
                pending,
                updater.items_processed()
            );
            std::thread::sleep(Duration::from_secs(2));
        }
        let items_processed = updater.items_processed();
        println!("Async workers drained ({} items processed)", items_processed);

        match Arc::try_unwrap(updater) {
            Ok(owned_updater) => owned_updater.shutdown(),
            Err(_) => println!("Note: Async workers still shutting down in background"),
        }
    }

    let build_time = start.elapsed().as_secs_f64();
    let final_rss = get_rss_bytes();
    peak_rss = peak_rss.max(final_rss);

    println!(
        "\n\n=== Index Build Complete ===\n\
         Vectors inserted: {}\n\
         Errors: {}\n\
         Duration: {:.2}s\n\
         Throughput: {:.1} vec/s\n\
         Peak RSS: {:.2} MB",
        inserted, errors, build_time,
        inserted as f64 / build_time,
        peak_rss as f64 / 1_000_000.0
    );

    // Save metadata
    let mut meta = BenchmarkMetadata::new(
        dim,
        &format!("{:?}", distance).to_lowercase(),
        false,
        0,
        false,
        args.m,
        args.ef_construction,
        &args.dataset,
    );
    meta.num_vectors = existing_count + inserted;
    meta.vector_seed = args.seed;
    meta.checkpoint(&args.db_path)?;
    println!("Metadata saved: {:?}", BenchmarkMetadata::path(&args.db_path));

    // Save JSON output if requested
    if let Some(output_path) = args.output {
        let json = serde_json::json!({
            "command": "index",
            "config": {
                "dataset": args.dataset,
                "num_vectors": args.num_vectors,
                "dim": dim,
                "hnsw_m": args.m,
                "hnsw_ef_construction": args.ef_construction,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "distance": format!("{:?}", distance).to_lowercase(),
                "async": args.r#async,
            },
            "results": {
                "vectors_inserted": inserted,
                "errors": errors,
                "duration_secs": build_time,
                "throughput_vec_per_sec": inserted as f64 / build_time,
                "peak_rss_bytes": peak_rss,
            }
        });

        std::fs::write(&output_path, serde_json::to_string_pretty(&json)?)
            .context("Failed to write output file")?;
        println!("Results saved to: {:?}", output_path);
    }

    Ok(())
}

/// Format ETA as human-readable string
fn format_eta(secs: f64) -> String {
    if secs > 86400.0 * 365.0 {
        "N/A".to_string()
    } else if secs > 3600.0 {
        format!("{}h{}m", (secs / 3600.0) as u64, ((secs % 3600.0) / 60.0) as u64)
    } else if secs > 60.0 {
        format!("{}m{}s", (secs / 60.0) as u64, (secs % 60.0) as u64)
    } else {
        format!("{}s", secs as u64)
    }
}

// ============================================================================
// Query Command
// ============================================================================

#[derive(Parser)]
pub struct QueryArgs {
    /// Database path
    #[arg(long)]
    pub db_path: PathBuf,

    /// Dataset for queries (to load test vectors). Use 'random' to generate from seed.
    #[arg(long)]
    pub dataset: String,

    /// Data directory (for dataset files, not used for 'random')
    #[arg(long, default_value = "./data")]
    pub data_dir: PathBuf,

    /// Number of results per query
    #[arg(long, default_value = "10")]
    pub k: usize,

    /// ef_search parameter for HNSW
    #[arg(long, default_value = "100")]
    pub ef_search: usize,

    /// Number of queries to run
    #[arg(long, default_value = "1000")]
    pub num_queries: usize,

    /// Random seed for query generation (only for 'random' dataset).
    /// If not specified, uses vector_seed + 1000000 from metadata.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Number of queries to sample for recall computation (0 uses all queries)
    #[arg(long, default_value = "0")]
    pub recall_sample_size: usize,

    /// Output results to JSON file
    #[arg(long)]
    pub output: Option<PathBuf>,
}

pub async fn query(args: QueryArgs) -> Result<()> {
    use motlie_db::vector::benchmark::scale::{get_rss_bytes, StreamingVectorGenerator};

    println!("=== Vector Search ===");
    println!("Database: {:?}", args.db_path);
    println!("Dataset: {}", args.dataset);
    println!("k={}, ef_search={}, num_queries={}", args.k, args.ef_search, args.num_queries);

    // Load metadata
    if !BenchmarkMetadata::exists(&args.db_path) {
        anyhow::bail!("No index found at {:?}. Run 'index' first.", args.db_path);
    }
    let meta = BenchmarkMetadata::load(&args.db_path)?;
    println!("Index: {} vectors, dim={}, distance={}", meta.num_vectors, meta.dim, meta.distance);

    // Parse distance
    let distance = match meta.distance.as_str() {
        "cosine" => Distance::Cosine,
        "l2" => Distance::L2,
        _ => Distance::Cosine,
    };

    // Load or generate queries and database vectors for ground truth
    let (db_vectors, queries, dim) = if args.dataset.to_lowercase() == "random" {
        // For random datasets, regenerate vectors from seed for ground truth
        let query_seed = args.seed.unwrap_or(meta.vector_seed + 1_000_000);
        println!("Regenerating {} vectors from seed {} for ground truth...", meta.num_vectors, meta.vector_seed);

        // Regenerate database vectors
        let mut db_gen = StreamingVectorGenerator::new_with_distance(
            meta.dim,
            meta.num_vectors,
            meta.vector_seed,
            distance,
        );
        let mut db_vectors: Vec<Vec<f32>> = Vec::with_capacity(meta.num_vectors);
        while let Some(batch) = db_gen.next_batch(10_000) {
            db_vectors.extend(batch);
            if db_vectors.len() % 100_000 == 0 {
                print!("\r  Regenerated {}/{} vectors...", db_vectors.len(), meta.num_vectors);
                std::io::stdout().flush().ok();
            }
        }
        println!("\r  Regenerated {} vectors.          ", db_vectors.len());

        // Generate query vectors
        println!("Generating {} query vectors from seed {}...", args.num_queries, query_seed);
        let mut query_gen = StreamingVectorGenerator::new_with_distance(
            meta.dim,
            args.num_queries,
            query_seed,
            distance,
        );
        let mut queries: Vec<Vec<f32>> = Vec::with_capacity(args.num_queries);
        while let Some(batch) = query_gen.next_batch(1000) {
            queries.extend(batch);
        }

        (db_vectors, queries, meta.dim)
    } else {
        // Load from dataset files
        load_dataset_for_query(
            &args.dataset,
            &args.data_dir,
            meta.num_vectors,
            args.num_queries,
        )?
    };
    println!("Loaded {} queries, dim={}", queries.len(), dim);

    // Compute ground truth (optionally sample for large datasets)
    let recall_queries = if args.recall_sample_size > 0 && args.recall_sample_size < queries.len() {
        args.recall_sample_size
    } else {
        queries.len()
    };

    println!("Computing ground truth for {} queries...", recall_queries);
    let ground_truth = compute_ground_truth(&db_vectors, &queries[..recall_queries], args.k, distance);

    // Open storage with Processor API
    let mut storage = Storage::readwrite(&args.db_path);
    storage.ready()?;
    let storage = Arc::new(storage);

    // Create processor to access the index properly
    let registry = storage.cache().clone();
    let processor = motlie_db::vector::Processor::new(storage.clone(), registry);

    // Get the embedding (registered during index build)
    let embedding = storage
        .cache()
        .get("bench", dim as u32, distance)
        .ok_or_else(|| anyhow::anyhow!("Embedding 'bench' not found in registry"))?;

    // Get the HNSW index with proper entry point loaded
    let index = processor
        .get_or_create_index(embedding.code())
        .ok_or_else(|| anyhow::anyhow!("Failed to get HNSW index"))?;

    // Run queries using low-level HNSW search (returns vec_ids directly for recall)
    let peak_rss_before = get_rss_bytes();
    println!("\nRunning {} queries (ef_search={})...", queries.len(), args.ef_search);
    let search_start = Instant::now();
    let mut latencies = Vec::with_capacity(queries.len());
    let mut search_results = Vec::with_capacity(recall_queries);

    for (qi, query) in queries.iter().enumerate() {
        let start = Instant::now();
        let results = index.search(&storage, query, args.ef_search, args.k)?;
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        latencies.push(latency_ms);
        // Only collect results for recall computation
        if qi < recall_queries {
            // HNSW search returns (distance, vec_id) - vec_id matches insertion order
            let result_ids: Vec<usize> = results.iter().map(|(_, vid)| *vid as usize).collect();
            search_results.push(result_ids);
        }

        if (qi + 1) % 100 == 0 || qi + 1 == queries.len() {
            print!("\r  {}/{} queries", qi + 1, queries.len());
            std::io::stdout().flush().ok();
        }
    }
    let search_duration = search_start.elapsed();
    let peak_rss_after = get_rss_bytes();
    println!();

    // Compute metrics
    let recall = compute_recall(&search_results, &ground_truth, args.k);
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let stats = LatencyStats::from_latencies(&latencies);

    println!("\n=== Results ===");
    println!("Recall@{}: {:.1}% ({} queries sampled)", args.k, recall * 100.0, recall_queries);
    println!("QPS: {:.1}", stats.qps);
    println!(
        "Latency: avg={:.2}ms, p50={:.2}ms, p95={:.2}ms, p99={:.2}ms",
        stats.avg_ms, stats.p50_ms, stats.p95_ms, stats.p99_ms
    );
    println!("Peak RSS: {:.2} MB", peak_rss_after.max(peak_rss_before) as f64 / 1_000_000.0);

    // Save JSON output if requested
    if let Some(output_path) = args.output {
        let json = serde_json::json!({
            "command": "query",
            "config": {
                "dataset": args.dataset,
                "num_queries": queries.len(),
                "recall_sample_size": recall_queries,
                "k": args.k,
                "ef_search": args.ef_search,
            },
            "results": {
                "recall_at_k": recall,
                "qps": stats.qps,
                "latency_avg_ms": stats.avg_ms,
                "latency_p50_ms": stats.p50_ms,
                "latency_p95_ms": stats.p95_ms,
                "latency_p99_ms": stats.p99_ms,
                "search_duration_secs": search_duration.as_secs_f64(),
                "peak_rss_bytes": peak_rss_after.max(peak_rss_before),
            }
        });

        std::fs::write(&output_path, serde_json::to_string_pretty(&json)?)
            .context("Failed to write output file")?;
        println!("Results saved to: {:?}", output_path);
    }

    Ok(())
}

// ============================================================================
// Sweep Command
// ============================================================================

#[derive(Parser)]
pub struct SweepArgs {
    /// Dataset: laion, sift, gist, random
    #[arg(long)]
    pub dataset: String,

    /// Data directory (not used for 'random' dataset)
    #[arg(long, default_value = "./data")]
    pub data_dir: PathBuf,

    /// Number of database vectors
    #[arg(long, default_value = "100000")]
    pub num_vectors: usize,

    /// Number of queries
    #[arg(long, default_value = "1000")]
    pub num_queries: usize,

    /// Vector dimension (only for 'random' dataset)
    #[arg(long, default_value = "1024")]
    pub dim: usize,

    /// Random seed (only for 'random' dataset)
    #[arg(long, default_value = "42")]
    pub seed: u64,

    /// ef_search values (comma-separated)
    #[arg(long, default_value = "50,100,200")]
    pub ef: String,

    /// k values for Recall@k (comma-separated)
    #[arg(long, default_value = "1,10")]
    pub k: String,

    /// Enable RaBitQ mode
    #[arg(long)]
    pub rabitq: bool,

    /// RaBitQ bits per dimension (comma-separated)
    #[arg(long, default_value = "1,2,4")]
    pub bits: String,

    /// Rerank factors (comma-separated)
    #[arg(long, default_value = "1,4,10,20")]
    pub rerank: String,

    /// HNSW M parameter
    #[arg(long, default_value = "16")]
    pub m: usize,

    /// HNSW ef_construction
    #[arg(long, default_value = "200")]
    pub ef_construction: usize,

    /// Results output directory
    #[arg(long, default_value = "./results")]
    pub results_dir: PathBuf,

    /// Use SIMD-optimized dot products (default: true)
    #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
    pub simd_dot: bool,

    /// Compare SIMD vs scalar (runs sweep twice)
    #[arg(long)]
    pub compare_simd: bool,

    /// Show Pareto frontier after sweep (optimal recall-QPS trade-offs)
    #[arg(long)]
    pub show_pareto: bool,

    /// Assert minimum recall threshold (exit code 1 if any recall < threshold)
    /// Example: --assert-recall 0.80 requires all recall values >= 80%
    #[arg(long)]
    pub assert_recall: Option<f64>,
}

pub async fn sweep(args: SweepArgs) -> Result<()> {
    println!("=== Parameter Sweep ===");
    println!("Dataset: {}", args.dataset);
    println!("Vectors: {}, Queries: {}", args.num_vectors, args.num_queries);

    // Parse parameters
    let ef_values: Vec<usize> = args.ef.split(',')
        .map(|s| s.trim().parse().expect("Invalid ef value"))
        .collect();
    let k_values: Vec<usize> = args.k.split(',')
        .map(|s| s.trim().parse().expect("Invalid k value"))
        .collect();
    let bits_values: Vec<u8> = args.bits.split(',')
        .map(|s| s.trim().parse().expect("Invalid bits value"))
        .collect();
    let rerank_values: Vec<usize> = args.rerank.split(',')
        .map(|s| s.trim().parse().expect("Invalid rerank value"))
        .collect();

    println!("ef_search: {:?}", ef_values);
    println!("k: {:?}", k_values);
    if args.rabitq {
        println!("RaBitQ bits: {:?}", bits_values);
        println!("Rerank factors: {:?}", rerank_values);
    }

    // Determine distance and dimension from dataset
    let (distance, dim) = match args.dataset.to_lowercase().as_str() {
        "laion" => (Distance::Cosine, benchmark::LAION_EMBEDDING_DIM),
        "sift" => (Distance::L2, benchmark::SIFT_EMBEDDING_DIM),
        "gist" => (Distance::L2, benchmark::GIST_EMBEDDING_DIM),
        "random" => (Distance::Cosine, args.dim),
        _ => anyhow::bail!("Unknown dataset: {}. Use: laion, sift, gist, random", args.dataset),
    };

    // Load dataset
    let (db_vectors, queries, _dim) = if args.dataset.to_lowercase() == "random" {
        let ds = RandomDataset::generate(args.num_vectors, args.num_queries, args.dim, args.seed);
        (ds.vectors, ds.queries, ds.dim)
    } else {
        load_dataset_for_query(
            &args.dataset,
            &args.data_dir,
            args.num_vectors,
            args.num_queries,
        )?
    };
    println!("Loaded {} db vectors, {} queries, dim={}", db_vectors.len(), queries.len(), dim);

    // Compute ground truth
    let max_k = *k_values.iter().max().unwrap_or(&10);
    println!("Computing ground truth (k={})...", max_k);
    let ground_truth = compute_ground_truth(&db_vectors, &queries, max_k, distance);

    // Create temp database
    let temp_path = std::env::temp_dir().join(format!("bench_vector_{}", std::process::id()));
    if temp_path.exists() {
        std::fs::remove_dir_all(&temp_path)?;
    }
    std::fs::create_dir_all(&temp_path)?;
    let db_path = &temp_path;
    println!("Database: {:?}", db_path);

    // Initialize storage
    let mut storage = Storage::readwrite(db_path);
    storage.ready()?;

    // Build index
    let nav_cache = Arc::new(NavigationCache::new());
    let embedding_code: EmbeddingCode = 1;
    let storage_type = VectorElementType::F16;

    let hnsw_config = hnsw::Config {
        dim,
        m: args.m,
        m_max: args.m * 2,
        m_max_0: args.m * 2,
        ef_construction: args.ef_construction,
        ..Default::default()
    };

    let index = hnsw::Index::with_storage_type(
        embedding_code,
        distance,
        storage_type,
        hnsw_config.clone(),
        nav_cache.clone(),
    );

    // Store vectors and build index
    let txn_db = storage.transaction_db()?;
    let vectors_cf = txn_db
        .cf_handle(Vectors::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("Vectors CF not found"))?;

    println!("\nBuilding HNSW index (M={}, ef_construction={})...", args.m, args.ef_construction);
    let start = Instant::now();

    for (i, vector) in db_vectors.iter().enumerate() {
        let vec_id = i as VecId;
        let key = VectorCfKey(embedding_code, vec_id);
        let value_bytes = Vectors::value_to_bytes_typed(vector, storage_type);

        // Use transaction for atomic vector storage + HNSW insert
        let txn = txn_db.transaction();
        txn.put_cf(&vectors_cf, Vectors::key_to_bytes(&key), value_bytes)?;
        let cache_update = hnsw::insert(&index, &txn, &txn_db, &storage, vec_id, vector)?;
        txn.commit()?;
        cache_update.apply(index.nav_cache());

        if (i + 1) % 10000 == 0 {
            println!("  {}/{} vectors", i + 1, db_vectors.len());
        }
    }

    let build_time = start.elapsed().as_secs_f64();
    println!("Index built in {:.2}s ({:.1} vec/s)", build_time, db_vectors.len() as f64 / build_time);

    // Create results directory
    std::fs::create_dir_all(&args.results_dir)?;

    // Track minimum recall across all sweeps
    let mut overall_min_recall: f64 = 1.0;

    if args.rabitq {
        if args.compare_simd {
            // Run both SIMD and scalar modes for comparison
            println!("\n=== SIMD vs Scalar Comparison ===\n");

            println!("--- SIMD Mode ---");
            let min_recall = run_rabitq_sweep(
                &index,
                &storage,
                &db_vectors,
                &queries,
                &ground_truth,
                &ef_values,
                &k_values,
                &bits_values,
                &rerank_values,
                dim,
                build_time,
                &args.results_dir,
                true, // use_simd_dot
                args.show_pareto,
            )?;
            if min_recall < overall_min_recall {
                overall_min_recall = min_recall;
            }

            println!("\n--- Scalar Mode ---");
            let min_recall = run_rabitq_sweep(
                &index,
                &storage,
                &db_vectors,
                &queries,
                &ground_truth,
                &ef_values,
                &k_values,
                &bits_values,
                &rerank_values,
                dim,
                build_time,
                &args.results_dir,
                false, // use_simd_dot
                args.show_pareto,
            )?;
            if min_recall < overall_min_recall {
                overall_min_recall = min_recall;
            }
        } else {
            // Single mode sweep
            let min_recall = run_rabitq_sweep(
                &index,
                &storage,
                &db_vectors,
                &queries,
                &ground_truth,
                &ef_values,
                &k_values,
                &bits_values,
                &rerank_values,
                dim,
                build_time,
                &args.results_dir,
                args.simd_dot,
                args.show_pareto,
            )?;
            overall_min_recall = min_recall;
        }
    } else {
        // Standard HNSW sweep
        let min_recall = run_hnsw_sweep(
            &index,
            &storage,
            &queries,
            &ground_truth,
            &ef_values,
            &k_values,
            build_time,
        )?;
        overall_min_recall = min_recall;
    }

    // Check recall assertion if threshold provided
    if let Some(threshold) = args.assert_recall {
        println!("\n=== Recall Assertion ===");
        println!("Minimum recall observed: {:.1}%", overall_min_recall * 100.0);
        println!("Required threshold: {:.1}%", threshold * 100.0);

        if overall_min_recall < threshold {
            anyhow::bail!(
                "Recall assertion FAILED: {:.1}% < {:.1}% threshold",
                overall_min_recall * 100.0,
                threshold * 100.0
            );
        }
        println!("✓ Recall assertion PASSED");
    }

    Ok(())
}

// ============================================================================
// Helper Functions
// ============================================================================

fn load_dataset_vectors(
    dataset: &str,
    data_dir: &PathBuf,
    max_vectors: usize,
) -> Result<(Vec<Vec<f32>>, usize)> {
    match dataset.to_lowercase().as_str() {
        "laion" => {
            let ds = LaionDataset::load(data_dir, max_vectors)?;
            let dim = ds.dim;
            Ok((ds.image_embeddings, dim))
        }
        "sift" => {
            let ds = SiftDataset::load(data_dir, max_vectors, 0)?;
            let dim = benchmark::SIFT_EMBEDDING_DIM;
            Ok((ds.base_vectors, dim))
        }
        "gist" => {
            let ds = GistDataset::load(data_dir, max_vectors, GIST_QUERIES)?;
            let dim = benchmark::GIST_EMBEDDING_DIM;
            Ok((ds.base_vectors, dim))
        }
        _ => anyhow::bail!("Unknown dataset: {}. Use: laion, sift, gist", dataset),
    }
}

/// Load dataset vectors with support for random dataset generation.
fn load_dataset_vectors_with_random(
    dataset: &str,
    data_dir: &PathBuf,
    max_vectors: usize,
    dim: usize,
    seed: u64,
) -> Result<(Vec<Vec<f32>>, usize)> {
    match dataset.to_lowercase().as_str() {
        "random" => {
            println!("Generating {} random vectors (dim={}, seed={})...", max_vectors, dim, seed);
            let ds = RandomDataset::generate(max_vectors, 0, dim, seed);
            Ok((ds.vectors, ds.dim))
        }
        _ => load_dataset_vectors(dataset, data_dir, max_vectors),
    }
}

fn load_dataset_for_query(
    dataset: &str,
    data_dir: &PathBuf,
    num_db: usize,
    num_queries: usize,
) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>, usize)> {
    match dataset.to_lowercase().as_str() {
        "laion" => {
            let ds = LaionDataset::load(data_dir, num_db)?;
            let dim = ds.dim;
            let queries: Vec<Vec<f32>> = ds.text_embeddings.into_iter().take(num_queries).collect();
            Ok((ds.image_embeddings, queries, dim))
        }
        "sift" => {
            let ds = SiftDataset::load(data_dir, num_db, num_queries)?;
            let dim = benchmark::SIFT_EMBEDDING_DIM;
            Ok((ds.base_vectors, ds.query_vectors, dim))
        }
        "gist" => {
            let ds = GistDataset::load(data_dir, num_db, num_queries)?;
            let dim = benchmark::GIST_EMBEDDING_DIM;
            let queries: Vec<Vec<f32>> = ds.query_vectors.into_iter().take(num_queries).collect();
            Ok((ds.base_vectors, queries, dim))
        }
        _ => anyhow::bail!("Unknown dataset: {}", dataset),
    }
}

fn compute_ground_truth(
    db_vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    k: usize,
    distance: Distance,
) -> Vec<Vec<usize>> {
    queries.iter().map(|query| {
        let mut distances: Vec<(usize, f32)> = db_vectors.iter()
            .enumerate()
            .map(|(i, v)| (i, distance.compute(query, v)))
            .collect();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.iter().take(k).map(|(i, _)| *i).collect()
    }).collect()
}

/// Run HNSW parameter sweep.
/// Returns the minimum recall observed across all configurations.
fn run_hnsw_sweep(
    index: &hnsw::Index,
    storage: &Storage,
    queries: &[Vec<f32>],
    ground_truth: &[Vec<usize>],
    ef_values: &[usize],
    k_values: &[usize],
    build_time: f64,
) -> Result<f64> {
    println!("\n=== HNSW Parameter Sweep ===");
    println!("{:<10} {:<10} {:<12} {:<10} {:<10} {:<10}", "ef_search", "k", "Recall", "QPS", "P50(ms)", "P99(ms)");
    println!("{}", "-".repeat(62));

    let max_k = *k_values.iter().max().unwrap_or(&10);
    let mut min_recall: f64 = 1.0;

    for &ef_search in ef_values {
        // Run all queries
        let mut latencies = Vec::with_capacity(queries.len());
        let mut search_results = Vec::with_capacity(queries.len());

        for query in queries {
            let start = Instant::now();
            let results = index.search(storage, query, ef_search, max_k)?;
            latencies.push(start.elapsed().as_secs_f64() * 1000.0);
            search_results.push(results.iter().map(|(_, id)| *id as usize).collect());
        }

        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let stats = LatencyStats::from_latencies(&latencies);

        for &k in k_values {
            let recall = compute_recall(&search_results, ground_truth, k);
            if recall < min_recall {
                min_recall = recall;
            }
            println!(
                "{:<10} {:<10} {:<12.1}% {:<10.1} {:<10.2} {:<10.2}",
                ef_search, k, recall * 100.0, stats.qps, stats.p50_ms, stats.p99_ms
            );
        }
    }

    println!("\nBuild time: {:.2}s", build_time);
    Ok(min_recall)
}

/// Run RaBitQ parameter sweep.
/// Returns the minimum recall observed across all configurations.
fn run_rabitq_sweep(
    index: &hnsw::Index,
    storage: &Storage,
    db_vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    ground_truth: &[Vec<usize>],
    ef_values: &[usize],
    k_values: &[usize],
    bits_values: &[u8],
    rerank_values: &[usize],
    dim: usize,
    build_time: f64,
    results_dir: &PathBuf,
    use_simd_dot: bool,
    show_pareto: bool,
) -> Result<f64> {
    let simd_mode = if use_simd_dot { "SIMD" } else { "scalar" };
    println!("\n=== RaBitQ Parameter Sweep ({}) ===", simd_mode);

    let embedding_code: EmbeddingCode = 1;
    let mut all_results: Vec<RabitqExperimentResult> = Vec::new();
    let mut min_recall: f64 = 1.0;

    for &bits in bits_values {
        println!("\n--- {} bits/dim ({}) ---", bits, simd_mode);

        // Create encoder with SIMD option
        let encoder = RaBitQ::with_options(dim, bits, 42, use_simd_dot);

        // Build binary code cache
        let cache = BinaryCodeCache::new();
        let encode_start = Instant::now();

        for (i, vector) in db_vectors.iter().enumerate() {
            let (code, correction) = encoder.encode_with_correction(vector);
            cache.put(embedding_code, i as VecId, code, correction);
        }

        let encode_time = encode_start.elapsed().as_secs_f64();
        let (cache_count, cache_bytes) = cache.stats();
        println!(
            "Encoded {} vectors in {:.2}s ({:.2} MB)",
            cache_count, encode_time, cache_bytes as f64 / 1_000_000.0
        );

        println!("{:<8} {:<8} {:<8} {:<10} {:<10} {:<10} {:<10}",
            "ef", "rerank", "k", "Recall", "QPS", "P50(ms)", "P99(ms)");
        println!("{}", "-".repeat(64));

        for &ef_search in ef_values {
            for &rerank_factor in rerank_values {
                for &k in k_values {
                    let candidates = k * rerank_factor;
                    let mut latencies = Vec::with_capacity(queries.len());
                    let mut search_results = Vec::with_capacity(queries.len());

                    for query in queries {
                        let start = Instant::now();
                        let results = index.search_with_rabitq_cached(
                            storage, query, &encoder, &cache,
                            k, ef_search, candidates,
                        )?;
                        latencies.push(start.elapsed().as_secs_f64() * 1000.0);
                        search_results.push(results.iter().map(|(_, id)| *id as usize).collect());
                    }

                    let recall = compute_recall(&search_results, ground_truth, k);
                    if recall < min_recall {
                        min_recall = recall;
                    }
                    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let stats = LatencyStats::from_latencies(&latencies);

                    println!(
                        "{:<8} {:<8} {:<8} {:<10.1}% {:<10.1} {:<10.2} {:<10.2}",
                        ef_search, rerank_factor, k, recall * 100.0, stats.qps, stats.p50_ms, stats.p99_ms
                    );

                    // Compute recall std (approximate)
                    let recalls: Vec<f64> = search_results.iter()
                        .zip(ground_truth.iter())
                        .map(|(res, gt)| {
                            let hits = res.iter().take(k).filter(|id| gt.contains(id)).count();
                            hits as f64 / k as f64
                        })
                        .collect();
                    let recall_std = std_dev(&recalls);

                    all_results.push(RabitqExperimentResult {
                        scale: db_vectors.len(),
                        bits_per_dim: bits,
                        ef_search,
                        rerank_factor,
                        k,
                        recall_mean: recall,
                        recall_std,
                        latency_avg_ms: stats.avg_ms,
                        latency_p50_ms: stats.p50_ms,
                        latency_p95_ms: stats.p95_ms,
                        latency_p99_ms: stats.p99_ms,
                        qps: stats.qps,
                        encode_time_s: encode_time,
                        build_time_s: build_time,
                    });
                }
            }
        }
    }

    // Save results
    let csv_path = results_dir.join("rabitq_sweep.csv");
    save_rabitq_results_csv(&all_results, &csv_path)?;
    println!("\nResults saved to {:?}", csv_path);

    // Show Pareto frontier if requested
    if show_pareto {
        // Convert to ParetoInput format
        let pareto_inputs: Vec<ParetoInput> = all_results
            .iter()
            .map(|r| ParetoInput {
                bits_per_dim: r.bits_per_dim,
                ef_search: r.ef_search,
                rerank_factor: r.rerank_factor,
                k: r.k,
                recall: r.recall_mean,
                qps: r.qps,
            })
            .collect();

        // Compute and display Pareto frontier for each k
        for &k in k_values {
            let frontier = benchmark::compute_pareto_frontier_for_k(&pareto_inputs, k);
            if !frontier.is_empty() {
                print_pareto_frontier(&frontier, &format!("Pareto Frontier (Recall@{} vs QPS)", k));
            }
        }
    }

    Ok(min_recall)
}

fn std_dev(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    variance.sqrt()
}

// ============================================================================
// Check Distribution Command
// ============================================================================

#[derive(Parser)]
pub struct CheckDistributionArgs {
    /// Dataset: laion, sift, gist, random
    #[arg(long)]
    pub dataset: String,

    /// Data directory (not used for 'random' dataset)
    #[arg(long, default_value = "./data")]
    pub data_dir: PathBuf,

    /// Number of vectors to sample
    #[arg(long, default_value = "1000")]
    pub sample_size: usize,

    /// Vector dimension (only for 'random' dataset)
    #[arg(long, default_value = "1024")]
    pub dim: usize,

    /// RaBitQ bits per dimension
    #[arg(long, default_value = "4")]
    pub bits: u8,

    /// Random seed
    #[arg(long, default_value = "42")]
    pub seed: u64,
}

/// Check RaBitQ rotation distribution to validate √D scaling.
///
/// This command samples vectors from a dataset, applies RaBitQ rotation,
/// and reports component variance statistics. For correctly scaled rotation:
/// - Component mean should be ≈ 0
/// - Component variance should be ≈ 1.0 (in range [0.8, 1.2])
///
/// If variance is too low (e.g., ≈ 1/D), the rotation matrix lacks √D scaling.
pub fn check_distribution(args: CheckDistributionArgs) -> Result<()> {
    println!("=== RaBitQ Distribution Check ===");
    println!("Dataset: {}", args.dataset);
    println!("Sample size: {}", args.sample_size);
    println!("Bits per dim: {}", args.bits);

    // Load vectors
    let (vectors, dim) = match args.dataset.to_lowercase().as_str() {
        "laion" => {
            let ds = LaionDataset::load(&args.data_dir, args.sample_size)?;
            (ds.image_embeddings, ds.dim)
        }
        "sift" => {
            let ds = SiftDataset::load(&args.data_dir, args.sample_size, 0)?;
            (ds.base_vectors, benchmark::SIFT_EMBEDDING_DIM)
        }
        "gist" => {
            let ds = GistDataset::load(&args.data_dir, args.sample_size, 0)?;
            (ds.base_vectors, benchmark::GIST_EMBEDDING_DIM)
        }
        "random" => {
            let ds = RandomDataset::generate(args.sample_size, 0, args.dim, args.seed);
            (ds.vectors, ds.dim)
        }
        _ => anyhow::bail!(
            "Unknown dataset: {}. Use: laion, sift, gist, random",
            args.dataset
        ),
    };

    println!("Loaded {} vectors, dim={}", vectors.len(), dim);

    // Create RaBitQ encoder
    let encoder = RaBitQ::new(dim, args.bits, args.seed);

    // Compute rotation statistics
    let stats = compute_rotated_variance(
        |v| encoder.rotate_query(v),
        &vectors,
        args.sample_size,
    );

    // Print results
    print_rotation_stats(&stats, &format!("RaBitQ Rotation Stats ({}-bit)", args.bits));

    // Provide interpretation
    println!("\nInterpretation:");
    if stats.scaling_valid {
        println!("  ✓ Rotation matrix has correct √D scaling");
        println!("  ✓ RaBitQ should work well on this dataset");
    } else if stats.component_variance < 0.1 {
        println!("  ✗ Variance too low ({:.4}) - rotation matrix may lack √D scaling", stats.component_variance);
        println!("  ✗ This will cause poor quantization quality");
    } else if stats.component_variance > 2.0 {
        println!("  ✗ Variance too high ({:.4}) - vectors may not be normalized", stats.component_variance);
        println!("  ✗ Consider normalizing vectors before indexing");
    } else {
        println!("  ⚠ Variance ({:.4}) outside expected range [0.8, 1.2]", stats.component_variance);
        println!("  ⚠ RaBitQ may have suboptimal recall on this dataset");
    }

    Ok(())
}

// ============================================================================
// Datasets Command
// ============================================================================
// Scale Command
// ============================================================================

#[derive(Parser)]
pub struct ScaleArgs {
    /// Number of vectors to insert
    #[arg(long)]
    pub num_vectors: usize,

    /// Vector dimension
    #[arg(long, default_value = "128")]
    pub dim: usize,

    /// Batch size for inserts
    #[arg(long, default_value = "1000")]
    pub batch_size: usize,

    /// Number of search queries after insert
    #[arg(long, default_value = "1000")]
    pub num_queries: usize,

    /// ef_search parameter
    #[arg(long, default_value = "100")]
    pub ef_search: usize,

    /// k for top-k search
    #[arg(long, default_value = "10")]
    pub k: usize,

    /// HNSW M parameter
    #[arg(long, default_value = "16")]
    pub m: usize,

    /// HNSW ef_construction parameter
    #[arg(long, default_value = "200")]
    pub ef_construction: usize,

    /// Random seed for reproducibility
    #[arg(long, default_value = "42")]
    pub seed: u64,

    /// Distance metric: cosine, l2, dot
    #[arg(long, default_value = "cosine")]
    pub distance: String,

    /// Progress reporting interval in seconds
    #[arg(long, default_value = "10")]
    pub progress_interval: u64,

    /// Database path (required)
    #[arg(long)]
    pub db_path: PathBuf,

    /// Output results to JSON file
    #[arg(long)]
    pub output: Option<PathBuf>,

    /// Use async inserts (deferred HNSW graph construction)
    #[arg(long, default_value = "false")]
    pub r#async: bool,

    /// Number of async workers (only used with --async). Default is 1 to avoid
    /// lock contention on single-embedding workloads. Use higher values only
    /// with multiple embeddings.
    #[arg(long, default_value = "1")]
    pub async_workers: usize,

    /// Number of queries to sample for recall computation (0 to disable)
    #[arg(long, default_value = "100")]
    pub recall_sample_size: usize,

    /// Rerank factor for search (candidates = k * rerank_factor)
    #[arg(long, default_value = "4")]
    pub rerank_factor: usize,
}

/// **DEPRECATED**: Use `index` + `query` commands instead.
///
/// The `scale` command is deprecated in favor of composable `index` and `query` commands:
/// ```bash
/// # Instead of: bench_vector scale --num-vectors 1000000 --dim 128 --db-path /tmp/bench
/// # Use:
/// bench_vector index --dataset random --num-vectors 1000000 --dim 128 --seed 42 \
///     --fresh --batch-size 5000 --db-path /tmp/bench --output /tmp/index.json
/// bench_vector query --db-path /tmp/bench --dataset random \
///     --num-queries 1000 --k 10 --ef-search 100 --output /tmp/query.json
/// ```
///
/// See `bins/bench_vector/BENCH2.md` for the full migration guide.
pub fn scale(args: ScaleArgs) -> Result<()> {
    use motlie_db::vector::benchmark::{AsyncMetrics, ScaleBenchmark, ScaleConfig};
    use motlie_db::vector::{AsyncGraphUpdater, AsyncUpdaterConfig, Distance, EmbeddingBuilder};
    use std::time::Duration;

    // Print deprecation warning
    eprintln!("\n⚠️  DEPRECATED: The 'scale' command is deprecated.");
    eprintln!("   Use 'index' + 'query' commands instead. See BENCH2.md for migration guide.\n");

    let distance = match args.distance.to_lowercase().as_str() {
        "cosine" => Distance::Cosine,
        "l2" | "euclidean" => Distance::L2,
        "dot" | "dotproduct" => Distance::DotProduct,
        _ => anyhow::bail!(
            "Invalid distance: {} (use cosine, l2, or dot)",
            args.distance
        ),
    };

    println!("=== Scale Benchmark ===");
    println!("Vectors: {}", args.num_vectors);
    println!("Dimension: {}D", args.dim);
    println!("HNSW: M={}, ef_construction={}", args.m, args.ef_construction);
    println!("Distance: {:?}", distance);
    println!("Batch size: {}", args.batch_size);
    if args.r#async {
        println!("Insert mode: ASYNC ({} workers)", args.async_workers);
    } else {
        println!("Insert mode: SYNC (immediate indexing)");
    }
    println!("Database: {:?}", args.db_path);
    println!();

    // Delete existing DB if exists (fresh benchmark)
    if args.db_path.exists() {
        std::fs::remove_dir_all(&args.db_path)
            .context("Failed to remove existing database")?;
    }

    let mut storage = motlie_db::vector::Storage::readwrite(&args.db_path);
    storage.ready().context("Failed to initialize storage")?;
    let storage = Arc::new(storage);

    // Register embedding
    let txn_db = storage.transaction_db().context("Failed to get transaction DB")?;
    let embedding = storage
        .cache()
        .register(
            EmbeddingBuilder::new("scale-bench", args.dim as u32, distance)
                .with_hnsw_m(args.m as u16)
                .with_hnsw_ef_construction(args.ef_construction as u16),
            &txn_db,
        )
        .context("Failed to register embedding")?;

    // Start async updater if async mode enabled (wrap in Arc for sharing)
    let async_updater: Option<Arc<AsyncGraphUpdater>> = if args.r#async {
        let async_config = AsyncUpdaterConfig::new()
            .with_num_workers(args.async_workers)
            .with_batch_size(100)
            .with_ef_construction(args.ef_construction)
            .with_process_on_startup(false); // Don't drain on startup

        let registry = storage.cache().clone();
        let nav_cache = Arc::new(motlie_db::vector::NavigationCache::new());

        Some(Arc::new(AsyncGraphUpdater::start(
            storage.clone(),
            registry,
            nav_cache,
            async_config,
        )))
    } else {
        None
    };

    // Build config
    let config = ScaleConfig::new(args.num_vectors, args.dim)
        .with_batch_size(args.batch_size)
        .with_seed(args.seed)
        .with_distance(distance)
        .with_progress_interval(Duration::from_secs(args.progress_interval))
        .with_num_queries(args.num_queries)
        .with_ef_search(args.ef_search)
        .with_k(args.k)
        .with_hnsw_m(args.m)
        .with_hnsw_ef_construction(args.ef_construction)
        .with_immediate_index(!args.r#async) // false = async inserts
        .with_recall_sample_size(args.recall_sample_size)
        .with_rerank_factor(args.rerank_factor);

    // Create async metrics callback if in async mode
    let async_metrics = if let Some(ref updater) = async_updater {
        // Clone Arc references for the closures
        let updater_pending = Arc::clone(updater);
        let updater_items = Arc::clone(updater);

        Some(AsyncMetrics {
            pending_queue_size: Box::new(move || updater_pending.pending_queue_size()),
            items_processed: Box::new(move || updater_items.items_processed()),
        })
    } else {
        None
    };

    // Run benchmark (inserts phase) with async metrics tracking
    let result = ScaleBenchmark::run_with_async_metrics(&storage, &embedding, config, async_metrics)?;

    // Wait for async workers to drain pending queue before reporting results
    if let Some(updater) = async_updater {
        println!("\nWaiting for async workers to drain pending queue...");
        let drain_start = std::time::Instant::now();

        // Poll until pending queue is empty
        loop {
            let pending = updater.pending_queue_size();
            if pending == 0 {
                break;
            }
            println!(
                "  Pending: {} vectors, processed: {} items",
                pending,
                updater.items_processed()
            );
            std::thread::sleep(Duration::from_secs(2));
        }

        let items_processed = updater.items_processed();
        println!(
            "Async workers drained in {:.1}s ({} items processed)",
            drain_start.elapsed().as_secs_f64(),
            items_processed
        );

        // Shutdown workers - unwrap Arc to get ownership
        match Arc::try_unwrap(updater) {
            Ok(owned_updater) => owned_updater.shutdown(),
            Err(_arc) => {
                // Arc still has other references - just drop it
                // Workers will be signaled to stop when Arc is dropped
                println!("Note: Async workers still shutting down in background");
            }
        }
    }

    // Print results
    println!("{}", result);

    // Save to JSON if requested
    if let Some(output_path) = args.output {
        let json = serde_json::json!({
            "config": {
                "num_vectors": result.config.num_vectors,
                "dim": result.config.dim,
                "batch_size": result.config.batch_size,
                "hnsw_m": result.config.hnsw_m,
                "hnsw_ef_construction": result.config.hnsw_ef_construction,
                "seed": result.config.seed,
            },
            "insert": {
                "vectors_inserted": result.vectors_inserted,
                "errors": result.insert_errors,
                "duration_secs": result.insert_duration.as_secs_f64(),
                "throughput_vec_per_sec": result.insert_throughput,
            },
            "search": {
                "queries_executed": result.queries_executed,
                "duration_secs": result.search_duration.as_secs_f64(),
                "qps": result.search_qps,
                "p50_ms": result.search_p50.as_secs_f64() * 1000.0,
                "p99_ms": result.search_p99.as_secs_f64() * 1000.0,
            },
            "memory": {
                "peak_rss_bytes": result.peak_memory_bytes,
                "nav_cache_bytes": result.nav_cache_bytes,
            },
            "recall": {
                "recall_at_k": result.recall_at_k,
                "k": result.config.k,
                "sample_queries": result.recall_queries,
            }
        });

        std::fs::write(&output_path, serde_json::to_string_pretty(&json)?)
            .context("Failed to write output file")?;
        println!("\nResults saved to: {:?}", output_path);
    }

    Ok(())
}

// ============================================================================
// List Datasets Command
// ============================================================================

pub fn list_datasets() -> Result<()> {
    println!("Available datasets:\n");

    println!("Core datasets (always available):");
    println!("  laion       - LAION-CLIP 512D (NPY, Cosine)");
    println!("  sift        - SIFT-1M 128D (fvecs, L2)");
    println!("  gist        - GIST-960 960D (fvecs, L2)");
    println!("  random      - Synthetic unit-normalized (configurable dim, Cosine)");

    #[cfg(feature = "parquet")]
    {
        println!("\nParquet datasets (--features parquet):");
        println!("  cohere      - Cohere Wikipedia 768D (Parquet, Cosine)");
    }

    #[cfg(not(feature = "parquet"))]
    {
        println!("\nParquet datasets: Enable with --features parquet");
    }

    #[cfg(feature = "hdf5")]
    {
        println!("\nHDF5 datasets (--features hdf5):");
        println!("  glove       - GloVe-100 100D (HDF5, Angular)");
    }

    #[cfg(not(feature = "hdf5"))]
    {
        println!("\nHDF5 datasets: Enable with --features hdf5 (requires libhdf5)");
    }

    println!("\n--- Usage Examples ---\n");

    println!("Download a dataset:");
    println!("  bench_vector download --dataset laion --data-dir ./data\n");

    println!("Build an index:");
    println!("  bench_vector index --dataset laion --num-vectors 100000 --db-path ./bench_db\n");

    println!("Run queries:");
    println!("  bench_vector query --db-path ./bench_db --dataset laion --k 10 --ef-search 100\n");

    println!("Parameter sweep (HNSW):");
    println!("  bench_vector sweep --dataset laion --num-vectors 50000 --ef 50,100,200 --k 1,10\n");

    println!("Parameter sweep (RaBitQ):");
    println!("  bench_vector sweep --dataset laion --num-vectors 50000 --rabitq --bits 1,2,4 --rerank 1,4,10,20\n");

    println!("Parameter sweep (Random dataset):");
    println!("  bench_vector sweep --dataset random --dim 1024 --num-vectors 100000 --num-queries 1000 --rabitq\n");

    println!("Parameter sweep with Pareto frontier:");
    println!("  bench_vector sweep --dataset laion --rabitq --show-pareto\n");

    println!("Check RaBitQ distribution (√D scaling validation):");
    println!("  bench_vector check-distribution --dataset random --dim 1024 --sample-size 1000");

    Ok(())
}
