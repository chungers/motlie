//! CLI command implementations for vector benchmarks.
//!
//! Provides full benchmark functionality:
//! - Download: Fetch benchmark datasets
//! - Index: Build HNSW index (incremental support)
//! - Query: Run queries on existing index
//! - Sweep: Parameter sweep over ef_search, bits, rerank

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use motlie_db::vector::benchmark::{
    self, compute_recall, BenchmarkMetadata,
    GistDataset, LaionDataset, LatencyStats, RabitqExperimentResult, RandomDataset, SiftDataset,
    save_rabitq_results_csv, GIST_QUERIES,
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
    /// Dataset: laion, sift, gist
    #[arg(long)]
    pub dataset: String,

    /// Number of vectors to index
    #[arg(long)]
    pub num_vectors: usize,

    /// Database path for RocksDB storage
    #[arg(long)]
    pub db_path: PathBuf,

    /// Data directory (for dataset files)
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
}

pub async fn index(args: IndexArgs) -> Result<()> {
    println!("=== Vector Index Build ===");
    println!("Dataset: {}", args.dataset);
    println!("Vectors: {}", args.num_vectors);
    println!("Database: {:?}", args.db_path);
    println!("HNSW: M={}, ef_construction={}", args.m, args.ef_construction);

    // Determine distance metric
    let distance = if args.cosine {
        Distance::Cosine
    } else if args.l2 {
        Distance::L2
    } else {
        // Auto-detect from dataset
        match args.dataset.to_lowercase().as_str() {
            "laion" | "cohere" => Distance::Cosine,
            "sift" | "gist" => Distance::L2,
            _ => Distance::Cosine,
        }
    };
    println!("Distance: {:?}", distance);

    // Load dataset
    let (vectors, dim) = load_dataset_vectors(&args.dataset, &args.data_dir, args.num_vectors)?;
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

    // Initialize storage
    let mut storage = Storage::readwrite(&args.db_path);
    storage.ready()?;

    // Create HNSW index
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
        hnsw_config,
        nav_cache,
    );

    // Get vectors CF
    let txn_db = storage.transaction_db()?;
    let vectors_cf = txn_db
        .cf_handle(Vectors::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("Vectors CF not found"))?;

    // Insert vectors (skip already indexed)
    let start = Instant::now();
    let vectors_to_add = &vectors[existing_count..];
    let total = vectors_to_add.len();

    println!("\nInserting {} vectors (starting from {})...", total, existing_count);

    for (i, vector) in vectors_to_add.iter().enumerate() {
        let vec_id = (existing_count + i) as VecId;

        // Store vector in Vectors CF
        let key = VectorCfKey(embedding_code, vec_id);
        let value_bytes = Vectors::value_to_bytes_typed(vector, storage_type);
        txn_db.put_cf(&vectors_cf, Vectors::key_to_bytes(&key), value_bytes)?;

        // Insert into HNSW index
        index.insert(&storage, vec_id, vector)?;

        if (i + 1) % 10000 == 0 || i + 1 == total {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = (i + 1) as f64 / elapsed;
            println!("  {}/{} vectors ({:.1} vec/s)", i + 1, total, rate);
        }
    }

    let build_time = start.elapsed().as_secs_f64();
    println!(
        "\nIndex built in {:.2}s ({:.1} vec/s)",
        build_time,
        total as f64 / build_time
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
    meta.num_vectors = args.num_vectors;
    meta.checkpoint(&args.db_path)?;
    println!("Metadata saved: {:?}", BenchmarkMetadata::path(&args.db_path));

    Ok(())
}

// ============================================================================
// Query Command
// ============================================================================

#[derive(Parser)]
pub struct QueryArgs {
    /// Database path
    #[arg(long)]
    pub db_path: PathBuf,

    /// Dataset for queries (to load test vectors)
    #[arg(long)]
    pub dataset: String,

    /// Data directory (for dataset files)
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
}

pub async fn query(args: QueryArgs) -> Result<()> {
    println!("=== Vector Search ===");
    println!("Database: {:?}", args.db_path);
    println!("k={}, ef_search={}", args.k, args.ef_search);

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

    // Load queries and ground truth
    let (db_vectors, queries, dim) = load_dataset_for_query(
        &args.dataset,
        &args.data_dir,
        meta.num_vectors,
        args.num_queries,
    )?;
    println!("Loaded {} queries, dim={}", queries.len(), dim);

    // Compute ground truth
    println!("Computing ground truth...");
    let ground_truth = compute_ground_truth(&db_vectors, &queries, args.k, distance);

    // Open storage
    let mut storage = Storage::readwrite(&args.db_path);
    storage.ready()?;

    // Create HNSW index handle
    let nav_cache = Arc::new(NavigationCache::new());
    let embedding_code: EmbeddingCode = 1;
    let storage_type = VectorElementType::F16;

    let hnsw_config = hnsw::Config {
        dim,
        m: meta.hnsw_m,
        m_max: meta.hnsw_m * 2,
        m_max_0: meta.hnsw_m * 2,
        ef_construction: meta.hnsw_ef_construction,
        ..Default::default()
    };

    let index = hnsw::Index::with_storage_type(
        embedding_code,
        distance,
        storage_type,
        hnsw_config,
        nav_cache,
    );

    // Run queries
    println!("\nRunning {} queries (ef_search={})...", queries.len(), args.ef_search);
    let mut latencies = Vec::with_capacity(queries.len());
    let mut search_results = Vec::with_capacity(queries.len());

    for (qi, query) in queries.iter().enumerate() {
        let start = Instant::now();
        let results = index.search(&storage, query, args.ef_search, args.k)?;
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        latencies.push(latency_ms);
        let result_ids: Vec<usize> = results.iter().map(|(_, id)| *id as usize).collect();
        search_results.push(result_ids);

        if (qi + 1) % 100 == 0 {
            println!("  {}/{} queries", qi + 1, queries.len());
        }
    }

    // Compute metrics
    let recall = compute_recall(&search_results, &ground_truth, args.k);
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let stats = LatencyStats::from_latencies(&latencies);

    println!("\n=== Results ===");
    println!("Recall@{}: {:.1}%", args.k, recall * 100.0);
    println!("QPS: {:.1}", stats.qps);
    println!(
        "Latency: avg={:.2}ms, p50={:.2}ms, p95={:.2}ms, p99={:.2}ms",
        stats.avg_ms, stats.p50_ms, stats.p95_ms, stats.p99_ms
    );

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
        txn_db.put_cf(&vectors_cf, Vectors::key_to_bytes(&key), value_bytes)?;
        index.insert(&storage, vec_id, vector)?;

        if (i + 1) % 10000 == 0 {
            println!("  {}/{} vectors", i + 1, db_vectors.len());
        }
    }

    let build_time = start.elapsed().as_secs_f64();
    println!("Index built in {:.2}s ({:.1} vec/s)", build_time, db_vectors.len() as f64 / build_time);

    // Create results directory
    std::fs::create_dir_all(&args.results_dir)?;

    if args.rabitq {
        if args.compare_simd {
            // Run both SIMD and scalar modes for comparison
            println!("\n=== SIMD vs Scalar Comparison ===\n");

            println!("--- SIMD Mode ---");
            run_rabitq_sweep(
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
            )?;

            println!("\n--- Scalar Mode ---");
            run_rabitq_sweep(
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
            )?;
        } else {
            // Single mode sweep
            run_rabitq_sweep(
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
            )?;
        }
    } else {
        // Standard HNSW sweep
        run_hnsw_sweep(
            &index,
            &storage,
            &queries,
            &ground_truth,
            &ef_values,
            &k_values,
            build_time,
        )?;
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
        _ => anyhow::bail!("Unknown dataset: {}", dataset),
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

fn run_hnsw_sweep(
    index: &hnsw::Index,
    storage: &Storage,
    queries: &[Vec<f32>],
    ground_truth: &[Vec<usize>],
    ef_values: &[usize],
    k_values: &[usize],
    build_time: f64,
) -> Result<()> {
    println!("\n=== HNSW Parameter Sweep ===");
    println!("{:<10} {:<10} {:<12} {:<10} {:<10} {:<10}", "ef_search", "k", "Recall", "QPS", "P50(ms)", "P99(ms)");
    println!("{}", "-".repeat(62));

    let max_k = *k_values.iter().max().unwrap_or(&10);

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
            println!(
                "{:<10} {:<10} {:<12.1}% {:<10.1} {:<10.2} {:<10.2}",
                ef_search, k, recall * 100.0, stats.qps, stats.p50_ms, stats.p99_ms
            );
        }
    }

    println!("\nBuild time: {:.2}s", build_time);
    Ok(())
}

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
) -> Result<()> {
    let simd_mode = if use_simd_dot { "SIMD" } else { "scalar" };
    println!("\n=== RaBitQ Parameter Sweep ({}) ===", simd_mode);

    let embedding_code: EmbeddingCode = 1;
    let mut all_results: Vec<RabitqExperimentResult> = Vec::new();

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
    save_rabitq_results_csv(&all_results, results_dir)?;
    println!("\nResults saved to {:?}", csv_path);

    Ok(())
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
// Datasets Command
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
    println!("  bench_vector sweep --dataset random --dim 1024 --num-vectors 100000 --num-queries 1000 --rabitq");

    Ok(())
}
