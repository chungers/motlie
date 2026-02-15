//! CLI command implementations for vector benchmarks.
//!
//! Provides full benchmark functionality:
//! - Download: Fetch benchmark datasets
//! - Index: Build HNSW index (incremental support)
//! - Query: Run queries on existing index
//! - Sweep: Parameter sweep over ef_search, bits, rerank

use anyhow::{Context, Result};
use clap::Parser;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use motlie_db::rocksdb::{ColumnFamily, ColumnFamilySerde};
use motlie_db::vector::benchmark::{
    self, compute_recall, compute_rotated_variance, print_pareto_frontier, print_rotation_stats,
    save_rabitq_results_csv, BenchmarkMetadata, GistDataset, LaionDataset, LatencyStats,
    ParetoInput, RandomDataset, SiftDataset,
};
use motlie_db::vector::schema::{EmbeddingSpec, EmbeddingSpecs, ExternalKey};
use motlie_db::vector::{
    create_reader_with_storage, create_writer, spawn_mutation_consumer_with_storage_autoreg,
    spawn_query_consumers_with_storage_autoreg, AsyncGraphUpdater, AsyncUpdaterConfig, Distance,
    EmbeddingBuilder, IdAllocator, InsertVectorBatch, NavigationCache, RaBitQ, ReaderConfig,
    Runnable, SearchKNN, Storage, VecId, VectorElementType, Vectors, WriterConfig,
};
use motlie_db::Id;

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
    std::fs::create_dir_all(&args.data_dir).context("Failed to create data directory")?;

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
        "cohere" | "cohere-wiki" => {
            println!("Downloading Cohere Wikipedia embeddings...");
            benchmark::CohereWikipediaDataset::download(&args.data_dir)?;
        }
        "glove" | "glove-100" => {
            println!("Downloading GloVe-100 angular dataset...");
            benchmark::GloveDataset::download(&args.data_dir)?;
        }
        "random" => {
            println!("Random dataset is generated at runtime, no download needed.");
            println!("Use: bench_vector index --dataset random --dim 1024 --num-vectors 100000");
            return Ok(());
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
    /// Dataset: laion, sift, gist, cohere, glove, random
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

    /// Stream random vectors instead of loading into memory (random dataset only)
    #[arg(long, default_value = "false")]
    pub stream: bool,

    /// Batch size for inserts (vectors per transaction)
    #[arg(long, default_value = "1000")]
    pub batch_size: usize,

    /// Progress reporting interval (seconds)
    #[arg(long, default_value = "10")]
    pub progress_interval: u64,

    /// Number of async workers (0 = sync inserts)
    #[arg(long, default_value = "0")]
    pub async_workers: usize,

    /// Drain pending async updates only (no new inserts)
    #[arg(long, default_value = "false")]
    pub drain_pending: bool,

    /// Vector storage type: f32 (full precision) or f16 (half precision, 50% smaller)
    #[arg(long, default_value = "f32")]
    pub storage_type: String,

    /// Output results to JSON file
    #[arg(long)]
    pub output: Option<PathBuf>,
}

fn parse_storage_type(value: &str) -> Result<VectorElementType> {
    match value.to_lowercase().as_str() {
        "f32" => Ok(VectorElementType::F32),
        "f16" => Ok(VectorElementType::F16),
        other => anyhow::bail!("Invalid storage type: {} (use f32 or f16)", other),
    }
}

pub async fn index(args: IndexArgs) -> Result<()> {
    use motlie_db::vector::benchmark::scale::get_rss_bytes;
    use std::time::Duration;

    let storage_type = parse_storage_type(&args.storage_type)?;

    println!("=== Vector Index Build ===");
    println!("Dataset: {}", args.dataset);
    println!("Vectors: {}", args.num_vectors);
    println!("Database: {:?}", args.db_path);
    println!(
        "HNSW: M={}, ef_construction={}",
        args.m, args.ef_construction
    );
    println!("Storage type: {:?}", storage_type);
    println!("Batch size: {}", args.batch_size);
    if args.async_workers > 0 {
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
            "laion" | "cohere" | "glove" | "random" => Distance::Cosine,
            "sift" | "gist" => Distance::L2,
            _ => Distance::Cosine,
        }
    };
    println!("Distance: {:?}", distance);

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

    if existing_count >= args.num_vectors && !args.drain_pending {
        println!(
            "Index already has {} vectors, nothing to do.",
            existing_count
        );
        return Ok(());
    }

    let is_random = args.dataset.to_lowercase() == "random";

    if args.stream && !is_random {
        anyhow::bail!("--stream is only supported for the random dataset.");
    }
    if args.async_workers > 0 && !args.stream && !args.drain_pending {
        anyhow::bail!("--async-workers requires --stream with the random dataset.");
    }
    if args.drain_pending && args.async_workers == 0 {
        anyhow::bail!("--drain-pending requires --async-workers > 0.");
    }
    if args.drain_pending && !args.db_path.exists() {
        anyhow::bail!("Database path does not exist for --drain-pending.");
    }

    if !args.drain_pending {
        // Create database directory
        std::fs::create_dir_all(&args.db_path)?;
    }

    // Initialize storage
    let mut storage = Storage::readwrite(&args.db_path);
    storage.ready()?;
    let storage = Arc::new(storage);

    if args.drain_pending {
        println!(
            "Draining pending async updates with {} worker(s)...",
            args.async_workers
        );

        let async_config = AsyncUpdaterConfig::new()
            .with_num_workers(args.async_workers)
            .with_batch_size(args.batch_size)
            .with_process_on_startup(true);
        let registry = storage.cache().clone();
        let nav_cache = Arc::new(NavigationCache::new());
        let updater = AsyncGraphUpdater::start(storage.clone(), registry, nav_cache, async_config);

        loop {
            let pending = updater.pending_queue_size();
            if pending == 0 {
                break;
            }
            println!("  Pending: {} vectors", pending);
            std::thread::sleep(std::time::Duration::from_secs(2));
        }

        updater.shutdown();
        println!("Pending queue drained.");
        return Ok(());
    }

    // Prepare embedding registry
    let registry = storage.cache().clone();
    registry.set_storage(storage.clone())?;
    let model = format!("bench-{}", args.dataset.to_lowercase());

    if is_random && args.stream {
        // ADMIN.md Issue #1 (chungers, 2025-01-27, FIXED):
        // Defer embedding registration until after validation checks pass.
        // Previously, registration happened before the existing_count check,
        // leaving orphan EmbeddingSpecs on abort.
        if existing_count > 0 {
            anyhow::bail!(
                "Streaming random index requires a fresh database. Use --fresh to rebuild."
            );
        }

        let dim = args.dim;
        let embedding = registry.register(
            EmbeddingBuilder::new(&model, dim as u32, distance)
                .with_hnsw_m(args.m as u16)
                .with_hnsw_ef_construction(args.ef_construction as u16)
                .with_storage_type(storage_type),
        )?;
        print_embedding_summary_from_db(
            storage.transaction_db()?,
            &args.db_path,
            embedding.code(),
        )?;

        if args.async_workers > 0 {
            println!("Insert mode: ASYNC ({} workers)", args.async_workers);
        }

        let async_updater = if args.async_workers > 0 {
            let async_config = AsyncUpdaterConfig::new()
                .with_num_workers(args.async_workers)
                .with_batch_size(100)
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

        let (writer, writer_rx) = create_writer(WriterConfig::default());
        let consumer_handle = spawn_mutation_consumer_with_storage_autoreg(
            writer_rx,
            WriterConfig::default(),
            storage.clone(),
        );

        let start = Instant::now();
        let mut generator =
            motlie_db::vector::benchmark::StreamingVectorGenerator::new_with_distance(
                args.dim,
                args.num_vectors,
                args.seed,
                distance,
            );
        let total = args.num_vectors;
        let mut inserted = 0usize;
        let errors = 0usize;
        let mut peak_rss: u64 = 0;
        let mut last_progress = Instant::now();
        let progress_interval = Duration::from_secs(args.progress_interval);
        let immediate = args.async_workers == 0;

        println!("\nStreaming {} vectors (dim={})...", total, args.dim);
        println!();

        while let Some(batch) = generator.next_batch(args.batch_size) {
            let batch_start = inserted;
            let batch_len = batch.len();
            let mut payload = Vec::with_capacity(batch_len);

            for (offset, vector) in batch.into_iter().enumerate() {
                let id = Id::from_bytes(((batch_start + offset) as u128).to_be_bytes());
                // T7.1: Use ExternalKey::NodeId for polymorphic ID mapping
                payload.push((ExternalKey::NodeId(id), vector));
            }

            let mutation = InsertVectorBatch::new(&embedding, payload);
            let mutation = if immediate {
                mutation.immediate()
            } else {
                mutation
            };
            writer.send(vec![mutation.into()]).await?;
            inserted += batch_len;

            if last_progress.elapsed() >= progress_interval {
                let rss = get_rss_bytes();
                peak_rss = peak_rss.max(rss);
                let elapsed = start.elapsed().as_secs_f64();
                let rate = inserted as f64 / elapsed.max(0.0001);
                let remaining = total.saturating_sub(inserted);
                let eta_secs = if rate > 0.0 {
                    remaining as f64 / rate
                } else {
                    0.0
                };
                let eta_str = format_eta(eta_secs);

                if let Some(ref updater) = async_updater {
                    let pending = updater.pending_queue_size();
                    print!(
                        "\r[{:>6.2}%] {}/{} vectors | {:.1} vec/s | Pending: {} | Errors: {} | ETA: {}    ",
                        (inserted as f64 / total as f64) * 100.0,
                        inserted,
                        total,
                        rate,
                        pending,
                        errors,
                        eta_str
                    );
                } else {
                    print!(
                        "\r[{:>6.2}%] {}/{} vectors | {:.1} vec/s | Errors: {} | ETA: {}    ",
                        (inserted as f64 / total as f64) * 100.0,
                        inserted,
                        total,
                        rate,
                        errors,
                        eta_str
                    );
                }
                std::io::stdout().flush().ok();
                last_progress = Instant::now();
            }
        }

        writer.flush().await?;
        drop(writer);
        consumer_handle.await??;

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
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
            match Arc::try_unwrap(updater) {
                Ok(owned) => owned.shutdown(),
                Err(_) => {
                    println!("Note: async workers still shutting down in background");
                }
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
            inserted,
            errors,
            build_time,
            inserted as f64 / build_time.max(0.0001),
            peak_rss as f64 / 1_000_000.0
        );

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
        meta.num_vectors = inserted;
        meta.embedding_code = Some(embedding.code());
        meta.storage_type = Some(format!("{:?}", storage_type).to_lowercase());
        if is_random {
            meta.vector_seed = args.seed;
            meta.query_seed = args.seed.saturating_add(1_000_000);
        }
        meta.checkpoint(&args.db_path)?;
        println!(
            "Metadata saved: {:?}",
            BenchmarkMetadata::path(&args.db_path)
        );

        if let Some(output_path) = args.output {
            let (embedding_code, embedding_spec) = resolve_embedding_spec_from_db(
                storage.transaction_db()?,
                Some(embedding.code()),
                None,
                None,
                None,
                None,
            )?;
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
                    "storage_type": format!("{:?}", storage_type).to_lowercase(),
                    "async_workers": args.async_workers,
                    "stream": args.stream,
                },
                "embedding": embedding_spec_json(embedding_code, &embedding_spec),
                "results": {
                    "vectors_inserted": inserted,
                    "errors": errors,
                    "duration_secs": build_time,
                    "throughput_vec_per_sec": inserted as f64 / build_time.max(0.0001),
                    "peak_rss_bytes": peak_rss,
                }
            });

            std::fs::write(&output_path, serde_json::to_string_pretty(&json)?)
                .context("Failed to write output file")?;
            println!("Results saved to: {:?}", output_path);
        }

        return Ok(());
    }

    // Load dataset
    let dataset = load_benchmark_dataset_for_sweep(
        &args.dataset,
        &args.data_dir,
        args.num_vectors,
        0, // index path only needs base vectors
        args.dim,
        args.seed,
    )?;
    let vectors = dataset.vectors().to_vec();
    let dim = dataset.dim();
    println!("Loaded {} vectors, dim={}", vectors.len(), dim);

    let embedding = registry.register(
        EmbeddingBuilder::new(&model, dim as u32, distance)
            .with_hnsw_m(args.m as u16)
            .with_hnsw_ef_construction(args.ef_construction as u16)
            .with_storage_type(storage_type),
    )?;
    print_embedding_summary_from_db(storage.transaction_db()?, &args.db_path, embedding.code())?;

    if existing_count > 0 {
        let txn_db = storage.transaction_db()?;
        let allocator = IdAllocator::recover(&txn_db, embedding.code())?;
        let next_id = allocator.next_id() as usize;
        if next_id < existing_count {
            anyhow::bail!(
                "Existing index allocator state ({}) behind metadata count ({}). Rebuild with --fresh.",
                next_id,
                existing_count
            );
        }
    }

    // Setup mutation writer/consumer
    let (writer, writer_rx) = create_writer(WriterConfig::default());
    let consumer_handle = spawn_mutation_consumer_with_storage_autoreg(
        writer_rx,
        WriterConfig::default(),
        storage.clone(),
    );

    // Insert vectors (skip already indexed)
    let start = Instant::now();
    let vectors_to_add = &vectors[existing_count..];
    let total = vectors_to_add.len();

    println!(
        "\nInserting {} vectors (starting from {})...",
        total, existing_count
    );

    let mut inserted = 0usize;
    let errors = 0usize;
    let mut peak_rss: u64 = 0;
    let mut last_progress = Instant::now();
    let progress_interval = Duration::from_secs(args.progress_interval);

    for batch in vectors_to_add.chunks(args.batch_size) {
        let batch_start = existing_count + inserted;
        let mut payload = Vec::with_capacity(batch.len());

        for (offset, vector) in batch.iter().enumerate() {
            let id = Id::from_bytes(((batch_start + offset) as u128).to_be_bytes());
            // T7.1: Use ExternalKey::NodeId for polymorphic ID mapping
            payload.push((ExternalKey::NodeId(id), vector.clone()));
        }

        let mutation = InsertVectorBatch::new(&embedding, payload).immediate();
        writer.send(vec![mutation.into()]).await?;
        inserted += batch.len();

        if last_progress.elapsed() >= progress_interval || inserted == total {
            let rss = get_rss_bytes();
            peak_rss = peak_rss.max(rss);
            let elapsed = start.elapsed().as_secs_f64();
            let rate = inserted as f64 / elapsed.max(0.0001);
            let remaining = total.saturating_sub(inserted);
            let eta_secs = if rate > 0.0 {
                remaining as f64 / rate
            } else {
                0.0
            };
            let eta_str = format_eta(eta_secs);
            print!(
                "\r[{:>6.2}%] {}/{} vectors | {:.1} vec/s | Errors: {} | ETA: {}    ",
                (inserted as f64 / total as f64) * 100.0,
                inserted,
                total,
                rate,
                errors,
                eta_str
            );
            std::io::stdout().flush().ok();
            last_progress = Instant::now();
        }
    }

    writer.flush().await?;
    drop(writer);
    consumer_handle.await??;

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
        inserted,
        errors,
        build_time,
        inserted as f64 / build_time.max(0.0001),
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
    meta.embedding_code = Some(embedding.code());
    meta.storage_type = Some(format!("{:?}", storage_type).to_lowercase());
    if is_random {
        meta.vector_seed = args.seed;
        meta.query_seed = args.seed.saturating_add(1_000_000);
    }
    meta.checkpoint(&args.db_path)?;
    println!(
        "Metadata saved: {:?}",
        BenchmarkMetadata::path(&args.db_path)
    );

    if let Some(output_path) = args.output {
        let (embedding_code, embedding_spec) = resolve_embedding_spec_from_db(
            storage.transaction_db()?,
            Some(embedding.code()),
            None,
            None,
            None,
            None,
        )?;
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
                "storage_type": format!("{:?}", storage_type).to_lowercase(),
                "async_workers": args.async_workers,
                "stream": args.stream,
            },
            "embedding": embedding_spec_json(embedding_code, &embedding_spec),
            "results": {
                "vectors_inserted": inserted,
                "errors": errors,
                "duration_secs": build_time,
                "throughput_vec_per_sec": inserted as f64 / build_time.max(0.0001),
                "peak_rss_bytes": peak_rss,
            }
        });

        std::fs::write(&output_path, serde_json::to_string_pretty(&json)?)
            .context("Failed to write output file")?;
        println!("Results saved to: {:?}", output_path);
    }

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

    /// Embedding code to use (overrides dataset-based lookup)
    #[arg(long)]
    pub embedding_code: Option<u64>,

    /// Expected storage type for embedding resolution: f32 or f16
    #[arg(long)]
    pub storage_type: Option<String>,

    /// Dataset for queries: laion, sift, gist, cohere, glove, random
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

    /// Random seed for random dataset queries (defaults to metadata if present)
    #[arg(long)]
    pub query_seed: Option<u64>,

    /// Random seed for random dataset vectors (defaults to metadata if present)
    #[arg(long)]
    pub vector_seed: Option<u64>,

    /// Convenience seed for random dataset (sets both vector/query seeds)
    #[arg(long)]
    pub seed: Option<u64>,

    /// Recall sample size for random dataset (0 disables)
    #[arg(long, default_value = "0")]
    pub recall_sample_size: usize,

    /// Skip recall/ground truth computation
    #[arg(long, default_value = "false")]
    pub skip_recall: bool,

    /// Output results to JSON file
    #[arg(long)]
    pub output: Option<PathBuf>,

    /// Read a single query vector from stdin as a JSON array
    #[arg(long, default_value = "false")]
    pub stdin: bool,
}

pub async fn query(args: QueryArgs) -> Result<()> {
    use motlie_db::vector::benchmark::scale::get_rss_bytes;

    println!("=== Vector Search ===");
    println!("Database: {:?}", args.db_path);
    println!("k={}, ef_search={}", args.k, args.ef_search);

    // Load metadata
    if !BenchmarkMetadata::exists(&args.db_path) {
        anyhow::bail!("No index found at {:?}. Run 'index' first.", args.db_path);
    }
    let meta = BenchmarkMetadata::load(&args.db_path)?;
    println!(
        "Index: {} vectors, dim={}, distance={}",
        meta.num_vectors, meta.dim, meta.distance
    );

    // Parse distance
    let distance = match meta.distance.as_str() {
        "cosine" => Distance::Cosine,
        "l2" => Distance::L2,
        _ => Distance::Cosine,
    };

    let meta_storage_type = match meta.storage_type.as_deref() {
        Some(value) => Some(
            parse_storage_type(value)
                .with_context(|| format!("Invalid storage_type in metadata: {}", value))?,
        ),
        None => None,
    };
    let expected_storage_type = match args.storage_type.as_deref() {
        Some(value) => Some(parse_storage_type(value)?),
        None => meta_storage_type,
    };

    let is_random = args.dataset.to_lowercase() == "random";
    let dim = meta.dim;

    let (queries, ground_truth, skip_recall, vector_seed, query_seed) = if is_random {
        let seed = args.seed;
        let vector_seed = seed.or(args.vector_seed).unwrap_or(meta.vector_seed);
        let query_seed = seed.or(args.query_seed).unwrap_or(meta.query_seed);
        let mut skip_recall = args.skip_recall;

        if args.recall_sample_size == 0 && !skip_recall {
            println!("Skipping recall for random dataset (set --recall-sample-size to enable).");
            skip_recall = true;
        }

        (Vec::new(), Vec::new(), skip_recall, vector_seed, query_seed)
    } else {
        // Load queries and ground truth
        let dataset = load_benchmark_dataset_for_sweep(
            &args.dataset,
            &args.data_dir,
            meta.num_vectors,
            args.num_queries,
            dim,
            args.seed.unwrap_or(42),
        )?;
        let db_vectors = dataset.vectors().to_vec();
        let queries = dataset.queries().to_vec();
        println!("Loaded {} queries, dim={}", queries.len(), dim);

        let ground_truth = if args.skip_recall {
            Vec::new()
        } else {
            println!("Computing ground truth...");
            benchmark::compute_ground_truth_parallel(&db_vectors, &queries, args.k, distance)
        };
        (queries, ground_truth, args.skip_recall, 0, 0)
    };

    // Open storage
    let mut storage = Storage::readwrite(&args.db_path);
    storage.ready()?;
    let storage = Arc::new(storage);

    // Resolve embedding spec without mutating the registry/DB.
    let registry = storage.cache().clone();
    registry.set_storage(storage.clone())?;
    let model = format!("bench-{}", args.dataset.to_lowercase());
    let txn_db = storage.transaction_db()?;
    let selected_embedding_code = args.embedding_code.or(meta.embedding_code);
    let (embedding_code, embedding_spec) = if let Some(code) = selected_embedding_code {
        resolve_embedding_spec_from_db(txn_db, Some(code), None, None, None, None)?
    } else {
        resolve_embedding_spec_from_db(
            txn_db,
            None,
            Some(&model),
            Some(dim as u32),
            Some(meta.distance.as_str()),
            expected_storage_type,
        )?
    };

    if embedding_spec.dim != dim as u32 || embedding_spec.distance != distance {
        anyhow::bail!(
            "Embedding spec mismatch for code {}: spec dim={} distance={:?}, metadata dim={} distance={:?}",
            embedding_code,
            embedding_spec.dim,
            embedding_spec.distance,
            dim,
            distance
        );
    }
    if let Some(storage_type) = expected_storage_type {
        if embedding_spec.storage_type != storage_type {
            anyhow::bail!(
                "Embedding storage_type mismatch for code {}: spec {:?}, expected {:?}",
                embedding_code,
                embedding_spec.storage_type,
                storage_type
            );
        }
    }

    if registry.get_by_code(embedding_code).is_none() {
        registry.prewarm()?;
    }
    let embedding = registry
        .get_by_code(embedding_code)
        .ok_or_else(|| anyhow::anyhow!("No embedding found for code {}", embedding_code))?;

    if args.stdin {
        let mut input = String::new();
        std::io::stdin()
            .read_to_string(&mut input)
            .context("Failed to read stdin")?;
        let query: Vec<f32> =
            serde_json::from_str(&input).context("Expected stdin as JSON array of floats")?;

        let (search_reader, query_rx) = create_reader_with_storage(ReaderConfig::default());
        let query_handles = spawn_query_consumers_with_storage_autoreg(
            query_rx,
            ReaderConfig::default(),
            storage.clone(),
            2,
        );

        let timeout = std::time::Duration::from_secs(30);
        let results = SearchKNN::new(&embedding, query, args.k)
            .with_ef(args.ef_search)
            .run(&search_reader, timeout)
            .await?;

        if let Some(output_path) = args.output {
            let json = serde_json::json!({
                "command": "query",
                "config": {
                    "dataset": args.dataset,
                    "k": args.k,
                    "ef_search": args.ef_search,
                    "stdin": true,
                    "storage_type": format!("{:?}", embedding_spec.storage_type).to_lowercase(),
                },
                // T7.2: JSON output with typed external keys
                "results": results.iter().map(|r| serde_json::json!({
                    "vec_id": r.vec_id,
                    "distance": r.distance,
                    "external_key": r.external_key,
                    "external_key_type": r.external_key.variant_name(),
                })).collect::<Vec<_>>(),
            });
            std::fs::write(&output_path, serde_json::to_string_pretty(&json)?)
                .context("Failed to write output file")?;
            println!("Results saved to: {:?}", output_path);
        } else {
            println!("Results:");
            for result in results {
                // T7.1: Use external_key with Debug formatting (supports all ExternalKey variants)
                println!(
                    "  vec_id={} external_key={:?} distance={:.6}",
                    result.vec_id, result.external_key, result.distance
                );
            }
        }

        drop(search_reader);
        for handle in query_handles {
            handle.await??;
        }

        return Ok(());
    }

    // Setup query reader/consumers
    let (search_reader, query_rx) = create_reader_with_storage(ReaderConfig::default());
    let query_handles = spawn_query_consumers_with_storage_autoreg(
        query_rx,
        ReaderConfig::default(),
        storage.clone(),
        2,
    );

    // Run queries
    let query_count = if is_random {
        args.num_queries
    } else {
        queries.len()
    };
    let peak_rss_before = get_rss_bytes();

    println!(
        "\nRunning {} queries (ef_search={})...",
        query_count, args.ef_search
    );
    let search_start = Instant::now();
    let mut latencies = Vec::with_capacity(query_count);
    let mut search_results: Vec<Vec<usize>> = Vec::new();

    let timeout = std::time::Duration::from_secs(30);

    if is_random {
        let mut query_gen =
            motlie_db::vector::benchmark::StreamingVectorGenerator::new_with_distance(
                dim,
                args.num_queries,
                query_seed,
                distance,
            );

        for qi in 0..args.num_queries {
            let query = query_gen.generate_query();
            let start = Instant::now();
            let _results = SearchKNN::new(&embedding, query, args.k)
                .with_ef(args.ef_search)
                .run(&search_reader, timeout)
                .await?;
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

            latencies.push(latency_ms);

            if (qi + 1) % 100 == 0 {
                println!("  {}/{} queries", qi + 1, args.num_queries);
            }
        }
    } else {
        for (qi, query) in queries.iter().enumerate() {
            let start = Instant::now();
            let results = SearchKNN::new(&embedding, query.clone(), args.k)
                .with_ef(args.ef_search)
                .run(&search_reader, timeout)
                .await?;
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

            latencies.push(latency_ms);
            if !skip_recall {
                let result_ids: Vec<usize> = results
                    .iter()
                    .map(|result| result.vec_id as usize)
                    .collect();
                search_results.push(result_ids);
            }

            if (qi + 1) % 100 == 0 {
                println!("  {}/{} queries", qi + 1, queries.len());
            }
        }
    }

    let search_duration = search_start.elapsed();
    let peak_rss_after = get_rss_bytes();

    // Compute metrics
    let mut recall_at_k: Option<f64> = None;

    if !skip_recall && !is_random {
        recall_at_k = Some(compute_recall(&search_results, &ground_truth, args.k));
    }

    if is_random && args.recall_sample_size > 0 && !skip_recall {
        println!(
            "\nComputing recall with {} random sample queries (brute-force ground truth)...",
            args.recall_sample_size
        );

        let mut db_gen = motlie_db::vector::benchmark::StreamingVectorGenerator::new_with_distance(
            dim,
            meta.num_vectors,
            vector_seed,
            distance,
        );
        let mut db_vectors: Vec<Vec<f32>> = Vec::with_capacity(meta.num_vectors);
        while let Some(batch) = db_gen.next_batch(10_000) {
            db_vectors.extend(batch);
            if db_vectors.len() % 100_000 == 0 {
                print!(
                    "\r  Regenerated {}/{} vectors for ground truth...",
                    db_vectors.len(),
                    meta.num_vectors
                );
                std::io::stdout().flush().ok();
            }
        }
        println!(
            "\r  Regenerated {} vectors for ground truth.          ",
            db_vectors.len()
        );

        let recall_seed = query_seed.saturating_add(1_000_000);
        let mut recall_query_gen =
            motlie_db::vector::benchmark::StreamingVectorGenerator::new_with_distance(
                dim,
                args.recall_sample_size,
                recall_seed,
                distance,
            );

        let mut all_search_results: Vec<Vec<usize>> = Vec::with_capacity(args.recall_sample_size);
        let mut all_ground_truth: Vec<Vec<usize>> = Vec::with_capacity(args.recall_sample_size);

        for qi in 0..args.recall_sample_size {
            let query = recall_query_gen.generate_query();

            let mut distances: Vec<(usize, f32)> = db_vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (i, distance.compute(&query, v)))
                .collect();
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let ground_truth: Vec<usize> = distances.iter().take(args.k).map(|(i, _)| *i).collect();
            all_ground_truth.push(ground_truth);

            let results = SearchKNN::new(&embedding, query, args.k)
                .with_ef(args.ef_search)
                .run(&search_reader, timeout)
                .await?;
            let result_indices: Vec<usize> = results
                .iter()
                .map(|result| result.vec_id as usize)
                .collect();
            all_search_results.push(result_indices);

            if (qi + 1) % 20 == 0 || qi + 1 == args.recall_sample_size {
                print!(
                    "\r  Computed {}/{} recall queries...",
                    qi + 1,
                    args.recall_sample_size
                );
                std::io::stdout().flush().ok();
            }
        }
        println!();

        recall_at_k = Some(compute_recall(
            &all_search_results,
            &all_ground_truth,
            args.k,
        ));
    }
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let stats = LatencyStats::from_latencies(&latencies);

    println!("\n=== Results ===");
    if let Some(recall) = recall_at_k {
        println!("Recall@{}: {:.1}%", args.k, recall * 100.0);
    } else {
        println!("Recall@{}: skipped", args.k);
    }
    println!("QPS: {:.1}", stats.qps);
    println!(
        "Latency: avg={:.2}ms, p50={:.2}ms, p95={:.2}ms, p99={:.2}ms",
        stats.avg_ms, stats.p50_ms, stats.p95_ms, stats.p99_ms
    );
    let peak_rss = peak_rss_after.max(peak_rss_before);
    println!("Peak RSS: {:.2} MB", peak_rss as f64 / 1_000_000.0);

    if let Some(output_path) = args.output {
        let json = serde_json::json!({
            "command": "query",
            "config": {
                "dataset": args.dataset,
                "num_queries": query_count,
                "recall_sample_size": args.recall_sample_size,
                "k": args.k,
                "ef_search": args.ef_search,
                "skip_recall": skip_recall,
                "embedding_code": embedding_code,
                "storage_type": format!("{:?}", embedding_spec.storage_type).to_lowercase(),
            },
            "embedding": embedding_spec_json(embedding_code, &embedding_spec),
            "results": {
                "recall_at_k": recall_at_k,
                "qps": stats.qps,
                "latency_avg_ms": stats.avg_ms,
                "latency_p50_ms": stats.p50_ms,
                "latency_p95_ms": stats.p95_ms,
                "latency_p99_ms": stats.p99_ms,
                "search_duration_secs": search_duration.as_secs_f64(),
                "peak_rss_bytes": peak_rss,
            }
        });

        std::fs::write(&output_path, serde_json::to_string_pretty(&json)?)
            .context("Failed to write output file")?;
        println!("Results saved to: {:?}", output_path);
    }

    drop(search_reader);
    for handle in query_handles {
        handle.await??;
    }

    Ok(())
}

// ============================================================================
// Sweep Command
// ============================================================================

#[derive(Parser)]
pub struct SweepArgs {
    /// Dataset: laion, sift, gist, cohere, glove, random
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

    /// Vector storage type: f32 (full precision) or f16 (half precision, 50% smaller)
    #[arg(long, default_value = "f32")]
    pub storage_type: String,
}

pub async fn sweep(args: SweepArgs) -> Result<()> {
    let storage_type = parse_storage_type(&args.storage_type)?;

    println!("=== Parameter Sweep ===");
    println!("Dataset: {}", args.dataset);
    println!(
        "Vectors: {}, Queries: {}",
        args.num_vectors, args.num_queries
    );
    println!("Storage type: {:?}", storage_type);

    // Parse parameters
    let ef_values: Vec<usize> = args
        .ef
        .split(',')
        .map(|s| s.trim().parse().expect("Invalid ef value"))
        .collect();
    let k_values: Vec<usize> = args
        .k
        .split(',')
        .map(|s| s.trim().parse().expect("Invalid k value"))
        .collect();
    let bits_values: Vec<u8> = args
        .bits
        .split(',')
        .map(|s| s.trim().parse().expect("Invalid bits value"))
        .collect();
    let rerank_values: Vec<usize> = args
        .rerank
        .split(',')
        .map(|s| s.trim().parse().expect("Invalid rerank value"))
        .collect();

    println!("ef_search: {:?}", ef_values);
    println!("k: {:?}", k_values);
    if args.rabitq {
        println!("RaBitQ bits: {:?}", bits_values);
        println!("Rerank factors: {:?}", rerank_values);
    }

    let dataset = load_benchmark_dataset_for_sweep(
        &args.dataset,
        &args.data_dir,
        args.num_vectors,
        args.num_queries,
        args.dim,
        args.seed,
    )?;
    let dim = dataset.dim();
    let distance = dataset.distance();
    println!(
        "Loaded dataset: {} db vectors, {} queries, dim={}, distance={:?}",
        dataset.vectors().len(),
        dataset.queries().len(),
        dim,
        distance
    );

    let max_k = *k_values.iter().max().unwrap_or(&10);
    let ground_truth = match dataset.ground_truth(max_k) {
        Some(gt) => gt,
        None => benchmark::compute_ground_truth_parallel(
            dataset.vectors(),
            dataset.queries(),
            max_k,
            distance,
        ),
    };

    // Create temp database
    let temp_path = std::env::temp_dir().join(format!("bench_vector_{}", std::process::id()));
    if temp_path.exists() {
        std::fs::remove_dir_all(&temp_path)?;
    }
    std::fs::create_dir_all(&temp_path)?;
    let db_path = &temp_path;
    println!("Database: {:?}", db_path);

    // Initialize storage + build index through benchmark crate
    let mut storage = Storage::readwrite(db_path);
    storage.ready()?;
    let storage = Arc::new(storage);
    let (index, embedding, build_time) = benchmark::build_hnsw_index(
        &storage,
        dataset.vectors(),
        dim,
        args.m,
        args.ef_construction,
        distance,
        storage_type,
    )?;
    println!(
        "Index built in {:.2}s ({:.1} vec/s)",
        build_time,
        dataset.vectors().len() as f64 / build_time.max(0.0001)
    );

    // Create results directory
    std::fs::create_dir_all(&args.results_dir)?;

    // Track minimum recall across all sweeps
    let mut overall_min_recall: f64 = 1.0;

    if args.rabitq {
        let mut config = benchmark::ExperimentConfig::default()
            .with_ef_search(ef_values.clone())
            .with_k_values(k_values.clone())
            .with_rabitq_bits(bits_values.clone())
            .with_rerank_factors(rerank_values.clone());
        config.dim = dim;
        config.distance = distance;

        let sweep_modes: Vec<(&str, bool)> = if args.compare_simd {
            vec![("simd", true), ("scalar", false)]
        } else {
            vec![("single", args.simd_dot)]
        };

        for (mode_name, use_simd_dot) in sweep_modes {
            if args.compare_simd {
                println!("\n=== RaBitQ mode: {} ===", mode_name);
            } else {
                println!("\n=== RaBitQ mode: simd_dot={} ===", use_simd_dot);
            }

            let rabitq_results = benchmark::run_rabitq_experiments(
                &config,
                &index,
                storage.as_ref(),
                dataset.as_ref(),
                &ground_truth,
                build_time,
                embedding.code(),
                use_simd_dot,
            )?;

            let csv_path = if args.compare_simd {
                args.results_dir
                    .join(format!("rabitq_sweep_{}.csv", mode_name))
            } else {
                args.results_dir.join("rabitq_sweep.csv")
            };
            save_rabitq_results_csv(&rabitq_results, &csv_path)?;
            println!("RaBitQ sweep saved to {:?}", csv_path);

            if let Some(min) = rabitq_results
                .iter()
                .map(|r| r.recall_mean)
                .min_by(|a, b| a.partial_cmp(b).unwrap())
            {
                overall_min_recall = overall_min_recall.min(min);
            }

            if args.show_pareto {
                let pareto_inputs: Vec<ParetoInput> = rabitq_results
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
                for &k in &k_values {
                    let frontier = benchmark::compute_pareto_frontier_for_k(&pareto_inputs, k);
                    if !frontier.is_empty() {
                        let title = if args.compare_simd {
                            format!("Pareto Frontier ({} mode, Recall@{} vs QPS)", mode_name, k)
                        } else {
                            format!("Pareto Frontier (Recall@{} vs QPS)", k)
                        };
                        print_pareto_frontier(&frontier, &title);
                    }
                }
            }
        }
    } else {
        println!("\n=== HNSW Parameter Sweep ===");
        for &ef_search in &ef_values {
            let result = benchmark::run_single_experiment(
                &embedding,
                &storage,
                dataset.as_ref(),
                &ground_truth,
                &k_values,
                ef_search,
                dataset.vectors().len(),
                build_time,
                &format!("HNSW-{}", distance),
                false,
            )?;
            if let Some(min) = result
                .recall_at_k
                .values()
                .cloned()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
            {
                overall_min_recall = overall_min_recall.min(min);
            }
        }

        let flat = benchmark::run_flat_baseline(
            dataset.as_ref(),
            &ground_truth,
            &k_values,
            dataset.vectors().len(),
            false,
        )?;
        if let Some(min) = flat
            .recall_at_k
            .values()
            .cloned()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
        {
            overall_min_recall = overall_min_recall.min(min);
        }
    }

    // Check recall assertion if threshold provided
    if let Some(threshold) = args.assert_recall {
        println!("\n=== Recall Assertion ===");
        println!(
            "Minimum recall observed: {:.1}%",
            overall_min_recall * 100.0
        );
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

fn format_eta(secs: f64) -> String {
    if secs > 86400.0 * 365.0 {
        "N/A".to_string()
    } else if secs > 3600.0 {
        format!(
            "{}h{}m",
            (secs / 3600.0) as u64,
            ((secs % 3600.0) / 60.0) as u64
        )
    } else if secs > 60.0 {
        format!("{}m{}s", (secs / 60.0) as u64, (secs % 60.0) as u64)
    } else {
        format!("{}s", secs as u64)
    }
}

fn parse_distance(value: &str) -> Result<Distance> {
    match value.to_lowercase().as_str() {
        "cosine" => Ok(Distance::Cosine),
        "l2" | "euclidean" => Ok(Distance::L2),
        "dot" | "dotproduct" => Ok(Distance::DotProduct),
        other => anyhow::bail!("Invalid distance: {} (use cosine, l2, or dot)", other),
    }
}

/// Load embedding specs from an already-open TransactionDB.
///
/// Use this when storage is already open to avoid double-locking RocksDB.
fn load_embedding_specs_from_db(db: &rocksdb::TransactionDB) -> Result<Vec<(u64, EmbeddingSpec)>> {
    let cf = db
        .cf_handle(EmbeddingSpecs::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("EmbeddingSpecs CF not found"))?;
    let iter = db.iterator_cf(&cf, rocksdb::IteratorMode::Start);

    let mut specs = Vec::new();
    for item in iter {
        let (key_bytes, value_bytes) = item?;
        let key = EmbeddingSpecs::key_from_bytes(&key_bytes)?;
        let value = EmbeddingSpecs::value_from_bytes(&value_bytes)?;
        specs.push((key.0, value.0));
    }

    Ok(specs)
}

/// Resolve an embedding spec from an already-open TransactionDB.
fn resolve_embedding_spec_from_db(
    db: &rocksdb::TransactionDB,
    code: Option<u64>,
    model: Option<&str>,
    dim: Option<u32>,
    distance: Option<&str>,
    storage_type: Option<VectorElementType>,
) -> Result<(u64, EmbeddingSpec)> {
    let specs = load_embedding_specs_from_db(db)?;
    find_embedding_spec(specs, code, model, dim, distance, storage_type)
}

fn find_embedding_spec(
    specs: Vec<(u64, EmbeddingSpec)>,
    code: Option<u64>,
    model: Option<&str>,
    dim: Option<u32>,
    distance: Option<&str>,
    storage_type: Option<VectorElementType>,
) -> Result<(u64, EmbeddingSpec)> {
    if let Some(code) = code {
        return specs
            .into_iter()
            .find(|(existing, _)| *existing == code)
            .ok_or_else(|| anyhow::anyhow!("No embedding found for code {}", code));
    }

    let model = model.ok_or_else(|| anyhow::anyhow!("--model is required without --code"))?;
    let dim = dim.ok_or_else(|| anyhow::anyhow!("--dim is required without --code"))?;
    let distance_str =
        distance.ok_or_else(|| anyhow::anyhow!("--distance is required without --code"))?;
    let distance = parse_distance(distance_str)?;

    let matches: Vec<(u64, EmbeddingSpec)> = specs
        .into_iter()
        .filter(|(_, spec)| {
            spec.model == model
                && spec.dim == dim
                && spec.distance == distance
                && storage_type.map_or(true, |st| spec.storage_type == st)
        })
        .collect();

    if matches.is_empty() {
        return match storage_type {
            Some(st) => Err(anyhow::anyhow!(
                "No embedding found for model={}, dim={}, distance={}, storage_type={:?}",
                model,
                dim,
                distance_str,
                st
            )),
            None => Err(anyhow::anyhow!(
                "No embedding found for model={}, dim={}, distance={}",
                model,
                dim,
                distance_str
            )),
        };
    }

    if matches.len() > 1 && storage_type.is_none() {
        let choices = matches
            .iter()
            .map(|(code, spec)| format!("code={} storage={:?}", code, spec.storage_type))
            .collect::<Vec<_>>()
            .join(", ");
        anyhow::bail!(
            "Multiple embeddings match model={}, dim={}, distance={}. Specify --embedding-code or --storage-type. Matches: {}",
            model,
            dim,
            distance_str,
            choices
        );
    }

    Ok(matches.into_iter().next().expect("non-empty matches"))
}

fn embedding_spec_json(code: u64, spec: &EmbeddingSpec) -> serde_json::Value {
    serde_json::json!({
        "code": code,
        "model": spec.model,
        "dim": spec.dim,
        "distance": format!("{:?}", spec.distance).to_lowercase(),
        "storage_type": format!("{:?}", spec.storage_type).to_lowercase(),
        "hnsw_m": spec.hnsw_m,
        "hnsw_ef_construction": spec.hnsw_ef_construction,
        "rabitq_bits": spec.rabitq_bits,
        "rabitq_seed": spec.rabitq_seed,
    })
}

/// Print embedding summary from an already-open TransactionDB.
fn print_embedding_summary_from_db(
    db: &rocksdb::TransactionDB,
    db_path: &PathBuf,
    code: u64,
) -> Result<()> {
    let (code, spec) = resolve_embedding_spec_from_db(db, Some(code), None, None, None, None)?;
    println!("\nEmbedding registered:");
    println!("  Code: {}", code);
    println!("  Model: {}", spec.model);
    println!("  Dim: {}", spec.dim);
    println!("  Distance: {:?}", spec.distance);
    println!("  Storage: {:?}", spec.storage_type);
    println!("  HNSW M: {}", spec.hnsw_m);
    println!("  HNSW ef_construction: {}", spec.hnsw_ef_construction);
    println!("  RaBitQ bits: {}", spec.rabitq_bits);
    println!("  RaBitQ seed: {}", spec.rabitq_seed);
    println!(
        "  Inspect: bench_vector embeddings inspect --db-path {} --code {}",
        db_path.display(),
        code
    );
    Ok(())
}

/// Load vectors from an already-open TransactionDB using reservoir sampling.
fn load_vectors_for_embedding_from_db(
    db: &rocksdb::TransactionDB,
    embedding: u64,
    storage_type: VectorElementType,
    limit: usize,
    seed: u64,
) -> Result<Vec<(VecId, Vec<f32>)>> {
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    let cf = db
        .cf_handle(Vectors::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("Vectors CF not found"))?;

    let prefix = embedding.to_be_bytes();
    let iter = db.iterator_cf(
        &cf,
        rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
    );

    // Reservoir sampling (Algorithm R) for unbiased random sample
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut reservoir: Vec<(VecId, Vec<f32>)> = Vec::with_capacity(limit);
    let mut count = 0usize;

    for item in iter {
        let (key_bytes, value_bytes) = item?;
        if !key_bytes.starts_with(&prefix) {
            break;
        }
        let key = Vectors::key_from_bytes(&key_bytes)?;
        if key.0 != embedding {
            continue;
        }
        let vector = Vectors::value_from_bytes_typed(&value_bytes, storage_type)?;

        if count < limit {
            // Fill reservoir
            reservoir.push((key.1, vector));
        } else {
            // Replace with probability limit / (count + 1)
            let j = rng.random_range(0..=count);
            if j < limit {
                reservoir[j] = (key.1, vector);
            }
        }
        count += 1;
    }

    println!(
        "Reservoir sampled {} vectors from {} total",
        reservoir.len(),
        count
    );
    Ok(reservoir)
}

fn load_benchmark_dataset_for_sweep(
    dataset: &str,
    data_dir: &PathBuf,
    num_vectors: usize,
    num_queries: usize,
    dim: usize,
    seed: u64,
) -> Result<Box<dyn benchmark::Dataset>> {
    match dataset.to_lowercase().as_str() {
        "laion" => {
            let ds = LaionDataset::load(data_dir, num_vectors)?;
            Ok(Box::new(ds.subset(num_vectors, num_queries)))
        }
        "sift" => {
            let ds = SiftDataset::load(data_dir, num_vectors, num_queries)?;
            Ok(Box::new(ds.subset(num_vectors, num_queries)))
        }
        "gist" => {
            let ds = GistDataset::load(data_dir, num_vectors, num_queries)?;
            Ok(Box::new(ds.subset(num_vectors, num_queries)))
        }
        "cohere" | "cohere-wiki" => {
            let ds = benchmark::CohereWikipediaDataset::load(data_dir, num_vectors, num_queries)?;
            Ok(Box::new(ds.subset(num_vectors, num_queries)))
        }
        "glove" | "glove-100" => {
            let ds = benchmark::GloveDataset::load(data_dir, num_vectors, num_queries)?;
            Ok(Box::new(ds.subset(num_vectors, num_queries)))
        }
        "random" => Ok(Box::new(RandomDataset::generate(
            num_vectors,
            num_queries,
            dim,
            seed,
        ))),
        _ => anyhow::bail!(
            "Unknown dataset: {}. Use: laion, sift, gist, cohere, glove, random",
            dataset
        ),
    }
}

// ============================================================================
// Check Distribution Command
// ============================================================================

#[derive(Parser)]
pub struct CheckDistributionArgs {
    /// Dataset: laion, sift, gist, cohere, glove, random
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

    // Load vectors through shared benchmark dataset loader
    let dataset = load_benchmark_dataset_for_sweep(
        &args.dataset,
        &args.data_dir,
        args.sample_size,
        0, // no query vectors needed for distribution check
        args.dim,
        args.seed,
    )?;
    let vectors = dataset.vectors().to_vec();
    let dim = dataset.dim();

    println!("Loaded {} vectors, dim={}", vectors.len(), dim);

    // Create RaBitQ encoder
    let encoder = RaBitQ::new(dim, args.bits, args.seed);

    // Compute rotation statistics
    let stats = compute_rotated_variance(|v| encoder.rotate_query(v), &vectors, args.sample_size);

    // Print results
    print_rotation_stats(
        &stats,
        &format!("RaBitQ Rotation Stats ({}-bit)", args.bits),
    );

    // Provide interpretation
    println!("\nInterpretation:");
    if stats.scaling_valid {
        println!("  ✓ Rotation matrix has correct √D scaling");
        println!("  ✓ RaBitQ should work well on this dataset");
    } else if stats.component_variance < 0.1 {
        println!(
            "  ✗ Variance too low ({:.4}) - rotation matrix may lack √D scaling",
            stats.component_variance
        );
        println!("  ✗ This will cause poor quantization quality");
    } else if stats.component_variance > 2.0 {
        println!(
            "  ✗ Variance too high ({:.4}) - vectors may not be normalized",
            stats.component_variance
        );
        println!("  ✗ Consider normalizing vectors before indexing");
    } else {
        println!(
            "  ⚠ Variance ({:.4}) outside expected range [0.8, 1.2]",
            stats.component_variance
        );
        println!("  ⚠ RaBitQ may have suboptimal recall on this dataset");
    }

    Ok(())
}

// ============================================================================
// Embeddings Command
// ============================================================================

#[derive(Parser)]
pub struct EmbeddingsArgs {
    #[command(subcommand)]
    pub command: EmbeddingsCommand,
}

#[derive(Parser)]
pub enum EmbeddingsCommand {
    /// List embedding specs in the registry
    List(EmbeddingsListArgs),
    /// Inspect a specific embedding spec
    Inspect(EmbeddingsInspectArgs),
    /// Generate ground truth vectors and matches for testing
    Groundtruth(EmbeddingsGroundtruthArgs),
}

#[derive(Parser)]
pub struct EmbeddingsListArgs {
    /// Database path
    #[arg(long)]
    pub db_path: PathBuf,

    /// Filter by model name
    #[arg(long)]
    pub model: Option<String>,

    /// Filter by dimension
    #[arg(long)]
    pub dim: Option<u32>,

    /// Filter by distance (cosine, l2, dot)
    #[arg(long)]
    pub distance: Option<String>,
}

#[derive(Parser)]
pub struct EmbeddingsInspectArgs {
    /// Database path
    #[arg(long)]
    pub db_path: PathBuf,

    /// Embedding code to inspect
    #[arg(long)]
    pub code: Option<u64>,

    /// Model name (required if code is not provided)
    #[arg(long)]
    pub model: Option<String>,

    /// Dimension (required if code is not provided)
    #[arg(long)]
    pub dim: Option<u32>,

    /// Distance (cosine, l2, dot) (required if code is not provided)
    #[arg(long)]
    pub distance: Option<String>,
}

#[derive(Parser)]
pub struct EmbeddingsGroundtruthArgs {
    /// Database path
    #[arg(long)]
    pub db_path: PathBuf,

    /// Embedding code to use
    #[arg(long)]
    pub code: Option<u64>,

    /// Model name (required if code is not provided)
    #[arg(long)]
    pub model: Option<String>,

    /// Dimension (required if code is not provided)
    #[arg(long)]
    pub dim: Option<u32>,

    /// Distance (cosine, l2, dot) (required if code is not provided)
    #[arg(long)]
    pub distance: Option<String>,

    /// Number of vectors to sample
    #[arg(long, default_value = "1000")]
    pub count: usize,

    /// Number of query vectors to generate
    #[arg(long, default_value = "10")]
    pub queries: usize,

    /// k for top-k matches
    #[arg(long, default_value = "10")]
    pub k: usize,

    /// RNG seed for query generation
    #[arg(long, default_value = "42")]
    pub seed: u64,

    /// Output JSON file (defaults to stdout)
    #[arg(long)]
    pub output: Option<PathBuf>,
}

pub fn embeddings(args: EmbeddingsArgs) -> Result<()> {
    match args.command {
        EmbeddingsCommand::List(args) => embeddings_list(args),
        EmbeddingsCommand::Inspect(args) => embeddings_inspect(args),
        EmbeddingsCommand::Groundtruth(args) => embeddings_groundtruth(args),
    }
}

fn embeddings_list(args: EmbeddingsListArgs) -> Result<()> {
    let mut storage = Storage::readwrite(&args.db_path);
    storage.ready()?;
    let db = storage.transaction_db()?;
    let specs = load_embedding_specs_from_db(db)?;
    let distance = match args.distance.as_deref() {
        Some(value) => Some(parse_distance(value)?),
        None => None,
    };

    let mut rows = Vec::new();
    for (code, spec) in specs {
        if let Some(ref model) = args.model {
            if spec.model != *model {
                continue;
            }
        }
        if let Some(dim) = args.dim {
            if spec.dim != dim {
                continue;
            }
        }
        if let Some(distance) = distance {
            if spec.distance != distance {
                continue;
            }
        }
        rows.push((code, spec));
    }

    if rows.is_empty() {
        println!("No embeddings found.");
        return Ok(());
    }

    println!(
        "{:<6} {:<24} {:<6} {:<8} {:<6} {:<6} {:<6} {:<6}",
        "Code", "Model", "Dim", "Distance", "M", "ef", "Bits", "Seed"
    );
    println!("{}", "-".repeat(84));

    for (code, spec) in rows {
        println!(
            "{:<6} {:<24} {:<6} {:<8} {:<6} {:<6} {:<6} {:<6}",
            code,
            spec.model,
            spec.dim,
            format!("{:?}", spec.distance).to_lowercase(),
            spec.hnsw_m,
            spec.hnsw_ef_construction,
            spec.rabitq_bits,
            spec.rabitq_seed
        );
    }

    Ok(())
}

fn embeddings_inspect(args: EmbeddingsInspectArgs) -> Result<()> {
    let mut storage = Storage::readwrite(&args.db_path);
    storage.ready()?;
    let db = storage.transaction_db()?;
    let (code, spec) = resolve_embedding_spec_from_db(
        db,
        args.code,
        args.model.as_deref(),
        args.dim,
        args.distance.as_deref(),
        None,
    )?;

    println!("Code: {}", code);
    println!("Model: {}", spec.model);
    println!("Dim: {}", spec.dim);
    println!("Distance: {:?}", spec.distance);
    println!("Storage: {:?}", spec.storage_type);
    println!("HNSW M: {}", spec.hnsw_m);
    println!("HNSW ef_construction: {}", spec.hnsw_ef_construction);
    println!("RaBitQ bits: {}", spec.rabitq_bits);
    println!("RaBitQ seed: {}", spec.rabitq_seed);

    Ok(())
}

fn embeddings_groundtruth(args: EmbeddingsGroundtruthArgs) -> Result<()> {
    let mut storage = Storage::readwrite(&args.db_path);
    storage.ready()?;
    let db = storage.transaction_db()?;
    let (code, spec) = resolve_embedding_spec_from_db(
        db,
        args.code,
        args.model.as_deref(),
        args.dim,
        args.distance.as_deref(),
        None,
    )?;

    let vectors =
        load_vectors_for_embedding_from_db(db, code, spec.storage_type, args.count, args.seed)?;
    if vectors.is_empty() {
        anyhow::bail!("No vectors found for embedding {}", code);
    }
    if vectors.len() < args.count {
        println!(
            "Warning: Requested {} vectors but only {} found for embedding {}",
            args.count,
            vectors.len(),
            code
        );
    }

    let mut query_gen = motlie_db::vector::benchmark::StreamingVectorGenerator::new_with_distance(
        spec.dim as usize,
        args.queries,
        args.seed,
        spec.distance,
    );

    let mut queries = Vec::with_capacity(args.queries);
    let mut results = Vec::with_capacity(args.queries);

    for _ in 0..args.queries {
        let query = query_gen.generate_query();

        let mut distances: Vec<(VecId, f32)> = vectors
            .iter()
            .map(|(vec_id, vec)| (*vec_id, spec.distance.compute(&query, vec)))
            .collect();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let matches: Vec<serde_json::Value> = distances
            .iter()
            .take(args.k)
            .map(|(vec_id, dist)| {
                serde_json::json!({
                    "vec_id": vec_id,
                    "distance": dist,
                })
            })
            .collect();

        queries.push(serde_json::json!({ "vector": query }));
        results.push(serde_json::json!({
            "matches": matches,
        }));
    }

    let payload = serde_json::json!({
        "embedding": {
            "code": code,
            "model": spec.model,
            "dim": spec.dim,
            "distance": format!("{:?}", spec.distance).to_lowercase(),
            "storage_type": format!("{:?}", spec.storage_type),
            "hnsw_m": spec.hnsw_m,
            "hnsw_ef_construction": spec.hnsw_ef_construction,
            "rabitq_bits": spec.rabitq_bits,
            "rabitq_seed": spec.rabitq_seed,
        },
        "vectors": vectors.iter().map(|(vec_id, vec)| serde_json::json!({
            "vec_id": vec_id,
            "vector": vec,
        })).collect::<Vec<_>>(),
        "queries": queries.iter().zip(results.iter()).map(|(query, result)| {
            serde_json::json!({
                "vector": query["vector"].clone(),
                "matches": result["matches"].clone(),
            })
        }).collect::<Vec<_>>(),
        "k": args.k,
        "seed": args.seed,
    });

    let output = serde_json::to_string_pretty(&payload)?;
    if let Some(path) = args.output {
        std::fs::write(&path, output)?;
        println!("Ground truth saved to: {:?}", path);
    } else {
        println!("{}", output);
    }

    Ok(())
}

// ============================================================================
// Datasets Command
// ============================================================================
// List Datasets Command
// ============================================================================

pub fn list_datasets() -> Result<()> {
    println!("Available datasets:");
    println!("  laion       - LAION-CLIP 512D (NPY, Cosine)");
    println!("  sift        - SIFT-1M 128D (fvecs, L2)");
    println!("  gist        - GIST-960 960D (fvecs, L2)");
    println!("  random      - Synthetic unit-normalized (configurable dim, Cosine)");
    println!("  cohere      - Cohere Wikipedia 768D (Parquet, Cosine)");
    println!("  glove       - GloVe-100 100D (HDF5, Angular)");

    // ADMIN.md Issue #2 (chungers, 2025-01-27, FIXED):
    // Updated examples to show --embedding-code for query command.
    println!("\n--- Usage Examples ---\n");

    println!("Download a dataset:");
    println!("  bench_vector download --dataset laion --data-dir ./data\n");

    println!("Build an index:");
    println!("  bench_vector index --dataset laion --num-vectors 100000 --db-path ./bench_db\n");

    println!("List embeddings (to find embedding code):");
    println!("  bench_vector embeddings list --db-path ./bench_db\n");

    println!("Run queries (use embedding code from 'embeddings list'):");
    println!("  bench_vector query --db-path ./bench_db --embedding-code 1 --dataset laion --k 10 --ef-search 100\n");

    println!("Parameter sweep (HNSW):");
    println!("  bench_vector sweep --dataset laion --num-vectors 50000 --ef 50,100,200 --k 1,10\n");

    println!("Parameter sweep (RaBitQ):");
    println!("  bench_vector sweep --dataset laion --num-vectors 50000 --rabitq --bits 1,2,4 --rerank 1,4,10,20\n");

    println!("Parameter sweep (Random dataset):");
    println!("  bench_vector sweep --dataset random --dim 1024 --num-vectors 100000 --num-queries 1000 --rabitq\n");

    println!("Parameter sweep with Pareto frontier:");
    println!("  bench_vector sweep --dataset laion --rabitq --show-pareto\n");

    println!("Check RaBitQ distribution (√D scaling validation):");
    println!("  bench_vector check-distribution --dataset random --dim 1024 --sample-size 1000\n");

    println!("Admin diagnostics:");
    println!("  bench_vector admin stats --db-path ./bench_db");
    println!("  bench_vector admin validate --db-path ./bench_db");

    Ok(())
}

// ============================================================================
// Admin Command
// ============================================================================

#[derive(Parser)]
pub struct AdminArgs {
    #[command(subcommand)]
    pub command: AdminCommand,
}

#[derive(Parser)]
pub enum AdminCommand {
    /// Show storage statistics for embeddings
    Stats(AdminStatsArgs),
    /// Deep inspection of a specific embedding
    Inspect(AdminInspectArgs),
    /// Vector-level diagnostics
    Vectors(AdminVectorsArgs),
    /// Run consistency validation checks
    Validate(AdminValidateArgs),
    /// RocksDB-level diagnostics
    Rocksdb(AdminRocksdbArgs),
    /// Migrate lifecycle counters from VecMeta scan (one-time upgrade)
    MigrateLifecycleCounts(AdminMigrateCountsArgs),
}

#[derive(Parser)]
pub struct AdminStatsArgs {
    /// Database path
    #[arg(long)]
    pub db_path: PathBuf,

    /// Embedding code (optional, shows all if not specified)
    #[arg(long)]
    pub code: Option<u64>,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,

    // ADMIN.md Read-Only Mode (chungers, 2025-01-27, FIXED):
    // Added --secondary flag for read-only access via secondary DB instance.
    // This avoids contention with live workloads and prevents accidental writes.
    /// Open database in secondary (read-only) mode
    #[arg(long)]
    pub secondary: bool,
}

#[derive(Parser)]
pub struct AdminInspectArgs {
    /// Database path
    #[arg(long)]
    pub db_path: PathBuf,

    /// Embedding code to inspect
    #[arg(long)]
    pub code: u64,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,

    /// Open database in secondary (read-only) mode
    #[arg(long)]
    pub secondary: bool,
}

#[derive(Parser)]
pub struct AdminVectorsArgs {
    /// Database path
    #[arg(long)]
    pub db_path: PathBuf,

    /// Embedding code
    #[arg(long)]
    pub code: u64,

    /// Filter by lifecycle state: indexed, pending, deleted
    #[arg(long)]
    pub state: Option<String>,

    /// Inspect specific vector ID
    #[arg(long)]
    pub vec_id: Option<u32>,

    /// Sample random vectors (specify count)
    #[arg(long)]
    pub sample: Option<usize>,

    /// Random seed for sampling
    #[arg(long, default_value = "42")]
    pub seed: u64,

    /// Maximum results to return
    #[arg(long, default_value = "20")]
    pub limit: usize,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,

    /// Open database in secondary (read-only) mode
    #[arg(long)]
    pub secondary: bool,
}

#[derive(Parser)]
pub struct AdminValidateArgs {
    /// Database path
    #[arg(long)]
    pub db_path: PathBuf,

    /// Embedding code (optional, validates all if not specified)
    #[arg(long)]
    pub code: Option<u64>,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,

    /// Open database in secondary (read-only) mode
    #[arg(long)]
    pub secondary: bool,

    // ADMIN.md Stronger Validation (chungers, 2025-01-27, ADDED):
    // --strict flag for full-scan validation without sampling.
    /// Strict mode: full scan without sampling (slower but complete)
    #[arg(long)]
    pub strict: bool,

    // ADMIN.md Validation UX (claude, 2025-01-29, ADDED):
    // Configurable sample size, max errors, and max entries.
    /// Sample size for non-strict validation (default 1000, 0 for no limit)
    #[arg(long, default_value = "1000")]
    pub sample_size: u32,

    /// Maximum errors before stopping validation (0 = no limit)
    #[arg(long, default_value = "0")]
    pub max_errors: u32,

    /// Maximum entries to check in strict mode (0 = no limit)
    #[arg(long, default_value = "0")]
    pub max_entries: u64,
}

#[derive(Parser)]
pub struct AdminRocksdbArgs {
    /// Database path
    #[arg(long)]
    pub db_path: PathBuf,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,

    /// Open database in secondary (read-only) mode
    #[arg(long)]
    pub secondary: bool,
}

#[derive(Parser)]
pub struct AdminMigrateCountsArgs {
    /// Database path
    #[arg(long)]
    pub db_path: PathBuf,

    /// Embedding code (optional, migrates all if not specified)
    #[arg(long)]
    pub code: Option<u64>,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

pub fn admin(args: AdminArgs) -> Result<()> {
    match args.command {
        AdminCommand::Stats(args) => admin_stats(args),
        AdminCommand::Inspect(args) => admin_inspect(args),
        AdminCommand::Vectors(args) => admin_vectors(args),
        AdminCommand::Validate(args) => admin_validate(args),
        AdminCommand::Rocksdb(args) => admin_rocksdb(args),
        AdminCommand::MigrateLifecycleCounts(args) => admin_migrate_lifecycle_counts(args),
    }
}

fn admin_stats(args: AdminStatsArgs) -> Result<()> {
    use motlie_db::vector::admin;

    // ADMIN.md Read-Only Mode (chungers, 2025-01-27, FIXED):
    // Use secondary mode for read-only access when --secondary flag is set.
    // ADMIN.md Secondary Cleanup (claude, 2025-01-27, FIXED):
    // Clean up temp directory after use to avoid accumulation.
    let stats = if args.secondary {
        let secondary_path = std::env::temp_dir().join(format!(
            "bench_vector_secondary_{}_{}_{:08x}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis(),
            rand::random::<u32>()
        ));
        let result = {
            let mut storage = Storage::secondary(&args.db_path, &secondary_path);
            storage.ready()?;

            if let Some(code) = args.code {
                vec![admin::get_embedding_stats_secondary(&storage, code)?]
            } else {
                admin::get_all_stats_secondary(&storage)?
            }
        };
        // Clean up secondary DB files
        let _ = std::fs::remove_dir_all(&secondary_path);
        result
    } else {
        let mut storage = Storage::readwrite(&args.db_path);
        storage.ready()?;

        if let Some(code) = args.code {
            vec![admin::get_embedding_stats(&storage, code)?]
        } else {
            admin::get_all_stats(&storage)?
        }
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&stats)?);
    } else {
        print_stats(&stats);
    }

    Ok(())
}

fn print_stats(stats: &[motlie_db::vector::admin::EmbeddingStats]) {
    if stats.is_empty() {
        println!("No embeddings found.");
        return;
    }

    println!("=== Vector Storage Statistics ===\n");

    for stat in stats {
        println!("Embedding {} ({}):", stat.code, stat.model);
        println!(
            "  Dimension: {}  Distance: {}  Storage: {}",
            stat.dim, stat.distance, stat.storage_type
        );
        println!("  Vectors: {} total", stat.total_vectors);
        println!(
            "    Indexed:        {:>8} ({:.1}%)",
            stat.indexed_count,
            if stat.total_vectors > 0 {
                stat.indexed_count as f64 / stat.total_vectors as f64 * 100.0
            } else {
                0.0
            }
        );
        println!(
            "    Pending:        {:>8} ({:.1}%)",
            stat.pending_count,
            if stat.total_vectors > 0 {
                stat.pending_count as f64 / stat.total_vectors as f64 * 100.0
            } else {
                0.0
            }
        );
        // ADMIN.md Lifecycle Accounting (chungers, 2025-01-27, FIXED):
        // Show PendingDeleted separately instead of folding into Deleted.
        println!(
            "    Deleted:        {:>8} ({:.1}%)",
            stat.deleted_count,
            if stat.total_vectors > 0 {
                stat.deleted_count as f64 / stat.total_vectors as f64 * 100.0
            } else {
                0.0
            }
        );
        println!(
            "    PendingDeleted: {:>8} ({:.1}%)",
            stat.pending_deleted_count,
            if stat.total_vectors > 0 {
                stat.pending_deleted_count as f64 / stat.total_vectors as f64 * 100.0
            } else {
                0.0
            }
        );
        println!("  HNSW Graph:");
        println!("    Entry point: {:?}", stat.entry_point);
        println!("    Max level: {}", stat.max_level);
        println!(
            "  Spec hash: {:?} (valid: {})",
            stat.spec_hash.map(|h| format!("0x{:016x}", h)),
            stat.spec_hash_valid
        );
        println!("  ID Allocator:");
        println!("    Next ID: {}", stat.next_id);
        println!("    Free IDs: {}", stat.free_id_count);
        println!();
    }
}

fn admin_inspect(args: AdminInspectArgs) -> Result<()> {
    use motlie_db::vector::admin;

    // ADMIN.md ADMIN-1 (claude, 2025-01-28, FIXED):
    // Secondary (read-only) mode now supported for inspect.
    let inspection = if args.secondary {
        let secondary_path = std::env::temp_dir().join(format!(
            "bench_vector_secondary_{}_{}_{:08x}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis(),
            rand::random::<u32>()
        ));
        let result = {
            let mut storage = Storage::secondary(&args.db_path, &secondary_path);
            storage.ready()?;
            admin::inspect_embedding_secondary(&storage, args.code)?
        };
        let _ = std::fs::remove_dir_all(&secondary_path);
        result
    } else {
        let mut storage = Storage::readwrite(&args.db_path);
        storage.ready()?;
        admin::inspect_embedding(&storage, args.code)?
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&inspection)?);
    } else {
        print_inspection(&inspection);
    }

    Ok(())
}

fn print_inspection(inspection: &motlie_db::vector::admin::EmbeddingInspection) {
    let stat = &inspection.stats;

    println!("=== Embedding Inspection ===\n");
    println!("Code: {}", stat.code);
    println!("Model: {}", stat.model);
    println!("Dimension: {}", stat.dim);
    println!("Distance: {}", stat.distance);
    println!("Storage type: {}", stat.storage_type);

    println!("\nHNSW Configuration:");
    println!("  M: {}", inspection.hnsw_m);
    println!("  ef_construction: {}", inspection.hnsw_ef_construction);

    println!("\nRaBitQ Configuration:");
    println!("  Bits: {}", inspection.rabitq_bits);
    println!("  Seed: {}", inspection.rabitq_seed);

    println!("\nGraph Metadata:");
    println!("  Entry point: {:?}", stat.entry_point);
    println!("  Max level: {}", stat.max_level);
    println!("  Count: {}", stat.total_vectors);
    println!(
        "  Spec hash: {:?} (valid: {})",
        stat.spec_hash.map(|h| format!("0x{:016x}", h)),
        if stat.spec_hash_valid { "✓" } else { "✗" }
    );

    println!("\nID Allocator:");
    println!("  Next ID: {}", stat.next_id);
    println!("  Free IDs: {}", stat.free_id_count);

    println!("\nPending Queue:");
    println!("  Depth: {} vectors", inspection.pending_queue_depth);
    if let Some(ts) = inspection.oldest_pending_timestamp {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let age_secs = (now - ts) / 1000;
        println!("  Oldest entry: {} seconds ago", age_secs);
    }

    println!("\nLayer Distribution:");
    if inspection.layer_distribution.counts.is_empty() {
        println!("  (no indexed vectors)");
    } else {
        for (layer, count) in inspection.layer_distribution.counts.iter().enumerate() {
            println!("  L{}: {} vectors", layer, count);
        }
    }
}

fn admin_vectors(args: AdminVectorsArgs) -> Result<()> {
    use motlie_db::vector::admin::{self, VecLifecycle};

    // ADMIN.md ADMIN-1 (claude, 2025-01-28, FIXED):
    // Secondary (read-only) mode now supported for vectors.
    let vectors = if args.secondary {
        let secondary_path = std::env::temp_dir().join(format!(
            "bench_vector_secondary_{}_{}_{:08x}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis(),
            rand::random::<u32>()
        ));
        let result = {
            let mut storage = Storage::secondary(&args.db_path, &secondary_path);
            storage.ready()?;

            if let Some(vec_id) = args.vec_id {
                match admin::get_vector_info_secondary(&storage, args.code, vec_id)? {
                    Some(info) => vec![info],
                    None => {
                        println!("Vector {} not found in embedding {}", vec_id, args.code);
                        return Ok(());
                    }
                }
            } else if let Some(sample_count) = args.sample {
                admin::sample_vectors_secondary(&storage, args.code, sample_count, args.seed)?
            } else if let Some(ref state_str) = args.state {
                let state = match state_str.to_lowercase().as_str() {
                    "indexed" => VecLifecycle::Indexed,
                    "pending" => VecLifecycle::Pending,
                    "deleted" => VecLifecycle::Deleted,
                    "pending_deleted" | "pendingdeleted" => VecLifecycle::PendingDeleted,
                    _ => anyhow::bail!(
                        "Unknown state: {}. Use: indexed, pending, deleted, pending_deleted",
                        state_str
                    ),
                };
                admin::list_vectors_by_state_secondary(&storage, args.code, state, args.limit)?
            } else {
                admin::sample_vectors_secondary(&storage, args.code, args.limit, args.seed)?
            }
        };
        let _ = std::fs::remove_dir_all(&secondary_path);
        result
    } else {
        let mut storage = Storage::readwrite(&args.db_path);
        storage.ready()?;

        if let Some(vec_id) = args.vec_id {
            match admin::get_vector_info(&storage, args.code, vec_id)? {
                Some(info) => vec![info],
                None => {
                    println!("Vector {} not found in embedding {}", vec_id, args.code);
                    return Ok(());
                }
            }
        } else if let Some(sample_count) = args.sample {
            admin::sample_vectors(&storage, args.code, sample_count, args.seed)?
        } else if let Some(ref state_str) = args.state {
            let state = match state_str.to_lowercase().as_str() {
                "indexed" => VecLifecycle::Indexed,
                "pending" => VecLifecycle::Pending,
                "deleted" => VecLifecycle::Deleted,
                "pending_deleted" | "pendingdeleted" => VecLifecycle::PendingDeleted,
                _ => anyhow::bail!(
                    "Unknown state: {}. Use: indexed, pending, deleted, pending_deleted",
                    state_str
                ),
            };
            admin::list_vectors_by_state(&storage, args.code, state, args.limit)?
        } else {
            admin::sample_vectors(&storage, args.code, args.limit, args.seed)?
        }
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&vectors)?);
    } else {
        print_vectors(&vectors);
    }

    Ok(())
}

fn print_vectors(vectors: &[motlie_db::vector::admin::VectorInfo]) {
    if vectors.is_empty() {
        println!("No vectors found.");
        return;
    }

    println!("=== Vector Information ===\n");
    println!(
        "{:>8}  {:>10}  {:>8}  {:>5}  {:>10}  {:>8}",
        "vec_id", "lifecycle", "layer", "edges", "binary", "external_id"
    );
    println!("{}", "-".repeat(70));

    for v in vectors {
        let total_edges: usize = v.edge_counts.iter().sum();
        // T6.1: VectorInfo now uses external_key_type: Option<String> for display
        let key_display = v.external_key_type.as_deref().unwrap_or("-");
        println!(
            "{:>8}  {:>10}  {:>8}  {:>5}  {:>10}  {}",
            v.vec_id,
            format!("{:?}", v.lifecycle).to_lowercase(),
            v.max_layer,
            total_edges,
            if v.has_binary_code { "yes" } else { "no" },
            key_display
        );
    }

    println!("\nShowing {} vector(s).", vectors.len());
}

fn admin_validate(args: AdminValidateArgs) -> Result<()> {
    use motlie_db::vector::admin::{self, ValidationStatus};

    // ADMIN.md ADMIN-1 (claude, 2025-01-28, FIXED):
    // Secondary (read-only) mode now supported for validate.
    // ADMIN.md Stronger Validation (chungers, 2025-01-27, ADDED):
    // Use strict mode for full-scan validation when --strict is passed.
    let results = if args.secondary {
        let secondary_path = std::env::temp_dir().join(format!(
            "bench_vector_secondary_{}_{}_{:08x}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis(),
            rand::random::<u32>()
        ));
        // Build validation options from CLI args
        let opts = admin::ValidationOptions {
            sample_size: if args.strict { 0 } else { args.sample_size },
            max_errors: args.max_errors,
            max_entries: args.max_entries,
        };

        let result = {
            let mut storage = Storage::secondary(&args.db_path, &secondary_path);
            storage.ready()?;

            if let Some(code) = args.code {
                vec![admin::validate_embedding_secondary_with_opts(
                    &storage, code, opts,
                )?]
            } else {
                admin::validate_all_secondary_with_opts(&storage, opts)?
            }
        };
        let _ = std::fs::remove_dir_all(&secondary_path);
        result
    } else {
        // Build validation options from CLI args
        let opts = admin::ValidationOptions {
            sample_size: if args.strict { 0 } else { args.sample_size },
            max_errors: args.max_errors,
            max_entries: args.max_entries,
        };

        let mut storage = Storage::readwrite(&args.db_path);
        storage.ready()?;

        if let Some(code) = args.code {
            vec![admin::validate_embedding_with_opts(&storage, code, opts)?]
        } else {
            admin::validate_all_with_opts(&storage, opts)?
        }
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&results)?);
    } else {
        print_validation_results(&results);
    }

    // Exit with error code if any check failed
    let has_errors = results
        .iter()
        .flat_map(|r| &r.checks)
        .any(|c| c.status == ValidationStatus::Error);

    if has_errors {
        std::process::exit(1);
    }

    Ok(())
}

fn print_validation_results(results: &[motlie_db::vector::admin::ValidationResult]) {
    use motlie_db::vector::admin::ValidationStatus;

    if results.is_empty() {
        println!("No embeddings found.");
        return;
    }

    println!("=== Validation Results ===\n");

    for result in results {
        println!("Embedding {} ({}):", result.code, result.model);

        let mut pass_count = 0;
        let mut warn_count = 0;
        let mut error_count = 0;

        for check in &result.checks {
            let symbol = match check.status {
                ValidationStatus::Pass => {
                    pass_count += 1;
                    "✓"
                }
                ValidationStatus::Warning => {
                    warn_count += 1;
                    "⚠"
                }
                ValidationStatus::Error => {
                    error_count += 1;
                    "✗"
                }
            };
            println!("  {} {}: {}", symbol, check.name, check.message);
        }

        println!(
            "\n  Summary: {} passed, {} warnings, {} errors\n",
            pass_count, warn_count, error_count
        );
    }
}

fn admin_rocksdb(args: AdminRocksdbArgs) -> Result<()> {
    use motlie_db::vector::admin;

    // ADMIN.md Read-Only Mode (chungers, 2025-01-27, FIXED):
    // Use secondary mode for read-only access when --secondary flag is set.
    // ADMIN.md Secondary Cleanup (claude, 2025-01-27, FIXED):
    // Clean up temp directory after use to avoid accumulation.
    let stats = if args.secondary {
        let secondary_path = std::env::temp_dir().join(format!(
            "bench_vector_secondary_{}_{}_{:08x}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis(),
            rand::random::<u32>()
        ));
        let result = {
            let mut storage = Storage::secondary(&args.db_path, &secondary_path);
            storage.ready()?;
            admin::get_rocksdb_stats_secondary(&storage)?
        };
        // Clean up secondary DB files
        let _ = std::fs::remove_dir_all(&secondary_path);
        result
    } else {
        let mut storage = Storage::readwrite(&args.db_path);
        storage.ready()?;
        admin::get_rocksdb_stats(&storage)?
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&stats)?);
    } else {
        print_rocksdb_stats(&stats);
    }

    Ok(())
}

fn admin_migrate_lifecycle_counts(args: AdminMigrateCountsArgs) -> Result<()> {
    use motlie_db::vector::admin;

    // Migration requires readwrite mode (not secondary)
    let mut storage = Storage::readwrite(&args.db_path);
    storage.ready()?;

    let results = if let Some(code) = args.code {
        vec![admin::migrate_lifecycle_counts(&storage, code)?]
    } else {
        admin::migrate_all_lifecycle_counts(&storage)?
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&results)?);
    } else {
        println!("=== Lifecycle Counter Migration ===\n");
        for r in &results {
            println!(
                "Embedding {}: indexed={}, pending={}, deleted={}, pending_deleted={}",
                r.code, r.indexed, r.pending, r.deleted, r.pending_deleted
            );
        }
        println!("\nMigrated {} embedding(s).", results.len());
    }

    Ok(())
}

fn print_rocksdb_stats(stats: &motlie_db::vector::admin::RocksDbStats) {
    // ADMIN.md Issue #4 (chungers, 2025-01-27, FIXED):
    // Now uses is_sampled field to clearly label sampled values vs exact counts.
    // Size column header changed to clarify these are sampled bytes, not on-disk totals.
    // ADMIN.md RocksDB Properties (chungers, 2025-01-27, ADDED):
    // Shows estimated_num_keys from RocksDB property when available (secondary mode).

    // Check if any CF has estimated_num_keys
    let has_estimates = stats
        .column_families
        .iter()
        .any(|cf| cf.estimated_num_keys.is_some());

    println!("=== RocksDB Statistics ===\n");
    if has_estimates {
        println!(
            "{:<30}  {:>14}  {:>14}  {:>14}",
            "Column Family", "Entries", "Est. Keys", "Bytes (sampled)"
        );
        println!("{}", "-".repeat(78));
    } else {
        println!(
            "{:<30}  {:>14}  {:>14}",
            "Column Family", "Entries", "Bytes (sampled)"
        );
        println!("{}", "-".repeat(62));
    }

    for cf in &stats.column_families {
        let size_str = format_size(cf.size_bytes);
        // Use is_sampled field to indicate partial counts
        let entry_str = if cf.is_sampled {
            format!("{}+", cf.entry_count)
        } else {
            format!("{}", cf.entry_count)
        };

        if has_estimates {
            let est_str = cf
                .estimated_num_keys
                .map(|n| n.to_string())
                .unwrap_or_else(|| "-".to_string());
            println!(
                "{:<30}  {:>14}  {:>14}  {:>14}",
                cf.name, entry_str, est_str, size_str
            );
        } else {
            println!("{:<30}  {:>14}  {:>14}", cf.name, entry_str, size_str);
        }
    }

    if has_estimates {
        println!("{}", "-".repeat(78));
        println!(
            "{:<30}  {:>14}  {:>14}  {:>14}",
            "Total (sampled)",
            "",
            "",
            format_size(stats.total_size_bytes)
        );
    } else {
        println!("{}", "-".repeat(62));
        println!(
            "{:<30}  {:>14}  {:>14}",
            "Total (sampled)",
            "",
            format_size(stats.total_size_bytes)
        );
    }
}

fn format_size(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.1} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}
