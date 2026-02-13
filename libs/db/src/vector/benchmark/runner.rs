//! Experiment runner for HNSW benchmarks.
//!
//! Provides configurable benchmark execution with standardized metrics.

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;

use super::dataset::{compute_ground_truth_parallel, LAION_EMBEDDING_DIM};
use super::metrics::{compute_recall, percentile, LatencyStats};
use super::Dataset;
use crate::vector::{
    hnsw, processor::Processor, query::SearchKNN, BinaryCodeCache, Distance, Embedding,
    EmbeddingBuilder, EmbeddingCode, RaBitQ, Storage, VecId, VectorElementType,
};
use crate::Id;
use crate::vector::schema::ExternalKey;

/// Experiment configuration.
#[derive(Debug, Clone)]
pub struct ExperimentConfig {
    /// Scales (number of vectors) to test.
    pub scales: Vec<usize>,
    /// ef_search values to test.
    pub ef_search_values: Vec<usize>,
    /// k values for Recall@k.
    pub k_values: Vec<usize>,
    /// Number of queries per experiment.
    pub num_queries: usize,
    /// HNSW M parameter.
    pub m: usize,
    /// HNSW ef_construction parameter.
    pub ef_construction: usize,
    /// Embedding dimension.
    pub dim: usize,
    /// Distance metric.
    pub distance: Distance,
    /// Vector storage type.
    pub storage_type: VectorElementType,
    /// Data directory for dataset files.
    pub data_dir: PathBuf,
    /// Results directory for output files.
    pub results_dir: PathBuf,
    /// Enable verbose output.
    pub verbose: bool,

    // RaBitQ parameters (Phase A)
    /// Enable RaBitQ quantization for search.
    pub use_rabitq: bool,
    /// Bits per dimension to test (1, 2, or 4).
    pub rabitq_bits: Vec<u8>,
    /// Rerank factors to test (e.g., [1, 4, 10, 20]).
    pub rerank_factors: Vec<usize>,
    /// Use in-memory binary code cache.
    pub use_binary_cache: bool,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            scales: vec![50_000, 100_000],
            ef_search_values: vec![50, 100, 200],
            k_values: vec![1, 5, 10],
            num_queries: 1000,
            m: 16,
            ef_construction: 100,
            dim: LAION_EMBEDDING_DIM,
            distance: Distance::Cosine,
            storage_type: VectorElementType::F16,
            data_dir: PathBuf::from("data"),
            results_dir: PathBuf::from("results"),
            verbose: false,
            // RaBitQ defaults
            use_rabitq: false,
            rabitq_bits: vec![1, 2, 4],
            rerank_factors: vec![1, 4, 10, 20],
            use_binary_cache: true,
        }
    }
}

impl ExperimentConfig {
    /// Set scales to test.
    pub fn with_scales(mut self, scales: Vec<usize>) -> Self {
        self.scales = scales;
        self
    }

    /// Set ef_search values to test.
    pub fn with_ef_search(mut self, ef_search: Vec<usize>) -> Self {
        self.ef_search_values = ef_search;
        self
    }

    /// Set k values for recall.
    pub fn with_k_values(mut self, k_values: Vec<usize>) -> Self {
        self.k_values = k_values;
        self
    }

    /// Set number of queries.
    pub fn with_num_queries(mut self, n: usize) -> Self {
        self.num_queries = n;
        self
    }

    /// Set HNSW M parameter.
    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self
    }

    /// Set HNSW ef_construction.
    pub fn with_ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    /// Set distance metric.
    pub fn with_distance(mut self, distance: Distance) -> Self {
        self.distance = distance;
        self
    }

    /// Set data directory.
    pub fn with_data_dir(mut self, path: PathBuf) -> Self {
        self.data_dir = path;
        self
    }

    /// Set results directory.
    pub fn with_results_dir(mut self, path: PathBuf) -> Self {
        self.results_dir = path;
        self
    }

    /// Enable/disable verbose output.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Enable RaBitQ mode with specified bits and rerank factors.
    pub fn with_rabitq(mut self, bits: Vec<u8>, rerank_factors: Vec<usize>) -> Self {
        self.use_rabitq = true;
        self.rabitq_bits = bits;
        self.rerank_factors = rerank_factors;
        self
    }

    /// Set RaBitQ bits per dimension to test.
    pub fn with_rabitq_bits(mut self, bits: Vec<u8>) -> Self {
        self.rabitq_bits = bits;
        self
    }

    /// Set rerank factors to test.
    pub fn with_rerank_factors(mut self, factors: Vec<usize>) -> Self {
        self.rerank_factors = factors;
        self
    }

    /// Enable/disable binary code cache.
    pub fn with_binary_cache(mut self, enabled: bool) -> Self {
        self.use_binary_cache = enabled;
        self
    }
}

/// Results from a single experiment run.
#[derive(Debug, Clone)]
pub struct ExperimentResult {
    /// Number of vectors in index.
    pub scale: usize,
    /// ef_search parameter used.
    pub ef_search: usize,
    /// Search strategy name.
    pub strategy: String,
    /// Recall at each k value.
    pub recall_at_k: HashMap<usize, f64>,
    /// Average latency in milliseconds.
    pub latency_avg_ms: f64,
    /// Median latency in milliseconds.
    pub latency_p50_ms: f64,
    /// 95th percentile latency.
    pub latency_p95_ms: f64,
    /// 99th percentile latency.
    pub latency_p99_ms: f64,
    /// Queries per second.
    pub qps: f64,
    /// Index build time in seconds.
    pub build_time_s: f64,
}

impl ExperimentResult {
    /// Get recall at a specific k.
    pub fn recall(&self, k: usize) -> f64 {
        *self.recall_at_k.get(&k).unwrap_or(&0.0)
    }

    /// Format recall as percentage.
    pub fn recall_percent(&self, k: usize) -> String {
        format!("{:.1}%", self.recall(k) * 100.0)
    }

    /// Get latency statistics.
    pub fn latency_stats(&self) -> LatencyStats {
        LatencyStats {
            avg_ms: self.latency_avg_ms,
            p50_ms: self.latency_p50_ms,
            p95_ms: self.latency_p95_ms,
            p99_ms: self.latency_p99_ms,
            qps: self.qps,
            ..Default::default()
        }
    }
}

/// Results from a RaBitQ experiment run.
///
/// Extends standard experiment results with RaBitQ-specific parameters.
#[derive(Debug, Clone)]
pub struct RabitqExperimentResult {
    /// Number of vectors in index.
    pub scale: usize,
    /// Bits per dimension (1, 2, or 4).
    pub bits_per_dim: u8,
    /// ef_search parameter used.
    pub ef_search: usize,
    /// Rerank factor (candidates = k * rerank_factor).
    pub rerank_factor: usize,
    /// k value for Recall@k.
    pub k: usize,
    /// Mean recall across all queries.
    pub recall_mean: f64,
    /// Standard deviation of recall.
    pub recall_std: f64,
    /// Average latency in milliseconds.
    pub latency_avg_ms: f64,
    /// Median latency in milliseconds.
    pub latency_p50_ms: f64,
    /// 95th percentile latency.
    pub latency_p95_ms: f64,
    /// 99th percentile latency.
    pub latency_p99_ms: f64,
    /// Queries per second.
    pub qps: f64,
    /// Binary code encoding time (seconds).
    pub encode_time_s: f64,
    /// Index build time in seconds.
    pub build_time_s: f64,
}

impl RabitqExperimentResult {
    /// Format as CSV row.
    pub fn to_csv_row(&self) -> String {
        format!(
            "{},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.1},{:.2},{:.2}",
            self.scale,
            self.bits_per_dim,
            self.ef_search,
            self.rerank_factor,
            self.k,
            self.recall_mean,
            self.recall_std,
            self.latency_avg_ms,
            self.latency_p50_ms,
            self.latency_p95_ms,
            self.latency_p99_ms,
            self.qps,
            self.encode_time_s,
            self.build_time_s,
        )
    }

    /// CSV header for RaBitQ results.
    pub fn csv_header() -> &'static str {
        "scale,bits,ef_search,rerank_factor,k,recall_mean,recall_std,latency_avg_ms,latency_p50_ms,latency_p95_ms,latency_p99_ms,qps,encode_time_s,build_time_s"
    }
}

/// Run all experiments according to configuration.
pub fn run_all_experiments(
    dataset: &dyn Dataset,
    config: &ExperimentConfig,
) -> Result<Vec<ExperimentResult>> {
    struct DatasetView<'a> {
        name: &'a str,
        dim: usize,
        distance: Distance,
        vectors: &'a [Vec<f32>],
        queries: &'a [Vec<f32>],
        ground_truth: Option<Vec<Vec<usize>>>,
    }

    impl Dataset for DatasetView<'_> {
        fn name(&self) -> &str {
            self.name
        }
        fn dim(&self) -> usize {
            self.dim
        }
        fn distance(&self) -> Distance {
            self.distance
        }
        fn vectors(&self) -> &[Vec<f32>] {
            self.vectors
        }
        fn queries(&self) -> &[Vec<f32>] {
            self.queries
        }
        fn ground_truth(&self, _k: usize) -> Option<Vec<Vec<usize>>> {
            self.ground_truth.clone()
        }
    }

    println!(
        "Dataset loaded: {} ({} vectors, {} queries, {}D, {:?})\n",
        dataset.name(),
        dataset.vectors().len(),
        dataset.queries().len(),
        dataset.dim(),
        dataset.distance(),
    );

    let mut all_results = Vec::new();
    let max_k = *config.k_values.iter().max().unwrap_or(&20);
    let full_gt = dataset.ground_truth(max_k);
    let full_vectors = dataset.vectors().len();
    let full_queries = dataset.queries().len();

    for &scale in &config.scales {
        let scale = scale.min(full_vectors);
        let query_count = config.num_queries.min(full_queries);

        println!("\n{}", "=".repeat(60));
        println!("Scale: {} vectors", scale);
        println!("{}\n", "=".repeat(60));

        let view = DatasetView {
            name: dataset.name(),
            dim: dataset.dim(),
            distance: dataset.distance(),
            vectors: &dataset.vectors()[..scale],
            queries: &dataset.queries()[..query_count],
            // Only trust precomputed GT at full vector scale.
            ground_truth: if scale == full_vectors {
                full_gt
                    .as_ref()
                    .map(|gt| gt.iter().take(query_count).cloned().collect())
            } else {
                None
            },
        };
        println!(
            "Subset: {} db vectors, {} queries ({})",
            view.vectors().len(),
            view.queries().len(),
            view.name()
        );

        // Compute ground truth (use pre-computed if available, else brute-force)
        let ground_truth = match view.ground_truth(max_k) {
            Some(gt) => gt,
            None => compute_ground_truth_parallel(
                view.vectors(),
                view.queries(),
                max_k,
                view.distance(),
            ),
        };

        // Create temp directory for this scale's database
        let temp_dir = tempfile::tempdir()?;
        let db_path = temp_dir.path().to_path_buf();
        println!("Database path: {:?}", db_path);

        // Initialize storage
        let mut storage = Storage::readwrite(&db_path);
        storage.ready()?;
        let storage = Arc::new(storage);

        // Build HNSW index
        println!(
            "\nBuilding HNSW index (M={}, ef_construction={})...",
            config.m, config.ef_construction
        );
        let (_index, embedding, build_time) = build_hnsw_index(
            &storage,
            view.vectors(),
            view.dim(),
            config.m,
            config.ef_construction,
            view.distance(),
            config.storage_type,
        )?;
        println!(
            "Index built in {:.2}s ({:.1} vec/s)",
            build_time,
            scale as f64 / build_time
        );

        // Run experiments for each ef_search value
        for &ef_search in &config.ef_search_values {
            println!("\n--- ef_search = {} ---", ef_search);

            let result = run_single_experiment(
                &embedding,
                &storage,
                &view,
                &ground_truth,
                &config.k_values,
                ef_search,
                scale,
                build_time,
                &format!("HNSW-{}", view.distance()),
                config.verbose,
            )?;
            all_results.push(result);
        }

        // Run flat baseline
        println!("\n--- Flat (brute-force) baseline ---");
        let flat_result = run_flat_baseline(
            &view,
            &ground_truth,
            &config.k_values,
            scale,
            config.verbose,
        )?;
        all_results.push(flat_result);
    }

    // Save results
    std::fs::create_dir_all(&config.results_dir)?;
    save_results_csv(&all_results, &config.results_dir)?;

    Ok(all_results)
}

/// Build HNSW index from vectors.
pub fn build_hnsw_index(
    storage: &Arc<Storage>,
    vectors: &[Vec<f32>],
    dim: usize,
    m: usize,
    ef_construction: usize,
    distance: Distance,
    storage_type: VectorElementType,
) -> Result<(hnsw::Index, Embedding, f64)> {
    if storage_type != VectorElementType::F16 {
        anyhow::bail!(
            "build_hnsw_index currently requires {:?} storage_type (got {:?})",
            VectorElementType::F16,
            storage_type
        );
    }

    let registry = storage.cache().clone();
    registry.set_storage(storage.clone())?;
    let embedding = registry.register(
        EmbeddingBuilder::new("benchmark", dim as u32, distance)
            .with_hnsw_m(m as u16)
            .with_hnsw_ef_construction(ef_construction as u16),
    )?;
    let processor = Processor::new(storage.clone(), registry);

    let start = Instant::now();

    for (batch_idx, batch) in vectors.chunks(1000).enumerate() {
        let base = batch_idx * 1000;
        let payload: Vec<(ExternalKey, Vec<f32>)> = batch
            .iter()
            .enumerate()
            .map(|(offset, vector)| {
                let id = Id::from_bytes(((base + offset) as u128).to_be_bytes());
                (ExternalKey::NodeId(id), vector.clone())
            })
            .collect();
        let _ = processor.insert_batch(&embedding, &payload, true)?;
        let inserted = (base + batch.len()).min(vectors.len());
        if inserted % 10000 == 0 || inserted == vectors.len() {
            println!("  Inserted {}/{} vectors", inserted, vectors.len());
        }
    }

    let index = processor
        .get_or_create_index(embedding.code())
        .ok_or_else(|| anyhow::anyhow!("Failed to acquire HNSW index after benchmark insert"))?;

    let build_time = start.elapsed().as_secs_f64();
    Ok((index, embedding, build_time))
}

/// Run a single experiment configuration.
pub fn run_single_experiment(
    embedding: &Embedding,
    storage: &Arc<Storage>,
    dataset: &dyn Dataset,
    ground_truth: &[Vec<usize>],
    k_values: &[usize],
    ef_search: usize,
    scale: usize,
    build_time: f64,
    strategy: &str,
    verbose: bool,
) -> Result<ExperimentResult> {
    let registry = storage.cache().clone();
    registry.set_storage(storage.clone())?;
    let processor = Processor::new(storage.clone(), registry);

    let mut latencies = Vec::with_capacity(dataset.queries().len());
    let mut search_results = Vec::with_capacity(dataset.queries().len());

    let max_k = *k_values.iter().max().unwrap_or(&20);

    // Run queries
    for (qi, query) in dataset.queries().iter().enumerate() {
        let start = Instant::now();
        let results = SearchKNN::new(embedding, query.clone(), max_k)
            .with_ef(ef_search)
            .exact()
            .execute_with_processor(&processor)?;
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        latencies.push(latency_ms);
        let result_ids: Vec<usize> = results.iter().map(|r| r.vec_id as usize).collect();
        search_results.push(result_ids);

        if verbose && (qi + 1) % 100 == 0 {
            println!(
                "  Query {}/{}: {:.2}ms",
                qi + 1,
                dataset.queries().len(),
                latency_ms
            );
        }
    }

    // Compute recall at each k
    let mut recall_at_k = HashMap::new();
    for &k in k_values {
        let recall = compute_recall(&search_results, ground_truth, k);
        recall_at_k.insert(k, recall);
        println!("  Recall@{}: {:.1}%", k, recall * 100.0);
    }

    // Compute latency statistics
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let latency_avg_ms = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let latency_p50_ms = percentile(&latencies, 50.0);
    let latency_p95_ms = percentile(&latencies, 95.0);
    let latency_p99_ms = percentile(&latencies, 99.0);
    let qps = 1000.0 / latency_avg_ms;

    println!(
        "  Latency: avg={:.2}ms, p50={:.2}ms, p95={:.2}ms, p99={:.2}ms",
        latency_avg_ms, latency_p50_ms, latency_p95_ms, latency_p99_ms
    );
    println!("  QPS: {:.1}", qps);

    Ok(ExperimentResult {
        scale,
        ef_search,
        strategy: strategy.to_string(),
        recall_at_k,
        latency_avg_ms,
        latency_p50_ms,
        latency_p95_ms,
        latency_p99_ms,
        qps,
        build_time_s: build_time,
    })
}

/// Run flat (brute-force) baseline for comparison.
pub fn run_flat_baseline(
    dataset: &dyn Dataset,
    ground_truth: &[Vec<usize>],
    k_values: &[usize],
    scale: usize,
    verbose: bool,
) -> Result<ExperimentResult> {
    let mut latencies = Vec::with_capacity(dataset.queries().len());
    let mut search_results = Vec::with_capacity(dataset.queries().len());

    let max_k = *k_values.iter().max().unwrap_or(&20);

    // Run brute-force queries
    for (qi, query) in dataset.queries().iter().enumerate() {
        let start = Instant::now();

        // Compute distances to all vectors
        let mut distances: Vec<(usize, f32)> = dataset
            .vectors()
            .iter()
            .enumerate()
            .map(|(i, v)| (i, dataset.distance().compute(query, v)))
            .collect();

        // Sort and take top-k
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let result_ids: Vec<usize> = distances.iter().take(max_k).map(|(i, _)| *i).collect();

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        latencies.push(latency_ms);
        search_results.push(result_ids);

        if verbose && (qi + 1) % 100 == 0 {
            println!(
                "  Query {}/{}: {:.2}ms",
                qi + 1,
                dataset.queries().len(),
                latency_ms
            );
        }
    }

    // Compute recall at each k (should be 100% for flat search)
    let mut recall_at_k = HashMap::new();
    for &k in k_values {
        let recall = compute_recall(&search_results, ground_truth, k);
        recall_at_k.insert(k, recall);
        println!("  Recall@{}: {:.1}%", k, recall * 100.0);
    }

    // Compute latency statistics
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let latency_avg_ms = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let latency_p50_ms = percentile(&latencies, 50.0);
    let latency_p95_ms = percentile(&latencies, 95.0);
    let latency_p99_ms = percentile(&latencies, 99.0);
    let qps = 1000.0 / latency_avg_ms;

    println!(
        "  Latency: avg={:.2}ms, p50={:.2}ms, p95={:.2}ms, p99={:.2}ms",
        latency_avg_ms, latency_p50_ms, latency_p95_ms, latency_p99_ms
    );
    println!("  QPS: {:.1}", qps);

    Ok(ExperimentResult {
        scale,
        ef_search: 0,
        strategy: "Flat".to_string(),
        recall_at_k,
        latency_avg_ms,
        latency_p50_ms,
        latency_p95_ms,
        latency_p99_ms,
        qps,
        build_time_s: 0.0,
    })
}

/// Run RaBitQ experiments with parameter sweep.
///
/// Iterates over all combinations of (bits_per_dim, ef_search, rerank_factor)
/// and records RaBitQ-specific metrics including encoding time.
///
/// # Arguments
///
/// * `config` - Experiment configuration including RaBitQ parameters
/// * `index` - Built HNSW index
/// * `storage` - Vector storage
/// * `dataset` - Dataset with queries and db vectors (via Dataset trait)
/// * `ground_truth` - Ground truth results for recall computation
/// * `build_time` - Time taken to build the HNSW index
/// * `embedding_code` - Embedding code used for binary code cache keys
///
/// # Returns
///
/// Vector of `RabitqExperimentResult` for each parameter combination.
pub fn run_rabitq_experiments(
    config: &ExperimentConfig,
    index: &hnsw::Index,
    storage: &Storage,
    dataset: &dyn Dataset,
    ground_truth: &[Vec<usize>],
    build_time: f64,
    embedding_code: EmbeddingCode,
    use_simd_dot: bool,
) -> Result<Vec<RabitqExperimentResult>> {
    let mut results = Vec::new();

    for &bits in &config.rabitq_bits {
        println!("\n  --- RaBitQ {} bits ---", bits);

        // Create encoder for this bit configuration
        let encoder = RaBitQ::with_options(dataset.dim(), bits, 42, use_simd_dot);
        println!(
            "  Encoder created: {} bits/dim (simd_dot={})",
            bits, use_simd_dot
        );

        // Create and populate binary code cache
        let cache = BinaryCodeCache::new();
        let encode_start = Instant::now();

        for (i, vector) in dataset.vectors().iter().enumerate() {
            let vec_id = i as VecId;
            let (code, correction) = encoder.encode_with_correction(vector);
            cache.put(embedding_code, vec_id, code, correction);
        }

        let encode_time_s = encode_start.elapsed().as_secs_f64();
        let (cache_count, cache_bytes) = cache.stats();
        println!(
            "  Binary codes cached: {} vectors, {:.2} MB, {:.2}s",
            cache_count,
            cache_bytes as f64 / 1_000_000.0,
            encode_time_s
        );

        // Run experiments for each (ef_search, rerank_factor, k) combination
        for &ef_search in &config.ef_search_values {
            for &rerank_factor in &config.rerank_factors {
                for &k in &config.k_values {
                    let mut latencies = Vec::with_capacity(dataset.queries().len());
                    let mut recalls = Vec::with_capacity(dataset.queries().len());

                    // Run all queries
                    for (qi, query) in dataset.queries().iter().enumerate() {
                        let start = Instant::now();
                        let search_results = index.search_with_rabitq_cached(
                            storage,
                            query,
                            &encoder,
                            &cache,
                            k,
                            ef_search,
                            rerank_factor,
                        )?;
                        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                        latencies.push(latency_ms);

                        // Compute recall for this query
                        let result_ids: Vec<usize> =
                            search_results.iter().map(|(_, id)| *id as usize).collect();
                        let gt = &ground_truth[qi];
                        let hits = result_ids
                            .iter()
                            .take(k)
                            .filter(|id| gt.iter().take(k).any(|gt_id| gt_id == *id))
                            .count();
                        let recall = hits as f64 / k.min(gt.len()) as f64;
                        recalls.push(recall);
                    }

                    // Compute statistics
                    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let latency_avg_ms = latencies.iter().sum::<f64>() / latencies.len() as f64;
                    let latency_p50_ms = percentile(&latencies, 50.0);
                    let latency_p95_ms = percentile(&latencies, 95.0);
                    let latency_p99_ms = percentile(&latencies, 99.0);
                    let qps = 1000.0 / latency_avg_ms;

                    let recall_mean = recalls.iter().sum::<f64>() / recalls.len() as f64;
                    let recall_std = if recalls.len() > 1 {
                        let variance = recalls
                            .iter()
                            .map(|r| (r - recall_mean).powi(2))
                            .sum::<f64>()
                            / (recalls.len() - 1) as f64;
                        variance.sqrt()
                    } else {
                        0.0
                    };

                    println!(
                        "    bits={}, ef={}, rerank={}, k={}: recall={:.1}%±{:.1}%, {:.2}ms, {:.0} QPS",
                        bits, ef_search, rerank_factor, k,
                        recall_mean * 100.0, recall_std * 100.0,
                        latency_avg_ms, qps
                    );

                    results.push(RabitqExperimentResult {
                        scale: dataset.vectors().len(),
                        bits_per_dim: bits,
                        ef_search,
                        rerank_factor,
                        k,
                        recall_mean,
                        recall_std,
                        latency_avg_ms,
                        latency_p50_ms,
                        latency_p95_ms,
                        latency_p99_ms,
                        qps,
                        encode_time_s,
                        build_time_s: build_time,
                    });
                }
            }
        }
    }

    Ok(results)
}

/// Save RaBitQ results to CSV file.
pub fn save_rabitq_results_csv(results: &[RabitqExperimentResult], csv_path: &Path) -> Result<()> {
    let mut file = File::create(csv_path)?;

    // Header
    writeln!(file, "{}", RabitqExperimentResult::csv_header())?;

    // Data rows
    for r in results {
        writeln!(file, "{}", r.to_csv_row())?;
    }

    println!("RaBitQ results saved to: {:?}", csv_path);
    Ok(())
}

/// Save results to CSV file.
fn save_results_csv(results: &[ExperimentResult], results_dir: &PathBuf) -> Result<()> {
    let csv_path = results_dir.join("benchmark_results.csv");
    let mut file = File::create(&csv_path)?;

    // Header
    writeln!(
        file,
        "scale,ef_search,strategy,recall@1,recall@5,recall@10,latency_avg_ms,latency_p50_ms,latency_p95_ms,latency_p99_ms,qps,build_time_s"
    )?;

    // Data rows
    for r in results {
        writeln!(
            file,
            "{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.2},{:.2}",
            r.scale,
            r.ef_search,
            r.strategy,
            r.recall_at_k.get(&1).unwrap_or(&0.0),
            r.recall_at_k.get(&5).unwrap_or(&0.0),
            r.recall_at_k.get(&10).unwrap_or(&0.0),
            r.latency_avg_ms,
            r.latency_p50_ms,
            r.latency_p95_ms,
            r.latency_p99_ms,
            r.qps,
            r.build_time_s,
        )?;
    }

    println!("\nResults saved to: {:?}", csv_path);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_experiment_config_default() {
        let config = ExperimentConfig::default();
        assert_eq!(config.m, 16);
        assert_eq!(config.ef_construction, 100);
        assert_eq!(config.distance, Distance::Cosine);
    }

    #[test]
    fn test_experiment_config_builder() {
        let config = ExperimentConfig::default()
            .with_scales(vec![1000, 5000])
            .with_m(32)
            .with_verbose(true);

        assert_eq!(config.scales, vec![1000, 5000]);
        assert_eq!(config.m, 32);
        assert!(config.verbose);
    }

    #[test]
    fn test_experiment_result_recall() {
        let mut recall_at_k = HashMap::new();
        recall_at_k.insert(1, 0.85);
        recall_at_k.insert(10, 0.95);

        let result = ExperimentResult {
            scale: 1000,
            ef_search: 100,
            strategy: "test".to_string(),
            recall_at_k,
            latency_avg_ms: 1.0,
            latency_p50_ms: 0.9,
            latency_p95_ms: 1.5,
            latency_p99_ms: 2.0,
            qps: 1000.0,
            build_time_s: 1.0,
        };

        assert!((result.recall(1) - 0.85).abs() < 1e-6);
        assert!((result.recall(10) - 0.95).abs() < 1e-6);
        assert!((result.recall(5) - 0.0).abs() < 1e-6); // Not set
    }
}
