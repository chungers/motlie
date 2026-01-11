//! Experiment runner for HNSW benchmarks.
//!
//! Provides configurable benchmark execution with standardized metrics.

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;

use super::dataset::{LaionDataset, LaionSubset, LAION_EMBEDDING_DIM};
use super::metrics::{compute_recall, percentile, LatencyStats};
use crate::rocksdb::ColumnFamily;
use crate::vector::{
    hnsw, Distance, EmbeddingCode, NavigationCache, Storage, VecId, VectorCfKey,
    VectorElementType, Vectors,
};

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

/// Run all experiments according to configuration.
pub fn run_all_experiments(config: &ExperimentConfig) -> Result<Vec<ExperimentResult>> {
    // Load full dataset
    let max_vectors = *config.scales.iter().max().unwrap_or(&200_000);
    println!("Loading LAION dataset (up to {} vectors)...", max_vectors);
    let dataset = LaionDataset::load(&config.data_dir, max_vectors)?;
    println!(
        "Dataset loaded: {} image embeddings, {} text embeddings\n",
        dataset.image_embeddings.len(),
        dataset.text_embeddings.len()
    );

    let mut all_results = Vec::new();

    for &scale in &config.scales {
        println!("\n{}", "=".repeat(60));
        println!("Scale: {} vectors", scale);
        println!("{}\n", "=".repeat(60));

        // Get subset for this scale
        let subset = dataset.subset(scale, config.num_queries);
        println!(
            "Subset: {} db vectors, {} queries",
            subset.db_vectors.len(),
            subset.queries.len()
        );

        // Compute brute-force ground truth
        let max_k = *config.k_values.iter().max().unwrap_or(&20);
        let ground_truth = subset.compute_ground_truth_topk(max_k);

        // Create temp directory for this scale's database
        let temp_dir = tempfile::tempdir()?;
        let db_path = temp_dir.path().to_path_buf();
        println!("Database path: {:?}", db_path);

        // Initialize storage
        let mut storage = Storage::readwrite(&db_path);
        storage.ready()?;

        // Build HNSW index
        println!(
            "\nBuilding HNSW index (M={}, ef_construction={})...",
            config.m, config.ef_construction
        );
        let hnsw_config = hnsw::Config {
            dim: config.dim,
            m: config.m,
            m_max: config.m * 2,
            m_max_0: config.m * 2,
            ef_construction: config.ef_construction,
            ..Default::default()
        };
        let (index, build_time) = build_hnsw_index(
            &storage,
            &subset.db_vectors,
            hnsw_config,
            config.distance,
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
                &index,
                &storage,
                &subset,
                &ground_truth,
                &config.k_values,
                ef_search,
                scale,
                build_time,
                &format!("HNSW-{}", config.distance),
                config.verbose,
            )?;
            all_results.push(result);
        }

        // Run flat baseline
        println!("\n--- Flat (brute-force) baseline ---");
        let flat_result = run_flat_baseline(
            &subset,
            &ground_truth,
            &config.k_values,
            scale,
            config.distance,
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
    storage: &Storage,
    vectors: &[Vec<f32>],
    hnsw_config: hnsw::Config,
    distance: Distance,
    storage_type: VectorElementType,
) -> Result<(hnsw::Index, f64)> {
    let nav_cache = Arc::new(NavigationCache::new());
    let embedding_code: EmbeddingCode = 1;

    let index = hnsw::Index::with_storage_type(
        embedding_code,
        distance,
        storage_type,
        hnsw_config,
        nav_cache,
    );

    // Store vectors in RocksDB
    let txn_db = storage.transaction_db()?;
    let vectors_cf = txn_db
        .cf_handle(Vectors::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("Vectors CF not found"))?;

    let start = Instant::now();

    for (i, vector) in vectors.iter().enumerate() {
        let vec_id = i as VecId;

        // Store vector in Vectors CF
        let key = VectorCfKey(embedding_code, vec_id);
        let value_bytes = Vectors::value_to_bytes_typed(vector, storage_type);
        txn_db.put_cf(&vectors_cf, Vectors::key_to_bytes(&key), value_bytes)?;

        // Insert into HNSW index
        index.insert(storage, vec_id, vector)?;

        if (i + 1) % 10000 == 0 {
            println!("  Inserted {}/{} vectors", i + 1, vectors.len());
        }
    }

    let build_time = start.elapsed().as_secs_f64();
    Ok((index, build_time))
}

/// Run a single experiment configuration.
pub fn run_single_experiment(
    index: &hnsw::Index,
    storage: &Storage,
    subset: &LaionSubset,
    ground_truth: &[Vec<usize>],
    k_values: &[usize],
    ef_search: usize,
    scale: usize,
    build_time: f64,
    strategy: &str,
    verbose: bool,
) -> Result<ExperimentResult> {
    let mut latencies = Vec::with_capacity(subset.queries.len());
    let mut search_results = Vec::with_capacity(subset.queries.len());

    let max_k = *k_values.iter().max().unwrap_or(&20);

    // Run queries
    for (qi, query) in subset.queries.iter().enumerate() {
        let start = Instant::now();
        let results = index.search(storage, query, ef_search, max_k)?;
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        latencies.push(latency_ms);
        let result_ids: Vec<usize> = results.iter().map(|(_, id)| *id as usize).collect();
        search_results.push(result_ids);

        if verbose && (qi + 1) % 100 == 0 {
            println!(
                "  Query {}/{}: {:.2}ms",
                qi + 1,
                subset.queries.len(),
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
    subset: &LaionSubset,
    ground_truth: &[Vec<usize>],
    k_values: &[usize],
    scale: usize,
    distance: Distance,
    verbose: bool,
) -> Result<ExperimentResult> {
    let mut latencies = Vec::with_capacity(subset.queries.len());
    let mut search_results = Vec::with_capacity(subset.queries.len());

    let max_k = *k_values.iter().max().unwrap_or(&20);

    // Run brute-force queries
    for (qi, query) in subset.queries.iter().enumerate() {
        let start = Instant::now();

        // Compute distances to all vectors
        let mut distances: Vec<(usize, f32)> = subset
            .db_vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, distance.compute(query, v)))
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
                subset.queries.len(),
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
