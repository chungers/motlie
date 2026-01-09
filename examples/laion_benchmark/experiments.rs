//! Experiment runner for HNSW at Scale benchmarks
//!
//! Reproduces experiments from the article with:
//! - Scales: 50K, 100K, 150K, 200K vectors
//! - ef_search: 10, 20, 40, 80, 160
//! - Recall@k for k = 1, 5, 10, 15, 20
//! - Latency measurements

use crate::loader::{cosine_distance, LaionDataset, LaionSubset, EMBEDDING_DIM};
use anyhow::Result;
use motlie_db::rocksdb::ColumnFamily;
use motlie_db::vector::{
    Distance, EmbeddingCode, HnswConfig, HnswIndex, NavigationCache,
    Storage, VecId, VectorCfKey, VectorCfValue, Vectors,
};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

/// Experiment configuration matching the article
pub struct ExperimentConfig {
    pub scales: Vec<usize>,
    pub ef_search_values: Vec<usize>,
    pub k_values: Vec<usize>,
    pub num_queries: usize,
    pub m: usize,
    pub ef_construction: usize,
    pub data_dir: PathBuf,
    pub results_dir: PathBuf,
    pub verbose: bool,
}

/// Results from a single experiment run
#[derive(Debug, Clone)]
pub struct ExperimentResult {
    pub scale: usize,
    pub ef_search: usize,
    pub strategy: String,
    pub recall_at_k: HashMap<usize, f64>,
    pub latency_avg_ms: f64,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_p99_ms: f64,
    pub qps: f64,
    pub build_time_s: f64,
}

/// Run all experiments according to config
pub fn run_all_experiments(config: &ExperimentConfig) -> Result<Vec<ExperimentResult>> {
    // Load full dataset
    let max_vectors = *config.scales.iter().max().unwrap_or(&200_000);
    println!("Loading LAION dataset (up to {} vectors)...", max_vectors);
    let dataset = LaionDataset::load(&config.data_dir, max_vectors)?;
    println!("Dataset loaded: {} image embeddings, {} text embeddings\n",
             dataset.image_embeddings.len(), dataset.text_embeddings.len());

    let mut all_results = Vec::new();

    for &scale in &config.scales {
        println!("\n{}", "=".repeat(60));
        println!("Scale: {} vectors", scale);
        println!("{}\n", "=".repeat(60));

        // Get subset for this scale
        let subset = dataset.subset(scale, config.num_queries);
        println!("Subset: {} db vectors, {} queries", subset.db_vectors.len(), subset.queries.len());

        // Compute brute-force ground truth for all k values
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
        println!("\nBuilding HNSW index (M={}, ef_construction={})...",
                 config.m, config.ef_construction);
        let (index, build_time) = build_hnsw_index(
            &storage,
            &subset.db_vectors,
            config.m,
            config.ef_construction,
        )?;
        println!("Index built in {:.2}s ({:.1} vec/s)",
                 build_time, scale as f64 / build_time);

        // Run experiments for each ef_search value
        for &ef_search in &config.ef_search_values {
            println!("\n--- ef_search = {} ---", ef_search);

            // HNSW with Cosine distance (our implementation after bug fix)
            let result = run_single_experiment(
                &index,
                &storage,
                &subset,
                &ground_truth,
                &config.k_values,
                ef_search,
                scale,
                build_time,
                "HNSW-Cosine",
                config.verbose,
            )?;
            all_results.push(result);
        }

        // Also run flat (brute-force) baseline for comparison
        println!("\n--- Flat (brute-force) baseline ---");
        let flat_result = run_flat_baseline(
            &subset,
            &ground_truth,
            &config.k_values,
            scale,
            config.verbose,
        )?;
        all_results.push(flat_result);

        // temp_dir is automatically cleaned up when it goes out of scope
        drop(temp_dir);
    }

    // Save results to CSV
    save_results_csv(&all_results, &config.results_dir)?;

    Ok(all_results)
}

/// Build HNSW index from vectors
fn build_hnsw_index(
    storage: &Storage,
    vectors: &[Vec<f32>],
    m: usize,
    ef_construction: usize,
) -> Result<(HnswIndex, f64)> {
    let hnsw_config = HnswConfig {
        dim: EMBEDDING_DIM,
        m,
        m_max: m * 2,
        m_max_0: m * 2,
        ef_construction,
        ..Default::default()
    };

    let nav_cache = Arc::new(NavigationCache::new());
    let embedding_code: EmbeddingCode = 1;

    let index = HnswIndex::new(
        embedding_code,
        Distance::Cosine,
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
        let value = VectorCfValue(vector.clone());
        txn_db.put_cf(
            &vectors_cf,
            Vectors::key_to_bytes(&key),
            Vectors::value_to_bytes(&value),
        )?;

        // Insert into HNSW index
        index.insert(storage, vec_id, vector)?;

        if (i + 1) % 10000 == 0 {
            println!("  Inserted {}/{} vectors", i + 1, vectors.len());
        }
    }

    let build_time = start.elapsed().as_secs_f64();
    Ok((index, build_time))
}

/// Run a single experiment configuration
fn run_single_experiment(
    index: &HnswIndex,
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
            println!("  Query {}/{}: {:.2}ms", qi + 1, subset.queries.len(), latency_ms);
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

    println!("  Latency: avg={:.2}ms, p50={:.2}ms, p95={:.2}ms, p99={:.2}ms",
             latency_avg_ms, latency_p50_ms, latency_p95_ms, latency_p99_ms);
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

/// Run flat (brute-force) baseline for comparison
fn run_flat_baseline(
    subset: &LaionSubset,
    ground_truth: &[Vec<usize>],
    k_values: &[usize],
    scale: usize,
    verbose: bool,
) -> Result<ExperimentResult> {
    let mut latencies = Vec::with_capacity(subset.queries.len());
    let mut search_results = Vec::with_capacity(subset.queries.len());

    let max_k = *k_values.iter().max().unwrap_or(&20);

    // Run brute-force queries
    for (qi, query) in subset.queries.iter().enumerate() {
        let start = Instant::now();

        // Compute distances to all vectors
        let mut distances: Vec<(usize, f32)> = subset.db_vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, cosine_distance(query, v)))
            .collect();

        // Sort and take top-k
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let result_ids: Vec<usize> = distances.iter().take(max_k).map(|(i, _)| *i).collect();

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        latencies.push(latency_ms);
        search_results.push(result_ids);

        if verbose && (qi + 1) % 100 == 0 {
            println!("  Query {}/{}: {:.2}ms", qi + 1, subset.queries.len(), latency_ms);
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

    println!("  Latency: avg={:.2}ms, p50={:.2}ms, p95={:.2}ms, p99={:.2}ms",
             latency_avg_ms, latency_p50_ms, latency_p95_ms, latency_p99_ms);
    println!("  QPS: {:.1}", qps);

    Ok(ExperimentResult {
        scale,
        ef_search: 0,  // N/A for flat search
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

/// Compute Recall@k
/// recall = |retrieved ∩ relevant| / |relevant|
fn compute_recall(
    search_results: &[Vec<usize>],
    ground_truth: &[Vec<usize>],
    k: usize,
) -> f64 {
    let mut total_recall = 0.0;

    for (results, truth) in search_results.iter().zip(ground_truth.iter()) {
        let retrieved: std::collections::HashSet<_> = results.iter().take(k).collect();
        let relevant: std::collections::HashSet<_> = truth.iter().take(k).collect();

        let intersection = retrieved.intersection(&relevant).count();
        let recall = intersection as f64 / relevant.len().min(k) as f64;
        total_recall += recall;
    }

    total_recall / search_results.len() as f64
}

/// Compute percentile from sorted values
fn percentile(sorted_values: &[f64], p: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted_values.len() - 1) as f64).round() as usize;
    sorted_values[idx.min(sorted_values.len() - 1)]
}

/// Save results to CSV files
fn save_results_csv(results: &[ExperimentResult], results_dir: &PathBuf) -> Result<()> {
    // Main results CSV
    let csv_path = results_dir.join("laion_benchmark_results.csv");
    let mut file = File::create(&csv_path)?;

    // Header
    writeln!(file, "scale,ef_search,strategy,recall@1,recall@5,recall@10,recall@15,recall@20,latency_avg_ms,latency_p50_ms,latency_p95_ms,latency_p99_ms,qps,build_time_s")?;

    // Data rows
    for r in results {
        writeln!(
            file,
            "{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.2},{:.2}",
            r.scale,
            r.ef_search,
            r.strategy,
            r.recall_at_k.get(&1).unwrap_or(&0.0),
            r.recall_at_k.get(&5).unwrap_or(&0.0),
            r.recall_at_k.get(&10).unwrap_or(&0.0),
            r.recall_at_k.get(&15).unwrap_or(&0.0),
            r.recall_at_k.get(&20).unwrap_or(&0.0),
            r.latency_avg_ms,
            r.latency_p50_ms,
            r.latency_p95_ms,
            r.latency_p99_ms,
            r.qps,
            r.build_time_s,
        )?;
    }

    println!("\nResults saved to: {:?}", csv_path);

    // Also save a summary markdown file
    let md_path = results_dir.join("laion_benchmark_summary.md");
    save_summary_markdown(results, &md_path)?;

    Ok(())
}

/// Save summary as markdown
fn save_summary_markdown(results: &[ExperimentResult], path: &PathBuf) -> Result<()> {
    let mut file = File::create(path)?;

    writeln!(file, "# LAION-CLIP Benchmark Results")?;
    writeln!(file)?;
    writeln!(file, "Reproducing experiments from \"HNSW at Scale\" article.")?;
    writeln!(file)?;
    writeln!(file, "## Configuration")?;
    writeln!(file)?;
    writeln!(file, "- Dataset: LAION-400M CLIP ViT-B-32 (512D)")?;
    writeln!(file, "- HNSW: M=16, ef_construction=100")?;
    writeln!(file, "- Distance: Cosine")?;
    writeln!(file)?;

    // Group by scale
    let mut scales: Vec<usize> = results.iter().map(|r| r.scale).collect();
    scales.sort();
    scales.dedup();

    for scale in scales {
        writeln!(file, "## {} Vectors", scale)?;
        writeln!(file)?;
        writeln!(file, "| Strategy | ef_search | Recall@5 | Recall@10 | Latency (ms) | QPS |")?;
        writeln!(file, "|----------|-----------|----------|-----------|--------------|-----|")?;

        for r in results.iter().filter(|r| r.scale == scale) {
            writeln!(
                file,
                "| {} | {} | {:.1}% | {:.1}% | {:.2} | {:.1} |",
                r.strategy,
                if r.ef_search == 0 { "N/A".to_string() } else { r.ef_search.to_string() },
                r.recall_at_k.get(&5).unwrap_or(&0.0) * 100.0,
                r.recall_at_k.get(&10).unwrap_or(&0.0) * 100.0,
                r.latency_avg_ms,
                r.qps,
            )?;
        }
        writeln!(file)?;
    }

    println!("Summary saved to: {:?}", path);
    Ok(())
}
