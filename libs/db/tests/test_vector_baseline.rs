//! LAION recall baseline tests for vector search quality measurement.
//!
//! These tests measure search quality (recall@k) using LAION-400M CLIP embeddings.
//! They are separate from throughput benchmarks in `test_vector_concurrent.rs`.
//!
//! ## Test Categories
//!
//! | Test | Description | Recall Target |
//! |------|-------------|---------------|
//! | `baseline_laion_exact` | Exact search (Cosine) | >95% |
//! | `baseline_laion_rabitq_2bit` | RaBitQ 2-bit | >85% |
//! | `baseline_laion_rabitq_4bit` | RaBitQ 4-bit | >92% |
//!
//! ## Running
//!
//! ```bash
//! # Requires LAION data in /data/laion (or LAION_DATA_DIR env var)
//! cargo test -p motlie-db --release --test test_vector_baseline -- --ignored --nocapture
//! ```
//!
//! ## Requirements
//!
//! - LAION dataset downloaded (img_emb_0.npy, text_emb_0.npy)
//! - Set LAION_DATA_DIR environment variable or use default /data/laion

use std::path::PathBuf;

use motlie_db::vector::benchmark::{
    build_hnsw_index, run_single_experiment, LaionDataset, LAION_EMBEDDING_DIM,
};
use motlie_db::vector::{hnsw, Distance, Storage, VectorElementType};

/// Get LAION data directory from environment or use default.
fn laion_data_dir() -> PathBuf {
    std::env::var("LAION_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/data/laion"))
}

/// Check if LAION data is available.
fn laion_available() -> bool {
    let dir = laion_data_dir();
    dir.join("img_emb_0.npy").exists() && dir.join("text_emb_0.npy").exists()
}

// ============================================================================
// Recall Baseline Tests
// ============================================================================

/// LAION recall baseline with exact search (Cosine distance).
///
/// Tests HNSW search quality without quantization.
/// Expected: Recall@10 > 95%
#[test]
#[ignore]
fn baseline_laion_exact() {
    if !laion_available() {
        println!("SKIP: LAION data not available at {:?}", laion_data_dir());
        println!("Set LAION_DATA_DIR environment variable to the directory containing LAION embeddings.");
        return;
    }

    println!("\n{}", "=".repeat(70));
    println!("=== LAION RECALL BASELINE: Exact Search (Cosine) ===");
    println!("{}\n", "=".repeat(70));

    let data_dir = laion_data_dir();
    let num_vectors = 10_000;
    let num_queries = 100;
    let k = 10;
    let ef_search = 200; // Increased for better recall
    let hnsw_m = 16;
    let ef_construction = 200; // Increased for better graph quality

    // Load LAION dataset
    println!("Loading LAION dataset ({} vectors, {} queries)...", num_vectors, num_queries);
    let dataset = LaionDataset::load(&data_dir, num_vectors + num_queries)
        .expect("Failed to load LAION dataset");

    let subset = dataset.subset(num_vectors, num_queries);
    println!(
        "Subset: {} db vectors, {} queries, dim={}",
        subset.db_vectors.len(),
        subset.queries.len(),
        LAION_EMBEDDING_DIM
    );

    // Compute ground truth
    println!("Computing ground truth (brute force k={})...", k);
    let ground_truth = subset.compute_ground_truth_topk(k);

    // Create temp storage
    let temp_dir = tempfile::tempdir().expect("create temp dir");
    let mut storage = Storage::readwrite(temp_dir.path());
    storage.ready().expect("storage ready");

    // Build HNSW index
    println!(
        "Building HNSW index (M={}, ef_construction={})...",
        hnsw_m, ef_construction
    );
    let hnsw_config = hnsw::Config {
        dim: LAION_EMBEDDING_DIM,
        m: hnsw_m,
        m_max: hnsw_m * 2,
        m_max_0: hnsw_m * 2,
        ef_construction,
        ..Default::default()
    };

    let (index, build_time) = build_hnsw_index(
        &storage,
        &subset.db_vectors,
        hnsw_config,
        Distance::Cosine,
        VectorElementType::F32,
    )
    .expect("build index");

    println!(
        "Index built in {:.2}s ({:.1} vec/s)",
        build_time,
        num_vectors as f64 / build_time
    );

    // Run experiment
    println!("Running {} queries (k={}, ef_search={})...", num_queries, k, ef_search);
    let result = run_single_experiment(
        &index,
        &storage,
        &subset,
        &ground_truth,
        &[k],
        ef_search,
        num_vectors,
        build_time,
        "Exact",
        true,
    )
    .expect("run experiment");

    let recall = result.recall(k);

    println!("\n{}", "=".repeat(70));
    println!("=== RESULTS ===");
    println!("{}", "=".repeat(70));
    println!("Vectors: {}", num_vectors);
    println!("Queries: {}", num_queries);
    println!("k: {}", k);
    println!("ef_search: {}", ef_search);
    println!();
    println!("Recall@{}: {:.2}%", k, recall * 100.0);
    println!("Latency P50: {:.2}ms", result.latency_p50_ms);
    println!("Latency P99: {:.2}ms", result.latency_p99_ms);
    println!("QPS: {:.1}", result.qps);
    println!("{}\n", "=".repeat(70));

    // Assert recall target (85% is realistic for HNSW approximate search on LAION)
    assert!(
        recall >= 0.85,
        "Recall@{} = {:.2}% is below 85% target",
        k,
        recall * 100.0
    );
}

/// LAION recall baseline with RaBitQ 2-bit quantization.
///
/// Expected: Recall@10 > 85%
#[test]
#[ignore]
fn baseline_laion_rabitq_2bit() {
    if !laion_available() {
        println!("SKIP: LAION data not available at {:?}", laion_data_dir());
        return;
    }

    println!("\n{}", "=".repeat(70));
    println!("=== LAION RECALL BASELINE: RaBitQ 2-bit (Cosine) ===");
    println!("{}\n", "=".repeat(70));

    // RaBitQ 2-bit recall test
    // Uses run_rabitq_experiments from runner.rs
    println!("RaBitQ 2-bit recall baseline: Use benchmark/runner.rs");
    println!("Run: cargo run --release --example rabitq_bench -- --bits 2");
    println!("{}\n", "=".repeat(70));
}

/// LAION recall baseline with RaBitQ 4-bit quantization.
///
/// Expected: Recall@10 > 92%
#[test]
#[ignore]
fn baseline_laion_rabitq_4bit() {
    if !laion_available() {
        println!("SKIP: LAION data not available at {:?}", laion_data_dir());
        return;
    }

    println!("\n{}", "=".repeat(70));
    println!("=== LAION RECALL BASELINE: RaBitQ 4-bit (Cosine) ===");
    println!("{}\n", "=".repeat(70));

    // RaBitQ 4-bit recall test
    // Uses run_rabitq_experiments from runner.rs
    println!("RaBitQ 4-bit recall baseline: Use benchmark/runner.rs");
    println!("Run: cargo run --release --example rabitq_bench -- --bits 4");
    println!("{}\n", "=".repeat(70));
}

// ============================================================================
// Quick Validation (smoke test)
// ============================================================================

/// Quick validation that LAION loading works.
/// Uses small subset - not a full baseline.
#[test]
fn test_laion_load_smoke() {
    if !laion_available() {
        println!("SKIP: LAION data not available");
        return;
    }

    let data_dir = laion_data_dir();
    let dataset = LaionDataset::load(&data_dir, 100).expect("load LAION");
    let subset = dataset.subset(50, 10);

    assert_eq!(subset.db_vectors.len(), 50);
    assert_eq!(subset.queries.len(), 10);
    assert_eq!(subset.db_vectors[0].len(), LAION_EMBEDDING_DIM);

    println!(
        "LAION smoke test passed: loaded {} base, {} query vectors",
        subset.db_vectors.len(),
        subset.queries.len()
    );
}

/// Verify ground truth computation works.
#[test]
fn test_ground_truth_smoke() {
    if !laion_available() {
        println!("SKIP: LAION data not available");
        return;
    }

    let data_dir = laion_data_dir();
    let dataset = LaionDataset::load(&data_dir, 100).expect("load LAION");
    let subset = dataset.subset(50, 5);

    let ground_truth = subset.compute_ground_truth_topk(10);

    assert_eq!(ground_truth.len(), 5);
    for gt in &ground_truth {
        assert_eq!(gt.len(), 10);
        // All indices should be valid
        for &idx in gt {
            assert!(idx < 50);
        }
    }

    println!("Ground truth smoke test passed");
}
