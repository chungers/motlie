#![cfg(feature = "benchmark")]
//! LAION smoke tests for vector search infrastructure.
//!
//! **NOTE:** Quality baseline tests (recall measurement) have been moved to the
//! `bench_vector` CLI tool. Use `bench_vector sweep --assert-recall` for CI integration.
//!
//! ## Running Quality Baselines (Recommended)
//!
//! ```bash
//! # Download LAION dataset
//! cargo run --release --bin bench_vector -- download --dataset laion --data-dir ~/data/laion
//!
//! # Run HNSW baseline with recall assertion
//! cargo run --release --bin bench_vector -- sweep \
//!     --dataset laion --data-dir ~/data/laion \
//!     --num-vectors 10000 --ef 100,200 --k 10 \
//!     --assert-recall 0.85
//!
//! # Run RaBitQ baseline with recall assertion
//! cargo run --release --bin bench_vector -- sweep \
//!     --dataset laion --data-dir ~/data/laion \
//!     --num-vectors 10000 --rabitq --bits 2,4 --rerank 10 \
//!     --assert-recall 0.80
//! ```
//!
//! ## Smoke Tests (This File)
//!
//! These tests verify LAION data loading works but do not measure recall.
//! They are fast and useful for development iteration.

use std::path::PathBuf;

use motlie_db::vector::benchmark::{LaionDataset, LAION_EMBEDDING_DIM};

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
// Smoke Tests
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
