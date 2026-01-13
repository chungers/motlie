//! CLI command implementations.
//!
//! Note: This CLI provides basic dataset management. For full benchmark
//! experiments, use the benchmark module API directly via examples/vector2.

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;

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
    use motlie_db::vector::benchmark;

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
// Index Command (placeholder)
// ============================================================================

#[derive(Parser)]
pub struct IndexArgs {
    /// Dataset: laion, sift, gist, cohere, glove
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
}

pub async fn index(args: IndexArgs) -> Result<()> {
    println!("=== Vector Index Build ===");
    println!("Dataset: {}", args.dataset);
    println!("Vectors: {}", args.num_vectors);
    println!("Database: {:?}", args.db_path);
    println!("\nNote: Use examples/vector2 for full index building with all parameters.");
    println!("This command will be fully implemented in a future version.");
    Ok(())
}

// ============================================================================
// Query Command (placeholder)
// ============================================================================

#[derive(Parser)]
pub struct QueryArgs {
    /// Database path
    #[arg(long)]
    pub db_path: PathBuf,

    /// Number of results per query
    #[arg(long, default_value = "10")]
    pub k: usize,
}

pub async fn query(args: QueryArgs) -> Result<()> {
    println!("=== Vector Search ===");
    println!("Database: {:?}", args.db_path);
    println!("k={}", args.k);
    println!("\nNote: Use examples/vector2 for full search functionality.");
    println!("This command will be fully implemented in a future version.");
    Ok(())
}

// ============================================================================
// Sweep Command (placeholder)
// ============================================================================

#[derive(Parser)]
pub struct SweepArgs {
    /// Dataset: laion, sift, gist, cohere, glove
    #[arg(long)]
    pub dataset: String,

    /// Data directory
    #[arg(long, default_value = "./data")]
    pub data_dir: PathBuf,

    /// Number of database vectors
    #[arg(long, default_value = "100000")]
    pub num_vectors: usize,

    /// Number of queries
    #[arg(long, default_value = "1000")]
    pub num_queries: usize,
}

pub async fn sweep(args: SweepArgs) -> Result<()> {
    println!("=== Parameter Sweep ===");
    println!("Dataset: {}", args.dataset);
    println!("Vectors: {}, Queries: {}", args.num_vectors, args.num_queries);
    println!("\nNote: Use examples/vector2 or the benchmark module API for full sweeps.");
    println!("Example:");
    println!("  cargo run --example vector2 -- --data-dir {:?} --num-vectors {} --num-queries {}",
             args.data_dir, args.num_vectors, args.num_queries);
    Ok(())
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

    println!("Run benchmarks (use examples/vector2 for now):");
    println!("  cargo run --example vector2 -- --data-dir ./data --num-vectors 100000\n");

    println!("Full benchmark with RaBitQ:");
    println!("  cargo run --example vector2 -- --data-dir ./data --scale 50000,100000 --rabitq");

    Ok(())
}
