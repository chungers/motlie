//! LAION-CLIP Benchmark for HNSW at Scale
//!
//! This benchmark reproduces experiments from the article:
//! "HNSW at Scale: Why Your RAG System Gets Worse as the Vector Database Grows"
//! https://towardsdatascience.com/hnsw-at-scale-why-your-rag-system-gets-worse-as-the-vector-database-grows/
//!
//! Dataset: LAION-400M CLIP ViT-B-32 embeddings (512D)
//! - Image embeddings: database vectors
//! - Text embeddings: query vectors (caption -> image retrieval)
//!
//! Usage:
//!   # Download data first (one-time)
//!   cargo run --release --example laion_benchmark -- --download
//!
//!   # Run experiments
//!   cargo run --release --example laion_benchmark -- --run-all
//!
//!   # Generate charts from existing CSV data
//!   cargo run --release --example laion_benchmark -- --charts-only

mod loader;
mod experiments;
mod charts;

use clap::Parser;
use std::path::PathBuf;

/// LAION-CLIP Benchmark: HNSW at Scale Experiments
#[derive(Parser, Debug)]
#[command(name = "laion_benchmark")]
#[command(about = "Reproduce HNSW at Scale experiments with LAION-CLIP data")]
struct Args {
    /// Download LAION-400M embeddings (first 200K)
    #[arg(long)]
    download: bool,

    /// Run all experiments (50K, 100K, 150K, 200K)
    #[arg(long)]
    run_all: bool,

    /// Run single experiment at specific scale
    #[arg(long)]
    scale: Option<usize>,

    /// Only generate charts from existing CSV data
    #[arg(long)]
    charts_only: bool,

    /// Data directory (default: examples/laion_benchmark/data)
    #[arg(long, default_value = "examples/laion_benchmark/data")]
    data_dir: PathBuf,

    /// Results directory (default: examples/laion_benchmark/results)
    #[arg(long, default_value = "examples/laion_benchmark/results")]
    results_dir: PathBuf,

    /// Number of queries to run (default: 500, matching article)
    #[arg(long, default_value = "500")]
    num_queries: usize,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();

    println!("=== LAION-CLIP Benchmark: HNSW at Scale ===\n");

    // Ensure directories exist
    std::fs::create_dir_all(&args.data_dir)?;
    std::fs::create_dir_all(&args.results_dir)?;

    if args.download {
        println!("Downloading LAION-400M embeddings...");
        loader::download_laion_embeddings(&args.data_dir)?;
        println!("Download complete!\n");
    }

    if args.charts_only {
        println!("Generating charts from existing CSV data...");
        charts::generate_all_charts(&args.results_dir)?;
        println!("Charts generated!\n");
        return Ok(());
    }

    if args.run_all {
        println!("Running experiments at all scales: 50K, 100K, 150K, 200K\n");

        // Article parameters
        let config = experiments::ExperimentConfig {
            scales: vec![50_000, 100_000, 150_000, 200_000],
            ef_search_values: vec![10, 20, 40, 80, 160],
            k_values: vec![1, 5, 10, 15, 20],
            num_queries: args.num_queries,
            m: 16,                    // Match article
            ef_construction: 100,     // Match article
            data_dir: args.data_dir.clone(),
            results_dir: args.results_dir.clone(),
            verbose: args.verbose,
        };

        experiments::run_all_experiments(&config)?;

        println!("\nGenerating charts...");
        charts::generate_all_charts(&args.results_dir)?;

    } else if let Some(scale) = args.scale {
        println!("Running experiment at scale: {}K\n", scale / 1000);

        let config = experiments::ExperimentConfig {
            scales: vec![scale],
            ef_search_values: vec![10, 20, 40, 80, 160],
            k_values: vec![1, 5, 10, 15, 20],
            num_queries: args.num_queries,
            m: 16,
            ef_construction: 100,
            data_dir: args.data_dir.clone(),
            results_dir: args.results_dir.clone(),
            verbose: args.verbose,
        };

        experiments::run_all_experiments(&config)?;
    } else {
        println!("No action specified. Use --help for options.");
        println!("\nQuick start:");
        println!("  1. Download data:  cargo run --release --example laion_benchmark -- --download");
        println!("  2. Run experiments: cargo run --release --example laion_benchmark -- --run-all");
    }

    println!("\n=== Benchmark Complete ===");
    Ok(())
}
