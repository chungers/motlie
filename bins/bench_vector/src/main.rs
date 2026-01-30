//! Vector benchmark CLI tool.
//!
//! Consolidated benchmark tool for HNSW + RaBitQ vector search.
//!
//! ## Commands
//!
//! ```bash
//! # Download datasets
//! bench_vector download --dataset laion --data-dir ./data
//!
//! # Build index with checkpointing
//! bench_vector index --dataset laion --num-vectors 100000 --db-path ./db
//!
//! # Run queries (use embedding code from index output)
//! bench_vector query --db-path ./db --embedding-code <CODE> --queries 1000 --k 10
//!
//! # Parameter sweep
//! bench_vector sweep --dataset laion --bits 1,2,4 --ef 50,100,200 --rerank 1,4,10
//! ```

use anyhow::Result;
use clap::{Parser, Subcommand};

mod commands;

#[derive(Parser)]
#[command(name = "bench_vector")]
#[command(version, about = "Vector search benchmark tool for HNSW + RaBitQ")]
struct Cli {
    /// Verbose output
    #[arg(short, long, global = true)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Download benchmark datasets
    Download(commands::DownloadArgs),

    /// Build HNSW index (with optional checkpointing)
    Index(commands::IndexArgs),

    /// Run search queries on existing index
    Query(commands::QueryArgs),

    /// Parameter sweep (grid search over bits, ef, rerank)
    Sweep(commands::SweepArgs),

    /// Check RaBitQ rotation distribution (validates √D scaling)
    CheckDistribution(commands::CheckDistributionArgs),

    /// List or inspect embedding specs
    Embeddings(commands::EmbeddingsArgs),

    /// List available datasets
    Datasets,

    /// Administrative and diagnostic commands
    Admin(commands::AdminArgs),
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Download(args) => commands::download(args).await,
        Commands::Index(args) => commands::index(args).await,
        Commands::Query(args) => commands::query(args).await,
        Commands::Sweep(args) => commands::sweep(args).await,
        Commands::CheckDistribution(args) => commands::check_distribution(args),
        Commands::Embeddings(args) => commands::embeddings(args),
        Commands::Datasets => commands::list_datasets(),
        Commands::Admin(args) => commands::admin(args),
    }
}
