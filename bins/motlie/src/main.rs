use clap::{Parser, Subcommand};

mod db;
mod fulltext;

#[allow(unused_imports)]
use tracing::{debug, error, info, trace, warn};

#[derive(Parser)]
#[clap(author = "chunger", version, about = "Motlie CLI utility")]
#[clap(propagate_version = true)]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Database inspection commands
    Db(db::Command),
    /// Fulltext search commands
    Fulltext(fulltext::Command),
}

fn main() {
    // Initialize tracing subscriber based on build configuration
    // For production builds with OpenTelemetry, enable the telemetry-otel feature
    // and set OTEL_EXPORTER_OTLP_ENDPOINT environment variable
    init_tracing();

    let cli = Cli::parse();

    tracing::info!("starting");

    match cli.command {
        Commands::Db(args) => {
            db::run(&args);
        }
        Commands::Fulltext(args) => {
            fulltext::run(&args);
        }
    }
}

/// Initialize tracing subscriber.
///
/// Uses OpenTelemetry when the `dtrace-otel` feature is enabled and
/// `DTRACE_ENDPOINT` is set; otherwise falls back to stderr logging.
fn init_tracing() {
    #[cfg(feature = "dtrace-otel")]
    {
        // Check if DTRACE endpoint is configured
        if let Ok(endpoint) = std::env::var("DTRACE_ENDPOINT") {
            let service_name = std::env::var("DTRACE_SERVICE_NAME")
                .unwrap_or_else(|_| "motlie".to_string());

            if let Err(e) = motlie_core::telemetry::init_otel_subscriber_with_env_filter(
                &service_name,
                &endpoint,
            ) {
                eprintln!("Failed to initialize OpenTelemetry: {}. Falling back to dev subscriber.", e);
                motlie_core::telemetry::init_dev_subscriber_with_env_filter();
            }
            return;
        }
    }

    // Default: use development subscriber with env filter
    motlie_core::telemetry::init_dev_subscriber_with_env_filter();
}
