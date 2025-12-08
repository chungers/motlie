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
    //
    // The guard must be held for the lifetime of the application to ensure
    // proper shutdown and flushing of telemetry data when using OpenTelemetry.
    let _telemetry_guard = init_tracing();

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
    // _telemetry_guard is dropped here, triggering OtelGuard::drop() which
    // flushes pending spans to the collector before the process exits.
}

/// Initialize tracing subscriber.
///
/// Uses OpenTelemetry when the `dtrace-otel` feature is enabled and
/// `DTRACE_ENDPOINT` is set; otherwise falls back to stderr logging.
///
/// Returns an optional `OtelGuard` that must be kept alive for the duration
/// of the application when using OpenTelemetry. The guard ensures proper
/// shutdown and flushing of spans when dropped.
#[cfg(feature = "dtrace-otel")]
fn init_tracing() -> Option<motlie_core::telemetry::OtelGuard> {
    // Check if DTRACE endpoint is configured
    if let Ok(endpoint) = std::env::var("DTRACE_ENDPOINT") {
        let service_name = std::env::var("DTRACE_SERVICE_NAME")
            .unwrap_or_else(|_| "motlie".to_string());

        match motlie_core::telemetry::init_otel_subscriber_with_env_filter(
            &service_name,
            &endpoint,
        ) {
            Ok(guard) => return Some(guard),
            Err(e) => {
                eprintln!("Failed to initialize OpenTelemetry: {}. Falling back to dev subscriber.", e);
            }
        }
    }

    // Default: use development subscriber with env filter
    motlie_core::telemetry::init_dev_subscriber_with_env_filter();
    None
}

/// Initialize tracing subscriber (non-OpenTelemetry build).
///
/// Falls back to stderr logging when the `dtrace-otel` feature is not enabled.
#[cfg(not(feature = "dtrace-otel"))]
fn init_tracing() -> Option<()> {
    motlie_core::telemetry::init_dev_subscriber_with_env_filter();
    None
}
