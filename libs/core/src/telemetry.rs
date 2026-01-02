//! Telemetry module providing tracing subscriber initialization and build metadata.
//!
//! This module provides:
//! - Subscriber initialization for dev and production
//! - Build metadata (git hash, SIMD level, version, etc.)
//!
//! # Build Metadata
//!
//! ```no_run
//! use motlie_core::telemetry::{BuildInfo, log_build_info};
//!
//! fn main() {
//!     // Get build info
//!     let info = BuildInfo::current();
//!     println!("Version: {} ({})", info.version, info.git_hash);
//!     println!("SIMD: {}", info.simd_level);
//!
//!     // Or log it at startup
//!     log_build_info();
//! }
//! ```
//!
//! # Subscriber Initialization
//!
//! ## Development (simple stderr logging)
//! ```no_run
//! use motlie_core::telemetry;
//!
//! fn main() {
//!     telemetry::init_dev_subscriber();
//!     // Application code...
//! }
//! ```
//!
//! ## Production (OpenTelemetry)
//! First, enable the feature in Cargo.toml:
//! ```toml
//! motlie-core = { path = "../libs/core", features = ["dtrace-otel"] }
//! ```
//!
//! Then initialize in your application. **Important**: Keep the returned guard alive
//! for the duration of your application to ensure proper shutdown and span flushing:
//! ```ignore
//! use motlie_core::telemetry;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // The guard ensures spans are flushed when main exits
//!     let _telemetry_guard = telemetry::init_otel_subscriber("my-service", "http://localhost:4317")?;
//!     // Application code...
//!     Ok(())
//! }
//! ```

use crate::distance;
use tracing::Level;
use tracing_subscriber::fmt;

// ============================================================================
// Build Metadata
// ============================================================================

/// Build and runtime information for the binary.
///
/// Provides metadata about the build that is useful for debugging,
/// observability, and version tracking.
///
/// # Example
///
/// ```no_run
/// use motlie_core::telemetry::BuildInfo;
///
/// let info = BuildInfo::current();
/// println!("Running {} v{}", info.package_name, info.version);
/// println!("Git: {}", info.git_hash);
/// println!("SIMD: {}", info.simd_level);
/// println!("Built: {}", info.build_timestamp);
/// ```
#[derive(Debug, Clone)]
pub struct BuildInfo {
    /// Package name from Cargo.toml
    pub package_name: &'static str,
    /// Package version from Cargo.toml
    pub version: &'static str,
    /// Git commit hash (short, with "-dirty" suffix if uncommitted changes)
    pub git_hash: &'static str,
    /// Build timestamp (RFC 3339 format)
    pub build_timestamp: &'static str,
    /// Target architecture (e.g., "x86_64", "aarch64")
    pub target_arch: &'static str,
    /// Target OS (e.g., "linux", "macos", "windows")
    pub target_os: &'static str,
    /// Active SIMD implementation (e.g., "NEON", "AVX2+FMA", "Scalar")
    pub simd_level: &'static str,
    /// Enabled Cargo features (comma-separated, or "default" if none)
    pub features: &'static str,
    /// Build profile (e.g., "debug", "release")
    pub profile: &'static str,
}

impl BuildInfo {
    /// Get build information for the current binary.
    ///
    /// This information is captured at compile time and reflects
    /// the state of the repository when the binary was built.
    pub fn current() -> Self {
        Self {
            package_name: env!("CARGO_PKG_NAME"),
            version: env!("CARGO_PKG_VERSION"),
            git_hash: env!("MOTLIE_GIT_HASH"),
            build_timestamp: env!("MOTLIE_BUILD_TIMESTAMP"),
            target_arch: std::env::consts::ARCH,
            target_os: std::env::consts::OS,
            simd_level: distance::simd_level(),
            features: env!("MOTLIE_FEATURES"),
            profile: if cfg!(debug_assertions) { "debug" } else { "release" },
        }
    }

    /// Format build info as a single-line summary.
    ///
    /// Example: `motlie-core v0.1.0 (abc1234) [NEON, aarch64-linux, release]`
    pub fn summary(&self) -> String {
        format!(
            "{} v{} ({}) [{}, {}-{}, {}]",
            self.package_name,
            self.version,
            self.git_hash,
            self.simd_level,
            self.target_arch,
            self.target_os,
            self.profile,
        )
    }

    /// Format build info as multiple lines for detailed output.
    pub fn detailed(&self) -> String {
        format!(
            "Package:    {} v{}\n\
             Git:        {}\n\
             Built:      {}\n\
             Target:     {}-{}\n\
             SIMD:       {}\n\
             Features:   {}\n\
             Profile:    {}",
            self.package_name,
            self.version,
            self.git_hash,
            self.build_timestamp,
            self.target_arch,
            self.target_os,
            self.simd_level,
            self.features,
            self.profile,
        )
    }
}

impl std::fmt::Display for BuildInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.summary())
    }
}

/// Log build information at INFO level.
///
/// Call this at application startup to record build metadata in logs/traces.
///
/// # Example
///
/// ```no_run
/// use motlie_core::telemetry;
///
/// fn main() {
///     telemetry::init_dev_subscriber();
///     telemetry::log_build_info();
///     // Application code...
/// }
/// ```
pub fn log_build_info() {
    let info = BuildInfo::current();
    tracing::info!(
        package = info.package_name,
        version = info.version,
        git_hash = info.git_hash,
        build_timestamp = info.build_timestamp,
        target_arch = info.target_arch,
        target_os = info.target_os,
        simd_level = info.simd_level,
        features = info.features,
        profile = info.profile,
        "Build info"
    );
}

/// Print build information to stderr (for use before logging is initialized).
///
/// # Example
///
/// ```no_run
/// use motlie_core::telemetry;
///
/// fn main() {
///     // Print build info before logging setup
///     telemetry::print_build_info();
///
///     telemetry::init_dev_subscriber();
///     // Application code...
/// }
/// ```
pub fn print_build_info() {
    eprintln!("{}", BuildInfo::current().detailed());
}

// ============================================================================
// Subsystem Configuration Reporting
// ============================================================================

/// Trait for subsystems to provide their configuration info for the `info` command.
///
/// Subsystems like graph database, fulltext search, or vector index implement
/// this trait to expose their runtime configuration in a consistent format.
///
/// # Example
///
/// ```ignore
/// use motlie_core::telemetry::SubsystemInfo;
///
/// pub struct MySubsystemInfo {
///     pub cache_size: usize,
/// }
///
/// impl SubsystemInfo for MySubsystemInfo {
///     fn name(&self) -> &'static str {
///         "My Subsystem"
///     }
///
///     fn info_lines(&self) -> Vec<(&'static str, String)> {
///         vec![
///             ("Cache Size", format!("{} MB", self.cache_size / (1024 * 1024))),
///         ]
///     }
/// }
/// ```
pub trait SubsystemInfo {
    /// Short name of the subsystem (e.g., "Graph DB", "Fulltext", "Vector Index")
    fn name(&self) -> &'static str;

    /// Key-value pairs describing the subsystem's configuration.
    /// Each tuple is (label, value) for display.
    fn info_lines(&self) -> Vec<(&'static str, String)>;
}

/// Format subsystem info for display.
///
/// Outputs the subsystem name as a section header followed by
/// indented key-value pairs.
///
/// # Example Output
///
/// ```text
/// [Graph Database (RocksDB)]
///   Block Cache Size:    256 MB
///   Graph Block Size:    4 KB
/// ```
pub fn format_subsystem_info(subsystem: &dyn SubsystemInfo) -> String {
    let mut output = format!("\n[{}]", subsystem.name());
    for (label, value) in subsystem.info_lines() {
        output.push_str(&format!("\n  {:<20} {}", format!("{}:", label), value));
    }
    output
}

// ============================================================================
// Tracing Subscribers
// ============================================================================

/// Guard that ensures proper shutdown of OpenTelemetry on drop.
///
/// When this guard is dropped, it calls `shutdown()` on the `TracerProvider`,
/// which flushes any pending spans to the collector. This is essential for
/// ensuring all telemetry data is exported before the application exits.
///
/// # Example
/// ```ignore
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let _guard = motlie_core::telemetry::init_otel_subscriber("my-service", "http://localhost:4317")?;
///     // Application runs...
///     // When main exits, _guard is dropped, triggering shutdown
///     Ok(())
/// }
/// ```
#[cfg(feature = "dtrace-otel")]
pub struct OtelGuard {
    provider: opentelemetry_sdk::trace::TracerProvider,
}

#[cfg(feature = "dtrace-otel")]
impl Drop for OtelGuard {
    fn drop(&mut self) {
        if let Err(e) = self.provider.shutdown() {
            eprintln!("Error shutting down OpenTelemetry tracer provider: {:?}", e);
        }
    }
}

/// Initialize a simple stderr subscriber for development.
///
/// This sets up a tracing subscriber that:
/// - Outputs to stderr
/// - Shows DEBUG level and above
/// - Includes target (module path), file, and line number
/// - Uses a compact format suitable for terminal output
///
/// Call this at application startup (not in the library).
///
/// # Panics
/// Panics if a global subscriber has already been set.
///
/// # Example
/// ```no_run
/// use motlie_core::telemetry;
///
/// fn main() {
///     telemetry::init_dev_subscriber();
///     tracing::info!("Application started");
/// }
/// ```
pub fn init_dev_subscriber() {
    let subscriber = fmt::Subscriber::builder()
        .with_max_level(Level::DEBUG)
        .with_writer(std::io::stderr)
        .with_target(true)
        .with_thread_ids(false)
        .with_file(true)
        .with_line_number(true)
        .finish();

    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set tracing subscriber");
}

/// Initialize a simple stderr subscriber for development with environment filter.
///
/// This is similar to `init_dev_subscriber` but respects the `RUST_LOG` environment
/// variable for filtering. If `RUST_LOG` is not set, defaults to DEBUG level.
///
/// # Example
/// ```no_run
/// use motlie_core::telemetry;
///
/// fn main() {
///     // Set RUST_LOG=info to only see info and above
///     // Set RUST_LOG=motlie_db=debug,info to see debug for motlie_db, info for others
///     telemetry::init_dev_subscriber_with_env_filter();
///     tracing::info!("Application started");
/// }
/// ```
pub fn init_dev_subscriber_with_env_filter() {
    use tracing_subscriber::EnvFilter;

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("debug"));

    let subscriber = fmt::Subscriber::builder()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_target(true)
        .with_thread_ids(false)
        .with_file(true)
        .with_line_number(true)
        .finish();

    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set tracing subscriber");
}

/// Initialize OpenTelemetry subscriber for production.
///
/// This sets up a tracing subscriber that:
/// - Exports spans to an OTLP endpoint (e.g., Jaeger, Tempo, or OpenTelemetry Collector)
/// - Also logs to stderr for local visibility
/// - Uses batch export for efficiency
///
/// Requires the `dtrace-otel` feature to be enabled.
///
/// **Important**: The returned [`OtelGuard`] must be kept alive for the duration of
/// your application. When the guard is dropped, it flushes pending spans and shuts
/// down the tracer provider gracefully.
///
/// # Arguments
/// * `service_name` - The name of your service (appears in traces)
/// * `endpoint` - The OTLP endpoint URL (e.g., "http://localhost:4317")
///
/// # Returns
/// An [`OtelGuard`] that must be held until application exit to ensure proper cleanup.
///
/// # Errors
/// Returns an error if:
/// - The OTLP exporter cannot be created
/// - The tracer provider cannot be built
/// - A global subscriber has already been set
///
/// # Example
/// ```ignore
/// use motlie_core::telemetry;
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Keep the guard alive for the entire application lifetime
///     let _guard = telemetry::init_otel_subscriber("my-service", "http://localhost:4317")?;
///     tracing::info!("Application started with OpenTelemetry");
///     // When main exits, _guard is dropped, flushing all pending spans
///     Ok(())
/// }
/// ```
#[cfg(feature = "dtrace-otel")]
pub fn init_otel_subscriber(
    service_name: &str,
    endpoint: &str,
) -> Result<OtelGuard, Box<dyn std::error::Error + Send + Sync>> {
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry::KeyValue;
    use opentelemetry_otlp::WithExportConfig;
    use opentelemetry_sdk::trace::TracerProvider;
    use opentelemetry_sdk::runtime;
    use opentelemetry_sdk::Resource;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    // Create the OTLP exporter
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .build()?;

    // Get build info for resource attributes
    let build_info = BuildInfo::current();

    // Create resource with service name and build metadata
    // Using OpenTelemetry semantic conventions where applicable
    let resource = Resource::new(vec![
        KeyValue::new("service.name", service_name.to_string()),
        KeyValue::new("service.version", build_info.version.to_string()),
        KeyValue::new("deployment.environment", build_info.profile.to_string()),
        // Custom attributes for motlie-specific metadata
        KeyValue::new("motlie.git_hash", build_info.git_hash.to_string()),
        KeyValue::new("motlie.build_timestamp", build_info.build_timestamp.to_string()),
        KeyValue::new("motlie.simd_level", build_info.simd_level.to_string()),
        KeyValue::new("host.arch", build_info.target_arch.to_string()),
        KeyValue::new("os.type", build_info.target_os.to_string()),
    ]);

    // Build the tracer provider with batch export and resource
    let provider = TracerProvider::builder()
        .with_batch_exporter(exporter, runtime::Tokio)
        .with_resource(resource)
        .build();

    // Create a tracer from the provider
    let tracer = provider.tracer(service_name.to_string());

    // Create the OpenTelemetry layer
    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    // Create the fmt layer for stderr output
    let fmt_layer = fmt::layer()
        .with_writer(std::io::stderr)
        .with_target(true)
        .with_file(true)
        .with_line_number(true);

    // Combine layers and set as global subscriber
    tracing_subscriber::registry()
        .with(otel_layer)
        .with(fmt_layer)
        .init();

    // Return guard to keep the provider alive and ensure proper shutdown
    Ok(OtelGuard { provider })
}

/// Initialize OpenTelemetry subscriber with environment filter for production.
///
/// Similar to `init_otel_subscriber` but respects the `RUST_LOG` environment variable.
///
/// **Important**: The returned [`OtelGuard`] must be kept alive for the duration of
/// your application. When the guard is dropped, it flushes pending spans and shuts
/// down the tracer provider gracefully.
///
/// # Arguments
/// * `service_name` - The name of your service (appears in traces)
/// * `endpoint` - The OTLP endpoint URL (e.g., "http://localhost:4317")
///
/// # Returns
/// An [`OtelGuard`] that must be held until application exit to ensure proper cleanup.
///
/// # Example
/// ```ignore
/// use motlie_core::telemetry;
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Set RUST_LOG=info for production
///     // Keep the guard alive for the entire application lifetime
///     let _guard = telemetry::init_otel_subscriber_with_env_filter("my-service", "http://localhost:4317")?;
///     tracing::info!("Application started with OpenTelemetry");
///     // When main exits, _guard is dropped, flushing all pending spans
///     Ok(())
/// }
/// ```
#[cfg(feature = "dtrace-otel")]
pub fn init_otel_subscriber_with_env_filter(
    service_name: &str,
    endpoint: &str,
) -> Result<OtelGuard, Box<dyn std::error::Error + Send + Sync>> {
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry::KeyValue;
    use opentelemetry_otlp::WithExportConfig;
    use opentelemetry_sdk::trace::TracerProvider;
    use opentelemetry_sdk::runtime;
    use opentelemetry_sdk::Resource;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    use tracing_subscriber::EnvFilter;

    // Create the OTLP exporter
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .build()?;

    // Get build info for resource attributes
    let build_info = BuildInfo::current();

    // Create resource with service name and build metadata
    // Using OpenTelemetry semantic conventions where applicable
    let resource = Resource::new(vec![
        KeyValue::new("service.name", service_name.to_string()),
        KeyValue::new("service.version", build_info.version.to_string()),
        KeyValue::new("deployment.environment", build_info.profile.to_string()),
        // Custom attributes for motlie-specific metadata
        KeyValue::new("motlie.git_hash", build_info.git_hash.to_string()),
        KeyValue::new("motlie.build_timestamp", build_info.build_timestamp.to_string()),
        KeyValue::new("motlie.simd_level", build_info.simd_level.to_string()),
        KeyValue::new("host.arch", build_info.target_arch.to_string()),
        KeyValue::new("os.type", build_info.target_os.to_string()),
    ]);

    // Build the tracer provider with batch export and resource
    let provider = TracerProvider::builder()
        .with_batch_exporter(exporter, runtime::Tokio)
        .with_resource(resource)
        .build();

    // Create a tracer from the provider
    let tracer = provider.tracer(service_name.to_string());

    // Create the OpenTelemetry layer
    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    // Create the fmt layer for stderr output
    let fmt_layer = fmt::layer()
        .with_writer(std::io::stderr)
        .with_target(true)
        .with_file(true)
        .with_line_number(true);

    // Create env filter
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    // Combine layers and set as global subscriber
    tracing_subscriber::registry()
        .with(filter)
        .with(otel_layer)
        .with(fmt_layer)
        .init();

    // Return guard to keep the provider alive and ensure proper shutdown
    Ok(OtelGuard { provider })
}

#[cfg(test)]
mod tests {
    // Note: We can't easily test subscriber initialization in unit tests
    // because set_global_default can only be called once per process.
    // These would be integration tests in practice.

    #[test]
    fn test_module_compiles() {
        // Just verify the module compiles correctly
        assert!(true);
    }
}
