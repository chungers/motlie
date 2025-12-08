//! Telemetry module providing tracing subscriber initialization for dev and production builds.
//!
//! This module provides two subscriber initialization functions:
//! - `init_dev_subscriber()` - Simple stderr logging for development
//! - `init_otel_subscriber()` - OpenTelemetry integration for production (requires `dtrace-otel` feature)
//!
//! # Usage
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

use tracing::Level;
use tracing_subscriber::fmt;

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

    // Create resource with service name (using the standard semantic convention key)
    let resource = Resource::new(vec![
        KeyValue::new("service.name", service_name.to_string()),
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

    // Create resource with service name (using the standard semantic convention key)
    let resource = Resource::new(vec![
        KeyValue::new("service.name", service_name.to_string()),
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
