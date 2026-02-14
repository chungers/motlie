//! Tracing Durability Test
//!
//! This example tests the durability of the telemetry guard pattern by running
//! a simple HTTP server and client that generate spans.
//!
//! ## Running with stderr logging (default)
//! ```bash
//! cargo run --example tracing_durability
//! ```
//!
//! ## Running with OpenTelemetry (requires a collector)
//! First, start an OTLP collector (e.g., Jaeger):
//! ```bash
//! docker run -d --name jaeger \
//!   -e COLLECTOR_OTLP_ENABLED=true \
//!   -p 16686:16686 \
//!   -p 4317:4317 \
//!   jaegertracing/all-in-one:1.51
//! ```
//!
//! Then run with the dtrace-otel feature:
//! ```bash
//! DTRACE_ENDPOINT=http://localhost:4317 \
//! DTRACE_SERVICE_NAME=tracing-durability-test \
//! RUST_LOG=info \
//! cargo run --example tracing_durability --features dtrace-otel
//! ```
//!
//! View traces at: http://localhost:16686
//!
//! ## What this tests
//!
//! 1. **Guard Pattern**: The `OtelGuard` is held in `main()` and dropped at exit,
//!    ensuring all buffered spans are flushed to the collector.
//!
//! 2. **Span Hierarchy**: Nested spans from server handlers create proper parent-child
//!    relationships visible in the trace UI.
//!
//! 3. **Async Instrumentation**: Both the server and client use `#[tracing::instrument]`
//!    to automatically create spans for async functions.
//!
//! 4. **Graceful Shutdown**: When the server shuts down, the guard ensures pending
//!    spans are exported before the process exits.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use rand::RngExt;
use tokio::sync::oneshot;
use tracing::{debug, error, info, info_span, instrument, warn, Instrument};
use warp::Filter;

/// Request counter for generating unique request IDs
static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Simulates some async work with variable latency
#[instrument(skip_all, fields(work_id = %work_id, duration_ms))]
async fn do_async_work(work_id: u64) -> String {
    let duration_ms = rand::rng().random_range(10..100);
    tracing::Span::current().record("duration_ms", duration_ms);

    debug!(duration_ms, "Starting async work");
    tokio::time::sleep(Duration::from_millis(duration_ms)).await;
    debug!("Async work completed");

    format!("Work {} completed in {}ms", work_id, duration_ms)
}

/// Simulates a database query
#[instrument(skip_all, fields(query_type = %query_type, rows_returned))]
async fn simulate_db_query(query_type: &str) -> Vec<String> {
    let latency = rand::rng().random_range(5..50);
    tokio::time::sleep(Duration::from_millis(latency)).await;

    let rows: Vec<String> = (0..rand::rng().random_range(1..10))
        .map(|i| format!("row_{}", i))
        .collect();

    tracing::Span::current().record("rows_returned", rows.len());
    debug!(rows = rows.len(), latency_ms = latency, "Query completed");

    rows
}

/// Handler for the /health endpoint
#[instrument(name = "health_check")]
async fn health_handler() -> Result<impl warp::Reply, warp::Rejection> {
    debug!("Health check requested");
    Ok(warp::reply::json(&serde_json::json!({
        "status": "healthy",
        "timestamp": chrono_lite_timestamp()
    })))
}

/// Handler for the /api/work endpoint
#[instrument(name = "api_work", fields(request_id))]
async fn work_handler() -> Result<impl warp::Reply, warp::Rejection> {
    let request_id = REQUEST_COUNTER.fetch_add(1, Ordering::SeqCst);
    tracing::Span::current().record("request_id", request_id);

    info!(request_id, "Processing work request");

    // Simulate nested async operations
    let work_result = do_async_work(request_id).await;

    // Simulate a database query
    let db_results = simulate_db_query("SELECT").await;

    debug!(
        request_id,
        work_result = %work_result,
        db_rows = db_results.len(),
        "Request processing complete"
    );

    Ok(warp::reply::json(&serde_json::json!({
        "request_id": request_id,
        "work_result": work_result,
        "db_rows": db_results.len()
    })))
}

/// Handler for the /api/error endpoint (demonstrates error tracing)
#[instrument(name = "api_error")]
async fn error_handler() -> Result<impl warp::Reply, warp::Rejection> {
    let should_fail = rand::rng().random_bool(0.5);

    if should_fail {
        warn!("Simulated error condition triggered");
        error!(error_type = "simulated", "Request failed");
        Ok(warp::reply::with_status(
            warp::reply::json(&serde_json::json!({
                "error": "Simulated failure",
                "recoverable": true
            })),
            warp::http::StatusCode::INTERNAL_SERVER_ERROR,
        ))
    } else {
        info!("Request succeeded (no simulated error)");
        Ok(warp::reply::with_status(
            warp::reply::json(&serde_json::json!({
                "status": "ok"
            })),
            warp::http::StatusCode::OK,
        ))
    }
}

/// Simple timestamp without external dependency
fn chrono_lite_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// Starts the HTTP server
async fn run_server(addr: SocketAddr, shutdown_rx: oneshot::Receiver<()>) {
    let health = warp::path("health")
        .and(warp::get())
        .and_then(health_handler)
        .boxed();

    let work = warp::path("api")
        .and(warp::path("work"))
        .and(warp::path::end())
        .and(warp::post())
        .and_then(work_handler)
        .boxed();

    let error = warp::path("api")
        .and(warp::path("error"))
        .and(warp::path::end())
        .and(warp::get())
        .and_then(error_handler)
        .boxed();

    let routes = health.or(work).or(error).with(warp::log("tracing_durability"));

    info!(%addr, "Server starting");
    warp::serve(routes)
        .bind(addr)
        .await
        .graceful(async {
            shutdown_rx.await.ok();
            info!("Server shutdown signal received");
        })
        .run()
        .await;
    info!("Server stopped");
}

/// HTTP client that makes test requests
#[instrument(name = "test_client", skip(base_url))]
async fn run_client(base_url: &str, num_requests: usize) {
    let client = reqwest::Client::new();

    info!(num_requests, "Starting client test run");

    for i in 0..num_requests {
        let span = info_span!("client_request", request_num = i);

        async {
            // Health check
            match client.get(format!("{}/health", base_url)).send().await {
                Ok(resp) => debug!(status = %resp.status(), "Health check response"),
                Err(e) => error!(error = %e, "Health check failed"),
            }

            // Work request
            match client.post(format!("{}/api/work", base_url)).send().await {
                Ok(resp) => {
                    let status = resp.status();
                    if let Ok(body) = resp.json::<serde_json::Value>().await {
                        info!(status = %status, request_id = %body["request_id"], "Work completed");
                    }
                }
                Err(e) => error!(error = %e, "Work request failed"),
            }

            // Error endpoint (50% chance of error)
            match client.get(format!("{}/api/error", base_url)).send().await {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        debug!(status = %status, "Error endpoint returned success");
                    } else {
                        warn!(status = %status, "Error endpoint returned error (expected)");
                    }
                }
                Err(e) => error!(error = %e, "Error endpoint request failed"),
            }

            // Small delay between request batches
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        .instrument(span)
        .await;
    }

    info!("Client test run complete");
}

/// Initialize tracing with the guard pattern
#[cfg(feature = "dtrace-otel")]
fn init_tracing() -> Option<motlie_core::telemetry::OtelGuard> {
    if let Ok(endpoint) = std::env::var("DTRACE_ENDPOINT") {
        let service_name = std::env::var("DTRACE_SERVICE_NAME")
            .unwrap_or_else(|_| "tracing-durability-test".to_string());

        match motlie_core::telemetry::init_otel_subscriber_with_env_filter(&service_name, &endpoint)
        {
            Ok(guard) => {
                info!(
                    service_name,
                    endpoint, "OpenTelemetry tracing initialized with guard"
                );
                return Some(guard);
            }
            Err(e) => {
                eprintln!("Failed to initialize OpenTelemetry: {}. Using stderr.", e);
            }
        }
    }

    motlie_core::telemetry::init_dev_subscriber_with_env_filter();
    info!("Using stderr tracing (no OpenTelemetry)");
    None
}

#[cfg(not(feature = "dtrace-otel"))]
fn init_tracing() -> Option<()> {
    motlie_core::telemetry::init_dev_subscriber_with_env_filter();
    info!("Using stderr tracing (dtrace-otel feature not enabled)");
    None
}

#[tokio::main]
async fn main() {
    // IMPORTANT: Hold the guard for the entire application lifetime.
    // When this guard is dropped at the end of main(), it triggers
    // OtelGuard::drop() which calls TracerProvider::shutdown() to
    // flush all pending spans to the collector.
    let _telemetry_guard = init_tracing();

    let root_span = info_span!("tracing_durability_test");
    let _root_guard = root_span.enter();

    info!("=== Tracing Durability Test ===");
    info!("This test verifies that spans are properly flushed on shutdown.");

    // Find an available port
    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);

    let base_url = format!("http://{}", addr);
    info!(base_url = %base_url, "Server will bind to");

    // Create shutdown channel
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    // Spawn server
    let server_handle = tokio::spawn(
        async move {
            run_server(addr, shutdown_rx).await;
        }
        .instrument(info_span!("server")),
    );

    // Give the server a moment to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Run client tests
    let num_requests = 5;
    run_client(&base_url, num_requests).await;

    // Shutdown server
    info!("Sending shutdown signal to server");
    let _ = shutdown_tx.send(());

    // Wait for server to stop
    if let Err(e) = server_handle.await {
        error!(error = %e, "Server task failed");
    }

    info!("=== Test Complete ===");
    info!("If using OpenTelemetry, check your collector UI for traces.");
    info!("The guard will now be dropped, flushing all pending spans...");

    // When main() exits, _telemetry_guard is dropped here.
    // This triggers the Drop implementation which calls provider.shutdown(),
    // ensuring all buffered spans are exported to the collector.
}
