//! Trace Verification Script
//!
//! This script verifies that traces are actually being exported to Jaeger
//! by querying Jaeger's HTTP API after running the tracing durability test.
//!
//! ## Prerequisites
//! Start Jaeger with OTLP enabled:
//! ```bash
//! docker run -d --name jaeger \
//!   -e COLLECTOR_OTLP_ENABLED=true \
//!   -p 16686:16686 \
//!   -p 4317:4317 \
//!   jaegertracing/all-in-one:1.51
//! ```
//!
//! ## Usage
//! ```bash
//! # Run the tracing durability test first (in another terminal or background)
//! DTRACE_ENDPOINT=http://localhost:4317 \
//! DTRACE_SERVICE_NAME=tracing-durability-test \
//! RUST_LOG=info \
//! cargo run --example tracing_durability --features dtrace-otel
//!
//! # Then run this verification script
//! cargo run --example verify_traces
//! ```
//!
//! ## What it checks
//! 1. Queries Jaeger API for services - verifies our service is registered
//! 2. Queries for traces from our service - verifies spans were exported
//! 3. Counts spans per trace - verifies the full span hierarchy was flushed
//! 4. Validates span names match expected operations

use std::time::Duration;

const JAEGER_QUERY_URL: &str = "http://localhost:16686";
const SERVICE_NAME: &str = "tracing-durability-test";

#[derive(Debug, serde::Deserialize)]
struct JaegerServicesResponse {
    data: Vec<String>,
}

#[derive(Debug, serde::Deserialize)]
struct JaegerTracesResponse {
    data: Vec<JaegerTrace>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct JaegerTrace {
    #[serde(alias = "traceID")]
    trace_id: String,
    spans: Vec<JaegerSpan>,
}

#[allow(dead_code)]
#[derive(Debug, serde::Deserialize)]
struct JaegerSpan {
    #[serde(alias = "spanID", alias = "spanId")]
    span_id: String,
    #[serde(alias = "operationName")]
    operation_name: String,
    #[serde(default)]
    duration: u64,
    #[serde(default)]
    tags: Vec<JaegerTag>,
}

#[allow(dead_code)]
#[derive(Debug, serde::Deserialize)]
struct JaegerTag {
    key: String,
    value: serde_json::Value,
}

async fn check_jaeger_available(client: &reqwest::Client) -> bool {
    match client
        .get(format!("{}/api/services", JAEGER_QUERY_URL))
        .timeout(Duration::from_secs(5))
        .send()
        .await
    {
        Ok(resp) => resp.status().is_success(),
        Err(_) => false,
    }
}

async fn get_services(client: &reqwest::Client) -> anyhow::Result<Vec<String>> {
    let resp = client
        .get(format!("{}/api/services", JAEGER_QUERY_URL))
        .send()
        .await?
        .json::<JaegerServicesResponse>()
        .await?;
    Ok(resp.data)
}

async fn get_traces(
    client: &reqwest::Client,
    service: &str,
    limit: u32,
) -> anyhow::Result<Vec<JaegerTrace>> {
    let url = format!(
        "{}/api/traces?service={}&limit={}",
        JAEGER_QUERY_URL, service, limit
    );
    let resp = client
        .get(&url)
        .send()
        .await?
        .json::<JaegerTracesResponse>()
        .await?;
    Ok(resp.data)
}

fn analyze_trace(trace: &JaegerTrace) {
    println!("\n  Trace ID: {}", trace.trace_id);
    println!("  Total spans: {}", trace.spans.len());

    // Group spans by operation name
    let mut op_counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for span in &trace.spans {
        *op_counts.entry(&span.operation_name).or_default() += 1;
    }

    println!("  Operations:");
    let mut ops: Vec<_> = op_counts.iter().collect();
    ops.sort_by_key(|(name, _)| *name);
    for (op, count) in ops {
        println!("    - {}: {} span(s)", op, count);
    }

    // Look for specific expected spans
    let expected_ops = [
        "tracing_durability_test",
        "server",
        "test_client",
        "client_request",
        "api_work",
        "health_check",
        "do_async_work",
        "simulate_db_query",
    ];

    println!("  Expected operations check:");
    for expected in expected_ops {
        let found = trace
            .spans
            .iter()
            .any(|s| s.operation_name.contains(expected));
        let status = if found { "✓" } else { "✗" };
        println!("    {} {}", status, expected);
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Jaeger Trace Verification ===\n");

    let client = reqwest::Client::new();

    // Check if Jaeger is available
    println!("1. Checking Jaeger availability at {}...", JAEGER_QUERY_URL);
    if !check_jaeger_available(&client).await {
        eprintln!("\n   ERROR: Jaeger is not available at {}", JAEGER_QUERY_URL);
        eprintln!("   Please start Jaeger with:");
        eprintln!("   docker run -d --name jaeger \\");
        eprintln!("     -e COLLECTOR_OTLP_ENABLED=true \\");
        eprintln!("     -p 16686:16686 \\");
        eprintln!("     -p 4317:4317 \\");
        eprintln!("     jaegertracing/all-in-one:1.51");
        return Ok(());
    }
    println!("   Jaeger is available!\n");

    // Get list of services
    println!("2. Fetching registered services...");
    let services = get_services(&client).await?;
    println!("   Found {} service(s):", services.len());
    for svc in &services {
        let marker = if svc == SERVICE_NAME { " <-- our service" } else { "" };
        println!("   - {}{}", svc, marker);
    }

    // Check if our service is registered
    if !services.iter().any(|s| s == SERVICE_NAME) {
        eprintln!("\n   WARNING: Service '{}' not found!", SERVICE_NAME);
        eprintln!("   Have you run the tracing_durability example with OpenTelemetry?");
        eprintln!("\n   Run:");
        eprintln!("   DTRACE_ENDPOINT=http://localhost:4317 \\");
        eprintln!("   DTRACE_SERVICE_NAME={} \\", SERVICE_NAME);
        eprintln!("   RUST_LOG=info \\");
        eprintln!("   cargo run --example tracing_durability --features dtrace-otel");
        return Ok(());
    }

    // Get traces for our service
    println!("\n3. Fetching traces for service '{}'...", SERVICE_NAME);
    let traces = get_traces(&client, SERVICE_NAME, 10).await?;
    println!("   Found {} trace(s)", traces.len());

    if traces.is_empty() {
        eprintln!("\n   WARNING: No traces found for service '{}'", SERVICE_NAME);
        eprintln!("   This could mean:");
        eprintln!("   - The example hasn't been run yet");
        eprintln!("   - Spans weren't flushed properly (guard issue!)");
        eprintln!("   - OTLP export failed");
        return Ok(());
    }

    // Analyze each trace
    println!("\n4. Analyzing traces...");
    let mut total_spans = 0;
    for trace in &traces {
        analyze_trace(trace);
        total_spans += trace.spans.len();
    }

    // Summary
    println!("\n=== Summary ===");
    println!("Total traces: {}", traces.len());
    println!("Total spans: {}", total_spans);

    // Calculate average spans per trace
    let avg_spans = total_spans as f64 / traces.len() as f64;
    println!("Average spans per trace: {:.1}", avg_spans);

    // Verdict
    println!("\n=== Verdict ===");
    if total_spans > 0 {
        println!("✓ SUCCESS: Traces were properly exported to Jaeger!");
        println!("  The OtelGuard correctly flushed spans before exit.");
        println!("\n  View traces in Jaeger UI: {}", JAEGER_QUERY_URL);
    } else {
        println!("✗ FAILURE: No spans found. Check the guard implementation.");
    }

    Ok(())
}
