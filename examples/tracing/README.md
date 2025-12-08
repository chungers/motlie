# Tracing Durability Examples

This directory contains examples that test and verify the durability of OpenTelemetry tracing exports, specifically the proper lifecycle management of the `TracerProvider` using the guard pattern.

## The Problem

When initializing OpenTelemetry tracing in Rust, a common mistake is to let the `TracerProvider` be dropped immediately after initialization:

```rust
// WRONG - TracerProvider dropped at end of function!
pub fn init_otel_subscriber(service_name: &str, endpoint: &str) -> Result<(), Error> {
    let provider = TracerProvider::builder()
        .with_batch_exporter(exporter, runtime::Tokio)
        .build();

    let tracer = provider.tracer(service_name.to_string());
    // ... setup subscriber ...

    Ok(())  // provider is dropped here!
}
```

This causes:
- **Lost spans**: Buffered spans in the batch exporter may never be sent
- **No graceful shutdown**: No way to flush pending spans before exit
- **Silent failures**: The application appears to work but traces are missing

## The Solution: Guard Pattern

The fix is to return a guard struct that holds the `TracerProvider` and implements `Drop` to call `shutdown()`:

```rust
pub struct OtelGuard {
    provider: TracerProvider,
}

impl Drop for OtelGuard {
    fn drop(&mut self) {
        if let Err(e) = self.provider.shutdown() {
            eprintln!("Error shutting down tracer provider: {:?}", e);
        }
    }
}

pub fn init_otel_subscriber(...) -> Result<OtelGuard, Error> {
    // ... setup ...
    Ok(OtelGuard { provider })
}
```

Then in `main()`:

```rust
fn main() {
    let _guard = init_otel_subscriber(...).unwrap();
    // Application runs...
    // When main exits, _guard is dropped, flushing all spans
}
```

## Examples

### tracing_durability.rs

A complete HTTP server + client that generates spans to test the guard pattern:

- **Server endpoints**: `/health`, `/api/work`, `/api/error`
- **Instrumentation**: Uses `#[tracing::instrument]` for automatic span creation
- **Nested spans**: Demonstrates span hierarchy with `do_async_work` and `simulate_db_query`
- **Graceful shutdown**: Server shutdown triggers guard drop

### verify_traces.rs

A verification script that queries Jaeger's HTTP API to confirm traces were exported:

- Checks Jaeger availability
- Verifies service registration
- Counts traces and spans
- Reports success/failure verdict

## Running the Examples

### Prerequisites

Start an OTLP collector. The examples below use Jaeger.

#### Using Docker

```bash
docker run -d --name jaeger \
  -e COLLECTOR_OTLP_ENABLED=true \
  -p 16686:16686 \
  -p 4317:4317 \
  jaegertracing/all-in-one:1.51
```

#### Using Podman

```bash
# Start Podman machine if needed
podman machine start

# Run Jaeger
podman run -d --name jaeger \
  -e COLLECTOR_OTLP_ENABLED=true \
  -p 16686:16686 \
  -p 4317:4317 \
  jaegertracing/all-in-one:1.51
```

### Run the Tracing Test

```bash
DTRACE_ENDPOINT=http://localhost:4317 \
DTRACE_SERVICE_NAME=tracing-durability-test \
RUST_LOG=info \
cargo run --example tracing_durability --features dtrace-otel
```

### Verify Traces Were Exported

```bash
cargo run --example verify_traces
```

### View Traces in Jaeger UI

Open http://localhost:16686 in your browser.

### Cleanup

```bash
# Docker
docker stop jaeger && docker rm jaeger

# Podman
podman stop jaeger && podman rm jaeger
```

## Verified Output

### Tracing Durability Test Output

```
2025-12-08T16:57:55.080Z INFO  tracing_durability: OpenTelemetry tracing initialized with guard service_name="tracing-durability-test" endpoint="http://localhost:4317"
2025-12-08T16:57:55.080Z INFO  tracing_durability_test: === Tracing Durability Test ===
2025-12-08T16:57:55.080Z INFO  tracing_durability_test: This test verifies that spans are properly flushed on shutdown.
2025-12-08T16:57:55.081Z INFO  tracing_durability_test: Server will bind to base_url=http://127.0.0.1:54193
2025-12-08T16:57:55.081Z INFO  tracing_durability_test:server: Server starting addr=127.0.0.1:54193
2025-12-08T16:57:55.213Z INFO  tracing_durability_test:test_client{num_requests=5}: Starting client test run num_requests=5
2025-12-08T16:57:55.215Z INFO  request{method=GET path=/health}: processing request
2025-12-08T16:57:55.215Z INFO  request{method=GET path=/health}: finished processing with success status=200
2025-12-08T16:57:55.216Z INFO  request{method=POST path=/api/work}:api_work{request_id=0}: Processing work request request_id=0
2025-12-08T16:57:55.347Z INFO  request{method=POST path=/api/work}: finished processing with success status=200
2025-12-08T16:57:55.347Z INFO  tracing_durability_test:test_client{num_requests=5}:client_request{request_num=0}: Work completed status=200 OK request_id=0
...
2025-12-08T16:57:55.632Z INFO  tracing_durability_test:test_client{num_requests=5}: Client test run complete
2025-12-08T16:57:55.632Z INFO  tracing_durability_test: Sending shutdown signal to server
2025-12-08T16:57:55.632Z INFO  tracing_durability_test:server: Server shutdown signal received
2025-12-08T16:57:55.633Z INFO  tracing_durability_test:server: Server stopped
2025-12-08T16:57:55.633Z INFO  tracing_durability_test: === Test Complete ===
2025-12-08T16:57:55.633Z INFO  tracing_durability_test: If using OpenTelemetry, check your collector UI for traces.
2025-12-08T16:57:55.633Z INFO  tracing_durability_test: The guard will now be dropped, flushing all pending spans...
```

### Verification Script Output

```
=== Jaeger Trace Verification ===

1. Checking Jaeger availability at http://localhost:16686...
   Jaeger is available!

2. Fetching registered services...
   Found 3 service(s):
   - jaeger-all-in-one
   - unknown_service
   - tracing-durability-test <-- our service

3. Fetching traces for service 'tracing-durability-test'...
   Found 10 trace(s)

4. Analyzing traces...

  Trace ID: 5a8b9c...
  Total spans: 4
  Operations:
    - api_work: 1 span(s)
    - do_async_work: 1 span(s)
    - request: 1 span(s)
    - simulate_db_query: 1 span(s)
  Expected operations check:
    ✓ api_work
    ✓ health_check
    ✓ do_async_work
    ✓ simulate_db_query

  ...

=== Summary ===
Total traces: 10
Total spans: 26
Average spans per trace: 2.6

=== Verdict ===
✓ SUCCESS: Traces were properly exported to Jaeger!
  The OtelGuard correctly flushed spans before exit.

  View traces in Jaeger UI: http://localhost:16686
```

## Jaeger UI Screenshots

After running the examples, you can view the traces in Jaeger's web UI:

### Service Selection
Navigate to http://localhost:16686 and select `tracing-durability-test` from the Service dropdown.

### Trace List
You'll see traces for the HTTP requests made during the test:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Service: tracing-durability-test                                        │
├─────────────────────────────────────────────────────────────────────────┤
│ ● api_work                                    4 Spans    131.45ms       │
│ ● api_work                                    4 Spans     87.32ms       │
│ ● api_work                                    4 Spans     60.12ms       │
│ ● api_work                                    4 Spans    128.67ms       │
│ ● api_work                                    4 Spans     69.83ms       │
│ ● health_check                                1 Span       0.12ms       │
│ ● api_error                                   2 Spans      0.15ms       │
│ ...                                                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

### Trace Detail View
Clicking on a trace shows the span hierarchy:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ api_work {request_id=0}                                      131.45ms   │
│ ├── do_async_work {work_id=0, duration_ms=87}                 87.32ms   │
│ └── simulate_db_query {query_type=SELECT, rows_returned=5}    34.21ms   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Takeaways

1. **Always return a guard** from OpenTelemetry init functions
2. **Hold the guard in main()** for the application's lifetime
3. **The guard's Drop impl** calls `provider.shutdown()` to flush spans
4. **Without the guard**, spans may be silently lost on exit

## References

- [Rust OpenTelemetry Tracing Guide](https://www.joshkasuboski.com/posts/rust-opentelemetry-tracing/)
- [tracing-opentelemetry crate](https://github.com/tokio-rs/tracing-opentelemetry)
- [OpenTelemetry Rust SDK](https://github.com/open-telemetry/opentelemetry-rust)
