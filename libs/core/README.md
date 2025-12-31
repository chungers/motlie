# motlie-core

Core utilities and infrastructure for the Motlie workspace.

## Modules

| Module | Description |
|--------|-------------|
| `distance` | SIMD-optimized vector distance computation (Euclidean, Cosine, Dot) |
| `telemetry` | Tracing subscriber initialization for development and production |

## Distance

The `distance` module provides hardware-accelerated distance functions with automatic platform detection.

```rust
use motlie_core::distance::{euclidean_squared, cosine, dot, DISTANCE};

let a = vec![1.0, 2.0, 3.0, 4.0];
let b = vec![5.0, 6.0, 7.0, 8.0];

let dist = euclidean_squared(&a, &b);
let cos_dist = cosine(&a, &b);
let dot_prod = dot(&a, &b);

println!("SIMD level: {}", DISTANCE.level());  // "NEON", "AVX2+FMA", etc.
```

**Supported platforms:**
- x86_64: AVX-512, AVX2+FMA, runtime detection
- ARM64: NEON (Apple Silicon, AWS Graviton)
- Fallback: Auto-vectorized scalar

**Performance Benchmarks (ARM64 NEON):**

| Distance | Dimension | Baseline (ns) | SIMD (ns) | Speedup |
|----------|-----------|---------------|-----------|---------|
| Euclidean | 128 | 39.4 | 28.2 | **1.39x** |
| Euclidean | 256 | 90.2 | 58.2 | **1.55x** |
| Euclidean | 512 | 220.9 | 119.0 | **1.86x** |
| Euclidean | 1024 | 485.8 | 237.4 | **2.05x** |
| Cosine | 128 | 96.8 | 28.4 | **3.41x** |
| Cosine | 256 | 252.4 | 58.2 | **4.34x** |
| Cosine | 512 | 655.1 | 117.0 | **5.60x** |
| Cosine | 1024 | 1428.8 | 247.3 | **5.78x** |

**Expected Performance (x86_64):**

| Implementation | 128-dim (ns) | 1024-dim (ns) | Speedup |
|----------------|--------------|---------------|---------|
| Baseline | ~50 | ~400 | 1x |
| AVX2 + FMA | ~15 | ~80 | 3-5x |
| AVX-512 | ~10 | ~50 | 5-8x |

See [src/distance/README.md](src/distance/README.md) for full documentation.

## Telemetry

The `telemetry` module provides helper functions for initializing tracing subscribers.

### Development (stderr logging)

```rust
use motlie_core::telemetry;

fn main() {
    // Simple stderr logging at DEBUG level
    telemetry::init_dev_subscriber();

    // Or with RUST_LOG environment variable support
    telemetry::init_dev_subscriber_with_env_filter();
}
```

### Production (OpenTelemetry)

Enable the `dtrace-otel` feature:

```toml
[dependencies]
motlie-core = { path = "libs/core", features = ["dtrace-otel"] }
```

Then initialize with the guard pattern:

```rust
use motlie_core::telemetry;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // IMPORTANT: Keep the guard alive for the entire application lifetime.
    // When dropped, it flushes pending spans and shuts down the tracer provider.
    let _guard = telemetry::init_otel_subscriber_with_env_filter("my-service", "http://localhost:4317")?;

    // Application code...

    Ok(())
    // _guard is dropped here, ensuring all spans are flushed before exit
}
```

**Why the guard matters**: The `OtelGuard` holds a reference to the `TracerProvider`. When dropped, it calls `shutdown()` to flush any buffered spans to the collector. Without holding this guard, spans may be lost on application exit.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RUST_LOG` | Log level filter (e.g., `debug`, `info`, `warn`) | `debug` (dev) / `info` (otel) |
| `DTRACE_ENDPOINT` | OTLP collector endpoint URL | None (required for otel) |
| `DTRACE_SERVICE_NAME` | Service name for traces | Application-defined |

## Integration Guide

This section covers how to integrate telemetry in binaries (applications) and libraries.

### Key Principle: Separation of Concerns

| Component | Responsibility |
|-----------|----------------|
| **Libraries** | Emit tracing spans and events (instrument code) |
| **Binaries** | Initialize the subscriber (configure where traces go) |

Libraries should **never** initialize a tracing subscriber. Only the final binary (application) should do this, typically in `main()`.

### For Binaries (Applications)

#### Cargo.toml Setup

```toml
[package]
name = "my-service"

[features]
default = []
dtrace-otel = ["motlie-core/dtrace-otel"]

[dependencies]
motlie-core = { path = "libs/core" }
tracing = "0.1"
```

#### main.rs Pattern

```rust
use tracing::{info, debug, error};

fn main() {
    // Initialize tracing FIRST, before any other code.
    // The guard must be held for the lifetime of the application to ensure
    // proper shutdown and flushing of telemetry data when using OpenTelemetry.
    let _telemetry_guard = init_tracing();

    info!("Service starting");

    // Application code...

    // When main exits, _telemetry_guard is dropped, flushing pending spans.
}

/// Initialize tracing subscriber based on environment and features.
/// Returns an optional OtelGuard that must be kept alive for proper shutdown.
#[cfg(feature = "dtrace-otel")]
fn init_tracing() -> Option<motlie_core::telemetry::OtelGuard> {
    // Check if DTRACE endpoint is configured
    if let Ok(endpoint) = std::env::var("DTRACE_ENDPOINT") {
        let service_name = std::env::var("DTRACE_SERVICE_NAME")
            .unwrap_or_else(|_| env!("CARGO_PKG_NAME").to_string());

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

#[cfg(not(feature = "dtrace-otel"))]
fn init_tracing() -> Option<()> {
    motlie_core::telemetry::init_dev_subscriber_with_env_filter();
    None
}
```

#### Build Commands

```bash
# Development build (stderr logging)
cargo build --bin my-service

# Production build with OpenTelemetry support
cargo build --release --bin my-service --features dtrace-otel
```

#### Runtime Configuration

```bash
# Development (uses RUST_LOG for filtering)
RUST_LOG=debug ./my-service

# Production with OpenTelemetry
DTRACE_ENDPOINT=http://localhost:4317 \
DTRACE_SERVICE_NAME=my-service \
RUST_LOG=info \
./my-service
```

### For Libraries

Libraries should only **emit** traces, never initialize subscribers.

#### Cargo.toml Setup

```toml
[package]
name = "my-library"

[dependencies]
tracing = "0.1"

# Note: Do NOT depend on tracing-subscriber or motlie-core's telemetry
# The binary will provide the subscriber
```

#### Instrumenting Functions

Use `#[tracing::instrument]` for automatic span creation:

```rust
use tracing::{debug, info, warn, error, instrument};

/// Automatically creates a span with function name and arguments.
#[instrument(skip(password))]  // Skip sensitive fields
pub fn authenticate_user(username: &str, password: &str) -> Result<User, AuthError> {
    debug!("Looking up user");

    let user = find_user(username)?;

    if !verify_password(&user, password) {
        warn!(username = %username, "Authentication failed");
        return Err(AuthError::InvalidCredentials);
    }

    info!(user_id = %user.id, "User authenticated successfully");
    Ok(user)
}

/// Control span name and included fields explicitly.
#[instrument(
    name = "db.query",
    skip(self),
    fields(table = %table_name, row_count)
)]
pub async fn query_table(&self, table_name: &str) -> Result<Vec<Row>> {
    let rows = self.execute_query(table_name).await?;

    // Record field value after computation
    tracing::Span::current().record("row_count", rows.len());

    Ok(rows)
}
```

#### Manual Span Creation

For more control, create spans manually:

```rust
use tracing::{span, Level, info_span, debug_span};

pub fn process_batch(items: &[Item]) -> Result<()> {
    let span = info_span!(
        "process_batch",
        batch_size = items.len(),
        batch_id = %generate_batch_id()
    );
    let _guard = span.enter();

    for (i, item) in items.iter().enumerate() {
        let item_span = debug_span!("process_item", index = i, item_id = %item.id);
        let _item_guard = item_span.enter();

        process_single_item(item)?;
    }

    Ok(())
}
```

#### Async Functions

For async code, use `.instrument()` or `#[instrument]`:

```rust
use tracing::{info_span, Instrument};

pub async fn fetch_data(url: &str) -> Result<Data> {
    let span = info_span!("fetch_data", url = %url);

    async {
        let response = client.get(url).await?;
        let data = response.json().await?;
        Ok(data)
    }
    .instrument(span)
    .await
}

// Or more concisely with the attribute:
#[instrument]
pub async fn fetch_data(url: &str) -> Result<Data> {
    let response = client.get(url).await?;
    response.json().await
}
```

#### Logging Events

Use tracing macros instead of `println!` or `log`:

```rust
use tracing::{trace, debug, info, warn, error};

pub fn process_request(request: &Request) -> Result<Response> {
    trace!(headers = ?request.headers, "Received request");

    debug!(method = %request.method, path = %request.path, "Processing request");

    if request.is_large() {
        info!(size = request.body.len(), "Processing large request");
    }

    match validate_request(request) {
        Ok(_) => {}
        Err(e) => {
            warn!(error = %e, "Request validation warning");
        }
    }

    match handle_request(request) {
        Ok(response) => {
            info!(status = response.status, "Request completed");
            Ok(response)
        }
        Err(e) => {
            error!(error = %e, request_id = %request.id, "Request failed");
            Err(e)
        }
    }
}
```

### Tracing Best Practices

#### Field Formatting

| Syntax | Output | Use Case |
|--------|--------|----------|
| `field = %value` | Display format | Human-readable values |
| `field = ?value` | Debug format | Complex structures |
| `field = value` | Direct value | Numbers, bools |
| `field` | Variable named `field` | Shorthand |

```rust
let user_id = 42;
let user = User { name: "Alice".into() };

info!(
    user_id,              // Same as user_id = user_id
    user_id = %user_id,   // Display: "42"
    user = ?user,         // Debug: "User { name: \"Alice\" }"
    count = 5,            // Direct: 5
);
```

#### Span Levels

| Level | Use Case | Example |
|-------|----------|---------|
| `ERROR` | Error recovery spans | Retry loops, fallback paths |
| `WARN` | Potentially problematic operations | Deprecated API usage |
| `INFO` | High-level operations | Request handling, batch processing |
| `DEBUG` | Detailed operations | Individual item processing |
| `TRACE` | Very detailed internals | Loop iterations, state changes |

#### Skip Sensitive Data

```rust
#[instrument(skip(password, api_key, credentials))]
pub fn login(username: &str, password: &str, api_key: &str) -> Result<Token> {
    // password and api_key won't appear in traces
}

#[instrument(skip_all, fields(user_id = %user.id))]
pub fn process_user(user: &User, sensitive_data: &SensitiveData) -> Result<()> {
    // Skip all args, only include user_id
}
```

#### Error Handling

```rust
#[instrument(err)]  // Automatically record errors
pub fn might_fail() -> Result<(), MyError> {
    do_something()?;
    Ok(())
}

#[instrument(err(Display))]  // Use Display instead of Debug for errors
pub fn might_fail_display() -> Result<(), MyError> {
    do_something()?;
    Ok(())
}
```

### Example: Library + Binary Integration

```
my-workspace/
├── libs/
│   └── my-lib/
│       ├── Cargo.toml          # Only depends on `tracing`
│       └── src/lib.rs          # Uses #[instrument], info!, debug!, etc.
└── bins/
    └── my-service/
        ├── Cargo.toml          # Depends on my-lib + motlie-core
        └── src/main.rs         # Calls init_tracing() in main()
```

**libs/my-lib/src/lib.rs:**
```rust
use tracing::{instrument, info};

#[instrument]
pub fn do_work(input: &str) -> Result<String, Error> {
    info!("Processing input");
    // ... library logic
    Ok(result)
}
```

**bins/my-service/src/main.rs:**
```rust
use my_lib::do_work;

fn main() {
    motlie_core::telemetry::init_dev_subscriber_with_env_filter();

    // Now all tracing from my_lib will be captured
    let result = do_work("test").unwrap();
}
```

## Distributed Tracing: Context Propagation

### Overview

OpenTelemetry provides **context propagation** for correlating traces across services. This enables you to follow a request as it flows through multiple services in a distributed system.

### How It Works

OpenTelemetry uses the **W3C Trace Context** standard with two HTTP headers:

| Header | Purpose |
|--------|---------|
| `traceparent` | Contains trace-id, parent-span-id, and trace flags |
| `tracestate` | Vendor-specific trace data |

Example `traceparent` header:
```
traceparent: 00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01
             ││ ││││││││││││││││││││││││││││││││ ││││││││││││││││ ││
             │└─────────────┬─────────────────┘ └───────┬───────┘ └┬┘
          version       trace-id (32 hex)      parent-id (16 hex) flags
```

- **trace-id**: Unique identifier for the entire distributed trace (same across all services)
- **parent-id**: Span ID of the parent span (links child spans to parents)
- **flags**: Sampling decisions (01 = sampled)

### Current Implementation Scope

The telemetry module currently provides:

| Feature | Status |
|---------|--------|
| Span creation and export to OTLP | ✅ Implemented |
| Local span correlation (within a service) | ✅ Automatic via `tracing` |
| Cross-service context propagation | ❌ Requires additional setup |

**Important**: Initializing a tracing subscriber only configures **export** to an OTLP collector. It does not automatically propagate trace context between services. For distributed tracing across services, you must explicitly:

1. **Extract** context from incoming requests
2. **Inject** context into outgoing requests

### Adding Context Propagation

#### Step 1: Set Global Propagator

Add to your initialization code (after setting up the subscriber):

```rust
use opentelemetry::propagation::TextMapPropagator;
use opentelemetry_sdk::propagation::TraceContextPropagator;

// Set the global propagator for W3C Trace Context
opentelemetry::global::set_text_map_propagator(TraceContextPropagator::new());
```

#### Step 2: Inject Context into Outgoing Requests

When making HTTP requests to other services:

```rust
use opentelemetry::propagation::TextMapPropagator;
use tracing_opentelemetry::OpenTelemetrySpanExt;

fn inject_trace_context(headers: &mut http::HeaderMap) {
    let span = tracing::Span::current();
    let context = span.context();

    opentelemetry::global::get_text_map_propagator(|propagator| {
        propagator.inject_context(&context, &mut HeaderInjector(headers));
    });
}

// Simple HeaderInjector implementation
struct HeaderInjector<'a>(&'a mut http::HeaderMap);

impl<'a> opentelemetry::propagation::Injector for HeaderInjector<'a> {
    fn set(&mut self, key: &str, value: String) {
        if let Ok(header_name) = http::header::HeaderName::from_bytes(key.as_bytes()) {
            if let Ok(header_value) = http::header::HeaderValue::from_str(&value) {
                self.0.insert(header_name, header_value);
            }
        }
    }
}
```

#### Step 3: Extract Context from Incoming Requests

When receiving HTTP requests:

```rust
use opentelemetry::propagation::TextMapPropagator;
use tracing_opentelemetry::OpenTelemetrySpanExt;

fn extract_trace_context(headers: &http::HeaderMap) -> opentelemetry::Context {
    opentelemetry::global::get_text_map_propagator(|propagator| {
        propagator.extract(&HeaderExtractor(headers))
    })
}

// In your request handler
async fn handle_request(headers: http::HeaderMap) {
    let parent_context = extract_trace_context(&headers);

    let span = tracing::info_span!("handle_request");
    span.set_parent(parent_context);

    // Process request within this span...
}

// Simple HeaderExtractor implementation
struct HeaderExtractor<'a>(&'a http::HeaderMap);

impl<'a> opentelemetry::propagation::Extractor for HeaderExtractor<'a> {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(|v| v.to_str().ok())
    }

    fn keys(&self) -> Vec<&str> {
        self.0.keys().map(|k| k.as_str()).collect()
    }
}
```

### Framework Integrations

Most Rust web frameworks have middleware or layers that handle context propagation automatically:

| Framework | Integration | Crate |
|-----------|-------------|-------|
| axum | Tower layer with OpenTelemetry | `axum-tracing-opentelemetry` |
| tower | TraceLayer + propagation | `tower-http`, `opentelemetry` |
| warp | Custom filter | Manual implementation |
| tonic (gRPC) | Built-in support | `tonic` + `opentelemetry` |
| reqwest | Middleware | `reqwest-middleware`, `reqwest-tracing` |
| hyper | Custom service wrapper | Manual implementation |

### Example: Full Distributed Trace Flow

```
Service A (API Gateway)                 Service B (Backend)
─────────────────────                   ──────────────────

[Span: api_request]
 trace_id: abc123
 span_id: span_001
       │
       │ HTTP Request
       │ traceparent: 00-abc123-span_001-01
       ▼
                                        [Span: handle_request]
                                         trace_id: abc123      ← Same trace!
                                         span_id: span_002
                                         parent_id: span_001   ← Linked!
                                               │
                                               ▼
                                        [Span: database_query]
                                         trace_id: abc123
                                         span_id: span_003
                                         parent_id: span_002
```

In the OTLP collector (Jaeger, Tempo, etc.), all three spans appear as a single connected trace.

### Troubleshooting

| Symptom | Cause | Solution |
|---------|-------|----------|
| Traces not connected across services | Missing propagator setup | Call `set_text_map_propagator()` |
| `traceparent` header missing | Not injecting into outgoing requests | Add injection code to HTTP client |
| New trace ID for each service | Not extracting from incoming requests | Add extraction code and `set_parent()` |
| Spans appear but unlinked | Propagator type mismatch | Use same propagator type everywhere |

### Future Enhancements

Potential additions to the telemetry module:

1. **Propagator initialization helper** - Include `set_text_map_propagator()` in init functions
2. **HTTP client middleware** - Automatic context injection for reqwest/hyper
3. **Warp filter** - Ready-to-use filter for context extraction
4. **Tower layer** - Compatible with axum/tonic

## Features

| Feature | Description |
|---------|-------------|
| (default) | Basic telemetry with stderr logging |
| `dtrace-otel` | OpenTelemetry support with OTLP export |

## Dependencies

### Required

- `tracing` (0.1) - Structured logging and tracing
- `tracing-subscriber` (0.3) - Subscriber implementations with env-filter

### Optional (dtrace-otel feature)

- `tracing-opentelemetry` (0.28) - OpenTelemetry integration
- `opentelemetry` (0.27) - OpenTelemetry API
- `opentelemetry-otlp` (0.27) - OTLP exporter
- `opentelemetry_sdk` (0.27) - OpenTelemetry SDK with Tokio runtime

## License

See workspace LICENSE file.
