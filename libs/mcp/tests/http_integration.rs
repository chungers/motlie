//! Integration tests for HTTP transport
//!
//! These tests verify the HTTP transport works correctly by starting
//! a real server and making HTTP requests to it.

use motlie_db::{Storage, StorageConfig};
use motlie_mcp::{HttpConfig, LazyResource, MotlieMcpServer};
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;

/// Helper to create a test server with a temporary database
fn create_test_server(db_path: &std::path::Path) -> MotlieMcpServer {
    let db_path = db_path.to_path_buf();

    let lazy_db = Arc::new(LazyResource::new(Box::new(move || {
        // Use the unified Storage API
        let storage = Storage::readwrite(&db_path);
        let handles = storage.ready(StorageConfig::default())?;

        Ok(handles)
    })));

    MotlieMcpServer::new(lazy_db, Duration::from_secs(5))
}

/// Find an available port by binding to port 0
async fn find_available_port() -> u16 {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    listener.local_addr().unwrap().port()
}

/// Start a test HTTP server and return the base URL
async fn start_test_server(server: MotlieMcpServer, port: u16) -> tokio::task::JoinHandle<()> {
    let config = HttpConfig::new(format!("127.0.0.1:{}", port).parse().unwrap())
        .with_sse_keep_alive(Some(Duration::from_secs(5)))
        .with_mcp_path("/mcp");

    tokio::spawn(async move {
        let _ = motlie_mcp::serve_http(server, config).await;
    })
}

#[tokio::test]
async fn test_http_server_initialize() {
    let temp_dir = TempDir::new().unwrap();
    let server = create_test_server(temp_dir.path());
    let port = find_available_port().await;

    let _handle = start_test_server(server, port).await;

    // Give server time to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    let client = reqwest::Client::new();
    let response = client
        .post(format!("http://127.0.0.1:{}/mcp", port))
        .header("Content-Type", "application/json")
        .header("Accept", "application/json, text/event-stream")
        .body(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}"#)
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success(), "Expected success, got {}", response.status());

    let body = response.text().await.unwrap();
    // SSE format: "data: {...}\n\n"
    assert!(body.starts_with("data:"), "Expected SSE format, got: {}", body);
    assert!(body.contains("protocolVersion"), "Expected protocolVersion in response");
    assert!(body.contains("2024-11-05"), "Expected protocol version 2024-11-05");
}

#[tokio::test]
async fn test_http_server_missing_accept_header() {
    let temp_dir = TempDir::new().unwrap();
    let server = create_test_server(temp_dir.path());
    let port = find_available_port().await;

    let _handle = start_test_server(server, port).await;

    // Give server time to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    let client = reqwest::Client::new();
    let response = client
        .post(format!("http://127.0.0.1:{}/mcp", port))
        .header("Content-Type", "application/json")
        // Missing Accept header!
        .body(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}"#)
        .send()
        .await
        .unwrap();

    // Should return 406 Not Acceptable
    assert_eq!(response.status().as_u16(), 406, "Expected 406 Not Acceptable");
}

#[tokio::test]
async fn test_http_server_returns_sse_format() {
    // Test that responses use SSE format with "data:" prefix
    let temp_dir = TempDir::new().unwrap();
    let server = create_test_server(temp_dir.path());
    let port = find_available_port().await;

    let _handle = start_test_server(server, port).await;

    // Give server time to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    let client = reqwest::Client::new();

    let response = client
        .post(format!("http://127.0.0.1:{}/mcp", port))
        .header("Content-Type", "application/json")
        .header("Accept", "application/json, text/event-stream")
        .body(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}"#)
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success());

    // Check content-type is SSE
    let content_type = response.headers().get("content-type").unwrap().to_str().unwrap();
    assert!(content_type.contains("text/event-stream"), "Expected SSE content-type, got: {}", content_type);

    let body = response.text().await.unwrap();
    // SSE format has multiple data: lines. The first is empty (connection open),
    // the second contains the actual JSON response.
    // Find the data line with actual JSON content (starts with '{')
    let json_str = body
        .lines()
        .filter(|line| line.starts_with("data:"))
        .map(|line| line.trim_start_matches("data:").trim())
        .find(|content| content.starts_with('{'))
        .expect(&format!("Expected SSE data: line with JSON, got: {}", body));
    let json: serde_json::Value = serde_json::from_str(json_str).expect(&format!("Should be valid JSON, got: {}", json_str));

    // Verify JSON-RPC structure
    assert_eq!(json["jsonrpc"], "2.0");
    assert_eq!(json["id"], 1);
    assert!(json["result"].is_object(), "Expected result object");
    assert!(json["result"]["serverInfo"].is_object(), "Expected serverInfo in result");
}

#[tokio::test]
async fn test_http_config_custom_path() {
    let temp_dir = TempDir::new().unwrap();
    let server = create_test_server(temp_dir.path());
    let port = find_available_port().await;

    // Use custom path
    let config = HttpConfig::new(format!("127.0.0.1:{}", port).parse().unwrap())
        .with_mcp_path("/api/v1/mcp");

    let _handle = tokio::spawn(async move {
        let _ = motlie_mcp::serve_http(server, config).await;
    });

    // Give server time to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    let client = reqwest::Client::new();

    // Request to default /mcp should fail
    let response = client
        .post(format!("http://127.0.0.1:{}/mcp", port))
        .header("Content-Type", "application/json")
        .header("Accept", "application/json, text/event-stream")
        .body(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}"#)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status().as_u16(), 404, "Expected 404 for wrong path");

    // Request to custom path should succeed
    let response = client
        .post(format!("http://127.0.0.1:{}/api/v1/mcp", port))
        .header("Content-Type", "application/json")
        .header("Accept", "application/json, text/event-stream")
        .body(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}"#)
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success(), "Expected success on custom path");
}

#[tokio::test]
async fn test_http_server_concurrent_requests() {
    let temp_dir = TempDir::new().unwrap();
    let server = create_test_server(temp_dir.path());
    let port = find_available_port().await;

    let _handle = start_test_server(server, port).await;

    // Give server time to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    let client = reqwest::Client::new();

    // Send multiple concurrent requests
    let mut handles = Vec::new();
    for i in 0..5 {
        let client = client.clone();
        let port = port;
        handles.push(tokio::spawn(async move {
            let response = client
                .post(format!("http://127.0.0.1:{}/mcp", port))
                .header("Content-Type", "application/json")
                .header("Accept", "application/json, text/event-stream")
                .body(format!(
                    r#"{{"jsonrpc":"2.0","id":{},"method":"initialize","params":{{"protocolVersion":"2024-11-05","capabilities":{{}},"clientInfo":{{"name":"test{}","version":"1.0"}}}}}}"#,
                    i, i
                ))
                .send()
                .await
                .unwrap();
            response.status().is_success()
        }));
    }

    // All requests should succeed
    for handle in handles {
        assert!(handle.await.unwrap(), "Concurrent request failed");
    }
}
