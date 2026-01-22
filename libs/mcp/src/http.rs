//! HTTP transport for MCP servers using Streamable HTTP protocol.
//!
//! This module provides HTTP transport support using rmcp's `StreamableHttpService`,
//! which implements the MCP Streamable HTTP protocol with Server-Sent Events (SSE)
//! for long-lived connections.
//!
//! # Example
//!
//! ```ignore
//! use motlie_mcp::{serve_http, HttpConfig};
//! use motlie_mcp::db::MotlieMcpServer;
//!
//! let server = MotlieMcpServer::new(lazy_db, Duration::from_secs(5));
//! let config = HttpConfig::new("127.0.0.1:8080".parse().unwrap());
//!
//! serve_http(server, config).await?;
//! ```

use anyhow::Result;
use rmcp::handler::server::ServerHandler;
use rmcp::transport::streamable_http_server::{
    session::local::LocalSessionManager, tower::StreamableHttpServerConfig,
    tower::StreamableHttpService,
};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

/// Configuration for HTTP transport.
#[derive(Debug, Clone)]
pub struct HttpConfig {
    /// Address to bind to
    pub addr: SocketAddr,
    /// SSE keep-alive interval (None to disable)
    pub sse_keep_alive: Option<Duration>,
    /// MCP endpoint path (e.g., "/mcp")
    pub mcp_path: String,
    /// Enable stateful session mode (maintains session across requests)
    pub stateful_mode: bool,
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            addr: "127.0.0.1:8080".parse().unwrap(),
            sse_keep_alive: Some(Duration::from_secs(30)),
            mcp_path: "/mcp".to_string(),
            stateful_mode: true,
        }
    }
}

impl HttpConfig {
    /// Create a new HTTP config with the given address.
    pub fn new(addr: SocketAddr) -> Self {
        Self {
            addr,
            ..Default::default()
        }
    }

    /// Set the SSE keep-alive interval.
    pub fn with_sse_keep_alive(mut self, interval: Option<Duration>) -> Self {
        self.sse_keep_alive = interval;
        self
    }

    /// Set the MCP endpoint path.
    pub fn with_mcp_path(mut self, path: impl Into<String>) -> Self {
        self.mcp_path = path.into();
        self
    }

    /// Enable or disable stateful session mode.
    pub fn with_stateful_mode(mut self, enabled: bool) -> Self {
        self.stateful_mode = enabled;
        self
    }
}

/// Serve an MCP server over HTTP with Streamable HTTP transport.
///
/// This function starts an HTTP server using axum that serves the MCP protocol
/// over the Streamable HTTP transport. It supports:
/// - Session management via `LocalSessionManager`
/// - SSE keep-alive for long-lived connections
/// - Graceful shutdown on Ctrl+C
///
/// # Type Parameters
///
/// * `S` - Any type implementing `ServerHandler + Clone + Send + Sync + 'static`
///
/// # Example
///
/// ```ignore
/// use motlie_mcp::{serve_http, HttpConfig};
/// use motlie_mcp::db::MotlieMcpServer;
///
/// let server = MotlieMcpServer::new(lazy_db, Duration::from_secs(5));
/// serve_http(server, HttpConfig::default()).await?;
/// ```
///
/// # Combined Server Example
///
/// ```ignore
/// use motlie_mcp::{serve_http, HttpConfig};
///
/// // Any type implementing ServerHandler works
/// let combined = MyCombinedServer::new(...);
/// serve_http(combined, HttpConfig::default()).await?;
/// ```
pub async fn serve_http<S>(server: S, config: HttpConfig) -> Result<()>
where
    S: ServerHandler + Clone + Send + Sync + 'static,
{
    let server = Arc::new(server);

    let http_config = StreamableHttpServerConfig {
        sse_keep_alive: config.sse_keep_alive,
        stateful_mode: config.stateful_mode,
    };

    // Create the StreamableHttpService with a factory that clones our server
    let service = StreamableHttpService::new(
        {
            let server = server.clone();
            move || Ok((*server).clone())
        },
        LocalSessionManager::default().into(),
        http_config,
    );

    // Build the axum router with the MCP service nested at the configured path
    let router = axum::Router::new().nest_service(&config.mcp_path, service);

    // Bind to the configured address
    let listener = tokio::net::TcpListener::bind(config.addr).await?;

    tracing::info!(
        "HTTP transport listening on http://{}{}",
        config.addr,
        config.mcp_path
    );
    tracing::info!("SSE keep-alive: {:?}", config.sse_keep_alive);
    tracing::info!("Stateful mode: {}", config.stateful_mode);

    // Serve with graceful shutdown on Ctrl+C
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            // kill -2 <PID> to gracefully shutdown the server.
            tokio::signal::ctrl_c().await.ok();
            tracing::info!("Graceful shutdown initiated");
        })
        .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_http_config_default() {
        let config = HttpConfig::default();
        assert_eq!(config.addr, "127.0.0.1:8080".parse::<SocketAddr>().unwrap());
        assert_eq!(config.sse_keep_alive, Some(Duration::from_secs(30)));
        assert_eq!(config.mcp_path, "/mcp");
        assert!(config.stateful_mode);
    }

    #[test]
    fn test_http_config_new() {
        let addr: SocketAddr = "192.168.1.1:3000".parse().unwrap();
        let config = HttpConfig::new(addr);

        // Should use provided address but defaults for other fields
        assert_eq!(config.addr, addr);
        assert_eq!(config.sse_keep_alive, Some(Duration::from_secs(30)));
        assert_eq!(config.mcp_path, "/mcp");
        assert!(config.stateful_mode);
    }

    #[test]
    fn test_http_config_builder() {
        let config = HttpConfig::new("0.0.0.0:9000".parse().unwrap())
            .with_sse_keep_alive(Some(Duration::from_secs(60)))
            .with_mcp_path("/api/mcp")
            .with_stateful_mode(false);

        assert_eq!(config.addr, "0.0.0.0:9000".parse::<SocketAddr>().unwrap());
        assert_eq!(config.sse_keep_alive, Some(Duration::from_secs(60)));
        assert_eq!(config.mcp_path, "/api/mcp");
        assert!(!config.stateful_mode);
    }

    #[test]
    fn test_http_config_disable_sse_keep_alive() {
        let config = HttpConfig::default().with_sse_keep_alive(None);
        assert!(config.sse_keep_alive.is_none());
    }

    #[test]
    fn test_http_config_custom_mcp_path() {
        let config = HttpConfig::default().with_mcp_path("/v1/mcp");
        assert_eq!(config.mcp_path, "/v1/mcp");

        // Test with String
        let config2 = HttpConfig::default().with_mcp_path(String::from("/v2/mcp"));
        assert_eq!(config2.mcp_path, "/v2/mcp");
    }

    #[test]
    fn test_http_config_clone() {
        let config = HttpConfig::new("127.0.0.1:9090".parse().unwrap())
            .with_mcp_path("/custom")
            .with_stateful_mode(false);

        let cloned = config.clone();
        assert_eq!(config.addr, cloned.addr);
        assert_eq!(config.mcp_path, cloned.mcp_path);
        assert_eq!(config.stateful_mode, cloned.stateful_mode);
    }

    #[test]
    fn test_http_config_debug() {
        let config = HttpConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("HttpConfig"));
        assert!(debug_str.contains("127.0.0.1:8080"));
        assert!(debug_str.contains("/mcp"));
    }

    #[test]
    fn test_http_config_various_addresses() {
        // IPv4 localhost
        let config = HttpConfig::new("127.0.0.1:8080".parse().unwrap());
        assert_eq!(config.addr.port(), 8080);

        // IPv4 all interfaces
        let config = HttpConfig::new("0.0.0.0:3000".parse().unwrap());
        assert_eq!(config.addr.port(), 3000);

        // IPv6 localhost
        let config = HttpConfig::new("[::1]:8080".parse().unwrap());
        assert_eq!(config.addr.port(), 8080);
        assert!(config.addr.is_ipv6());
    }
}
