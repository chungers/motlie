//! Motlie Database MCP Server
//!
//! A dedicated MCP server exposing only Motlie graph database tools.
//!
//! # Usage
//!
//! ```bash
//! # Stdio transport (for Claude Desktop, Claude Code, Cursor)
//! cargo run --example motlie_db -- --db-path /path/to/db --transport stdio
//!
//! # HTTP transport (for remote access)
//! cargo run --example motlie_db -- --db-path /path/to/db --transport http --port 8080
//! ```
//!
//! # Graceful Shutdown
//!
//! This server properly handles Ctrl+C to gracefully shut down the database,
//! ensuring all pending writes are flushed and no data corruption occurs.

use anyhow::Result;
use clap::{Parser, ValueEnum};
use motlie_db::{Storage, StorageConfig};
use motlie_mcp::db::MotlieMcpServer;
use motlie_mcp::{stdio, ManagedResource, ServiceExt};

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

/// Transport protocol for MCP server
#[derive(Debug, Clone, ValueEnum)]
enum Transport {
    /// Standard input/output (for local process communication)
    Stdio,
    /// HTTP with Streamable HTTP protocol (for remote access)
    Http,
}

impl std::fmt::Display for Transport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Transport::Stdio => write!(f, "stdio"),
            Transport::Http => write!(f, "http"),
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "motlie-db-mcp")]
#[command(about = "Motlie graph database MCP server", long_about = None)]
struct Args {
    /// Path to the RocksDB database directory
    #[arg(short, long)]
    db_path: String,

    /// Transport protocol (stdio or http)
    ///
    /// - stdio: Standard input/output for local process communication (default)
    /// - http: HTTP with Streamable HTTP protocol for remote access
    #[arg(short, long, value_enum, default_value = "stdio")]
    transport: Transport,

    /// Port for HTTP transport (only used when transport=http)
    #[arg(short, long, default_value = "8080")]
    port: u16,

    /// Host address for HTTP transport (only used when transport=http)
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// MCP endpoint path for HTTP transport (only used when transport=http)
    #[arg(long, default_value = "/mcp")]
    mcp_path: String,

    /// Query timeout in seconds
    #[arg(long, default_value = "30")]
    query_timeout: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging with tracing
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,rmcp=debug".into()),
        )
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(std::io::stderr)
                .with_ansi(false),
        )
        .init();

    // Parse command-line arguments
    let args = Args::parse();

    // Capture configuration for lazy initialization
    let db_path = PathBuf::from(&args.db_path);

    // Create managed database resource with proper lifecycle
    // The actual database initialization happens on first tool use,
    // allowing the MCP handshake to complete quickly.
    let managed_db = ManagedResource::new(Box::new(move || {
        tracing::info!("Initializing database at: {:?}", db_path);

        let storage = Storage::readwrite(&db_path);
        let handles = storage.ready(StorageConfig::default())?;

        tracing::info!("Database initialized successfully");
        Ok(handles)
    }));

    // Build the MCP server with lazy database
    let query_timeout = Duration::from_secs(args.query_timeout);
    let server = MotlieMcpServer::new(managed_db.lazy(), query_timeout);

    tracing::info!("Starting Motlie Database MCP server...");
    tracing::info!("Server: motlie-db-mcp");
    tracing::info!("Version: {}", env!("CARGO_PKG_VERSION"));
    tracing::info!("Database: {} (lazy initialization)", args.db_path);
    tracing::info!("Tools: 15 (7 mutations + 8 queries)");
    tracing::info!("Transport: {}", args.transport);

    // Start the server with the selected transport
    // Use tokio::select! to handle graceful shutdown on Ctrl+C
    let shutdown_result = match args.transport {
        Transport::Stdio => {
            tracing::info!("Listening on stdio (standard input/output)");

            let service = server.serve(stdio()).await.inspect_err(|e| {
                tracing::error!("serving error: {:?}", e);
            })?;

            // Wait for service to complete or Ctrl+C
            tokio::select! {
                result = service.waiting() => {
                    result?;
                    tracing::info!("Service completed normally");
                }
                _ = tokio::signal::ctrl_c() => {
                    tracing::info!("Received Ctrl+C, initiating graceful shutdown...");
                }
            }
            Ok(())
        }
        Transport::Http => {
            let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
            tracing::info!("HTTP endpoint: http://{}{}", addr, args.mcp_path);

            let http_config = motlie_mcp::HttpConfig::new(addr)
                .with_mcp_path(&args.mcp_path)
                .with_sse_keep_alive(Some(Duration::from_secs(30)));

            // Run HTTP server until Ctrl+C
            tokio::select! {
                result = motlie_mcp::serve_http(server, http_config) => {
                    result?;
                    tracing::info!("HTTP server completed normally");
                }
                _ = tokio::signal::ctrl_c() => {
                    tracing::info!("Received Ctrl+C, initiating graceful shutdown...");
                }
            }
            Ok(())
        }
    };

    // Graceful shutdown - critical for data integrity!
    tracing::info!("Shutting down database...");
    if let Err(e) = managed_db.shutdown().await {
        tracing::error!("Database shutdown error: {}", e);
        return Err(e);
    }
    tracing::info!("Database shutdown complete");

    shutdown_result
}
