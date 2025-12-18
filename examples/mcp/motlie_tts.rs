//! Motlie TTS MCP Server
//!
//! A dedicated MCP server exposing only text-to-speech tools (macOS only).
//!
//! # Usage
//!
//! ```bash
//! # HTTP transport (default, recommended for TTS)
//! cargo run --example motlie_tts
//! cargo run --example motlie_tts -- --port 8081
//!
//! # Stdio transport (for Claude Desktop, Claude Code, Cursor)
//! # Note: Client must keep stdin open until speech completes
//! cargo run --example motlie_tts -- --transport stdio
//! ```
//!
//! # Platform Support
//!
//! This server is only supported on macOS. On other platforms, tool calls
//! will return an error indicating the platform is not supported.
//!
//! # Why HTTP is Default
//!
//! HTTP transport is recommended for TTS because the `say` tool can take
//! 10-30+ seconds to complete. With stdio transport, if the client closes
//! stdin before speech finishes, the server terminates and speech may be
//! interrupted. HTTP connections remain open until the response is sent.
//!
//! # Graceful Shutdown
//!
//! This server properly handles Ctrl+C for graceful shutdown.

use anyhow::Result;
use clap::{Parser, ValueEnum};
use motlie_mcp::tts::{TtsEngine, TtsMcpServer};
use motlie_mcp::{stdio, ManagedResource, ServiceExt};

use std::net::SocketAddr;
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
#[command(name = "motlie-tts-mcp")]
#[command(about = "Text-to-speech MCP server (macOS only)", long_about = None)]
struct Args {
    /// Transport protocol (stdio or http)
    ///
    /// - stdio: Standard input/output for local process communication
    /// - http: HTTP with Streamable HTTP protocol for remote access (default, recommended)
    #[arg(short, long, value_enum, default_value = "http")]
    transport: Transport,

    /// Port for HTTP transport (only used when transport=http)
    #[arg(short, long, default_value = "8081")]
    port: u16,

    /// Host address for HTTP transport (only used when transport=http)
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// MCP endpoint path for HTTP transport (only used when transport=http)
    #[arg(long, default_value = "/mcp")]
    mcp_path: String,
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

    // Create managed TTS resource with proper lifecycle
    // Platform validation happens on first tool use
    let managed_tts = ManagedResource::new(Box::new(|| TtsEngine::new()));

    // Build the MCP server
    let server = TtsMcpServer::new(managed_tts.lazy());

    tracing::info!("Starting Motlie TTS MCP server...");
    tracing::info!("Server: motlie-tts-mcp");
    tracing::info!("Version: {}", env!("CARGO_PKG_VERSION"));
    tracing::info!("Platform: macOS only (validated on first tool use)");
    tracing::info!("Tools: 2 (say, list_voices)");
    tracing::info!("Transport: {}", args.transport);

    // Start the server with the selected transport
    // Use tokio::select! to handle graceful shutdown on Ctrl+C
    match args.transport {
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
        }
    }

    // Graceful shutdown (no-op for TTS, but follows the pattern)
    tracing::info!("Shutting down TTS engine...");
    if let Err(e) = managed_tts.shutdown().await {
        tracing::error!("TTS shutdown error: {}", e);
        return Err(e);
    }
    tracing::info!("TTS shutdown complete");

    Ok(())
}
