//! Combined Motlie MCP Server
//!
//! A composite MCP server exposing tools from multiple domains:
//! - Database tools (15): Graph operations on Motlie database
//! - TTS tools (2): Text-to-speech using macOS speech synthesis
//!
//! # Usage
//!
//! ```bash
//! # Stdio transport (for Claude Desktop, Claude Code, Cursor)
//! cargo run --example motlie_all -- --db-path /path/to/db --transport stdio
//!
//! # HTTP transport (for remote access)
//! cargo run --example motlie_all -- --db-path /path/to/db --transport http --port 8080
//! ```
//!
//! # Architecture
//!
//! This example demonstrates how to compose tools from multiple modules
//! (`db` and `tts`) into a single MCP server using the `ToolCall` trait.
//!
//! # Graceful Shutdown
//!
//! This server properly handles Ctrl+C to gracefully shut down all resources,
//! ensuring database writes are flushed and no data corruption occurs.

use anyhow::Result;
use clap::{Parser, ValueEnum};
use motlie_db::{Storage, StorageConfig};
use motlie_mcp::db::{self, DbResource};
use motlie_mcp::tts::{self, TtsEngine, TtsResource};
use motlie_mcp::{stdio, ManagedResource, ServiceExt, ToolCall};
use rmcp::{
    handler::server::{tool::ToolRouter, wrapper::Parameters},
    model::*,
    tool, tool_handler, tool_router,
    ErrorData as McpError, ServerHandler,
};

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
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
#[command(name = "motlie-all-mcp")]
#[command(about = "Combined MCP server with database and TTS tools", long_about = None)]
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

/// Combined MCP server with database and TTS tools.
///
/// This server demonstrates how to compose tools from multiple domains
/// using the `ToolCall` trait for uniform dispatch.
#[derive(Clone)]
struct CombinedServer {
    db_resource: Arc<DbResource>,
    tts_resource: Arc<TtsResource>,
    tool_router: ToolRouter<Self>,
}

impl CombinedServer {
    fn new(db: Arc<db::LazyDb>, tts: Arc<tts::LazyTts>, query_timeout: Duration) -> Self {
        Self {
            db_resource: Arc::new(DbResource::new(db, query_timeout)),
            tts_resource: Arc::new(TtsResource::new(tts)),
            tool_router: Self::tool_router(),
        }
    }
}

#[tool_router]
impl CombinedServer {
    // ==================== Database Mutation Tools ====================

    #[tool(
        description = "Create a new node in the graph with name and optional temporal validity range"
    )]
    async fn add_node(
        &self,
        Parameters(params): Parameters<db::AddNodeParams>,
    ) -> Result<CallToolResult, McpError> {
        params.call(&self.db_resource).await
    }

    #[tool(description = "Create an edge between two nodes with optional weight and temporal validity")]
    async fn add_edge(
        &self,
        Parameters(params): Parameters<db::AddEdgeParams>,
    ) -> Result<CallToolResult, McpError> {
        params.call(&self.db_resource).await
    }

    #[tool(description = "Add a text content fragment to an existing node")]
    async fn add_node_fragment(
        &self,
        Parameters(params): Parameters<db::AddNodeFragmentParams>,
    ) -> Result<CallToolResult, McpError> {
        params.call(&self.db_resource).await
    }

    #[tool(description = "Add a text content fragment to an existing edge")]
    async fn add_edge_fragment(
        &self,
        Parameters(params): Parameters<db::AddEdgeFragmentParams>,
    ) -> Result<CallToolResult, McpError> {
        params.call(&self.db_resource).await
    }

    #[tool(description = "Update the temporal validity range of a node")]
    async fn update_node_valid_range(
        &self,
        Parameters(params): Parameters<db::UpdateNodeActivePeriodParams>,
    ) -> Result<CallToolResult, McpError> {
        params.call(&self.db_resource).await
    }

    #[tool(description = "Update the temporal validity range of an edge")]
    async fn update_edge_valid_range(
        &self,
        Parameters(params): Parameters<db::UpdateEdgeActivePeriodParams>,
    ) -> Result<CallToolResult, McpError> {
        params.call(&self.db_resource).await
    }

    #[tool(description = "Update the weight of an edge for graph algorithms")]
    async fn update_edge_weight(
        &self,
        Parameters(params): Parameters<db::UpdateEdgeWeightParams>,
    ) -> Result<CallToolResult, McpError> {
        params.call(&self.db_resource).await
    }

    // ==================== Database Query Tools ====================

    #[tool(description = "Retrieve a node by its ID")]
    async fn query_node_by_id(
        &self,
        Parameters(params): Parameters<db::QueryNodeByIdParams>,
    ) -> Result<CallToolResult, McpError> {
        params.call(&self.db_resource).await
    }

    #[tool(description = "Retrieve an edge by its source, destination, and name")]
    async fn query_edge(
        &self,
        Parameters(params): Parameters<db::QueryEdgeParams>,
    ) -> Result<CallToolResult, McpError> {
        params.call(&self.db_resource).await
    }

    #[tool(description = "Get all outgoing edges from a node")]
    async fn query_outgoing_edges(
        &self,
        Parameters(params): Parameters<db::QueryOutgoingEdgesParams>,
    ) -> Result<CallToolResult, McpError> {
        params.call(&self.db_resource).await
    }

    #[tool(description = "Get all incoming edges to a node")]
    async fn query_incoming_edges(
        &self,
        Parameters(params): Parameters<db::QueryIncomingEdgesParams>,
    ) -> Result<CallToolResult, McpError> {
        params.call(&self.db_resource).await
    }

    #[tool(description = "Search for nodes by name or name prefix")]
    async fn query_nodes_by_name(
        &self,
        Parameters(params): Parameters<db::QueryNodesByNameParams>,
    ) -> Result<CallToolResult, McpError> {
        params.call(&self.db_resource).await
    }

    #[tool(description = "Search for edges by name or name prefix")]
    async fn query_edges_by_name(
        &self,
        Parameters(params): Parameters<db::QueryEdgesByNameParams>,
    ) -> Result<CallToolResult, McpError> {
        params.call(&self.db_resource).await
    }

    #[tool(description = "Get content fragments for a node within a time range")]
    async fn query_node_fragments(
        &self,
        Parameters(params): Parameters<db::QueryNodeFragmentsParams>,
    ) -> Result<CallToolResult, McpError> {
        params.call(&self.db_resource).await
    }

    #[tool(description = "Get content fragments for an edge within a time range")]
    async fn query_edge_fragments(
        &self,
        Parameters(params): Parameters<db::QueryEdgeFragmentsParams>,
    ) -> Result<CallToolResult, McpError> {
        params.call(&self.db_resource).await
    }

    // ==================== TTS Tools ====================

    #[tool(
        description = "Speak text aloud using macOS text-to-speech. Supports multiple phrases, voice selection, and rate control."
    )]
    async fn say(
        &self,
        Parameters(params): Parameters<tts::SayParams>,
    ) -> Result<CallToolResult, McpError> {
        params.call(&self.tts_resource).await
    }

    #[tool(description = "List available text-to-speech voices on the system")]
    async fn list_voices(
        &self,
        Parameters(params): Parameters<tts::ListVoicesParams>,
    ) -> Result<CallToolResult, McpError> {
        params.call(&self.tts_resource).await
    }
}

#[tool_handler]
impl ServerHandler for CombinedServer {
    fn get_info(&self) -> ServerInfo {
        let combined_instructions = format!(
            "# Combined Motlie MCP Server\n\n\
             This server provides tools from multiple domains.\n\n\
             ## Database Tools (15)\n{}\n\n\
             ## TTS Tools (2)\n{}",
            db::INSTRUCTIONS,
            tts::INSTRUCTIONS
        );

        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation::from_build_env(),
            instructions: Some(combined_instructions),
        }
    }
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
    let managed_db = ManagedResource::new(Box::new(move || {
        tracing::info!("Initializing database at: {:?}", db_path);

        let storage = Storage::readwrite(&db_path);
        let handles = storage.ready(StorageConfig::default())?;

        tracing::info!("Database initialized successfully");
        Ok(handles)
    }));

    // Create managed TTS resource with proper lifecycle
    let managed_tts = ManagedResource::new(Box::new(|| TtsEngine::new()));

    // Build the combined MCP server
    let query_timeout = Duration::from_secs(args.query_timeout);
    let server = CombinedServer::new(managed_db.lazy(), managed_tts.lazy(), query_timeout);

    tracing::info!("Starting Combined Motlie MCP server...");
    tracing::info!("Server: motlie-all-mcp");
    tracing::info!("Version: {}", env!("CARGO_PKG_VERSION"));
    tracing::info!("Database: {} (lazy initialization)", args.db_path);
    tracing::info!("TTS: macOS only (validated on first use)");
    tracing::info!("Tools: 17 total (15 database + 2 TTS)");
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

    // Graceful shutdown of all resources
    // Database shutdown is critical for data integrity!
    tracing::info!("Shutting down resources...");

    // Shutdown TTS first (quick, no-op)
    if let Err(e) = managed_tts.shutdown().await {
        tracing::error!("TTS shutdown error: {}", e);
    }

    // Shutdown database (important - flushes writes)
    if let Err(e) = managed_db.shutdown().await {
        tracing::error!("Database shutdown error: {}", e);
        return Err(e);
    }

    tracing::info!("All resources shutdown complete");

    Ok(())
}
