use anyhow::Result;
use clap::{Parser, ValueEnum};
use motlie_db::{create_mutation_writer, create_query_reader, Graph, ReaderConfig, Storage, WriterConfig, spawn_graph_consumer_with_graph, spawn_query_consumer_pool_shared};
use motlie_mcp::MotlieMcpServer;
use pmcp::server::streamable_http_server::StreamableHttpServer;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Transport protocol for MCP server
#[derive(Debug, Clone, ValueEnum)]
enum Transport {
    /// Standard input/output (for local process communication)
    Stdio,
    /// HTTP with Server-Sent Events (for remote access)
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
#[command(name = "motlie-mcp-server")]
#[command(about = "Motlie graph database MCP server using pmcp SDK", long_about = None)]
struct Args {
    /// Path to the RocksDB database directory
    #[arg(short, long)]
    db_path: String,

    /// Optional authentication token for secure access (Bearer token)
    ///
    /// When provided, all MCP tool requests must include authentication via
    /// pmcp's built-in auth context mechanism. The token is validated at the
    /// protocol level by pmcp.
    #[arg(short, long)]
    auth_token: Option<String>,

    /// Transport protocol (stdio or http)
    ///
    /// - stdio: Standard input/output for local process communication (default)
    /// - http: HTTP with Server-Sent Events for remote access
    #[arg(short, long, value_enum, default_value = "stdio")]
    transport: Transport,

    /// Port for HTTP transport (only used when transport=http)
    #[arg(short, long, default_value = "8080")]
    port: u16,

    /// Host address for HTTP transport (only used when transport=http)
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Mutation channel buffer size
    #[arg(long, default_value = "100")]
    mutation_buffer_size: usize,

    /// Query channel buffer size
    #[arg(long, default_value = "100")]
    query_buffer_size: usize,

    /// Number of concurrent query worker threads
    ///
    /// Controls how many worker threads process queries in parallel.
    /// All workers share a single readwrite TransactionDB via Arc<Graph>
    /// for 99%+ read-after-write consistency (vs 25-30% with readonly mode).
    /// Default is the number of CPU cores for optimal throughput.
    #[arg(long, default_value_t = num_cpus::get())]
    query_workers: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    // Parse command-line arguments
    let args = Args::parse();

    // Initialize storage and database components
    log::info!("Initializing database at: {}", args.db_path);
    let db_path = Path::new(&args.db_path);

    // Create writer and reader configurations
    let writer_config = WriterConfig {
        channel_buffer_size: args.mutation_buffer_size,
    };
    let reader_config = ReaderConfig {
        channel_buffer_size: args.query_buffer_size,
    };

    // Create ONE shared readwrite Storage for both mutations and queries
    // RocksDB TransactionDB acquires exclusive lock - cannot have multiple instances
    log::info!("Opening shared readwrite Storage for mutations and queries");
    let mut storage = Storage::readwrite(db_path);
    storage.ready()?;
    let storage = Arc::new(storage);
    let graph = Arc::new(Graph::new(storage));

    // Create writer with background consumer using shared graph
    let (writer, mutation_receiver) = create_mutation_writer(writer_config.clone());
    let _mutation_handle = spawn_graph_consumer_with_graph(
        mutation_receiver,
        writer_config,
        graph.clone(),
    );

    // Create reader with background consumer pool for queries
    // Uses same shared TransactionDB via Arc<Graph> for high consistency
    let (reader, query_receiver) = create_query_reader(reader_config.clone());
    let _query_handles = spawn_query_consumer_pool_shared(
        query_receiver,
        graph.clone(),
        args.query_workers,
    );

    log::info!(
        "Started {} query worker threads (shared TransactionDB, 99%+ consistency)",
        args.query_workers
    );

    // Create the MCP server handler
    let mcp_server = MotlieMcpServer::new(writer, reader);

    // Build the pmcp server with all tools
    let server = mcp_server.build_server(args.auth_token.clone())?;

    log::info!("Starting Motlie MCP server using pmcp SDK...");
    log::info!("Server: motlie-mcp-server");
    log::info!("Version: {}", env!("CARGO_PKG_VERSION"));

    if args.auth_token.is_some() {
        log::info!("Authentication: ENABLED (Bearer token required)");
        log::info!("  pmcp will validate authentication at the protocol level");
    } else {
        log::warn!("Authentication: DISABLED (not recommended for production)");
    }

    log::info!("Available tools: 15 (7 mutations + 8 queries)");
    log::info!("Transport: {}", args.transport);

    // Start the server with the selected transport
    match args.transport {
        Transport::Stdio => {
            log::info!("Listening on stdio (standard input/output)");
            log::info!("Ready for MCP client connections via stdin/stdout");

            // Start the server with stdio transport
            server.run_stdio().await?;
        }
        Transport::Http => {
            let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
            log::info!("Starting HTTP server on {}", addr);
            log::info!("MCP endpoint: http://{}", addr);
            log::info!("Ready for remote MCP client connections");

            // Wrap server in Arc<Mutex<>> for HTTP transport
            let server = Arc::new(Mutex::new(server));

            // Create the streamable HTTP server (stateless mode by default)
            let http_server = StreamableHttpServer::new(addr, server);

            // Start the server
            let (bound_addr, server_handle) = http_server
                .start()
                .await
                .map_err(|e| anyhow::anyhow!("Failed to start HTTP server: {}", e))?;

            log::info!("✓ HTTP server started successfully");
            log::info!("  Bound to: http://{}", bound_addr);
            log::info!("  Press Ctrl+C to stop the server");

            // Keep the server running until interrupted
            server_handle
                .await
                .map_err(|e| anyhow::anyhow!("Server error: {}", e))?;
        }
    }

    Ok(())
}
