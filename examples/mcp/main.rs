use anyhow::Result;
use clap::{Parser, ValueEnum};
use motlie_db::{
    create_mutation_writer, create_query_reader, spawn_graph_consumer_with_graph,
    spawn_query_consumer_pool_shared, Graph, ReaderConfig, Storage, WriterConfig,
};
use motlie_mcp::{stdio, LazyDatabase, MotlieMcpServer, ServiceExt, INFO_TEXT};

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
#[command(name = "motlie-mcp-server")]
#[command(about = "Motlie graph database MCP server using rmcp SDK", long_about = None)]
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

    // Parse command-line arguments FIRST (this is fast)
    let args = Args::parse();

    // Capture configuration for lazy initialization
    let db_path = PathBuf::from(&args.db_path);
    let mutation_buffer_size = args.mutation_buffer_size;
    let query_buffer_size = args.query_buffer_size;
    let query_workers = args.query_workers;

    // Create lazy database initializer
    // The actual database initialization will happen on first tool use
    let lazy_db = Arc::new(LazyDatabase::new(Box::new(move || {
        tracing::info!("Initializing database at: {:?}", db_path);

        let writer_config = WriterConfig {
            channel_buffer_size: mutation_buffer_size,
        };
        let reader_config = ReaderConfig {
            channel_buffer_size: query_buffer_size,
        };

        // Create ONE shared readwrite Storage for both mutations and queries
        tracing::info!("Opening shared readwrite Storage for mutations and queries");
        let mut storage = Storage::readwrite(&db_path);
        storage.ready()?;
        let storage = Arc::new(storage);
        let graph = Arc::new(Graph::new(storage));

        // Create writer with background consumer using shared graph
        let (writer, mutation_receiver) = create_mutation_writer(writer_config.clone());
        let _mutation_handle =
            spawn_graph_consumer_with_graph(mutation_receiver, writer_config, graph.clone());

        // Create reader with background consumer pool for queries
        let (reader, query_receiver) = create_query_reader(reader_config.clone());
        let _query_handles =
            spawn_query_consumer_pool_shared(query_receiver, graph.clone(), query_workers);

        tracing::info!(
            "Started {} query worker threads (shared TransactionDB, 99%+ consistency)",
            query_workers
        );

        Ok((writer, reader))
    })));

    // Build the MCP server with lazy database - this is FAST
    // The server can start listening immediately
    let query_timeout = Duration::from_secs(5);
    let server = MotlieMcpServer::new(lazy_db, query_timeout);

    tracing::info!("Starting Motlie MCP server using rmcp SDK...");
    tracing::info!("Server: motlie-mcp-server");
    tracing::info!("Version: {}", env!("CARGO_PKG_VERSION"));
    tracing::info!("Database will be initialized on first tool use (lazy initialization)");
    tracing::info!("Available tools: 15 (7 mutations + 8 queries)");
    tracing::info!("Transport: {}", args.transport);
    tracing::info!("Instructions: {}", INFO_TEXT);

    // Start the server with the selected transport
    match args.transport {
        Transport::Stdio => {
            tracing::info!("Listening on stdio (standard input/output)");
            tracing::info!("Ready for MCP client connections via stdin/stdout");

            // Start the server with stdio transport IMMEDIATELY
            // No database initialization blocking this!
            let service = server.serve(stdio()).await.inspect_err(|e| {
                tracing::error!("serving error: {:?}", e);
            })?;

            service.waiting().await?;
        }
        Transport::Http => {
            let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
            tracing::info!("Ready for MCP client connections via HTTP");

            let http_config = motlie_mcp::HttpConfig::new(addr)
                .with_mcp_path(&args.mcp_path)
                .with_sse_keep_alive(Some(Duration::from_secs(30)));

            motlie_mcp::serve_http(server, http_config).await?;
        }
    }

    Ok(())
}
