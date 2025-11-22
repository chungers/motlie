use anyhow::Result;
use clap::Parser;
use motlie_db::{create_mutation_writer, create_query_reader, ReaderConfig, WriterConfig, spawn_graph_consumer, spawn_query_consumer};
use motlie_mcp::MotlieMcpServer;
use rust_mcp_sdk::schema::{Implementation, InitializeResult, ServerCapabilities, ServerCapabilitiesTools, LATEST_PROTOCOL_VERSION};
use rust_mcp_sdk::{mcp_server::server_runtime, McpServer, StdioTransport, TransportOptions};
use std::path::Path;
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(name = "motlie-mcp-server")]
#[command(about = "Motlie graph database MCP server", long_about = None)]
struct Args {
    /// Path to the RocksDB database directory
    #[arg(short, long)]
    db_path: String,

    /// Optional authentication token for secure access
    #[arg(short, long)]
    auth_token: Option<String>,

    /// Mutation channel buffer size
    #[arg(long, default_value = "100")]
    mutation_buffer_size: usize,

    /// Query channel buffer size
    #[arg(long, default_value = "100")]
    query_buffer_size: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

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

    // Create writer with background consumer for mutations
    let (writer, mutation_receiver) = create_mutation_writer(writer_config.clone());
    let _mutation_handle = spawn_graph_consumer(mutation_receiver, writer_config, db_path);

    // Create reader with background consumer for queries
    let (reader, query_receiver) = create_query_reader(reader_config.clone());
    let _query_handle = spawn_query_consumer(query_receiver, reader_config, db_path);

    // Create the MCP server handler
    let handler = MotlieMcpServer::new(
        writer,
        reader,
        args.auth_token.clone(),
    );

    // Define server details and capabilities
    let server_details = InitializeResult {
        server_info: Implementation {
            name: "Motlie MCP Server".to_string(),
            version: "0.1.0".to_string(),
            title: Some("Motlie Graph Database MCP Server".to_string()),
        },
        capabilities: ServerCapabilities {
            tools: Some(ServerCapabilitiesTools { list_changed: None }),
            ..Default::default()
        },
        meta: None,
        instructions: Some(
            "MCP server for the Motlie graph database. Provides tools for adding nodes, edges, \
             and fragments, as well as querying the graph structure."
                .to_string(),
        ),
        protocol_version: LATEST_PROTOCOL_VERSION.to_string(),
    };

    // Create stdio transport
    let transport = StdioTransport::new(TransportOptions::default())
        .map_err(|e| anyhow::anyhow!("Failed to create transport: {}", e))?;

    // Create the MCP server runtime
    let server: Arc<dyn McpServer> =
        server_runtime::create_server(server_details, transport, handler);

    log::info!("Starting Motlie MCP server...");
    if args.auth_token.is_some() {
        log::info!("Authentication enabled");
    } else {
        log::warn!("Running without authentication - not recommended for production");
    }

    // Start the server
    if let Err(start_error) = server.start().await {
        eprintln!(
            "{}",
            start_error
                .rpc_error_message()
                .unwrap_or(&start_error.to_string())
        );
        std::process::exit(1);
    }

    Ok(())
}
