//! MCP server library for building Model Context Protocol servers.
//!
//! This crate provides infrastructure for building MCP servers with support for
//! multiple tool domains. It includes:
//!
//! - **Generic infrastructure**: `LazyResource`, `ToolCall` trait, HTTP transport
//! - **db**: Motlie graph database tools
//! - **tts**: Text-to-speech tools (macOS)
//!
//! # Architecture
//!
//! The crate uses a trait-based architecture where each tool's parameter type
//! implements the `ToolCall` trait. This ensures compile-time verification that
//! every parameter type has a corresponding tool implementation.
//!
//! ## Key Components
//!
//! - [`LazyResource<R>`]: Deferred resource initialization for fast MCP handshake
//! - [`ToolCall`]: Trait binding parameter types to their execution logic
//! - [`db::MotlieMcpServer`]: Ready-to-use server for database tools
//! - [`tts::TtsMcpServer`]: Ready-to-use server for TTS tools
//! - [`serve_http`]: HTTP transport for any `ServerHandler`
//!
//! # Quick Start: Using a Pre-built Server
//!
//! ```ignore
//! use motlie_mcp::db::{MotlieMcpServer, LazyDb};
//! use motlie_mcp::{LazyResource, serve_http, HttpConfig};
//! use std::sync::Arc;
//! use std::time::Duration;
//!
//! let db = Arc::new(LazyResource::new(Box::new(|| {
//!     Ok((writer, reader))
//! })));
//!
//! let server = MotlieMcpServer::new(db, Duration::from_secs(30));
//! serve_http(server, HttpConfig::default()).await?;
//! ```
//!
//! # Composing Tools from Multiple Domains
//!
//! ```ignore
//! use motlie_mcp::{db, tts, ToolCall};
//!
//! #[derive(Clone)]
//! struct CombinedServer {
//!     db_resource: Arc<db::DbResource>,
//!     tts_resource: Arc<tts::TtsResource>,
//!     tool_router: ToolRouter<Self>,
//! }
//!
//! #[tool_router]
//! impl CombinedServer {
//!     #[tool(description = "Add a node")]
//!     async fn add_node(&self, Parameters(p): Parameters<db::AddNodeParams>) -> Result<CallToolResult, McpError> {
//!         p.call(&self.db_resource).await
//!     }
//!
//!     #[tool(description = "Speak text")]
//!     async fn say(&self, Parameters(p): Parameters<tts::SayParams>) -> Result<CallToolResult, McpError> {
//!         p.call(&self.tts_resource).await
//!     }
//! }
//! ```
//!
//! # Transports
//!
//! The library supports two transport modes:
//! - **stdio**: Standard input/output for local process communication
//! - **http**: HTTP with Streamable HTTP protocol (using axum)

pub mod db;
pub mod http;
pub mod tts;

use async_trait::async_trait;
use rmcp::{model::CallToolResult, ErrorData as McpError};
use std::sync::Mutex;
use tokio::sync::OnceCell;

/// Generic initialization function type for lazy resources.
pub type ResourceInitFn<R> = Box<dyn FnOnce() -> anyhow::Result<R> + Send + Sync>;

/// Generic lazy resource holder for deferred initialization.
///
/// This is essential for MCP servers communicating over stdio transport, where
/// slow resource initialization (e.g., opening databases, establishing connections)
/// will cause the agent-tool handshake to timeout and fail. By deferring initialization
/// until the first tool invocation, the server can complete the MCP handshake quickly
/// and only pay the initialization cost when actually needed.
///
/// The resource is initialized on first access via the `resource()` method.
/// Initialization is thread-safe and runs at most once.
///
/// # Example
///
/// ```ignore
/// use motlie_mcp::LazyResource;
/// use std::sync::Arc;
///
/// // Create lazy resource - this is fast, no initialization yet
/// let lazy = Arc::new(LazyResource::new(Box::new(|| {
///     // This expensive initialization runs only on first access
///     Ok(MyDatabase::open("/path/to/db")?)
/// })));
///
/// // Server can start immediately and complete MCP handshake
/// let server = MyMcpServer::new(lazy);
///
/// // Later, when a tool is invoked:
/// let resource = lazy.resource().await?;
/// resource.query(...);
/// ```
pub struct LazyResource<R> {
    init_fn: Mutex<Option<ResourceInitFn<R>>>,
    resource: OnceCell<R>,
}

impl<R> LazyResource<R> {
    /// Create a new lazy resource with an initialization function.
    pub fn new(init_fn: ResourceInitFn<R>) -> Self {
        Self {
            init_fn: Mutex::new(Some(init_fn)),
            resource: OnceCell::new(),
        }
    }

    /// Get or initialize the resource.
    ///
    /// This method is idempotent - the initialization function runs at most once.
    /// Subsequent calls return the cached resource.
    pub async fn resource(&self) -> Result<&R, McpError> {
        self.ensure_initialized().await?;
        self.resource
            .get()
            .ok_or_else(|| McpError::internal_error("Resource not initialized", None))
    }

    /// Check if the resource has been initialized.
    pub fn is_initialized(&self) -> bool {
        self.resource.initialized()
    }

    async fn ensure_initialized(&self) -> Result<(), McpError> {
        // Check if already initialized
        if self.resource.initialized() {
            return Ok(());
        }

        // Take the init function (only runs once)
        let init_fn = {
            let mut guard = self.init_fn.lock().unwrap();
            guard.take()
        };

        if let Some(init_fn) = init_fn {
            tracing::info!("Initializing resource (lazy initialization on first access)...");
            let resource = init_fn().map_err(|e| {
                McpError::internal_error(format!("Resource initialization failed: {}", e), None)
            })?;
            let _ = self.resource.set(resource);
            tracing::info!("Resource initialization complete");
        }

        Ok(())
    }
}

/// Trait for MCP tool parameters that can be executed against a resource.
///
/// Each parameter type implements this trait to define its tool execution logic.
/// This ensures a 1:1 correspondence between parameter types and tool implementations,
/// verified at compile time.
///
/// # Design
///
/// The trait uses an associated type `Resource` to specify what resource the tool
/// needs to execute. This allows different tool domains (e.g., database, TTS) to
/// have their own resource types while sharing the same execution pattern.
///
/// # Example
///
/// ```ignore
/// use motlie_mcp::ToolCall;
/// use async_trait::async_trait;
///
/// #[derive(Deserialize, JsonSchema)]
/// pub struct MyToolParams {
///     pub message: String,
/// }
///
/// #[async_trait]
/// impl ToolCall for MyToolParams {
///     type Resource = MyResource;
///
///     async fn call(self, res: &Self::Resource) -> Result<CallToolResult, McpError> {
///         // Tool implementation here
///         Ok(CallToolResult::success(vec![Content::text("Done")]))
///     }
/// }
/// ```
///
/// # Composing Tools
///
/// When building a combined server, use this trait to delegate tool execution:
///
/// ```ignore
/// #[tool(description = "My tool")]
/// async fn my_tool(&self, Parameters(params): Parameters<MyToolParams>) -> Result<CallToolResult, McpError> {
///     params.call(&self.my_resource).await
/// }
/// ```
#[async_trait]
pub trait ToolCall: Sized + Send {
    /// The resource type required to execute this tool.
    type Resource: Sync;

    /// Execute the tool against the resource.
    async fn call(self, resource: &Self::Resource) -> Result<CallToolResult, McpError>;
}

// Re-exports for convenience
pub use http::{serve_http, HttpConfig};
pub use rmcp::{transport::stdio, ServiceExt};

// Backwards compatibility re-exports (keep at crate root for existing users)
pub use db::{LazyDb, MotlieMcpServer};

// Re-export DbResource for composition
pub use db::DbResource;
