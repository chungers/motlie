//! MCP tools for Motlie graph database
//!
//! This module provides MCP tool implementations for interacting with the Motlie
//! graph database. It includes:
//!
//! - **types**: Parameter types for all database tools, each implementing `ToolCall`
//! - **server**: `MotlieMcpServer` - a ready-to-use MCP server exposing all database tools
//!
//! # Architecture
//!
//! Each parameter type (e.g., `AddNodeParams`) implements the `ToolCall` trait,
//! which binds the parameter to its execution logic. This ensures a 1:1 correspondence
//! between parameter types and tool implementations.
//!
//! # Example: Using the default server
//!
//! ```ignore
//! use motlie_mcp::db::{MotlieMcpServer, LazyDb};
//! use motlie_mcp::LazyResource;
//! use std::sync::Arc;
//! use std::time::Duration;
//!
//! let db = Arc::new(LazyResource::new(Box::new(|| {
//!     // Initialize database
//!     Ok((writer, reader))
//! })));
//!
//! let server = MotlieMcpServer::new(db, Duration::from_secs(30));
//! ```
//!
//! # Example: Composing tools into a custom server
//!
//! ```ignore
//! use motlie_mcp::db::{DbResource, AddNodeParams};
//! use motlie_mcp::ToolCall;
//!
//! // In your custom server's #[tool] method:
//! #[tool(description = "Add a node")]
//! async fn add_node(&self, Parameters(params): Parameters<AddNodeParams>) -> Result<CallToolResult, McpError> {
//!     params.call(&self.db_resource).await
//! }
//! ```

pub mod server;
pub mod types;

pub use server::MotlieMcpServer;
pub use types::*;

use crate::LazyResource;
use motlie_db::reader::Reader;
use motlie_db::writer::Writer;
use rmcp::ErrorData as McpError;
use std::sync::Arc;
use std::time::Duration;

/// Type alias for lazy database initialization.
///
/// The database is lazily initialized on first tool invocation to allow
/// fast MCP handshake completion.
pub type LazyDb = LazyResource<(Writer, Reader)>;

/// Resource context for database tool execution.
///
/// This struct wraps the lazy database and configuration needed by all database tools.
/// It is passed to each tool's `ToolCall::call` method.
pub struct DbResource {
    db: Arc<LazyDb>,
    query_timeout: Duration,
}

impl DbResource {
    /// Create a new database resource context.
    pub fn new(db: Arc<LazyDb>, query_timeout: Duration) -> Self {
        Self { db, query_timeout }
    }

    /// Get the writer for mutation operations.
    pub async fn writer(&self) -> Result<&Writer, McpError> {
        let (writer, _) = self.db.resource().await?;
        Ok(writer)
    }

    /// Get the reader for query operations.
    pub async fn reader(&self) -> Result<&Reader, McpError> {
        let (_, reader) = self.db.resource().await?;
        Ok(reader)
    }

    /// Get the query timeout duration.
    pub fn query_timeout(&self) -> Duration {
        self.query_timeout
    }
}

/// Instructions for AI assistants using database tools.
pub const INSTRUCTIONS: &str = r#"
This server provides tools to interact with the Motlie graph database.

## Timestamps
- All timestamps are in milliseconds since Unix epoch.

## Entities and Relationships
- Nodes are entities, Edges are relationships.
- To avoid duplication, ALWAYS query an object by name first (if its ID is not in context).
- Confirm with the user before calling add_node to model a new entity. Nodes cannot be deleted.
- Relationships (edges) can have different names but similar meanings. Reuse string names when possible.

## Fragments
- Fragments model properties, attributes, and state changes of entities (nodes) and relationships (edges).
- A node or edge's fragments form a queryable history by time range.
- Fragments are ONLY for contexts, properties, and attributes about an entity or relationship.

## Temporal Validity
- No deletion of nodes or edges. ONLY invalidation by setting a validity time range.
- Example: If an event node (NeurIPS 2023) should disappear after 2024-01-01, set its 'valid until' timestamp.

## Graph Modeling Guidelines
- Think hard about how you model the user's text using the RDF model: subject, predicate, object.
- Subject and object are entities (nodes), predicate/action forms a relationship (edge).
- A fragment MUST NOT store content containing concepts outside the scope of the entity or relationship.
- Prefer creating new nodes and edges over using fragments for complex relationships.

### Example
"Johnny Rabbit loves ice cream in the summer. He is 7 years old. He started Reed Elementary School in Sept 2025."

**BEST**:
- Node1 = Johnny, Fragment1_1 = "7 years old", Fragment1_2 = "Last name is Rabbit"
- Node2 = Ice Cream, Edge2 = Johnny --[loves]--> Ice Cream, EdgeFragment2_1 = "in the summer"
- Node3 = Reed Elementary School, Edge3 = Johnny --[Started]--> Reed Elementary School, EdgeFragment3_1 = "started in 09/2025"

**BAD**:
- Node1 = Johnny Rabbit, Fragment1_1 = "7 years old. Loves ice cream and started Reed Elementary School in 2025"
"#;
