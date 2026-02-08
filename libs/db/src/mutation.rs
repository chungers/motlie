//! Unified mutation module exposing all mutation types.
//!
//! This module provides convenient access to all mutation types through a single
//! import path, following the same pattern as the unified query module.
//!
//! # Overview
//!
//! The unified mutation API provides:
//! - All mutation types (`AddNode`, `AddEdge`, etc.)
//! - The `Runnable` trait for executing mutations
//! - The `Mutation` enum for batch operations
//! - Common schema types used with mutations
//!
//! # Mutation Types
//!
//! | Mutation | Description |
//! |----------|-------------|
//! | [`AddNode`] | Create a new node with name and summary |
//! | [`AddEdge`] | Create an edge between two nodes |
//! | [`AddNodeFragment`] | Add timestamped content fragment to a node |
//! | [`AddEdgeFragment`] | Add timestamped content fragment to an edge |
//! | [`UpdateNode`] | Update node (active period and/or summary) |
//! | [`UpdateEdge`] | Update edge (weight, active period, and/or summary) |
//!
//! # Usage
//!
//! All mutations implement [`Runnable`], allowing execution with `mutation.run(&writer)`:
//!
//! ```ignore
//! use motlie_db::{Storage, StorageConfig};
//! use motlie_db::mutation::{AddNode, AddEdge, Runnable, NodeSummary, EdgeSummary};
//! use motlie_db::{Id, TimestampMilli};
//!
//! // Initialize unified storage in read-write mode
//! // Storage takes a single path and derives <path>/graph and <path>/fulltext
//! let storage = Storage::readwrite(db_path);
//! let handles = storage.ready(StorageConfig::default())?;  // ReadWriteHandles
//!
//! // Create a node - writer() returns &Writer directly, no unwrap needed!
//! let node_id = Id::new();
//! AddNode {
//!     id: node_id,
//!     ts_millis: TimestampMilli::now(),
//!     name: "Alice".to_string(),
//!     summary: NodeSummary::from_text("A person named Alice"),
//!     valid_range: None,
//! }
//! .run(handles.writer())
//! .await?;
//!
//! // Create another node
//! let other_id = Id::new();
//! AddNode {
//!     id: other_id,
//!     ts_millis: TimestampMilli::now(),
//!     name: "Bob".to_string(),
//!     summary: NodeSummary::from_text("A person named Bob"),
//!     valid_range: None,
//! }
//! .run(handles.writer())
//! .await?;
//!
//! // Create an edge between them
//! AddEdge {
//!     source_node_id: node_id,
//!     target_node_id: other_id,
//!     ts_millis: TimestampMilli::now(),
//!     name: "knows".to_string(),
//!     summary: EdgeSummary::from_text("Alice knows Bob"),
//!     weight: Some(1.0),
//!     valid_range: None,
//! }
//! .run(handles.writer())
//! .await?;
//!
//! // Clean shutdown
//! handles.shutdown().await?;
//! ```
//!
//! # Batch Mutations
//!
//! Use [`MutationBatch`] to send multiple mutations atomically:
//!
//! ```ignore
//! use motlie_db::mutation::{MutationBatch, Mutation, AddNode, AddEdge, Runnable};
//!
//! let mut batch = MutationBatch::new();
//! batch.push(Mutation::AddNode(AddNode { /* ... */ }));
//! batch.push(Mutation::AddEdge(AddEdge { /* ... */ }));
//! batch.run(handles.writer()).await?;
//! ```
//!
//! # See Also
//!
//! - [`Storage::ready()`](crate::Storage::ready) - Initialize unified storage
//! - [`graph::mutation`](crate::graph::mutation) - Graph mutation internals

// Re-export Runnable trait from writer module
pub use crate::writer::Runnable;

// Re-export all mutation types from graph::mutation
pub use crate::graph::mutation::{
    AddEdge, AddEdgeFragment, AddNode, AddNodeFragment, Mutation, MutationBatch,
    UpdateNode, UpdateEdge,
};

// Re-export schema types commonly used with mutations
pub use crate::graph::schema::{EdgeName, EdgeSummary, NodeName, NodeSummary, ActivePeriod};
