//! Unified writer module for mutation execution.
//!
//! This module provides the [`Runnable`] trait for executing mutations
//! against a [`Writer`](crate::graph::writer::Writer).

use anyhow::Result;

use crate::graph::writer::Writer;

// ============================================================================
// Runnable Trait - Execute mutations against a Writer
// ============================================================================

/// Trait for mutations that can be executed against a Writer.
///
/// This trait follows the same pattern as the Query API's Runnable trait,
/// enabling mutations to be constructed separately from execution.
///
/// # Examples
///
/// ```rust,ignore
/// use motlie_db::graph::mutation::AddNode;
/// use motlie_db::writer::Runnable;
/// use motlie_db::{Id, TimestampMilli};
///
/// // Construct mutation
/// let mutation = AddNode {
///     id: Id::new(),
///     name: "Alice".to_string(),
///     ts_millis: TimestampMilli::now(),
///     valid_range: None,
/// };
///
/// // Execute it
/// mutation.run(&writer).await?;
/// ```
#[async_trait::async_trait]
pub trait Runnable {
    /// Execute this mutation against the writer
    async fn run(self, writer: &Writer) -> Result<()>;
}
