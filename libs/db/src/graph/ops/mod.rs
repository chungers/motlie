//! Graph business logic (single source of truth for ops).
//!
//! MutationExecutor and QueryExecutor impls should delegate to these helpers.
//!
//! # Module Organization
//!
//! - `edge`: Edge mutation operations (add, update, delete, restore)
//! - `fragment`: Fragment operations for nodes and edges
//! - `name`: Name write and resolve operations
//! - `node`: Node mutation operations (add, update, delete, restore)
//! - `read`: Unified read operations with StorageAccess abstraction
//! - `summary`: Summary write and resolve operations
//! - `util`: Utility functions

pub(crate) mod edge;
pub(crate) mod fragment;
pub(crate) mod name;
pub(crate) mod node;
pub(crate) mod read;
pub(crate) mod summary;
pub(crate) mod util;

#[cfg(test)]
mod tests;
