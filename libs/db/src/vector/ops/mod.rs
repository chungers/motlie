//! Shared operation helpers — the single source of truth for vector business logic.
//!
//! This module provides the core implementation logic for vector operations,
//! used by `Processor` methods (public API), `MutationExecutor` implementations
//! (internal dispatch), and `QueryExecutor` implementations (query dispatch).
//!
//! # Design
//!
//! ## Mutation helpers (`insert`, `delete`, `embedding`)
//! - Take a transaction reference (caller manages commit)
//! - Perform all validation (dimension, duplicates, spec hash drift)
//! - Return cache updates to be applied AFTER transaction commit
//!
//! ## Read helpers (`read`)
//! - Take storage/registry references directly
//! - Perform point lookups (IdForward, IdReverse, Vectors, etc.)
//! - Used by QueryExecutor implementations
//!
//! ## Search helpers (`search`)
//! - Take resolved dependencies (storage, index, encoder, etc.)
//! - Contain the core search algorithms (HNSW, RaBitQ, pending scan)
//! - Processor resolves resources and delegates here
//!
//! # Module Structure
//!
//! - `insert` - Vector insertion operations (`ops::insert::vector`, `ops::insert::batch`)
//! - `delete` - Vector deletion operations (`ops::delete::vector`)
//! - `embedding` - Embedding spec operations (`ops::embedding::spec`)
//! - `read` - Point lookup operations (`ops::read::get_vector`, etc.)
//! - `search` - Search algorithms (`ops::search::search_exact`, etc.)

pub mod delete;
pub mod embedding;
pub mod insert;
pub mod read;
pub mod search;

// Re-export result types for convenience
pub use delete::DeleteResult;
pub use insert::{InsertBatchResult, InsertResult};
