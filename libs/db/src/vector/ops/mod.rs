//! Shared transaction-aware operation helpers.
//!
//! This module provides the core implementation logic for vector operations,
//! used by both `Processor` methods (public API) and `MutationExecutor`
//! implementations (internal dispatch).
//!
//! # Design
//!
//! Each helper function:
//! - Takes a transaction reference (caller manages commit)
//! - Performs all validation (dimension, duplicates, spec hash drift)
//! - Returns cache updates to be applied AFTER transaction commit
//!
//! This ensures **consistency** between the public API and mutation dispatch
//! paths - same validation, same behavior.
//!
//! # Module Structure
//!
//! - `insert` - Vector insertion operations (`ops::insert::vector`, `ops::insert::batch`)
//! - `delete` - Vector deletion operations (`ops::delete::vector`)
//! - `embedding` - Embedding spec operations (`ops::embedding::spec`)
//!
//! # Example
//!
//! ```rust,ignore
//! use crate::vector::ops;
//!
//! // In Processor::insert_vector():
//! let txn = txn_db.transaction();
//! let result = ops::insert::vector(&txn, &txn_db, processor, ...)?;
//! txn.commit()?;
//! result.apply_cache_updates(embedding, nav_cache, code_cache);
//!
//! // In MutationExecutor for InsertVector:
//! let result = ops::insert::vector(txn, txn_db, processor, ...)?;
//! // CacheUpdate returned for Consumer to apply after commit
//! ```

pub mod delete;
pub mod embedding;
pub mod insert;

// Re-export result types for convenience
pub use delete::DeleteResult;
pub use insert::{InsertBatchResult, InsertResult};
