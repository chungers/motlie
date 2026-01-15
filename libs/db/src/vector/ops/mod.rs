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
//! - `insert` - Vector insertion operations
//! - `delete` - Vector deletion operations
//! - `embedding` - Embedding spec operations
//!
//! # Example
//!
//! ```rust,ignore
//! use crate::vector::ops;
//!
//! // In Processor::insert_vector():
//! let txn = txn_db.transaction();
//! let result = ops::insert_vector_in_txn(&txn, &txn_db, processor, ...)?;
//! txn.commit()?;
//! result.apply_cache_updates(processor);
//!
//! // In MutationExecutor for InsertVector:
//! let result = ops::insert_vector_in_txn(txn, txn_db, processor, ...)?;
//! // CacheUpdate returned for Consumer to apply after commit
//! ```

pub mod delete;
pub mod embedding;
pub mod insert;

// Re-export main types for convenience
pub use delete::{delete_vector_in_txn, DeleteResult};
pub use embedding::add_embedding_spec_in_txn;
pub use insert::{insert_batch_in_txn, insert_vector_in_txn, InsertBatchResult, InsertResult};
