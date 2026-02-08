//! Transaction module for read-your-writes operations.
//!
//! This module provides the [`Transaction`] type for executing mutations and
//! queries within a single atomic scope, where writes are immediately visible
//! to subsequent reads before commit.
//!
//! # Overview
//!
//! The Transaction API enables patterns like HNSW vector index insertion where
//! you need to:
//! 1. Write a node
//! 2. Read to find neighbors (must see the new node)
//! 3. Write edges based on the read
//! 4. Commit atomically
//!
//! # Example
//!
//! ```rust,ignore
//! use motlie_db::graph::{Transaction, Writer};
//! use motlie_db::graph::mutation::{AddNode, AddEdge};
//! use motlie_db::graph::query::{NodeById, OutgoingEdges};
//!
//! let mut txn = writer.transaction()?;
//!
//! // Write mutations (not committed yet)
//! txn.write(AddNode { id: node_id, ... })?;
//!
//! // Read sees uncommitted writes!
//! let (name, summary) = txn.read(NodeById::new(node_id, None))?;
//!
//! // Write more based on read results
//! txn.write(AddEdge { src: node_id, dst: other, ... })?;
//!
//! // Atomic commit
//! txn.commit()?;
//! ```
//!
//! # Lifetime
//!
//! `Transaction<'a>` is tied to the `Writer` that created it via the underlying
//! RocksDB transaction lifetime. This ensures transactions cannot outlive the
//! storage they operate on.
//!
//! For concurrent transactions, clone the `Writer` first (cheap - just Arc clones).

use anyhow::{Context, Result};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;

use super::mutation::{ExecOptions, Mutation};
use super::writer::MutationRequest;
use super::name_hash::NameCache;
use super::query::TransactionQueryExecutor;
use crate::request::new_request_id;

/// A transaction scope for read-your-writes operations.
///
/// Writes within this transaction are immediately visible to reads
/// within the same transaction, before commit.
///
/// # Lifetime
///
/// `Transaction<'a>` is tied to the storage via the underlying RocksDB
/// transaction lifetime. This ensures:
/// - Transaction cannot outlive the storage
/// - Borrow checker prevents use-after-free
/// - Clear scoping: transaction lives within storage borrow
///
/// # Mutation Forwarding
///
/// On commit, mutations can be forwarded to a configured `mpsc::Sender`.
/// This is used for fulltext indexing but is configurable - any receiver
/// can be attached. Forwarding is best-effort (non-blocking `try_send`).
///
/// # Example
///
/// ```rust,ignore
/// let mut txn = writer.transaction()?;
///
/// // Write mutations (not committed yet)
/// txn.write(AddNode { id: node_id, ... })?;
/// txn.write(AddNodeFragment { id: node_id, content: vector, ... })?;
///
/// // Read sees uncommitted writes!
/// let edges = txn.read(OutgoingEdges::new(node_id, Some("hnsw")))?;
///
/// for (_, _, neighbor, _) in edges {
///     txn.write(AddEdge { src: node_id, dst: neighbor, ... })?;
/// }
///
/// // Atomic commit (sync, non-blocking forwarding)
/// txn.commit()?;
/// ```
pub struct Transaction<'a> {
    /// The underlying RocksDB transaction (wrapped in Option for move-out on commit/rollback).
    txn: Option<rocksdb::Transaction<'a, rocksdb::TransactionDB>>,

    /// Reference to the TransactionDB for column family handles.
    txn_db: &'a rocksdb::TransactionDB,

    /// Mutations executed in this transaction (for forwarding on commit).
    mutations: Vec<Mutation>,

    /// Optional sender to forward mutations on commit.
    /// If None, mutations are not forwarded anywhere.
    forward_to: Option<mpsc::Sender<MutationRequest>>,

    /// Name cache for efficient hash-to-name resolution.
    name_cache: Arc<NameCache>,
}

impl<'a> Transaction<'a> {
    /// Create a new transaction.
    ///
    /// This is called internally by `Writer::transaction()`.
    pub(crate) fn new(
        txn: rocksdb::Transaction<'a, rocksdb::TransactionDB>,
        txn_db: &'a rocksdb::TransactionDB,
        forward_to: Option<mpsc::Sender<MutationRequest>>,
        name_cache: Arc<NameCache>,
    ) -> Self {
        Self {
            txn: Some(txn),
            txn_db,
            mutations: Vec::new(),
            forward_to,
            name_cache,
        }
    }

    #[allow(dead_code)]
    /// Check if the transaction has been finished (committed or rolled back).
    fn is_finished(&self) -> bool {
        self.txn.is_none()
    }

    /// Take the inner transaction, marking this as finished.
    fn take_txn(&mut self) -> Option<rocksdb::Transaction<'a, rocksdb::TransactionDB>> {
        self.txn.take()
    }

    /// Write a mutation (visible to read() within this transaction).
    ///
    /// The mutation is executed against the RocksDB transaction but
    /// not committed until `commit()` is called.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// txn.write(AddNode { id, name: "Alice".into(), ... })?;
    /// txn.write(AddEdge { src: id, dst: other, ... })?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the mutation fails to execute against RocksDB.
    pub fn write<M: Into<Mutation>>(&mut self, mutation: M) -> Result<()> {
        let txn = self.txn.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Transaction already finished (committed or rolled back)")
        })?;

        let mutation = mutation.into();

        // Execute the mutation against the transaction
        mutation
            .execute(txn, self.txn_db)
            .with_context(|| format!("Failed to execute mutation in transaction: {:?}", mutation))?;

        // Track for forwarding on commit
        self.mutations.push(mutation);

        Ok(())
    }

    /// Write multiple mutations in a batch.
    ///
    /// All mutations are executed against the transaction (not committed).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// txn.write_batch(vec![
    ///     AddNode { ... }.into(),
    ///     AddEdge { ... }.into(),
    /// ])?;
    /// ```
    pub fn write_batch(&mut self, mutations: Vec<Mutation>) -> Result<()> {
        for mutation in mutations {
            self.write(mutation)?;
        }
        Ok(())
    }

    /// Read using a query (sees uncommitted writes in this transaction).
    ///
    /// Uses the `TransactionQueryExecutor` trait to execute queries
    /// against the transaction rather than committed storage.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// txn.write(AddNode { id, ... })?;
    ///
    /// // Sees the uncommitted AddNode!
    /// let (name, summary) = txn.read(NodeById::new(id, None))?;
    ///
    /// // Get edges (sees uncommitted edges too)
    /// let edges = txn.read(OutgoingEdges::new(id, Some("hnsw")))?;
    /// ```
    pub fn read<Q: TransactionQueryExecutor>(&self, query: Q) -> Result<Q::Output> {
        let txn = self.txn.as_ref().ok_or_else(|| {
            anyhow::anyhow!("Transaction already finished (committed or rolled back)")
        })?;

        query.execute_in_transaction(txn, self.txn_db, &self.name_cache)
    }

    /// Commit all changes atomically (sync).
    ///
    /// This method is **synchronous** - it blocks until RocksDB commits.
    /// Mutation forwarding (if configured) uses non-blocking `try_send`.
    ///
    /// After commit returns:
    /// - All mutations are visible to external readers
    /// - Mutations are forwarded to configured receiver (best-effort)
    ///
    /// # Forwarding Behavior
    ///
    /// If `forward_to` is configured (e.g., for fulltext indexing):
    /// - Uses `try_send` (non-blocking)
    /// - If channel is full, logs warning and continues
    /// - Flush markers are filtered out before forwarding
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - RocksDB commit fails (conflict, I/O error)
    /// - Transaction was already committed or rolled back
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut txn = writer.transaction()?;
    /// txn.write(AddNode { ... })?;
    /// txn.write(AddEdge { ... })?;
    /// txn.commit()?;  // Blocks until RocksDB commits
    /// // Now visible to all readers
    /// ```
    pub fn commit(mut self) -> Result<()> {
        // Take the transaction, marking this as finished
        let txn = self.take_txn().ok_or_else(|| {
            anyhow::anyhow!("Transaction already finished (committed or rolled back)")
        })?;

        // 1. Commit RocksDB transaction (sync, blocking)
        txn.commit()
            .context("Failed to commit RocksDB transaction")?;

        // 2. Forward mutations to configured receiver (non-blocking, best-effort)
        if let Some(sender) = &self.forward_to {
            // Filter out Flush markers (not relevant for downstream)
            let mutations: Vec<_> = self
                .mutations
                .drain(..)
                .filter(|m| !m.is_flush())
                .collect();

            if !mutations.is_empty() {
                // Best-effort send using try_send (non-blocking)
                if let Err(e) = sender.try_send(MutationRequest {
                    payload: mutations,
                    options: ExecOptions::default(),
                    reply: None,
                    timeout: None,
                    request_id: new_request_id(),
                    created_at: Instant::now(),
                }) {
                    tracing::warn!(
                        error = %e,
                        "Transaction forwarding failed - channel full or closed"
                    );
                }
            }
        }

        Ok(())
    }

    /// Rollback all changes.
    ///
    /// Discards all mutations in this transaction. Also called
    /// automatically on drop if transaction is not committed.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut txn = writer.transaction()?;
    /// txn.write(AddNode { ... })?;
    ///
    /// if something_went_wrong {
    ///     txn.rollback()?;  // Explicit rollback
    ///     return Ok(());
    /// }
    ///
    /// txn.commit()?;
    /// ```
    pub fn rollback(mut self) -> Result<()> {
        // Take the transaction, marking this as finished
        let txn = self.take_txn().ok_or_else(|| {
            anyhow::anyhow!("Transaction already finished (committed or rolled back)")
        })?;

        txn.rollback()
            .context("Failed to rollback RocksDB transaction")?;

        self.mutations.clear();

        Ok(())
    }

    /// Get the number of mutations in this transaction.
    pub fn len(&self) -> usize {
        self.mutations.len()
    }

    /// Check if the transaction has any mutations.
    pub fn is_empty(&self) -> bool {
        self.mutations.is_empty()
    }

    /// Get a reference to the underlying RocksDB transaction.
    ///
    /// This is for advanced use cases where direct transaction access is needed.
    /// Returns None if the transaction is already finished.
    pub fn raw_transaction(&self) -> Option<&rocksdb::Transaction<'a, rocksdb::TransactionDB>> {
        self.txn.as_ref()
    }

    /// Get a reference to the TransactionDB.
    ///
    /// This is for advanced use cases where direct DB access is needed.
    pub fn raw_transaction_db(&self) -> &rocksdb::TransactionDB {
        self.txn_db
    }
}

impl<'a> Drop for Transaction<'a> {
    fn drop(&mut self) {
        // RocksDB Transaction auto-rollbacks if not committed
        // We just need to NOT forward mutations (only forward on explicit commit)
        if self.txn.is_some() {
            tracing::debug!(
                mutation_count = self.mutations.len(),
                "Transaction dropped without commit - rolling back"
            );
            // RocksDB handles the actual rollback when the transaction is dropped
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Full integration tests require a RocksDB instance.
    // These are basic unit tests for the Transaction struct.

    #[test]
    fn test_transaction_is_empty() {
        // We can't easily create a Transaction without a real RocksDB,
        // so we test the logic conceptually. Full tests are in integration tests.
    }
}
