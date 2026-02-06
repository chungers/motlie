//! Graph module - RocksDB-based graph storage.
//!
//! This module provides the graph-specific implementation for processing mutations from
//! the MPSC queue and writing them to the graph store.
//!
//! ## Module Structure
//!
//! - `mod.rs` - Storage, Graph struct, and module exports
//! - `schema.rs` - RocksDB schema definitions (column families)
//! - `mutation.rs` - Mutation types (AddNode, AddEdge, etc.)
//! - `writer.rs` - Writer infrastructure and mutation consumers
//! - `query.rs` - Query types (NodeById, EdgeSummaryBySrcDstName, etc.)
//! - `reader.rs` - Reader infrastructure and query consumers
//! - `scan.rs` - Scan API for pagination

use std::sync::Arc;

use anyhow::Result;

// Re-export CF traits from rocksdb module
pub(crate) use crate::rocksdb::{
    ColumnFamily, ColumnFamilyConfig, ColumnFamilySerde, HotColumnFamilyRecord,
};

// Submodules
pub mod mutation;
pub mod name_hash;
pub mod query;
pub mod reader;
pub mod schema;
pub mod subsystem;
pub mod summary_hash;
pub mod transaction;
pub mod writer;

/// Scan API for iterating over column families with pagination support.
pub mod scan;

/// Garbage collection for stale summary index entries and tombstones.
pub mod gc;

/// Repair module for graph index consistency checking.
pub mod repair;

#[cfg(test)]
mod tests;

// Re-export commonly used types from submodules
pub use mutation::{
    AddEdge, AddEdgeFragment, AddNode, AddNodeFragment, Mutation, MutationBatch,
    UpdateEdgeValidSinceUntil, UpdateEdgeWeight, UpdateNodeValidSinceUntil,
    // CONTENT-ADDRESS: Update/Delete mutations with optimistic locking
    UpdateNodeSummary, UpdateEdgeSummary, DeleteNode, DeleteEdge,
};
pub use crate::writer::Runnable;
pub use query::{
    EdgeFragmentsByIdTimeRange, EdgeSummaryBySrcDstName, IncomingEdges, NodeById,
    NodeFragmentsByIdTimeRange, OutgoingEdges, Query, TransactionQueryExecutor,
    // CONTENT-ADDRESS reverse lookup query types
    NodesBySummaryHash, NodeSummaryLookupResult,
    EdgesBySummaryHash, EdgeSummaryLookupResult,
};
pub use transaction::Transaction;
pub use crate::reader::Runnable as QueryRunnable;
pub use reader::{
    // Query consumer functions
    create_query_consumer,
    create_query_consumer_readwrite,
    create_query_reader,
    spawn_query_consumer,
    spawn_query_consumer_pool_readonly,
    spawn_query_consumer_pool_shared,
    spawn_query_consumer_readwrite,
    spawn_query_consumer_with_graph,
    Consumer as QueryConsumer,
    Processor as ReaderProcessor,
    QueryExecutor,
    QueryProcessor,
    QueryWithTimeout,
    Reader,
    ReaderConfig,
};
pub use name_hash::{NameCache, NameHash};
pub use summary_hash::SummaryHash;
pub use schema::{DstId, EdgeName, EdgeSummary, FragmentContent, NodeName, NodeSummary, RefCount, SrcId, Version};

// Subsystem exports for use with rocksdb::Storage<S> and StorageBuilder
pub use subsystem::{GraphBlockCacheConfig, NameCacheConfig, Subsystem};

// CONTENT-ADDRESS: Garbage collection for stale index entries
pub use gc::{GraphGarbageCollector, GraphGcConfig, GcMetrics, GcMetricsSnapshot};

// CONTENT-ADDRESS: Repair for forward/reverse edge consistency
pub use repair::{GraphRepairer, RepairConfig, RepairMetrics, RepairMetricsSnapshot};

/// Storage type alias using generic rocksdb::Storage
pub type Storage = crate::rocksdb::Storage<Subsystem>;

pub use writer::{
    // Mutation consumer functions
    create_mutation_consumer,
    create_mutation_consumer_with_next,
    create_mutation_writer,
    spawn_mutation_consumer,
    spawn_mutation_consumer_with_graph,
    spawn_mutation_consumer_with_next,
    Consumer as MutationConsumer,
    MutationExecutor,
    Processor as MutationProcessor,
    Writer,
    WriterConfig,
};

// Internal imports
use writer::Processor;



/// Trait implemented by column families that supports patching of ActivePeriod.
pub(crate) trait ActivePeriodPatchable {
    fn patch_valid_range(
        &self,
        old_value: &[u8],
        new_range: schema::ActivePeriod,
    ) -> Result<Vec<u8>, anyhow::Error>;
}

// Note: SystemInfo functionality is now in Subsystem which implements SubsystemInfo

// ============================================================================
// Graph - Mutation Processor
// ============================================================================

/// Graph-specific mutation processor
pub struct Graph {
    storage: Arc<Storage>,
}

impl Graph {
    /// Create a new GraphProcessor
    pub fn new(storage: Arc<Storage>) -> Self {
        Self { storage }
    }

    /// Get a reference to the storage.
    ///
    /// This is used internally for transaction support.
    pub fn storage(&self) -> &Arc<Storage> {
        &self.storage
    }
}

impl Clone for Graph {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(), // Arc<Storage> is already Clone
        }
    }
}

#[async_trait::async_trait]
impl Processor for Graph {
    /// Process a batch of mutations
    #[tracing::instrument(skip(self, mutations), fields(mutation_count = mutations.len()))]
    async fn process_mutations(&self, mutations: &[Mutation]) -> Result<()> {
        if mutations.is_empty() {
            return Ok(());
        }

        tracing::info!(count = mutations.len(), "[Graph] About to insert mutations");

        // Get transaction and name cache
        let txn_db = self.storage.transaction_db()?;
        let txn = txn_db.transaction();
        let name_cache = self.storage.cache();

        // Each mutation executes itself with cache access for name deduplication
        for mutation in mutations {
            mutation.execute_with_cache(&txn, txn_db, name_cache)?;
        }

        // Single commit for all mutations
        txn.commit()?;

        tracing::info!(
            count = mutations.len(),
            "[Graph] Successfully committed mutations"
        );
        Ok(())
    }
}

/// Implement query processor for Graph
impl reader::Processor for Graph {
    fn storage(&self) -> &Storage {
        &self.storage
    }
}
