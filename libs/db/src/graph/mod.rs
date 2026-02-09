//! Graph module - RocksDB-based graph storage.
//!
//! This module provides the graph-specific implementation for processing mutations from
//! the MPSC queue and writing them to the graph store.
//!
//! ## Module Structure
//!
//! - `mod.rs` - Storage, Processor, and module exports
//! - `processor.rs` - Processor struct (central state hub with Storage + NameCache)
// (claude, 2026-02-07, FIXED: Updated header to reflect Graph→Processor migration per codex eval)
//! - `ops/` - Business logic helpers (single source of truth for mutations/queries)
//! - `schema.rs` - RocksDB schema definitions (column families)
//! - `mutation.rs` - Mutation types (AddNode, AddEdge, etc.)
//! - `writer.rs` - Writer infrastructure and mutation consumers
//! - `query.rs` - Query types (NodeById, EdgeSummaryBySrcDstName, etc.)
//! - `reader.rs` - Reader infrastructure and query consumers
//! - `scan.rs` - Scan API for pagination


// Re-export CF traits from rocksdb module
pub(crate) use crate::rocksdb::{
    ColumnFamily, ColumnFamilyConfig, ColumnFamilySerde, HotColumnFamilyRecord,
};

// Submodules
pub mod mutation;
pub mod name_hash;
pub(crate) mod ops;
pub mod processor;
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
    AddEdge, AddEdgeFragment, AddNode, AddNodeFragment, ExecOptions, Mutation,
    MutationResult, RunnableWithResult,
    // CONTENT-ADDRESS: Update/Delete mutations with optimistic locking
    UpdateNode, UpdateEdge, DeleteNode, DeleteEdge,
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
    create_reader_with_storage,
    spawn_query_consumers_with_storage,
    spawn_consumer_pool_with_processor,
    spawn_query_consumer,
    spawn_query_consumer_pool_shared,
    spawn_query_consumer_pool_readonly,
    spawn_query_consumer_with_processor,
    Consumer as QueryConsumer,
    QueryExecutor,
    Reader,
    ReaderConfig,
};
pub use name_hash::{NameCache, NameHash};
pub use summary_hash::SummaryHash;
pub use schema::{DstId, EdgeName, EdgeSummary, EdgeWeight, FragmentContent, NodeName, NodeSummary, RefCount, SrcId, Version};

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
    create_mutation_writer,
    spawn_mutation_consumer_with_storage,
    spawn_mutation_consumer,
    spawn_mutation_consumer_with_next,
    spawn_mutation_consumer_with_receiver,
    Consumer as MutationConsumer,
    MutationExecutor,
    Processor as MutationProcessor,
    Writer,
    WriterConfig,
};

// Processor struct - the central graph processing hub
pub use processor::Processor;



// Note: SystemInfo functionality is now in Subsystem which implements SubsystemInfo
// Graph struct removed - use processor::Processor instead (claude, 2026-02-07)
