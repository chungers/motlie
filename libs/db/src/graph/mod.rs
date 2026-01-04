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
use rkyv::validation::validators::DefaultValidator;
use rkyv::{Archive, CheckBytes, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};

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

#[cfg(test)]
mod tests;

// Re-export commonly used types from submodules
pub use mutation::{
    AddEdge, AddEdgeFragment, AddNode, AddNodeFragment, Mutation, MutationBatch,
    UpdateEdgeValidSinceUntil, UpdateEdgeWeight, UpdateNodeValidSinceUntil,
};
pub use crate::writer::Runnable;
pub use query::{
    EdgeFragmentsByIdTimeRange, EdgeSummaryBySrcDstName, IncomingEdges, NodeById,
    NodeFragmentsByIdTimeRange, OutgoingEdges, Query, TransactionQueryExecutor,
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
pub use schema::{DstId, EdgeName, EdgeSummary, FragmentContent, NodeName, NodeSummary, SrcId};

// Subsystem exports for use with rocksdb::Storage<S> and StorageBuilder
pub use subsystem::{GraphBlockCacheConfig, NameCacheConfig, Subsystem};

/// Storage type alias using generic rocksdb::Storage
pub type Storage = crate::rocksdb::Storage<Subsystem>;

/// BlockCacheConfig re-export (alias to GraphBlockCacheConfig for backwards compat)
pub type BlockCacheConfig = GraphBlockCacheConfig;

/// Component type for use with StorageBuilder
pub type Component = crate::rocksdb::ComponentWrapper<Subsystem>;

/// Convenience constructor for graph component
pub fn component() -> Component {
    Component::new()
}
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
use schema::ALL_COLUMN_FAMILIES;
use writer::Processor;

/// Trait for column family record types that can create and serialize key-value pairs.
///
/// This trait uses direct byte concatenation for keys (to enable RocksDB prefix extractors)
/// and MessagePack for values (where self-describing format is beneficial).
pub(crate) trait ColumnFamilyRecord {
    const CF_NAME: &'static str;

    /// The key type for this column family
    type Key: Serialize + for<'de> Deserialize<'de>;

    /// The value type for this column family
    type Value: Serialize + for<'de> Deserialize<'de>;

    /// The argument type for creating records
    type CreateOp;

    /// Create a key-value pair from arguments
    fn record_from(args: &Self::CreateOp) -> (Self::Key, Self::Value);

    /// Serialize the key to bytes using direct concatenation (no MessagePack).
    /// This enables constant-length prefixes for RocksDB prefix extractors.
    fn key_to_bytes(key: &Self::Key) -> Vec<u8>;

    /// Deserialize the key from bytes (direct format, no MessagePack).
    fn key_from_bytes(bytes: &[u8]) -> Result<Self::Key, anyhow::Error>;

    /// Serialize the value to bytes using MessagePack, then compress with LZ4.
    fn value_to_bytes(value: &Self::Value) -> Result<Vec<u8>, rmp_serde::encode::Error> {
        let msgpack_bytes = rmp_serde::to_vec(value)?;
        let compressed = lz4::block::compress(&msgpack_bytes, None, true).map_err(|e| {
            rmp_serde::encode::Error::Syntax(format!("LZ4 compression failed: {}", e))
        })?;
        Ok(compressed)
    }

    /// Decompress with LZ4, then deserialize the value from bytes using MessagePack.
    fn value_from_bytes(bytes: &[u8]) -> Result<Self::Value, rmp_serde::decode::Error> {
        let decompressed = lz4::block::decompress(bytes, None).map_err(|e| {
            rmp_serde::decode::Error::Syntax(format!("LZ4 decompression failed: {}", e))
        })?;
        rmp_serde::from_slice(&decompressed)
    }

    /// Create and serialize to bytes using direct encoding for keys, compressed MessagePack for values.
    fn create_bytes(args: &Self::CreateOp) -> Result<(Vec<u8>, Vec<u8>), rmp_serde::encode::Error> {
        let (key, value) = Self::record_from(args);
        let key_bytes = Self::key_to_bytes(&key);
        let value_bytes = Self::value_to_bytes(&value)?;
        Ok((key_bytes, value_bytes))
    }

    /// Configure RocksDB options for this column family.
    /// Each implementer specifies their own prefix extractor and bloom filter settings.
    fn column_family_options() -> rocksdb::Options {
        rocksdb::Options::default()
    }
}

// ============================================================================
// Hot Column Family Record Trait (rkyv zero-copy serialization)
// ============================================================================

/// Trait for hot column families using rkyv (zero-copy serialization).
///
/// Hot CFs store small, frequently-accessed data (topology, weights, temporal ranges).
/// Values are serialized with rkyv for zero-copy access during graph traversal.
///
/// # Example
///
/// ```rust,ignore
/// // Zero-copy access (hot path)
/// let archived = Nodes::value_archived(&value_bytes)?;
/// if archived.temporal_range.as_ref().map_or(true, |tr| tr.is_valid_at(now)) {
///     // Use archived data directly without allocation
/// }
///
/// // Full deserialization when ownership is needed (cold path)
/// let value: NodeCfValue = Nodes::value_from_bytes(&value_bytes)?;
/// ```
pub(crate) trait HotColumnFamilyRecord {
    /// Column family name (constant for each implementor)
    const CF_NAME: &'static str;

    /// The key type for this column family
    type Key;

    /// The value type for this column family (must implement rkyv traits)
    type Value: Archive + RkyvSerialize<rkyv::ser::serializers::AllocSerializer<256>>;

    /// Serialize key to bytes.
    fn key_to_bytes(key: &Self::Key) -> Vec<u8>;

    /// Deserialize key from bytes.
    fn key_from_bytes(bytes: &[u8]) -> Result<Self::Key>;

    /// Zero-copy value access - returns archived reference without allocation.
    ///
    /// This is the hot path for graph traversal. The returned reference
    /// is valid as long as the input bytes are valid.
    fn value_archived(bytes: &[u8]) -> Result<&<Self::Value as Archive>::Archived>
    where
        <Self::Value as Archive>::Archived: for<'a> CheckBytes<DefaultValidator<'a>>,
    {
        rkyv::check_archived_root::<Self::Value>(bytes)
            .map_err(|e| anyhow::anyhow!("rkyv validation failed: {}", e))
    }

    /// Full deserialization when mutation/ownership is needed.
    ///
    /// This allocates and copies data. Use sparingly - prefer value_archived()
    /// for read-only access.
    fn value_from_bytes(bytes: &[u8]) -> Result<Self::Value>
    where
        <Self::Value as Archive>::Archived: for<'a> CheckBytes<DefaultValidator<'a>>,
        <Self::Value as Archive>::Archived: RkyvDeserialize<Self::Value, rkyv::Infallible>,
    {
        let archived = Self::value_archived(bytes)?;
        Ok(archived.deserialize(&mut rkyv::Infallible).expect("Infallible"))
    }

    /// Serialize value to bytes using rkyv.
    fn value_to_bytes(value: &Self::Value) -> Result<rkyv::AlignedVec> {
        rkyv::to_bytes::<_, 256>(value)
            .map_err(|e| anyhow::anyhow!("rkyv serialization failed: {}", e))
    }
}

/// Trait implemented by column families that supports patching of TemporalRange.
pub(crate) trait ValidRangePatchable {
    fn patch_valid_range(
        &self,
        old_value: &[u8],
        new_range: schema::TemporalRange,
    ) -> Result<Vec<u8>, anyhow::Error>;
}

// ============================================================================
// SystemInfo - Telemetry
// ============================================================================

/// Static configuration info for the graph database subsystem.
///
/// Used by the `motlie info` command to display graph DB settings.
/// Implements [`motlie_core::telemetry::SubsystemInfo`] for consistent formatting.
///
/// # Example
///
/// ```ignore
/// use motlie_db::graph::SystemInfo;
/// use motlie_core::telemetry::{format_subsystem_info, SubsystemInfo};
///
/// let info = SystemInfo::default();
/// println!("{}", format_subsystem_info(&info));
/// ```
#[derive(Debug, Clone)]
pub struct SystemInfo {
    /// Block cache configuration
    pub block_cache_config: BlockCacheConfig,
    /// Name cache configuration
    pub name_cache_config: NameCacheConfig,
    /// List of column families
    pub column_families: Vec<&'static str>,
}

impl Default for SystemInfo {
    fn default() -> Self {
        Self {
            block_cache_config: BlockCacheConfig::default(),
            name_cache_config: NameCacheConfig::default(),
            column_families: ALL_COLUMN_FAMILIES.to_vec(),
        }
    }
}

impl motlie_core::telemetry::SubsystemInfo for SystemInfo {
    fn name(&self) -> &'static str {
        "Graph Database (RocksDB)"
    }

    fn info_lines(&self) -> Vec<(&'static str, String)> {
        vec![
            ("Block Cache Size", format_bytes(self.block_cache_config.cache_size_bytes)),
            ("Graph Block Size", format_bytes(self.block_cache_config.graph_block_size)),
            ("Fragment Block Size", format_bytes(self.block_cache_config.fragment_block_size)),
            ("Cache Index/Filter", self.block_cache_config.cache_index_and_filter_blocks.to_string()),
            ("Pin L0 Blocks", self.block_cache_config.pin_l0_filter_and_index.to_string()),
            ("Name Cache Prewarm", self.name_cache_config.prewarm_limit.to_string()),
            ("Column Families", self.column_families.join(", ")),
        ]
    }
}

/// Format a byte count as a human-readable string.
fn format_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{} GB", bytes / (1024 * 1024 * 1024))
    } else if bytes >= 1024 * 1024 {
        format!("{} MB", bytes / (1024 * 1024))
    } else if bytes >= 1024 {
        format!("{} KB", bytes / 1024)
    } else {
        format!("{} B", bytes)
    }
}

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
