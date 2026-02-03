//! Mutation module providing mutation types and their business logic implementations.
//!
//! This module contains only business logic - mutation type definitions and their
//! MutationExecutor implementations. Infrastructure (traits, Writer, Consumer, spawn
//! functions) is in the `writer` module.

use std::sync::Mutex;

use anyhow::Result;
use tokio::sync::oneshot;

use crate::rocksdb::MutationCodec;

use super::name_hash::NameHash;
use super::schema::{
    self, EdgeFragmentCfKey, EdgeFragmentCfValue, EdgeFragments, NodeFragmentCfKey,
    NodeFragmentCfValue, NodeFragments,
};
use super::writer::{MutationExecutor, Writer};
use super::{ColumnFamily, ColumnFamilySerde, HotColumnFamilyRecord};
use crate::writer::Runnable;
use crate::{Id, TimestampMilli};

// ============================================================================
// Flush Marker
// ============================================================================

/// Marker for flush synchronization.
///
/// Contains a oneshot sender that signals when the flush completes.
/// Uses `Mutex<Option<...>>` to allow taking ownership from a shared reference,
/// since `process_batch` receives `&[Mutation]`.
///
/// # Usage
///
/// ```rust,ignore
/// let (tx, rx) = oneshot::channel();
/// let marker = FlushMarker::new(tx);
///
/// // Later, in consumer:
/// if let Some(completion) = marker.take_completion() {
///     completion.send(()).ok();
/// }
/// ```
pub struct FlushMarker {
    completion: Mutex<Option<oneshot::Sender<()>>>,
}

impl FlushMarker {
    /// Create a new flush marker with completion channel.
    pub fn new(completion: oneshot::Sender<()>) -> Self {
        Self {
            completion: Mutex::new(Some(completion)),
        }
    }

    /// Take the completion sender (can only be called once).
    ///
    /// Returns `None` if already taken or if the mutex is poisoned.
    pub fn take_completion(&self) -> Option<oneshot::Sender<()>> {
        self.completion.lock().ok()?.take()
    }
}

impl std::fmt::Debug for FlushMarker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let has_completion = self
            .completion
            .lock()
            .map(|guard| guard.is_some())
            .unwrap_or(false);
        f.debug_struct("FlushMarker")
            .field("has_completion", &has_completion)
            .finish()
    }
}

// FlushMarker cannot be Clone since oneshot::Sender is not Clone
// We implement a manual Clone that creates an "empty" marker
impl Clone for FlushMarker {
    fn clone(&self) -> Self {
        // Cloning a FlushMarker creates an empty one (no completion channel)
        // This is intentional - only the original can signal completion
        Self {
            completion: Mutex::new(None),
        }
    }
}

// ============================================================================
// Mutation Enum
// ============================================================================

#[derive(Debug, Clone)]
pub enum Mutation {
    AddNode(AddNode),
    AddEdge(AddEdge),
    AddNodeFragment(AddNodeFragment),
    AddEdgeFragment(AddEdgeFragment),
    UpdateNodeValidSinceUntil(UpdateNodeValidSinceUntil),
    UpdateEdgeValidSinceUntil(UpdateEdgeValidSinceUntil),
    UpdateEdgeWeight(UpdateEdgeWeight),
    // CONTENT-ADDRESS: Update/Delete with optimistic locking
    UpdateNodeSummary(UpdateNodeSummary),
    UpdateEdgeSummary(UpdateEdgeSummary),
    DeleteNode(DeleteNode),
    DeleteEdge(DeleteEdge),

    /// Flush marker for synchronization.
    ///
    /// When the consumer processes this mutation, it signals completion
    /// via the oneshot channel. This is used by `Writer::flush()` to wait
    /// for all pending mutations to be committed.
    ///
    /// This mutation is NOT persisted to storage - it only serves as a
    /// synchronization mechanism.
    Flush(FlushMarker),
}

#[derive(Debug, Clone)]
pub struct AddNode {
    /// The UUID of the Node
    pub id: Id,

    /// The timestamp as number of milliseconds since the Unix epoch
    pub ts_millis: TimestampMilli,

    /// The name of the Node
    pub name: schema::NodeName,

    /// The temporal validity range for this node
    pub valid_range: Option<schema::TemporalRange>,

    /// The summary information for this node
    pub summary: schema::NodeSummary,
}

#[derive(Debug, Clone)]
pub struct AddEdge {
    /// The UUID of the source Node
    pub source_node_id: Id,

    /// The UUID of the target Node
    pub target_node_id: Id,

    /// The timestamp as number of milliseconds since the Unix epoch
    pub ts_millis: TimestampMilli,

    /// The name of the Edge
    pub name: schema::EdgeName,

    /// The temporal validity range for this edge
    pub valid_range: Option<schema::TemporalRange>,

    /// The summary information for this edge (moved from Edges CF)
    pub summary: schema::EdgeSummary,

    /// Optional weight for weighted graph algorithms
    pub weight: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct AddNodeFragment {
    /// The UUID of the Node this fragment belongs to
    pub id: Id,

    /// The timestamp as number of milliseconds since the Unix epoch
    pub ts_millis: TimestampMilli,

    /// The body of the Fragment
    pub content: crate::DataUrl,

    /// The temporal validity range for this fragment
    pub valid_range: Option<schema::TemporalRange>,
}

#[derive(Debug, Clone)]
pub struct AddEdgeFragment {
    /// The UUID of the source Node
    pub src_id: Id,

    /// The UUID of the destination Node
    pub dst_id: Id,

    /// The name of the Edge
    pub edge_name: schema::EdgeName,

    /// The timestamp as number of milliseconds since the Unix epoch
    pub ts_millis: TimestampMilli,

    /// The body of the Fragment
    pub content: crate::DataUrl,

    /// The temporal validity range for this fragment
    pub valid_range: Option<schema::TemporalRange>,
}

#[derive(Debug, Clone)]
pub struct UpdateNodeValidSinceUntil {
    /// The UUID of the Node
    pub id: Id,

    /// The temporal validity range for this fragment
    pub temporal_range: schema::TemporalRange,

    /// The reason for invalidation
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct UpdateEdgeValidSinceUntil {
    /// The UUID of the source Node
    pub src_id: Id,

    /// The UUID of the destination Node
    pub dst_id: Id,

    /// The name of the Edge
    pub name: schema::EdgeName,

    /// The temporal validity range for this edge
    pub temporal_range: schema::TemporalRange,

    /// The reason for invalidation
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct UpdateEdgeWeight {
    /// The UUID of the source Node
    pub src_id: Id,

    /// The UUID of the destination Node
    pub dst_id: Id,

    /// The name of the Edge
    pub name: schema::EdgeName,

    /// The new weight value
    pub weight: f64,
}

// ============================================================================
// CONTENT-ADDRESS: Update/Delete Mutations with Optimistic Locking
// (claude, 2026-02-02, implementing)
// ============================================================================

/// Update a node's summary with optimistic locking.
///
/// This mutation:
/// 1. Reads the current node to get version and old summary hash
/// 2. Verifies version matches expected_version (optimistic lock check)
/// 3. Increments version
/// 4. Writes new summary to NodeSummaries CF
/// 5. Flips old index entry to STALE
/// 6. Writes new index entry with CURRENT marker
/// 7. Updates Nodes CF with new version and summary hash
#[derive(Debug, Clone)]
pub struct UpdateNodeSummary {
    /// The UUID of the Node to update
    pub id: Id,

    /// The new summary content
    pub new_summary: schema::NodeSummary,

    /// Expected version for optimistic locking
    /// If the current version doesn't match, the update fails
    pub expected_version: schema::Version,
}

/// Update an edge's summary with optimistic locking.
#[derive(Debug, Clone)]
pub struct UpdateEdgeSummary {
    /// The UUID of the source Node
    pub src_id: Id,

    /// The UUID of the destination Node
    pub dst_id: Id,

    /// The name of the Edge
    pub name: schema::EdgeName,

    /// The new summary content
    pub new_summary: schema::EdgeSummary,

    /// Expected version for optimistic locking
    pub expected_version: schema::Version,
}

/// Delete a node with tombstone semantics.
///
/// This mutation:
/// 1. Reads the current node to verify version
/// 2. Sets deleted=true flag (tombstone)
/// 3. Increments version
/// 4. Flips current index entry to STALE
///
/// The node remains in the database for audit/time-travel until GC.
#[derive(Debug, Clone)]
pub struct DeleteNode {
    /// The UUID of the Node to delete
    pub id: Id,

    /// Expected version for optimistic locking
    pub expected_version: schema::Version,
}

/// Delete an edge with tombstone semantics.
#[derive(Debug, Clone)]
pub struct DeleteEdge {
    /// The UUID of the source Node
    pub src_id: Id,

    /// The UUID of the destination Node
    pub dst_id: Id,

    /// The name of the Edge
    pub name: schema::EdgeName,

    /// Expected version for optimistic locking
    pub expected_version: schema::Version,
}

// ============================================================================
// MutationCodec Implementations
// ============================================================================

impl MutationCodec for AddNodeFragment {
    type Cf = NodeFragments;

    fn to_record(&self) -> (NodeFragmentCfKey, NodeFragmentCfValue) {
        let key = NodeFragmentCfKey(self.id, self.ts_millis);
        let value = NodeFragmentCfValue(self.valid_range.clone(), self.content.clone());
        (key, value)
    }
}

impl MutationCodec for AddEdgeFragment {
    type Cf = EdgeFragments;

    fn to_record(&self) -> (EdgeFragmentCfKey, EdgeFragmentCfValue) {
        let name_hash = NameHash::from_name(&self.edge_name);
        let key = EdgeFragmentCfKey(self.src_id, self.dst_id, name_hash, self.ts_millis);
        let value = EdgeFragmentCfValue(self.valid_range.clone(), self.content.clone());
        (key, value)
    }
}

// ============================================================================
// Helper Functions - Shared logic for mutation execution
// ============================================================================

/// Helper function to write a name to the Names CF.
///
/// This is idempotent: writing the same name twice has no effect since
/// the same hash always maps to the same name.
fn write_name_to_cf(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    name: &str,
) -> Result<NameHash> {
    use super::schema::{Names, NameCfKey, NameCfValue};

    let name_hash = NameHash::from_name(name);

    let names_cf = txn_db
        .cf_handle(Names::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("Column family '{}' not found", Names::CF_NAME))?;

    let key_bytes = Names::key_to_bytes(&NameCfKey(name_hash));
    let value_bytes = Names::value_to_bytes(&NameCfValue(name.to_string()))?;

    txn.put_cf(names_cf, key_bytes, value_bytes)?;

    Ok(name_hash)
}

/// Helper function to write a name to the Names CF with cache optimization.
///
/// Uses the cache to:
/// 1. Check if the name is already interned (skip DB write)
/// 2. Intern new names for future lookups
///
/// Returns the NameHash and whether a DB write was performed.
fn write_name_to_cf_cached(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    name: &str,
    cache: &super::name_hash::NameCache,
) -> Result<NameHash> {
    use super::schema::{Names, NameCfKey, NameCfValue};

    // Check cache first - if already interned, skip DB write
    let (name_hash, is_new) = cache.intern_if_new(name);

    if is_new {
        // New name - write to Names CF
        let names_cf = txn_db
            .cf_handle(Names::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("Column family '{}' not found", Names::CF_NAME))?;

        let key_bytes = Names::key_to_bytes(&NameCfKey(name_hash));
        let value_bytes = Names::value_to_bytes(&NameCfValue(name.to_string()))?;

        txn.put_cf(names_cf, key_bytes, value_bytes)?;
        tracing::trace!(name = %name, "Wrote new name to Names CF");
    } else {
        tracing::trace!(name = %name, "Name already cached, skipping DB write");
    }

    Ok(name_hash)
}

/// Helper function to update TemporalRange for a single node.
/// Updates the Nodes CF.
fn update_node_valid_range(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    node_id: Id,
    new_range: schema::TemporalRange,
) -> Result<()> {
    use super::ValidRangePatchable;
    use super::schema::{NodeCfKey, Nodes};

    // Patch Nodes CF
    let nodes_cf = txn_db
        .cf_handle(Nodes::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("Nodes CF not found"))?;

    let node_key = NodeCfKey(node_id);
    let node_key_bytes = Nodes::key_to_bytes(&node_key);

    let node_value_bytes = txn
        .get_cf(nodes_cf, &node_key_bytes)?
        .ok_or_else(|| anyhow::anyhow!("Node not found for id: {}", node_id))?;

    let nodes = Nodes;
    let patched_node_bytes = nodes.patch_valid_range(&node_value_bytes, new_range)?;
    txn.put_cf(nodes_cf, &node_key_bytes, patched_node_bytes)?;

    Ok(())
}

/// Helper function to update TemporalRange for a single edge in ForwardEdges and ReverseEdges CFs.
/// This is the core logic shared by UpdateEdgeValidSinceUntil and UpdateNodeValidSinceUntil.
///
/// Note: This function accepts a NameHash directly. The caller is responsible for
/// computing the hash from the edge name string.
fn update_edge_valid_range(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    src_id: Id,
    dst_id: Id,
    name_hash: NameHash,
    new_range: schema::TemporalRange,
) -> Result<()> {
    use super::ValidRangePatchable;
    use super::schema::{ForwardEdgeCfKey, ForwardEdges, ReverseEdgeCfKey, ReverseEdges};

    // Patch ForwardEdges CF
    let forward_cf = txn_db
        .cf_handle(ForwardEdges::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("ForwardEdges CF not found"))?;

    let forward_key = ForwardEdgeCfKey(src_id, dst_id, name_hash);
    let forward_key_bytes = ForwardEdges::key_to_bytes(&forward_key);

    let forward_value_bytes = txn.get_cf(forward_cf, &forward_key_bytes)?.ok_or_else(|| {
        anyhow::anyhow!(
            "ForwardEdge not found: src={}, dst={}, name_hash={}",
            src_id,
            dst_id,
            name_hash
        )
    })?;

    let forward_edges = ForwardEdges;
    let patched_forward_bytes = forward_edges.patch_valid_range(&forward_value_bytes, new_range)?;
    txn.put_cf(forward_cf, &forward_key_bytes, patched_forward_bytes)?;

    // Patch ReverseEdges CF
    let reverse_cf = txn_db
        .cf_handle(ReverseEdges::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("ReverseEdges CF not found"))?;

    let reverse_key = ReverseEdgeCfKey(dst_id, src_id, name_hash);
    let reverse_key_bytes = ReverseEdges::key_to_bytes(&reverse_key);

    let reverse_value_bytes = txn.get_cf(reverse_cf, &reverse_key_bytes)?.ok_or_else(|| {
        anyhow::anyhow!(
            "ReverseEdge not found: src={}, dst={}, name_hash={}",
            src_id,
            dst_id,
            name_hash
        )
    })?;

    let reverse_edges = ReverseEdges;
    let patched_reverse_bytes = reverse_edges.patch_valid_range(&reverse_value_bytes, new_range)?;
    txn.put_cf(reverse_cf, &reverse_key_bytes, patched_reverse_bytes)?;

    Ok(())
}

/// Helper function to find all edges connected to a node.
/// Returns a deduplicated list of (src_id, dst_id, name_hash) tuples.
fn find_connected_edges(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    node_id: Id,
) -> Result<Vec<(Id, Id, NameHash)>> {

    use super::schema::{ForwardEdgeCfKey, ForwardEdges, ReverseEdgeCfKey, ReverseEdges};

    let mut edge_topologies = Vec::new();

    // Find outgoing edges (where this node is the source)
    let forward_cf = txn_db
        .cf_handle(ForwardEdges::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("ForwardEdges CF not found"))?;
    let forward_prefix = node_id.into_bytes().to_vec();
    let forward_iter = txn.prefix_iterator_cf(forward_cf, &forward_prefix);

    for item in forward_iter {
        let (key_bytes, _) = item?;
        let key: ForwardEdgeCfKey = ForwardEdges::key_from_bytes(&key_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize ForwardEdge key: {}", e))?;
        edge_topologies.push((key.0, key.1, key.2));
    }

    // Find incoming edges (where this node is the destination)
    let reverse_cf = txn_db
        .cf_handle(ReverseEdges::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("ReverseEdges CF not found"))?;
    let reverse_prefix = node_id.into_bytes().to_vec();
    let reverse_iter = txn.prefix_iterator_cf(reverse_cf, &reverse_prefix);

    for item in reverse_iter {
        let (key_bytes, _) = item?;
        let key: ReverseEdgeCfKey = ReverseEdges::key_from_bytes(&key_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize ReverseEdge key: {}", e))?;
        // ReverseEdgeCfKey is (dst_id, src_id, name_hash), extract as (src_id, dst_id, name_hash)
        edge_topologies.push((key.1, key.0, key.2));
    }

    // Deduplicate edges
    let unique_edges: std::collections::HashSet<_> = edge_topologies.into_iter().collect();
    Ok(unique_edges.into_iter().collect())
}

// ============================================================================
// MutationExecutor Implementations
// ============================================================================

impl MutationExecutor for AddNode {
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
    ) -> Result<()> {
        tracing::debug!(id = %self.id, name = %self.name, "Executing AddNode mutation");

        use super::schema::{
            Nodes, NodeSummaries, NodeSummaryCfKey, NodeSummaryCfValue,
            NodeSummaryIndex, NodeSummaryIndexCfKey, NodeSummaryIndexCfValue,
        };
        use super::summary_hash::SummaryHash;

        // Write name to Names CF (idempotent)
        write_name_to_cf(txn, txn_db, &self.name)?;

        // Write summary to NodeSummaries CF (cold) if non-empty
        // Also write reverse index entry with CURRENT marker
        // (claude, 2026-02-02, in-progress) CONTENT-ADDRESS index writes
        if !self.summary.is_empty() {
            if let Ok(summary_hash) = SummaryHash::from_summary(&self.summary) {
                let summaries_cf = txn_db
                    .cf_handle(NodeSummaries::CF_NAME)
                    .ok_or_else(|| anyhow::anyhow!("Column family '{}' not found", NodeSummaries::CF_NAME))?;
                let summary_key = NodeSummaries::key_to_bytes(&NodeSummaryCfKey(summary_hash));
                let summary_value = NodeSummaries::value_to_bytes(&NodeSummaryCfValue(self.summary.clone()))?;
                // Content-addressable: same content = same hash = idempotent write
                txn.put_cf(summaries_cf, summary_key, summary_value)?;

                // Write reverse index entry with CURRENT marker (version=1 for new nodes)
                let index_cf = txn_db
                    .cf_handle(NodeSummaryIndex::CF_NAME)
                    .ok_or_else(|| anyhow::anyhow!("Column family '{}' not found", NodeSummaryIndex::CF_NAME))?;
                let index_key = NodeSummaryIndexCfKey(summary_hash, self.id, 1);
                let index_key_bytes = NodeSummaryIndex::key_to_bytes(&index_key);
                let index_value_bytes = NodeSummaryIndex::value_to_bytes(&NodeSummaryIndexCfValue::current())?;
                txn.put_cf(index_cf, index_key_bytes, index_value_bytes)?;
            }
        }

        // Write to Nodes CF (hot)
        let nodes_cf = txn_db
            .cf_handle(Nodes::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("Column family '{}' not found", Nodes::CF_NAME))?;
        let (node_key, node_value) = Nodes::create_bytes(self)?;
        txn.put_cf(nodes_cf, node_key, node_value)?;

        Ok(())
    }

    fn execute_with_cache(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        cache: &super::name_hash::NameCache,
    ) -> Result<()> {
        tracing::debug!(id = %self.id, name = %self.name, "Executing AddNode mutation (cached)");

        use super::schema::{
            Nodes, NodeSummaries, NodeSummaryCfKey, NodeSummaryCfValue,
            NodeSummaryIndex, NodeSummaryIndexCfKey, NodeSummaryIndexCfValue,
        };
        use super::summary_hash::SummaryHash;

        // Write name to Names CF with cache optimization
        write_name_to_cf_cached(txn, txn_db, &self.name, cache)?;

        // Write summary to NodeSummaries CF (cold) if non-empty
        // Also write reverse index entry with CURRENT marker
        // (claude, 2026-02-02, in-progress) CONTENT-ADDRESS index writes
        if !self.summary.is_empty() {
            if let Ok(summary_hash) = SummaryHash::from_summary(&self.summary) {
                let summaries_cf = txn_db
                    .cf_handle(NodeSummaries::CF_NAME)
                    .ok_or_else(|| anyhow::anyhow!("Column family '{}' not found", NodeSummaries::CF_NAME))?;
                let summary_key = NodeSummaries::key_to_bytes(&NodeSummaryCfKey(summary_hash));
                let summary_value = NodeSummaries::value_to_bytes(&NodeSummaryCfValue(self.summary.clone()))?;
                txn.put_cf(summaries_cf, summary_key, summary_value)?;

                // Write reverse index entry with CURRENT marker (version=1 for new nodes)
                let index_cf = txn_db
                    .cf_handle(NodeSummaryIndex::CF_NAME)
                    .ok_or_else(|| anyhow::anyhow!("Column family '{}' not found", NodeSummaryIndex::CF_NAME))?;
                let index_key = NodeSummaryIndexCfKey(summary_hash, self.id, 1);
                let index_key_bytes = NodeSummaryIndex::key_to_bytes(&index_key);
                let index_value_bytes = NodeSummaryIndex::value_to_bytes(&NodeSummaryIndexCfValue::current())?;
                txn.put_cf(index_cf, index_key_bytes, index_value_bytes)?;
            }
        }

        // Write to Nodes CF (hot)
        let nodes_cf = txn_db
            .cf_handle(Nodes::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("Column family '{}' not found", Nodes::CF_NAME))?;
        let (node_key, node_value) = Nodes::create_bytes(self)?;
        txn.put_cf(nodes_cf, node_key, node_value)?;

        Ok(())
    }
}

impl MutationExecutor for AddEdge {
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
    ) -> Result<()> {
        tracing::debug!(
            src = %self.source_node_id,
            dst = %self.target_node_id,
            name = %self.name,
            "Executing AddEdge mutation"
        );

        use super::schema::{
            ForwardEdges, ReverseEdges, EdgeSummaries, EdgeSummaryCfKey, EdgeSummaryCfValue,
            EdgeSummaryIndex, EdgeSummaryIndexCfKey, EdgeSummaryIndexCfValue,
        };
        use super::summary_hash::SummaryHash;

        // Write name to Names CF (idempotent)
        let name_hash = write_name_to_cf(txn, txn_db, &self.name)?;

        // Write summary to EdgeSummaries CF (cold) if non-empty
        // Also write reverse index entry with CURRENT marker
        // (claude, 2026-02-02, in-progress) CONTENT-ADDRESS index writes
        if !self.summary.is_empty() {
            if let Ok(summary_hash) = SummaryHash::from_summary(&self.summary) {
                let summaries_cf = txn_db
                    .cf_handle(EdgeSummaries::CF_NAME)
                    .ok_or_else(|| anyhow::anyhow!("Column family '{}' not found", EdgeSummaries::CF_NAME))?;
                let summary_key = EdgeSummaries::key_to_bytes(&EdgeSummaryCfKey(summary_hash));
                let summary_value = EdgeSummaries::value_to_bytes(&EdgeSummaryCfValue(self.summary.clone()))?;
                // Content-addressable: same content = same hash = idempotent write
                txn.put_cf(summaries_cf, summary_key, summary_value)?;

                // Write reverse index entry with CURRENT marker (version=1 for new edges)
                let index_cf = txn_db
                    .cf_handle(EdgeSummaryIndex::CF_NAME)
                    .ok_or_else(|| anyhow::anyhow!("Column family '{}' not found", EdgeSummaryIndex::CF_NAME))?;
                let index_key = EdgeSummaryIndexCfKey(
                    summary_hash,
                    self.source_node_id,
                    self.target_node_id,
                    name_hash,
                    1, // version=1 for new edges
                );
                let index_key_bytes = EdgeSummaryIndex::key_to_bytes(&index_key);
                let index_value_bytes = EdgeSummaryIndex::value_to_bytes(&EdgeSummaryIndexCfValue::current())?;
                txn.put_cf(index_cf, index_key_bytes, index_value_bytes)?;
            }
        }

        // Write to ForwardEdges CF (hot)
        let forward_cf = txn_db.cf_handle(ForwardEdges::CF_NAME).ok_or_else(|| {
            anyhow::anyhow!("Column family '{}' not found", ForwardEdges::CF_NAME)
        })?;
        let (forward_key, forward_value) = ForwardEdges::create_bytes(self)?;
        txn.put_cf(forward_cf, forward_key, forward_value)?;

        // Write to ReverseEdges CF (hot)
        let reverse_cf = txn_db.cf_handle(ReverseEdges::CF_NAME).ok_or_else(|| {
            anyhow::anyhow!("Column family '{}' not found", ReverseEdges::CF_NAME)
        })?;
        let (reverse_key, reverse_value) = ReverseEdges::create_bytes(self)?;
        txn.put_cf(reverse_cf, reverse_key, reverse_value)?;

        Ok(())
    }

    fn execute_with_cache(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        cache: &super::name_hash::NameCache,
    ) -> Result<()> {
        tracing::debug!(
            src = %self.source_node_id,
            dst = %self.target_node_id,
            name = %self.name,
            "Executing AddEdge mutation (cached)"
        );

        use super::schema::{
            ForwardEdges, ReverseEdges, EdgeSummaries, EdgeSummaryCfKey, EdgeSummaryCfValue,
            EdgeSummaryIndex, EdgeSummaryIndexCfKey, EdgeSummaryIndexCfValue,
        };
        use super::summary_hash::SummaryHash;

        // Write name to Names CF with cache optimization
        let name_hash = write_name_to_cf_cached(txn, txn_db, &self.name, cache)?;

        // Write summary to EdgeSummaries CF (cold) if non-empty
        // Also write reverse index entry with CURRENT marker
        // (claude, 2026-02-02, in-progress) CONTENT-ADDRESS index writes
        if !self.summary.is_empty() {
            if let Ok(summary_hash) = SummaryHash::from_summary(&self.summary) {
                let summaries_cf = txn_db
                    .cf_handle(EdgeSummaries::CF_NAME)
                    .ok_or_else(|| anyhow::anyhow!("Column family '{}' not found", EdgeSummaries::CF_NAME))?;
                let summary_key = EdgeSummaries::key_to_bytes(&EdgeSummaryCfKey(summary_hash));
                let summary_value = EdgeSummaries::value_to_bytes(&EdgeSummaryCfValue(self.summary.clone()))?;
                txn.put_cf(summaries_cf, summary_key, summary_value)?;

                // Write reverse index entry with CURRENT marker (version=1 for new edges)
                let index_cf = txn_db
                    .cf_handle(EdgeSummaryIndex::CF_NAME)
                    .ok_or_else(|| anyhow::anyhow!("Column family '{}' not found", EdgeSummaryIndex::CF_NAME))?;
                let index_key = EdgeSummaryIndexCfKey(
                    summary_hash,
                    self.source_node_id,
                    self.target_node_id,
                    name_hash,
                    1, // version=1 for new edges
                );
                let index_key_bytes = EdgeSummaryIndex::key_to_bytes(&index_key);
                let index_value_bytes = EdgeSummaryIndex::value_to_bytes(&EdgeSummaryIndexCfValue::current())?;
                txn.put_cf(index_cf, index_key_bytes, index_value_bytes)?;
            }
        }

        // Write to ForwardEdges CF (hot)
        let forward_cf = txn_db.cf_handle(ForwardEdges::CF_NAME).ok_or_else(|| {
            anyhow::anyhow!("Column family '{}' not found", ForwardEdges::CF_NAME)
        })?;
        let (forward_key, forward_value) = ForwardEdges::create_bytes(self)?;
        txn.put_cf(forward_cf, forward_key, forward_value)?;

        // Write to ReverseEdges CF (hot)
        let reverse_cf = txn_db.cf_handle(ReverseEdges::CF_NAME).ok_or_else(|| {
            anyhow::anyhow!("Column family '{}' not found", ReverseEdges::CF_NAME)
        })?;
        let (reverse_key, reverse_value) = ReverseEdges::create_bytes(self)?;
        txn.put_cf(reverse_cf, reverse_key, reverse_value)?;

        Ok(())
    }
}

impl MutationExecutor for AddNodeFragment {
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
    ) -> Result<()> {
        tracing::debug!(
            id = %self.id,
            ts = %self.ts_millis.0,
            content_len = self.content.as_ref().len(),
            "Executing AddNodeFragment mutation"
        );

        let cf = txn_db.cf_handle(NodeFragments::CF_NAME).ok_or_else(|| {
            anyhow::anyhow!("Column family '{}' not found", NodeFragments::CF_NAME)
        })?;
        let (key, value) = self.to_cf_bytes()?;
        txn.put_cf(cf, key, value)?;

        Ok(())
    }
}

impl MutationExecutor for AddEdgeFragment {
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
    ) -> Result<()> {
        tracing::debug!(
            src = %self.src_id,
            dst = %self.dst_id,
            edge_name = %self.edge_name,
            ts = %self.ts_millis.0,
            content_len = self.content.as_ref().len(),
            "Executing AddEdgeFragment mutation"
        );

        // Write name to Names CF (idempotent)
        write_name_to_cf(txn, txn_db, &self.edge_name)?;

        let cf = txn_db.cf_handle(EdgeFragments::CF_NAME).ok_or_else(|| {
            anyhow::anyhow!("Column family '{}' not found", EdgeFragments::CF_NAME)
        })?;
        let (key, value) = self.to_cf_bytes()?;
        txn.put_cf(cf, key, value)?;

        Ok(())
    }

    fn execute_with_cache(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        cache: &super::name_hash::NameCache,
    ) -> Result<()> {
        tracing::debug!(
            src = %self.src_id,
            dst = %self.dst_id,
            edge_name = %self.edge_name,
            ts = %self.ts_millis.0,
            content_len = self.content.as_ref().len(),
            "Executing AddEdgeFragment mutation (cached)"
        );

        // Write name to Names CF with cache optimization
        write_name_to_cf_cached(txn, txn_db, &self.edge_name, cache)?;

        let cf = txn_db.cf_handle(EdgeFragments::CF_NAME).ok_or_else(|| {
            anyhow::anyhow!("Column family '{}' not found", EdgeFragments::CF_NAME)
        })?;
        let (key, value) = self.to_cf_bytes()?;
        txn.put_cf(cf, key, value)?;

        Ok(())
    }
}

impl MutationExecutor for UpdateNodeValidSinceUntil {
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
    ) -> Result<()> {
        let node_id = self.id;
        let new_range = self.temporal_range;

        // 1. Update the node (1 operation)
        update_node_valid_range(txn, txn_db, node_id, new_range)?;

        // 2. Find all connected edges (N edges) - returns (src_id, dst_id, name_hash)
        let edges = find_connected_edges(txn, txn_db, node_id)?;

        tracing::info!(
            node_id = %node_id,
            edge_count = edges.len(),
            "[UpdateNodeValidSinceUntil] Updating node and connected edges"
        );

        // 3. Update each edge (N × 2 operations = 2N operations)
        for (src_id, dst_id, name_hash) in edges {
            update_edge_valid_range(txn, txn_db, src_id, dst_id, name_hash, new_range)?;
        }

        Ok(())
    }
}

impl MutationExecutor for UpdateEdgeValidSinceUntil {
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
    ) -> Result<()> {
        tracing::debug!(
            src = %self.src_id,
            dst = %self.dst_id,
            name = %self.name,
            reason = %self.reason,
            "Executing UpdateEdgeValidSinceUntil mutation"
        );

        // Compute hash from edge name
        let name_hash = NameHash::from_name(&self.name);

        // Simply delegate to the helper
        update_edge_valid_range(
            txn,
            txn_db,
            self.src_id,
            self.dst_id,
            name_hash,
            self.temporal_range,
        )
    }
}

impl MutationExecutor for UpdateEdgeWeight {
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
    ) -> Result<()> {
        tracing::debug!(
            src = %self.src_id,
            dst = %self.dst_id,
            name = %self.name,
            weight = self.weight,
            "Executing UpdateEdgeWeight mutation"
        );


        use super::schema::{ForwardEdgeCfKey, ForwardEdgeCfValue, ForwardEdges};

        let cf = txn_db.cf_handle(ForwardEdges::CF_NAME).ok_or_else(|| {
            anyhow::anyhow!("Column family '{}' not found", ForwardEdges::CF_NAME)
        })?;

        // Compute hash from edge name
        let name_hash = NameHash::from_name(&self.name);
        let key = ForwardEdgeCfKey(self.src_id, self.dst_id, name_hash);
        let key_bytes = ForwardEdges::key_to_bytes(&key);

        // Read current value
        let current_value_bytes = txn
            .get_cf(cf, &key_bytes)?
            .ok_or_else(|| anyhow::anyhow!("Edge not found for update"))?;

        // Deserialize, modify weight field, reserialize
        let mut value: ForwardEdgeCfValue = ForwardEdges::value_from_bytes(&current_value_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize: {}", e))?;

        value.1 = Some(self.weight); // Update weight (field 1)

        let new_value_bytes = ForwardEdges::value_to_bytes(&value)
            .map_err(|e| anyhow::anyhow!("Failed to serialize: {}", e))?;

        txn.put_cf(cf, key_bytes, new_value_bytes)?;

        Ok(())
    }
}

// ============================================================================
// CONTENT-ADDRESS: MutationExecutor Implementations for Update/Delete
// (claude, 2026-02-02, implementing)
// ============================================================================

impl MutationExecutor for UpdateNodeSummary {
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
    ) -> Result<()> {
        tracing::debug!(
            id = %self.id,
            expected_version = self.expected_version,
            "Executing UpdateNodeSummary mutation"
        );

        use super::schema::{
            NodeCfKey, NodeCfValue, Nodes, NodeSummaries, NodeSummaryCfKey, NodeSummaryCfValue,
            NodeSummaryIndex, NodeSummaryIndexCfKey, NodeSummaryIndexCfValue, VERSION_MAX,
        };
        use super::summary_hash::SummaryHash;

        // 1. Read current node
        let nodes_cf = txn_db
            .cf_handle(Nodes::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("Nodes CF not found"))?;
        let node_key = NodeCfKey(self.id);
        let node_key_bytes = Nodes::key_to_bytes(&node_key);

        let current_bytes = txn
            .get_cf(nodes_cf, &node_key_bytes)?
            .ok_or_else(|| anyhow::anyhow!("Node not found: {}", self.id))?;
        let current: NodeCfValue = Nodes::value_from_bytes(&current_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize node: {}", e))?;

        let current_version = current.3;
        let old_hash = current.2;
        let is_deleted = current.4;

        // 2. Check if already deleted
        if is_deleted {
            return Err(anyhow::anyhow!("Cannot update deleted node: {}", self.id));
        }

        // 3. Optimistic lock check
        if current_version != self.expected_version {
            return Err(anyhow::anyhow!(
                "Version mismatch for node {}: expected {}, actual {}",
                self.id,
                self.expected_version,
                current_version
            ));
        }

        // 4. Version overflow check
        if current_version == VERSION_MAX {
            return Err(anyhow::anyhow!("Version overflow for node: {}", self.id));
        }

        // 5. Compute new version and hash
        let new_version = current_version + 1;
        let new_hash = SummaryHash::from_summary(&self.new_summary)?;

        // 6. Write new summary to NodeSummaries CF (cold)
        let summaries_cf = txn_db
            .cf_handle(NodeSummaries::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("NodeSummaries CF not found"))?;
        let summary_key = NodeSummaries::key_to_bytes(&NodeSummaryCfKey(new_hash));
        let summary_value = NodeSummaries::value_to_bytes(&NodeSummaryCfValue(self.new_summary.clone()))?;
        txn.put_cf(summaries_cf, summary_key, summary_value)?;

        // 7. Flip old index entry to STALE (if exists)
        if let Some(old_h) = old_hash {
            let index_cf = txn_db
                .cf_handle(NodeSummaryIndex::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!("NodeSummaryIndex CF not found"))?;
            let old_index_key = NodeSummaryIndexCfKey(old_h, self.id, current_version);
            let old_index_key_bytes = NodeSummaryIndex::key_to_bytes(&old_index_key);
            let stale_value_bytes = NodeSummaryIndex::value_to_bytes(&NodeSummaryIndexCfValue::stale())?;
            txn.put_cf(index_cf, old_index_key_bytes, stale_value_bytes)?;
        }

        // 8. Write new index entry with CURRENT marker
        let index_cf = txn_db
            .cf_handle(NodeSummaryIndex::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("NodeSummaryIndex CF not found"))?;
        let new_index_key = NodeSummaryIndexCfKey(new_hash, self.id, new_version);
        let new_index_key_bytes = NodeSummaryIndex::key_to_bytes(&new_index_key);
        let current_value_bytes = NodeSummaryIndex::value_to_bytes(&NodeSummaryIndexCfValue::current())?;
        txn.put_cf(index_cf, new_index_key_bytes, current_value_bytes)?;

        // 9. Update Nodes CF with new version and summary hash
        let new_node_value = NodeCfValue(current.0, current.1, Some(new_hash), new_version, false);
        let new_node_bytes = Nodes::value_to_bytes(&new_node_value)
            .map_err(|e| anyhow::anyhow!("Failed to serialize node: {}", e))?;
        txn.put_cf(nodes_cf, node_key_bytes, new_node_bytes)?;

        tracing::info!(
            id = %self.id,
            old_version = current_version,
            new_version = new_version,
            "UpdateNodeSummary completed"
        );

        Ok(())
    }
}

impl MutationExecutor for UpdateEdgeSummary {
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
    ) -> Result<()> {
        tracing::debug!(
            src = %self.src_id,
            dst = %self.dst_id,
            name = %self.name,
            expected_version = self.expected_version,
            "Executing UpdateEdgeSummary mutation"
        );

        use super::schema::{
            ForwardEdgeCfKey, ForwardEdgeCfValue, ForwardEdges,
            EdgeSummaries, EdgeSummaryCfKey, EdgeSummaryCfValue,
            EdgeSummaryIndex, EdgeSummaryIndexCfKey, EdgeSummaryIndexCfValue, VERSION_MAX,
        };
        use super::summary_hash::SummaryHash;

        let name_hash = NameHash::from_name(&self.name);

        // 1. Read current edge
        let forward_cf = txn_db
            .cf_handle(ForwardEdges::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("ForwardEdges CF not found"))?;
        let edge_key = ForwardEdgeCfKey(self.src_id, self.dst_id, name_hash);
        let edge_key_bytes = ForwardEdges::key_to_bytes(&edge_key);

        let current_bytes = txn
            .get_cf(forward_cf, &edge_key_bytes)?
            .ok_or_else(|| anyhow::anyhow!("Edge not found: {}→{}", self.src_id, self.dst_id))?;
        let current: ForwardEdgeCfValue = ForwardEdges::value_from_bytes(&current_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize edge: {}", e))?;

        let current_version = current.3;
        let old_hash = current.2;
        let is_deleted = current.4;

        // 2. Check if already deleted
        if is_deleted {
            return Err(anyhow::anyhow!("Cannot update deleted edge: {}→{}", self.src_id, self.dst_id));
        }

        // 3. Optimistic lock check
        if current_version != self.expected_version {
            return Err(anyhow::anyhow!(
                "Version mismatch for edge {}→{}: expected {}, actual {}",
                self.src_id,
                self.dst_id,
                self.expected_version,
                current_version
            ));
        }

        // 4. Version overflow check
        if current_version == VERSION_MAX {
            return Err(anyhow::anyhow!("Version overflow for edge: {}→{}", self.src_id, self.dst_id));
        }

        // 5. Compute new version and hash
        let new_version = current_version + 1;
        let new_hash = SummaryHash::from_summary(&self.new_summary)?;

        // 6. Write new summary to EdgeSummaries CF (cold)
        let summaries_cf = txn_db
            .cf_handle(EdgeSummaries::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("EdgeSummaries CF not found"))?;
        let summary_key = EdgeSummaries::key_to_bytes(&EdgeSummaryCfKey(new_hash));
        let summary_value = EdgeSummaries::value_to_bytes(&EdgeSummaryCfValue(self.new_summary.clone()))?;
        txn.put_cf(summaries_cf, summary_key, summary_value)?;

        // 7. Flip old index entry to STALE (if exists)
        if let Some(old_h) = old_hash {
            let index_cf = txn_db
                .cf_handle(EdgeSummaryIndex::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!("EdgeSummaryIndex CF not found"))?;
            let old_index_key = EdgeSummaryIndexCfKey(old_h, self.src_id, self.dst_id, name_hash, current_version);
            let old_index_key_bytes = EdgeSummaryIndex::key_to_bytes(&old_index_key);
            let stale_value_bytes = EdgeSummaryIndex::value_to_bytes(&EdgeSummaryIndexCfValue::stale())?;
            txn.put_cf(index_cf, old_index_key_bytes, stale_value_bytes)?;
        }

        // 8. Write new index entry with CURRENT marker
        let index_cf = txn_db
            .cf_handle(EdgeSummaryIndex::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("EdgeSummaryIndex CF not found"))?;
        let new_index_key = EdgeSummaryIndexCfKey(new_hash, self.src_id, self.dst_id, name_hash, new_version);
        let new_index_key_bytes = EdgeSummaryIndex::key_to_bytes(&new_index_key);
        let current_value_bytes = EdgeSummaryIndex::value_to_bytes(&EdgeSummaryIndexCfValue::current())?;
        txn.put_cf(index_cf, new_index_key_bytes, current_value_bytes)?;

        // 9. Update ForwardEdges CF with new version and summary hash
        let new_edge_value = ForwardEdgeCfValue(current.0, current.1, Some(new_hash), new_version, false);
        let new_edge_bytes = ForwardEdges::value_to_bytes(&new_edge_value)
            .map_err(|e| anyhow::anyhow!("Failed to serialize edge: {}", e))?;
        txn.put_cf(forward_cf, edge_key_bytes, new_edge_bytes)?;

        tracing::info!(
            src = %self.src_id,
            dst = %self.dst_id,
            old_version = current_version,
            new_version = new_version,
            "UpdateEdgeSummary completed"
        );

        Ok(())
    }
}

impl MutationExecutor for DeleteNode {
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
    ) -> Result<()> {
        tracing::debug!(
            id = %self.id,
            expected_version = self.expected_version,
            "Executing DeleteNode mutation"
        );

        use super::schema::{
            NodeCfKey, NodeCfValue, Nodes,
            NodeSummaryIndex, NodeSummaryIndexCfKey, NodeSummaryIndexCfValue, VERSION_MAX,
        };

        // 1. Read current node
        let nodes_cf = txn_db
            .cf_handle(Nodes::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("Nodes CF not found"))?;
        let node_key = NodeCfKey(self.id);
        let node_key_bytes = Nodes::key_to_bytes(&node_key);

        let current_bytes = txn
            .get_cf(nodes_cf, &node_key_bytes)?
            .ok_or_else(|| anyhow::anyhow!("Node not found: {}", self.id))?;
        let current: NodeCfValue = Nodes::value_from_bytes(&current_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize node: {}", e))?;

        let current_version = current.3;
        let current_hash = current.2;
        let is_deleted = current.4;

        // 2. Check if already deleted
        if is_deleted {
            return Err(anyhow::anyhow!("Node already deleted: {}", self.id));
        }

        // 3. Optimistic lock check
        if current_version != self.expected_version {
            return Err(anyhow::anyhow!(
                "Version mismatch for node {}: expected {}, actual {}",
                self.id,
                self.expected_version,
                current_version
            ));
        }

        // 4. Version overflow check
        if current_version == VERSION_MAX {
            return Err(anyhow::anyhow!("Version overflow for node: {}", self.id));
        }

        // 5. Increment version and set deleted flag
        let new_version = current_version + 1;
        let new_node_value = NodeCfValue(current.0, current.1, current.2, new_version, true);
        let new_node_bytes = Nodes::value_to_bytes(&new_node_value)
            .map_err(|e| anyhow::anyhow!("Failed to serialize node: {}", e))?;
        txn.put_cf(nodes_cf, &node_key_bytes, new_node_bytes)?;

        // 6. Flip current index entry to STALE
        if let Some(hash) = current_hash {
            let index_cf = txn_db
                .cf_handle(NodeSummaryIndex::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!("NodeSummaryIndex CF not found"))?;
            let index_key = NodeSummaryIndexCfKey(hash, self.id, current_version);
            let index_key_bytes = NodeSummaryIndex::key_to_bytes(&index_key);
            let stale_value_bytes = NodeSummaryIndex::value_to_bytes(&NodeSummaryIndexCfValue::stale())?;
            txn.put_cf(index_cf, index_key_bytes, stale_value_bytes)?;
        }

        tracing::info!(
            id = %self.id,
            old_version = current_version,
            new_version = new_version,
            "DeleteNode completed (tombstoned)"
        );

        Ok(())
    }
}

impl MutationExecutor for DeleteEdge {
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
    ) -> Result<()> {
        tracing::debug!(
            src = %self.src_id,
            dst = %self.dst_id,
            name = %self.name,
            expected_version = self.expected_version,
            "Executing DeleteEdge mutation"
        );

        use super::schema::{
            ForwardEdgeCfKey, ForwardEdgeCfValue, ForwardEdges,
            EdgeSummaryIndex, EdgeSummaryIndexCfKey, EdgeSummaryIndexCfValue, VERSION_MAX,
        };

        let name_hash = NameHash::from_name(&self.name);

        // 1. Read current edge
        let forward_cf = txn_db
            .cf_handle(ForwardEdges::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("ForwardEdges CF not found"))?;
        let edge_key = ForwardEdgeCfKey(self.src_id, self.dst_id, name_hash);
        let edge_key_bytes = ForwardEdges::key_to_bytes(&edge_key);

        let current_bytes = txn
            .get_cf(forward_cf, &edge_key_bytes)?
            .ok_or_else(|| anyhow::anyhow!("Edge not found: {}→{}", self.src_id, self.dst_id))?;
        let current: ForwardEdgeCfValue = ForwardEdges::value_from_bytes(&current_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize edge: {}", e))?;

        let current_version = current.3;
        let current_hash = current.2;
        let is_deleted = current.4;

        // 2. Check if already deleted
        if is_deleted {
            return Err(anyhow::anyhow!("Edge already deleted: {}→{}", self.src_id, self.dst_id));
        }

        // 3. Optimistic lock check
        if current_version != self.expected_version {
            return Err(anyhow::anyhow!(
                "Version mismatch for edge {}→{}: expected {}, actual {}",
                self.src_id,
                self.dst_id,
                self.expected_version,
                current_version
            ));
        }

        // 4. Version overflow check
        if current_version == VERSION_MAX {
            return Err(anyhow::anyhow!("Version overflow for edge: {}→{}", self.src_id, self.dst_id));
        }

        // 5. Increment version and set deleted flag
        let new_version = current_version + 1;
        let new_edge_value = ForwardEdgeCfValue(current.0, current.1, current.2, new_version, true);
        let new_edge_bytes = ForwardEdges::value_to_bytes(&new_edge_value)
            .map_err(|e| anyhow::anyhow!("Failed to serialize edge: {}", e))?;
        txn.put_cf(forward_cf, &edge_key_bytes, new_edge_bytes)?;

        // 6. Flip current index entry to STALE
        if let Some(hash) = current_hash {
            let index_cf = txn_db
                .cf_handle(EdgeSummaryIndex::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!("EdgeSummaryIndex CF not found"))?;
            let index_key = EdgeSummaryIndexCfKey(hash, self.src_id, self.dst_id, name_hash, current_version);
            let index_key_bytes = EdgeSummaryIndex::key_to_bytes(&index_key);
            let stale_value_bytes = EdgeSummaryIndex::value_to_bytes(&EdgeSummaryIndexCfValue::stale())?;
            txn.put_cf(index_cf, index_key_bytes, stale_value_bytes)?;
        }

        tracing::info!(
            src = %self.src_id,
            dst = %self.dst_id,
            old_version = current_version,
            new_version = new_version,
            "DeleteEdge completed (tombstoned)"
        );

        Ok(())
    }
}

impl Mutation {
    /// Execute this mutation directly against storage.
    /// Delegates to the specific mutation type's executor.
    pub fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
    ) -> Result<()> {
        match self {
            Mutation::AddNode(m) => m.execute(txn, txn_db),
            Mutation::AddEdge(m) => m.execute(txn, txn_db),
            Mutation::AddNodeFragment(m) => m.execute(txn, txn_db),
            Mutation::AddEdgeFragment(m) => m.execute(txn, txn_db),
            Mutation::UpdateNodeValidSinceUntil(m) => m.execute(txn, txn_db),
            Mutation::UpdateEdgeValidSinceUntil(m) => m.execute(txn, txn_db),
            Mutation::UpdateEdgeWeight(m) => m.execute(txn, txn_db),
            // CONTENT-ADDRESS: Update/Delete with optimistic locking
            Mutation::UpdateNodeSummary(m) => m.execute(txn, txn_db),
            Mutation::UpdateEdgeSummary(m) => m.execute(txn, txn_db),
            Mutation::DeleteNode(m) => m.execute(txn, txn_db),
            Mutation::DeleteEdge(m) => m.execute(txn, txn_db),
            // Flush is not a storage operation - it's handled by the consumer
            // for synchronization purposes only
            Mutation::Flush(_) => Ok(()),
        }
    }

    /// Execute this mutation with access to the name cache.
    ///
    /// Uses the cache to:
    /// 1. Skip redundant Names CF writes for already-interned names
    /// 2. Intern new names for future lookups
    pub fn execute_with_cache(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        cache: &super::name_hash::NameCache,
    ) -> Result<()> {
        match self {
            Mutation::AddNode(m) => m.execute_with_cache(txn, txn_db, cache),
            Mutation::AddEdge(m) => m.execute_with_cache(txn, txn_db, cache),
            Mutation::AddNodeFragment(m) => m.execute(txn, txn_db), // No names
            Mutation::AddEdgeFragment(m) => m.execute_with_cache(txn, txn_db, cache),
            Mutation::UpdateNodeValidSinceUntil(m) => m.execute(txn, txn_db), // No new names
            Mutation::UpdateEdgeValidSinceUntil(m) => m.execute(txn, txn_db), // No new names
            Mutation::UpdateEdgeWeight(m) => m.execute(txn, txn_db), // No new names
            // CONTENT-ADDRESS: Update/Delete with optimistic locking (no new names)
            Mutation::UpdateNodeSummary(m) => m.execute(txn, txn_db),
            Mutation::UpdateEdgeSummary(m) => m.execute(txn, txn_db),
            Mutation::DeleteNode(m) => m.execute(txn, txn_db),
            Mutation::DeleteEdge(m) => m.execute(txn, txn_db),
            Mutation::Flush(_) => Ok(()),
        }
    }

    /// Returns true if this mutation is a Flush marker.
    pub fn is_flush(&self) -> bool {
        matches!(self, Mutation::Flush(_))
    }
}

// ============================================================================
// Runnable Trait Implementations
// ============================================================================

// Implement Runnable for individual mutation types
#[async_trait::async_trait]
impl Runnable for AddNode {
    async fn run(self, writer: &Writer) -> Result<()> {
        writer.send(vec![Mutation::AddNode(self)]).await
    }
}

#[async_trait::async_trait]
impl Runnable for AddEdge {
    async fn run(self, writer: &Writer) -> Result<()> {
        writer.send(vec![Mutation::AddEdge(self)]).await
    }
}

#[async_trait::async_trait]
impl Runnable for AddNodeFragment {
    async fn run(self, writer: &Writer) -> Result<()> {
        writer.send(vec![Mutation::AddNodeFragment(self)]).await
    }
}

#[async_trait::async_trait]
impl Runnable for AddEdgeFragment {
    async fn run(self, writer: &Writer) -> Result<()> {
        writer.send(vec![Mutation::AddEdgeFragment(self)]).await
    }
}

#[async_trait::async_trait]
impl Runnable for UpdateNodeValidSinceUntil {
    async fn run(self, writer: &Writer) -> Result<()> {
        writer
            .send(vec![Mutation::UpdateNodeValidSinceUntil(self)])
            .await
    }
}

#[async_trait::async_trait]
impl Runnable for UpdateEdgeValidSinceUntil {
    async fn run(self, writer: &Writer) -> Result<()> {
        writer
            .send(vec![Mutation::UpdateEdgeValidSinceUntil(self)])
            .await
    }
}

#[async_trait::async_trait]
impl Runnable for UpdateEdgeWeight {
    async fn run(self, writer: &Writer) -> Result<()> {
        writer.send(vec![Mutation::UpdateEdgeWeight(self)]).await
    }
}

// CONTENT-ADDRESS: Runnable for Update/Delete mutations
#[async_trait::async_trait]
impl Runnable for UpdateNodeSummary {
    async fn run(self, writer: &Writer) -> Result<()> {
        writer.send(vec![Mutation::UpdateNodeSummary(self)]).await
    }
}

#[async_trait::async_trait]
impl Runnable for UpdateEdgeSummary {
    async fn run(self, writer: &Writer) -> Result<()> {
        writer.send(vec![Mutation::UpdateEdgeSummary(self)]).await
    }
}

#[async_trait::async_trait]
impl Runnable for DeleteNode {
    async fn run(self, writer: &Writer) -> Result<()> {
        writer.send(vec![Mutation::DeleteNode(self)]).await
    }
}

#[async_trait::async_trait]
impl Runnable for DeleteEdge {
    async fn run(self, writer: &Writer) -> Result<()> {
        writer.send(vec![Mutation::DeleteEdge(self)]).await
    }
}

// ============================================================================
// From Trait - Automatic conversion to Mutation enum
// ============================================================================

impl From<AddNode> for Mutation {
    fn from(m: AddNode) -> Self {
        Mutation::AddNode(m)
    }
}

impl From<AddEdge> for Mutation {
    fn from(m: AddEdge) -> Self {
        Mutation::AddEdge(m)
    }
}

impl From<AddNodeFragment> for Mutation {
    fn from(m: AddNodeFragment) -> Self {
        Mutation::AddNodeFragment(m)
    }
}

impl From<AddEdgeFragment> for Mutation {
    fn from(m: AddEdgeFragment) -> Self {
        Mutation::AddEdgeFragment(m)
    }
}

impl From<UpdateNodeValidSinceUntil> for Mutation {
    fn from(m: UpdateNodeValidSinceUntil) -> Self {
        Mutation::UpdateNodeValidSinceUntil(m)
    }
}

impl From<UpdateEdgeValidSinceUntil> for Mutation {
    fn from(m: UpdateEdgeValidSinceUntil) -> Self {
        Mutation::UpdateEdgeValidSinceUntil(m)
    }
}

impl From<UpdateEdgeWeight> for Mutation {
    fn from(m: UpdateEdgeWeight) -> Self {
        Mutation::UpdateEdgeWeight(m)
    }
}

// CONTENT-ADDRESS: From for Update/Delete mutations
impl From<UpdateNodeSummary> for Mutation {
    fn from(m: UpdateNodeSummary) -> Self {
        Mutation::UpdateNodeSummary(m)
    }
}

impl From<UpdateEdgeSummary> for Mutation {
    fn from(m: UpdateEdgeSummary) -> Self {
        Mutation::UpdateEdgeSummary(m)
    }
}

impl From<DeleteNode> for Mutation {
    fn from(m: DeleteNode) -> Self {
        Mutation::DeleteNode(m)
    }
}

impl From<DeleteEdge> for Mutation {
    fn from(m: DeleteEdge) -> Self {
        Mutation::DeleteEdge(m)
    }
}

// ============================================================================
// MutationBatch - Zero-overhead batching
// ============================================================================

/// A batch of mutations to be executed atomically.
///
/// This type provides zero-overhead batching of mutations without requiring
/// heap allocation via boxing (unlike implementing Runnable on Vec<Mutation>).
///
/// # Examples
///
/// ```rust,ignore
/// use motlie_db::{MutationBatch, AddNode, AddEdge, Mutation, Runnable};
///
/// // Manual construction
/// let batch = MutationBatch(vec![
///     Mutation::AddNode(AddNode { /* ... */ }),
///     Mutation::AddEdge(AddEdge { /* ... */ }),
/// ]);
/// batch.run(&writer).await?;
///
/// // Using the mutations![] macro (recommended)
/// mutations![
///     AddNode { /* ... */ },
///     AddEdge { /* ... */ },
/// ].run(&writer).await?;
/// ```
#[derive(Debug, Clone)]
pub struct MutationBatch(pub Vec<Mutation>);

impl MutationBatch {
    /// Create a new empty mutation batch
    pub fn new() -> Self {
        Self(Vec::new())
    }

    /// Create a new mutation batch with the given capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    /// Add a mutation to the batch
    pub fn push(&mut self, mutation: impl Into<Mutation>) {
        self.0.push(mutation.into());
    }

    /// Get the number of mutations in the batch
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Check if the batch is empty
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Default for MutationBatch {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl Runnable for MutationBatch {
    async fn run(self, writer: &Writer) -> Result<()> {
        writer.send(self.0).await
    }
}

// ============================================================================
// mutations![] Macro - Ergonomic batch construction
// ============================================================================

/// Convenience macro for creating a MutationBatch with automatic type conversion.
///
/// This macro provides `vec![]`-like syntax for creating mutation batches,
/// with automatic conversion from mutation types to the Mutation enum.
///
/// # Examples
///
/// ```rust,ignore
/// use motlie_db::{mutations, AddNode, AddEdge, Runnable};
///
/// // Empty batch
/// let batch = mutations![];
///
/// // Single mutation
/// mutations![
///     AddNode {
///         id: Id::new(),
///         name: "Alice".to_string(),
///         ts_millis: TimestampMilli::now(),
///         temporal_range: None,
///     }
/// ].run(&writer).await?;
///
/// // Multiple mutations
/// mutations![
///     AddNode { /* ... */ },
///     AddEdge { /* ... */ },
///     AddFragment { /* ... */ },
/// ].run(&writer).await?;
/// ```
#[macro_export]
macro_rules! mutations {
    () => {
        $crate::MutationBatch::new()
    };
    ($($mutation:expr),+ $(,)?) => {
        $crate::MutationBatch(vec![$($mutation.into()),+])
    };
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::writer::{
        create_mutation_writer, spawn_consumer, Consumer, Processor, WriterConfig,
    };
    use crate::Id;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tokio::sync::mpsc;
    use tokio::time::Duration;

    // Mock processor for testing
    struct TestProcessor {
        delay_ms: u64,
        processed_count: Arc<AtomicUsize>,
    }

    impl TestProcessor {
        fn new(delay_ms: u64) -> Self {
            Self {
                delay_ms,
                processed_count: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn with_counter(delay_ms: u64, counter: Arc<AtomicUsize>) -> Self {
            Self {
                delay_ms,
                processed_count: counter,
            }
        }

        fn count(&self) -> usize {
            self.processed_count.load(Ordering::SeqCst)
        }
    }

    #[async_trait::async_trait]
    impl Processor for TestProcessor {
        async fn process_mutations(&self, mutations: &[Mutation]) -> Result<()> {
            if self.delay_ms > 0 {
                tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            }
            self.processed_count
                .fetch_add(mutations.len(), Ordering::SeqCst);
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_generic_consumer_basic() {
        let config = WriterConfig {
            channel_buffer_size: 10,
        };

        let (writer, receiver) = {
            let (sender, receiver) = mpsc::channel(config.channel_buffer_size);
            let writer = super::super::writer::Writer::new(sender);
            (writer, receiver)
        };

        let processor = TestProcessor::new(1);
        let consumer = Consumer::new(receiver, config, processor);

        // Spawn consumer
        let consumer_handle = spawn_consumer(consumer);

        // Send a mutation
        let node_args = AddNode {
            id: Id::new(),
            ts_millis: TimestampMilli::now(),
            name: "test_node".to_string(),
            valid_range: None,
            summary: super::schema::NodeSummary::from_text("test summary"),
        };
        node_args.run(&writer).await.unwrap();

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Drop writer to close channel
        drop(writer);

        // Wait for consumer to finish
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_graph_processor_continues_when_fulltext_slow() {
        // This test verifies that the graph processor can continue processing mutations
        // even when the fulltext consumer isn't keeping up and the MPSC buffer is full.

        let graph_config = WriterConfig {
            channel_buffer_size: 100,
        };

        let fulltext_config = WriterConfig {
            channel_buffer_size: 2, // Very small buffer to force overflow
        };

        // Create channels
        let (graph_sender, graph_receiver) = mpsc::channel(graph_config.channel_buffer_size);
        let (fulltext_sender, fulltext_receiver) =
            mpsc::channel(fulltext_config.channel_buffer_size);

        // Create writer
        let writer = super::super::writer::Writer::new(graph_sender);

        // Create graph processor (fast, no delay)
        let graph_counter = Arc::new(AtomicUsize::new(0));
        let graph_processor = TestProcessor::with_counter(0, graph_counter.clone());

        // Create fulltext processor (slow, 50ms delay per mutation)
        let fulltext_counter = Arc::new(AtomicUsize::new(0));
        let fulltext_processor = TestProcessor::with_counter(50, fulltext_counter.clone());

        // Create consumers
        let graph_consumer = Consumer::with_next(
            graph_receiver,
            graph_config,
            graph_processor,
            fulltext_sender,
        );
        let fulltext_consumer =
            Consumer::new(fulltext_receiver, fulltext_config, fulltext_processor);

        // Spawn consumers
        let graph_handle = spawn_consumer(graph_consumer);
        let fulltext_handle = spawn_consumer(fulltext_consumer);

        // Send many mutations quickly (more than fulltext buffer can handle)
        let num_mutations = 20;
        for i in 0..num_mutations {
            let node_args = AddNode {
                id: Id::new(),
                ts_millis: TimestampMilli::now(),
                name: format!("test_node_{}", i),
                valid_range: None,
                summary: super::schema::NodeSummary::from_text(&format!("summary {}", i)),
            };
            node_args.run(&writer).await.unwrap();
        }

        // Give graph processor time to process all mutations
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Verify that graph processor processed all mutations
        let graph_processed = graph_counter.load(Ordering::SeqCst);
        assert_eq!(
            graph_processed, num_mutations,
            "Graph processor should have processed all {} mutations, but processed {}",
            num_mutations, graph_processed
        );

        // Verify that fulltext processor received fewer mutations (due to buffer overflow)
        let fulltext_processed = fulltext_counter.load(Ordering::SeqCst);
        assert!(
            fulltext_processed < num_mutations,
            "Fulltext processor should have processed fewer than {} mutations due to buffer overflow, but processed {}",
            num_mutations, fulltext_processed
        );

        println!(
            "Graph processed: {}, Fulltext processed: {}, Dropped: {}",
            graph_processed,
            fulltext_processed,
            graph_processed - fulltext_processed
        );

        // Drop writer to close channels
        drop(writer);

        // Wait for graph consumer to finish (should finish quickly)
        let graph_result = tokio::time::timeout(Duration::from_secs(5), graph_handle).await;
        assert!(
            graph_result.is_ok(),
            "Graph consumer should finish promptly"
        );
        graph_result.unwrap().unwrap().unwrap();

        // Give fulltext consumer some time to finish processing remaining items
        tokio::time::sleep(Duration::from_millis(500)).await;

        // The fulltext consumer will continue running until its channel is closed
        // Since we dropped the graph consumer's sender to fulltext, the fulltext receiver
        // channel should now be closed, allowing fulltext consumer to exit
        let fulltext_result = tokio::time::timeout(Duration::from_secs(5), fulltext_handle).await;
        assert!(
            fulltext_result.is_ok(),
            "Fulltext consumer should finish after channel closed"
        );
        fulltext_result.unwrap().unwrap().unwrap();
    }
}
