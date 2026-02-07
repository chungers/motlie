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
    // (codex, 2026-02-07, eval: VERSIONING plan includes RestoreNode/RestoreEdge and rollback mutations; missing here, so rollback API is incomplete.)
    // (claude, 2026-02-07, FIXED: Added RestoreNode/RestoreEdge/RestoreEdges mutations below)
    RestoreNode(RestoreNode),
    RestoreEdge(RestoreEdge),
    RestoreEdges(RestoreEdges),

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
    pub valid_range: Option<schema::ActivePeriod>,

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
    pub valid_range: Option<schema::ActivePeriod>,

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
    pub valid_range: Option<schema::ActivePeriod>,
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
    pub valid_range: Option<schema::ActivePeriod>,
}

#[derive(Debug, Clone)]
pub struct UpdateNodeValidSinceUntil {
    /// The UUID of the Node
    pub id: Id,

    /// The temporal validity range for this fragment
    pub temporal_range: schema::ActivePeriod,

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
    pub temporal_range: schema::ActivePeriod,

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

// (claude, 2026-02-07, FIXED: Added RestoreNode/RestoreEdge for VERSIONING rollback API)
// (claude, 2026-02-07, FIXED: Changed to as_of timestamp per Codex Item 1 review)

/// Restore a node to state at a previous time.
///
/// VERSIONING rollback: Finds the version that was active at `as_of` time
/// in NodeVersionHistory, creates a new current version with that state.
#[derive(Debug, Clone)]
pub struct RestoreNode {
    /// The UUID of the Node to restore
    pub id: Id,

    /// The timestamp to restore to (finds version active at this time)
    pub as_of: crate::TimestampMilli,
}

/// Restore an edge to state at a previous time.
///
/// VERSIONING rollback: Finds the version that was active at `as_of` time
/// in EdgeVersionHistory, creates a new current edge with that state.
#[derive(Debug, Clone)]
pub struct RestoreEdge {
    /// The UUID of the source Node
    pub src_id: Id,

    /// The UUID of the destination Node
    pub dst_id: Id,

    /// The name of the Edge
    pub name: schema::EdgeName,

    /// The timestamp to restore to (finds version active at this time)
    pub as_of: crate::TimestampMilli,
}

/// Restore all outgoing edges from a source node to state at a previous time.
/// (claude, 2026-02-07, FIXED: Added RestoreEdges batch API per VERSIONING.md)
///
/// VERSIONING rollback: Finds all edges from src (optionally filtered by name),
/// looks up their state at `as_of` time in EdgeVersionHistory, and creates
/// new current edges with that state.
#[derive(Debug, Clone)]
pub struct RestoreEdges {
    /// The UUID of the source Node
    pub src_id: Id,

    /// Optional edge name filter (None = restore all edge names)
    pub name: Option<schema::EdgeName>,

    /// The timestamp to restore to (finds version active at this time)
    pub as_of: crate::TimestampMilli,
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

// ============================================================================
// Summary Helper Functions for Content-Addressed Summaries
// ============================================================================
// (VERSIONING: RefCount replaced with OrphanSummaries-based GC)

use super::summary_hash::SummaryHash;
use super::schema::{NodeSummary, EdgeSummary};

/// Ensure a node summary exists in the summaries CF.
/// (VERSIONING: No refcount - OrphanSummaries handles deferred deletion)
///
/// This is idempotent - if the summary already exists, this is a no-op.
// (codex, 2026-02-07, eval: VERSIONING now specifies RefCount + OrphanSummaries retention; this path neither increments RefCount nor records orphan candidates, so rollback-safe GC is incomplete.)
fn ensure_node_summary(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    hash: SummaryHash,
    summary: &NodeSummary,
) -> Result<()> {
    use super::schema::{NodeSummaries, NodeSummaryCfKey, NodeSummaryCfValue};

    let cf = txn_db
        .cf_handle(NodeSummaries::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("NodeSummaries CF not found"))?;
    let key_bytes = NodeSummaries::key_to_bytes(&NodeSummaryCfKey(hash));

    // Check if summary already exists
    if txn.get_cf(cf, &key_bytes)?.is_some() {
        // Summary already exists - content-addressed, so data is identical
        tracing::trace!(hash = ?hash, "Node summary already exists");
        return Ok(());
    }

    // Create new summary entry
    let value = NodeSummaryCfValue(summary.clone());
    let value_bytes = NodeSummaries::value_to_bytes(&value)?;
    txn.put_cf(cf, key_bytes, value_bytes)?;

    tracing::trace!(hash = ?hash, "Created node summary");
    Ok(())
}

/// Mark a node summary as potentially orphaned.
/// Mark a node summary as potentially orphaned.
/// (VERSIONING: Actual deletion is deferred to GC via OrphanSummaries CF)
///
/// Writes to OrphanSummaries CF with current timestamp. GC will delete
/// entries older than retention period if they have no remaining references.
// (claude, 2026-02-07, FIXED: Implemented OrphanSummaries write - Codex Items 2-5)
fn mark_node_summary_orphan_candidate(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    hash: SummaryHash,
) -> Result<()> {
    use super::schema::{OrphanSummaries, OrphanSummaryCfKey, OrphanSummaryCfValue, SummaryKind};

    let cf = txn_db
        .cf_handle(OrphanSummaries::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("OrphanSummaries CF not found"))?;

    let now = crate::TimestampMilli::now();
    let key = OrphanSummaryCfKey(now, hash);
    let key_bytes = OrphanSummaries::key_to_bytes(&key);
    let value = OrphanSummaryCfValue(SummaryKind::Node);
    let value_bytes = OrphanSummaries::value_to_bytes(&value)?;

    txn.put_cf(cf, key_bytes, value_bytes)?;

    tracing::trace!(hash = ?hash, "Wrote node summary orphan candidate to OrphanSummaries CF");
    Ok(())
}

/// Ensure an edge summary exists in the summaries CF.
/// (VERSIONING: No refcount - OrphanSummaries handles deferred deletion)
///
/// This is idempotent - if the summary already exists, this is a no-op.
// (codex, 2026-02-07, eval: Missing RefCount increment and orphan-index bookkeeping; violates VERSIONING GC plan and can leak summaries indefinitely.)
fn ensure_edge_summary(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    hash: SummaryHash,
    summary: &EdgeSummary,
) -> Result<()> {
    use super::schema::{EdgeSummaries, EdgeSummaryCfKey, EdgeSummaryCfValue};

    let cf = txn_db
        .cf_handle(EdgeSummaries::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("EdgeSummaries CF not found"))?;
    let key_bytes = EdgeSummaries::key_to_bytes(&EdgeSummaryCfKey(hash));

    // Check if summary already exists
    if txn.get_cf(cf, &key_bytes)?.is_some() {
        // Summary already exists - content-addressed, so data is identical
        tracing::trace!(hash = ?hash, "Edge summary already exists");
        return Ok(());
    }

    // Create new summary entry
    let value = EdgeSummaryCfValue(summary.clone());
    let value_bytes = EdgeSummaries::value_to_bytes(&value)?;
    txn.put_cf(cf, key_bytes, value_bytes)?;

    tracing::trace!(hash = ?hash, "Created edge summary");
    Ok(())
}

/// Mark an edge summary as potentially orphaned.
/// (VERSIONING: Actual deletion is deferred to GC via OrphanSummaries CF)
///
/// Writes to OrphanSummaries CF with current timestamp. GC will delete
/// entries older than retention period if they have no remaining references.
// (claude, 2026-02-07, FIXED: Implemented OrphanSummaries write - Codex Items 2-5)
fn mark_edge_summary_orphan_candidate(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    hash: SummaryHash,
) -> Result<()> {
    use super::schema::{OrphanSummaries, OrphanSummaryCfKey, OrphanSummaryCfValue, SummaryKind};

    let cf = txn_db
        .cf_handle(OrphanSummaries::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("OrphanSummaries CF not found"))?;

    let now = crate::TimestampMilli::now();
    let key = OrphanSummaryCfKey(now, hash);
    let key_bytes = OrphanSummaries::key_to_bytes(&key);
    let value = OrphanSummaryCfValue(SummaryKind::Edge);
    let value_bytes = OrphanSummaries::value_to_bytes(&value)?;

    txn.put_cf(cf, key_bytes, value_bytes)?;

    tracing::trace!(hash = ?hash, "Wrote edge summary orphan candidate to OrphanSummaries CF");
    Ok(())
}

/// Remove a summary from OrphanSummaries when it's being reused (e.g., by restore).
/// (claude, 2026-02-07, FIXED: Rollback safety - remove from orphan index when summary reused)
///
/// Per VERSIONING.md: "Rollback reuses old summary | 0 → 1 | Remove `(*, hash)` from orphan index"
/// This scans for any entry with the given hash and removes it.
fn remove_summary_from_orphans(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    hash: SummaryHash,
) -> Result<()> {
    use super::schema::{OrphanSummaries, OrphanSummaryCfKey};

    let cf = txn_db
        .cf_handle(OrphanSummaries::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("OrphanSummaries CF not found"))?;

    // Scan entire CF to find entries with this hash
    // Key is (TimestampMilli, SummaryHash), so we need full scan to find by hash
    let iter = txn.iterator_cf(cf, rocksdb::IteratorMode::Start);
    let mut keys_to_delete = Vec::new();

    for item in iter {
        let (key_bytes, _value_bytes) = item?;
        let key: OrphanSummaryCfKey = OrphanSummaries::key_from_bytes(&key_bytes)?;
        if key.1 == hash {
            keys_to_delete.push(key_bytes.to_vec());
        }
    }

    // Delete all matching entries
    for key_bytes in keys_to_delete {
        txn.delete_cf(cf, &key_bytes)?;
        tracing::trace!(hash = ?hash, "Removed summary from OrphanSummaries CF (reused by restore)");
    }

    Ok(())
}

/// Verify that a node summary exists in NodeSummaries CF.
/// (claude, 2026-02-07, FIXED: Item 19 - Restore must verify summary exists before referencing)
fn verify_node_summary_exists(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    hash: SummaryHash,
) -> Result<bool> {
    use super::schema::{NodeSummaries, NodeSummaryCfKey};

    let cf = txn_db
        .cf_handle(NodeSummaries::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("NodeSummaries CF not found"))?;

    let key = NodeSummaryCfKey(hash);
    let key_bytes = NodeSummaries::key_to_bytes(&key);

    // Check if the key exists in the CF
    Ok(txn.get_cf(cf, &key_bytes)?.is_some())
}

/// Verify that an edge summary exists in EdgeSummaries CF.
/// (claude, 2026-02-07, FIXED: Item 19 - Restore must verify summary exists before referencing)
fn verify_edge_summary_exists(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    hash: SummaryHash,
) -> Result<bool> {
    use super::schema::{EdgeSummaries, EdgeSummaryCfKey};

    let cf = txn_db
        .cf_handle(EdgeSummaries::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("EdgeSummaries CF not found"))?;

    let key = EdgeSummaryCfKey(hash);
    let key_bytes = EdgeSummaries::key_to_bytes(&key);

    // Check if the key exists in the CF
    Ok(txn.get_cf(cf, &key_bytes)?.is_some())
}

/// Helper function to find the current version of a node.
/// (claude, 2026-02-06, in-progress: VERSIONING prefix scan for current version)
///
/// Returns (full_key_bytes, value) for the current version (ValidUntil = None).
fn find_current_node_version(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    node_id: Id,
) -> Result<Option<(Vec<u8>, schema::NodeCfValue)>> {
    use super::schema::Nodes;

    let nodes_cf = txn_db
        .cf_handle(Nodes::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("Nodes CF not found"))?;

    // Prefix scan on node_id (first 16 bytes)
    let prefix = node_id.into_bytes().to_vec();
    let iter = txn.prefix_iterator_cf(nodes_cf, &prefix);

    for item in iter {
        let (key_bytes, value_bytes) = item?;
        // Stop if we've gone past our prefix
        if !key_bytes.starts_with(&prefix) {
            break;
        }
        let value: schema::NodeCfValue = Nodes::value_from_bytes(&value_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize node value: {}", e))?;
        // Current version has ValidUntil = None
        if value.0.is_none() {
            return Ok(Some((key_bytes.to_vec(), value)));
        }
    }
    Ok(None)
}

/// Helper function to find the current version of a forward edge.
/// (claude, 2026-02-06, in-progress: VERSIONING prefix scan for current version)
///
/// Returns (full_key_bytes, value) for the current version (ValidUntil = None).
fn find_current_forward_edge_version(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    src_id: Id,
    dst_id: Id,
    name_hash: NameHash,
) -> Result<Option<(Vec<u8>, schema::ForwardEdgeCfValue)>> {
    use super::schema::ForwardEdges;

    let forward_cf = txn_db
        .cf_handle(ForwardEdges::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("ForwardEdges CF not found"))?;

    // Prefix scan on (src_id, dst_id, name_hash) = 40 bytes
    let mut prefix = Vec::with_capacity(40);
    prefix.extend_from_slice(&src_id.into_bytes());
    prefix.extend_from_slice(&dst_id.into_bytes());
    prefix.extend_from_slice(name_hash.as_bytes());
    let iter = txn.prefix_iterator_cf(forward_cf, &prefix);

    for item in iter {
        let (key_bytes, value_bytes) = item?;
        // Stop if we've gone past our prefix
        if !key_bytes.starts_with(&prefix) {
            break;
        }
        let value: schema::ForwardEdgeCfValue = ForwardEdges::value_from_bytes(&value_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize edge value: {}", e))?;
        // Current version has ValidUntil = None (field 0)
        if value.0.is_none() {
            return Ok(Some((key_bytes.to_vec(), value)));
        }
    }
    Ok(None)
}

/// Helper function to find the current version of a reverse edge.
/// (claude, 2026-02-06, in-progress: VERSIONING prefix scan for current version)
///
/// Returns (full_key_bytes, value) for the current version (ValidUntil = None).
fn find_current_reverse_edge_version(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    dst_id: Id,
    src_id: Id,
    name_hash: NameHash,
) -> Result<Option<(Vec<u8>, schema::ReverseEdgeCfValue)>> {
    use super::schema::ReverseEdges;

    let reverse_cf = txn_db
        .cf_handle(ReverseEdges::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("ReverseEdges CF not found"))?;

    // Prefix scan on (dst_id, src_id, name_hash) = 40 bytes
    let mut prefix = Vec::with_capacity(40);
    prefix.extend_from_slice(&dst_id.into_bytes());
    prefix.extend_from_slice(&src_id.into_bytes());
    prefix.extend_from_slice(name_hash.as_bytes());
    let iter = txn.prefix_iterator_cf(reverse_cf, &prefix);

    for item in iter {
        let (key_bytes, value_bytes) = item?;
        // Stop if we've gone past our prefix
        if !key_bytes.starts_with(&prefix) {
            break;
        }
        let value: schema::ReverseEdgeCfValue = ReverseEdges::value_from_bytes(&value_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize reverse edge value: {}", e))?;
        // Current version has ValidUntil = None (field 0)
        if value.0.is_none() {
            return Ok(Some((key_bytes.to_vec(), value)));
        }
    }
    Ok(None)
}

/// Helper function to update ActivePeriod for a single node.
/// Updates the Nodes CF.
/// (claude, 2026-02-06, in-progress: VERSIONING uses prefix scan)
/// Update node ActivePeriod with proper VERSIONING semantics.
/// (claude, 2026-02-07, FIXED: Create new version + history snapshot per Codex Item 17)
fn update_node_valid_range(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    node_id: Id,
    new_range: schema::ActivePeriod,
) -> Result<()> {
    use super::schema::{
        Nodes, NodeCfKey, NodeCfValue,
        NodeVersionHistory, NodeVersionHistoryCfKey, NodeVersionHistoryCfValue,
    };

    let nodes_cf = txn_db
        .cf_handle(Nodes::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("Nodes CF not found"))?;
    let history_cf = txn_db
        .cf_handle(NodeVersionHistory::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("NodeVersionHistory CF not found"))?;

    // Find current version via prefix scan
    let (current_key_bytes, current) = find_current_node_version(txn, txn_db, node_id)?
        .ok_or_else(|| anyhow::anyhow!("Node not found for id: {}", node_id))?;

    let now = crate::TimestampMilli::now();
    let new_version = current.4 + 1;

    // 1. Mark old row as superseded
    let old_node_value = NodeCfValue(
        Some(now),          // ValidUntil = now
        current.1.clone(),  // ActivePeriod (old)
        current.2,          // NameHash
        current.3,          // SummaryHash
        current.4,          // Version
        current.5,          // Deleted
    );
    let old_node_bytes = Nodes::value_to_bytes(&old_node_value)?;
    txn.put_cf(nodes_cf, &current_key_bytes, old_node_bytes)?;

    // 2. Create new row with updated ActivePeriod
    let new_node_key = NodeCfKey(node_id, now);
    let new_node_key_bytes = Nodes::key_to_bytes(&new_node_key);
    let new_node_value = NodeCfValue(
        None,               // ValidUntil = None (current)
        Some(new_range),    // New ActivePeriod
        current.2,          // NameHash (unchanged)
        current.3,          // SummaryHash (unchanged)
        new_version,        // Incremented version
        current.5,          // Deleted (unchanged)
    );
    let new_node_bytes = Nodes::value_to_bytes(&new_node_value)?;
    txn.put_cf(nodes_cf, new_node_key_bytes, new_node_bytes)?;

    // 3. Write NodeVersionHistory snapshot
    let history_key = NodeVersionHistoryCfKey(node_id, now, new_version);
    let history_key_bytes = NodeVersionHistory::key_to_bytes(&history_key);
    let history_value = NodeVersionHistoryCfValue(
        now,                // UpdatedAt
        current.3,          // SummaryHash (unchanged)
        current.2,          // NameHash (unchanged)
        Some(new_range),    // New ActivePeriod
    );
    let history_value_bytes = NodeVersionHistory::value_to_bytes(&history_value)?;
    txn.put_cf(history_cf, history_key_bytes, history_value_bytes)?;

    Ok(())
}

/// Update edge ActivePeriod with proper VERSIONING semantics.
/// (claude, 2026-02-07, FIXED: Create new version + history snapshot per Codex Item 17)
///
/// Note: This function accepts a NameHash directly. The caller is responsible for
/// computing the hash from the edge name string.
fn update_edge_valid_range(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    src_id: Id,
    dst_id: Id,
    name_hash: NameHash,
    new_range: schema::ActivePeriod,
) -> Result<()> {
    use super::schema::{
        ForwardEdges, ForwardEdgeCfKey, ForwardEdgeCfValue,
        ReverseEdges, ReverseEdgeCfKey, ReverseEdgeCfValue,
        EdgeVersionHistory, EdgeVersionHistoryCfKey, EdgeVersionHistoryCfValue,
    };

    let forward_cf = txn_db
        .cf_handle(ForwardEdges::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("ForwardEdges CF not found"))?;
    let reverse_cf = txn_db
        .cf_handle(ReverseEdges::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("ReverseEdges CF not found"))?;
    let history_cf = txn_db
        .cf_handle(EdgeVersionHistory::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("EdgeVersionHistory CF not found"))?;

    // Find current forward edge via prefix scan
    let (current_forward_key_bytes, current_forward) = find_current_forward_edge_version(txn, txn_db, src_id, dst_id, name_hash)?
        .ok_or_else(|| anyhow::anyhow!(
            "ForwardEdge not found: src={}, dst={}, name_hash={}",
            src_id, dst_id, name_hash
        ))?;

    // Find current reverse edge
    let (current_reverse_key_bytes, current_reverse) = find_current_reverse_edge_version(txn, txn_db, dst_id, src_id, name_hash)?
        .ok_or_else(|| anyhow::anyhow!(
            "ReverseEdge not found: src={}, dst={}, name_hash={}",
            src_id, dst_id, name_hash
        ))?;

    let now = crate::TimestampMilli::now();
    let new_version = current_forward.4 + 1;

    // 1. Mark old forward edge as superseded
    let old_forward_value = ForwardEdgeCfValue(
        Some(now),                  // ValidUntil = now
        current_forward.1.clone(),  // ActivePeriod (old)
        current_forward.2,          // Weight
        current_forward.3,          // SummaryHash
        current_forward.4,          // Version
        current_forward.5,          // Deleted
    );
    let old_forward_bytes = ForwardEdges::value_to_bytes(&old_forward_value)?;
    txn.put_cf(forward_cf, &current_forward_key_bytes, old_forward_bytes)?;

    // 2. Mark old reverse edge as superseded
    let old_reverse_value = ReverseEdgeCfValue(
        Some(now),                  // ValidUntil = now
        current_reverse.1.clone(),  // ActivePeriod (old)
    );
    let old_reverse_bytes = ReverseEdges::value_to_bytes(&old_reverse_value)?;
    txn.put_cf(reverse_cf, &current_reverse_key_bytes, old_reverse_bytes)?;

    // 3. Create new forward edge with updated ActivePeriod
    let new_forward_key = ForwardEdgeCfKey(src_id, dst_id, name_hash, now);
    let new_forward_key_bytes = ForwardEdges::key_to_bytes(&new_forward_key);
    let new_forward_value = ForwardEdgeCfValue(
        None,                       // ValidUntil = None (current)
        Some(new_range.clone()),    // New ActivePeriod
        current_forward.2,          // Weight (unchanged)
        current_forward.3,          // SummaryHash (unchanged)
        new_version,                // Incremented version
        current_forward.5,          // Deleted (unchanged)
    );
    let new_forward_bytes = ForwardEdges::value_to_bytes(&new_forward_value)?;
    txn.put_cf(forward_cf, new_forward_key_bytes, new_forward_bytes)?;

    // 4. Create new reverse edge
    let new_reverse_key = ReverseEdgeCfKey(dst_id, src_id, name_hash, now);
    let new_reverse_key_bytes = ReverseEdges::key_to_bytes(&new_reverse_key);
    let new_reverse_value = ReverseEdgeCfValue(
        None,                       // ValidUntil = None (current)
        Some(new_range.clone()),    // New ActivePeriod
    );
    let new_reverse_bytes = ReverseEdges::value_to_bytes(&new_reverse_value)?;
    txn.put_cf(reverse_cf, new_reverse_key_bytes, new_reverse_bytes)?;

    // 5. Write EdgeVersionHistory snapshot
    let history_key = EdgeVersionHistoryCfKey(src_id, dst_id, name_hash, now, new_version);
    let history_key_bytes = EdgeVersionHistory::key_to_bytes(&history_key);
    let history_value = EdgeVersionHistoryCfValue(
        now,                        // UpdatedAt
        current_forward.3,          // SummaryHash (unchanged)
        current_forward.2,          // Weight (unchanged)
        Some(new_range),            // New ActivePeriod
    );
    let history_value_bytes = EdgeVersionHistory::value_to_bytes(&history_value)?;
    txn.put_cf(history_cf, history_key_bytes, history_value_bytes)?;

    Ok(())
}

/// Helper function to find all current edges connected to a node.
/// (claude, 2026-02-06, in-progress: VERSIONING only returns current edges with ValidUntil=None)
/// Returns a deduplicated list of (src_id, dst_id, name_hash) tuples.
fn find_connected_edges(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    node_id: Id,
) -> Result<Vec<(Id, Id, NameHash)>> {

    use super::schema::{
        ForwardEdgeCfKey, ForwardEdgeCfValue, ForwardEdges,
        ReverseEdgeCfKey, ReverseEdgeCfValue, ReverseEdges,
    };

    let mut edge_topologies = std::collections::HashSet::new();

    // Find outgoing edges (where this node is the source)
    let forward_cf = txn_db
        .cf_handle(ForwardEdges::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("ForwardEdges CF not found"))?;
    let forward_prefix = node_id.into_bytes().to_vec();
    let forward_iter = txn.prefix_iterator_cf(forward_cf, &forward_prefix);

    for item in forward_iter {
        let (key_bytes, value_bytes) = item?;
        // Stop if we've gone past our prefix (src_id = 16 bytes)
        if !key_bytes.starts_with(&forward_prefix) {
            break;
        }
        let key: ForwardEdgeCfKey = ForwardEdges::key_from_bytes(&key_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize ForwardEdge key: {}", e))?;
        let value: ForwardEdgeCfValue = ForwardEdges::value_from_bytes(&value_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize ForwardEdge value: {}", e))?;
        // Only include current versions (ValidUntil = None at index 0)
        if value.0.is_none() {
            edge_topologies.insert((key.0, key.1, key.2));
        }
    }

    // Find incoming edges (where this node is the destination)
    let reverse_cf = txn_db
        .cf_handle(ReverseEdges::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("ReverseEdges CF not found"))?;
    let reverse_prefix = node_id.into_bytes().to_vec();
    let reverse_iter = txn.prefix_iterator_cf(reverse_cf, &reverse_prefix);

    for item in reverse_iter {
        let (key_bytes, value_bytes) = item?;
        // Stop if we've gone past our prefix (dst_id = 16 bytes)
        if !key_bytes.starts_with(&reverse_prefix) {
            break;
        }
        let key: ReverseEdgeCfKey = ReverseEdges::key_from_bytes(&key_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize ReverseEdge key: {}", e))?;
        let value: ReverseEdgeCfValue = ReverseEdges::value_from_bytes(&value_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize ReverseEdge value: {}", e))?;
        // Only include current versions (ValidUntil = None at index 0)
        // ReverseEdgeCfKey is (dst_id, src_id, name_hash), extract as (src_id, dst_id, name_hash)
        if value.0.is_none() {
            edge_topologies.insert((key.1, key.0, key.2));
        }
    }

    Ok(edge_topologies.into_iter().collect())
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
            Nodes, NodeSummaryIndex, NodeSummaryIndexCfKey, NodeSummaryIndexCfValue,
            NodeVersionHistory, NodeVersionHistoryCfKey, NodeVersionHistoryCfValue,
        };

        // Write name to Names CF (idempotent)
        let name_hash = write_name_to_cf(txn, txn_db, &self.name)?;

        // Write summary to NodeSummaries CF (cold) if non-empty
        // Increment refcount (or create with refcount=1)
        // Also write reverse index entry with CURRENT marker
        if !self.summary.is_empty() {
            if let Ok(summary_hash) = SummaryHash::from_summary(&self.summary) {
                // Increment refcount (creates row if missing)
                ensure_node_summary(txn, txn_db, summary_hash, &self.summary)?;

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

        // (claude, 2026-02-07, FIXED: Added initial NodeVersionHistory snapshot - Codex Items 6-7)
        // Write initial version snapshot to NodeVersionHistory CF for rollback support
        let history_cf = txn_db
            .cf_handle(NodeVersionHistory::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("NodeVersionHistory CF not found"))?;
        let summary_hash = if !self.summary.is_empty() {
            SummaryHash::from_summary(&self.summary).ok()
        } else {
            None
        };
        let history_key = NodeVersionHistoryCfKey(self.id, self.ts_millis, 1); // version=1 for new nodes
        let history_key_bytes = NodeVersionHistory::key_to_bytes(&history_key);
        let history_value = NodeVersionHistoryCfValue(
            self.ts_millis,     // UpdatedAt = creation time
            summary_hash,       // SummaryHash
            name_hash,          // NameHash
            self.valid_range.clone(), // ActivePeriod
        );
        let history_value_bytes = NodeVersionHistory::value_to_bytes(&history_value)?;
        txn.put_cf(history_cf, history_key_bytes, history_value_bytes)?;

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
            Nodes, NodeSummaryIndex, NodeSummaryIndexCfKey, NodeSummaryIndexCfValue,
            NodeVersionHistory, NodeVersionHistoryCfKey, NodeVersionHistoryCfValue,
        };

        // Write name to Names CF with cache optimization
        let name_hash = write_name_to_cf_cached(txn, txn_db, &self.name, cache)?;

        // Write summary to NodeSummaries CF (cold) if non-empty
        // Increment refcount (or create with refcount=1)
        // Also write reverse index entry with CURRENT marker
        if !self.summary.is_empty() {
            if let Ok(summary_hash) = SummaryHash::from_summary(&self.summary) {
                // Increment refcount (creates row if missing)
                ensure_node_summary(txn, txn_db, summary_hash, &self.summary)?;

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

        // (claude, 2026-02-07, FIXED: Added initial NodeVersionHistory snapshot - Codex Items 6-7)
        // Write initial version snapshot to NodeVersionHistory CF for rollback support
        let history_cf = txn_db
            .cf_handle(NodeVersionHistory::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("NodeVersionHistory CF not found"))?;
        let summary_hash = if !self.summary.is_empty() {
            SummaryHash::from_summary(&self.summary).ok()
        } else {
            None
        };
        let history_key = NodeVersionHistoryCfKey(self.id, self.ts_millis, 1); // version=1 for new nodes
        let history_key_bytes = NodeVersionHistory::key_to_bytes(&history_key);
        let history_value = NodeVersionHistoryCfValue(
            self.ts_millis,     // UpdatedAt = creation time
            summary_hash,       // SummaryHash
            name_hash,          // NameHash
            self.valid_range.clone(), // ActivePeriod
        );
        let history_value_bytes = NodeVersionHistory::value_to_bytes(&history_value)?;
        txn.put_cf(history_cf, history_key_bytes, history_value_bytes)?;

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
            ForwardEdges, ReverseEdges,
            EdgeSummaryIndex, EdgeSummaryIndexCfKey, EdgeSummaryIndexCfValue,
            EdgeVersionHistory, EdgeVersionHistoryCfKey, EdgeVersionHistoryCfValue,
        };

        // Write name to Names CF (idempotent)
        let name_hash = write_name_to_cf(txn, txn_db, &self.name)?;

        // Write summary to EdgeSummaries CF (cold) if non-empty
        // Increment refcount (or create with refcount=1)
        // Also write reverse index entry with CURRENT marker
        if !self.summary.is_empty() {
            if let Ok(summary_hash) = SummaryHash::from_summary(&self.summary) {
                // Increment refcount (creates row if missing)
                ensure_edge_summary(txn, txn_db, summary_hash, &self.summary)?;

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

        // (claude, 2026-02-07, FIXED: Added initial EdgeVersionHistory snapshot - Codex Items 8-9)
        // Write initial version snapshot to EdgeVersionHistory CF for rollback support
        let history_cf = txn_db
            .cf_handle(EdgeVersionHistory::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("EdgeVersionHistory CF not found"))?;
        let summary_hash = if !self.summary.is_empty() {
            SummaryHash::from_summary(&self.summary).ok()
        } else {
            None
        };
        let history_key = EdgeVersionHistoryCfKey(
            self.source_node_id,
            self.target_node_id,
            name_hash,
            self.ts_millis, // ValidSince
            1,              // version=1 for new edges
        );
        let history_key_bytes = EdgeVersionHistory::key_to_bytes(&history_key);
        let history_value = EdgeVersionHistoryCfValue(
            self.ts_millis,     // UpdatedAt = creation time
            summary_hash,       // SummaryHash
            self.weight,        // Weight
            self.valid_range.clone(), // ActivePeriod
        );
        let history_value_bytes = EdgeVersionHistory::value_to_bytes(&history_value)?;
        txn.put_cf(history_cf, history_key_bytes, history_value_bytes)?;

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
            ForwardEdges, ReverseEdges,
            EdgeSummaryIndex, EdgeSummaryIndexCfKey, EdgeSummaryIndexCfValue,
            EdgeVersionHistory, EdgeVersionHistoryCfKey, EdgeVersionHistoryCfValue,
        };

        // Write name to Names CF with cache optimization
        let name_hash = write_name_to_cf_cached(txn, txn_db, &self.name, cache)?;

        // Write summary to EdgeSummaries CF (cold) if non-empty
        // Increment refcount (or create with refcount=1)
        // Also write reverse index entry with CURRENT marker
        if !self.summary.is_empty() {
            if let Ok(summary_hash) = SummaryHash::from_summary(&self.summary) {
                // Increment refcount (creates row if missing)
                ensure_edge_summary(txn, txn_db, summary_hash, &self.summary)?;

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

        // (claude, 2026-02-07, FIXED: Added initial EdgeVersionHistory snapshot - Codex Items 8-9)
        // Write initial version snapshot to EdgeVersionHistory CF for rollback support
        let history_cf = txn_db
            .cf_handle(EdgeVersionHistory::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("EdgeVersionHistory CF not found"))?;
        let summary_hash = if !self.summary.is_empty() {
            SummaryHash::from_summary(&self.summary).ok()
        } else {
            None
        };
        let history_key = EdgeVersionHistoryCfKey(
            self.source_node_id,
            self.target_node_id,
            name_hash,
            self.ts_millis, // ValidSince
            1,              // version=1 for new edges
        );
        let history_key_bytes = EdgeVersionHistory::key_to_bytes(&history_key);
        let history_value = EdgeVersionHistoryCfValue(
            self.ts_millis,     // UpdatedAt = creation time
            summary_hash,       // SummaryHash
            self.weight,        // Weight
            self.valid_range.clone(), // ActivePeriod
        );
        let history_value_bytes = EdgeVersionHistory::value_to_bytes(&history_value)?;
        txn.put_cf(history_cf, history_key_bytes, history_value_bytes)?;

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
    /// Update edge weight with proper VERSIONING semantics.
    /// (claude, 2026-02-07, FIXED: Create new version + history snapshot per Codex Item 17)
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

        use super::schema::{
            ForwardEdges, ForwardEdgeCfKey, ForwardEdgeCfValue,
            ReverseEdges, ReverseEdgeCfKey, ReverseEdgeCfValue,
            EdgeVersionHistory, EdgeVersionHistoryCfKey, EdgeVersionHistoryCfValue,
        };

        let forward_cf = txn_db
            .cf_handle(ForwardEdges::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("ForwardEdges CF not found"))?;
        let reverse_cf = txn_db
            .cf_handle(ReverseEdges::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("ReverseEdges CF not found"))?;
        let history_cf = txn_db
            .cf_handle(EdgeVersionHistory::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("EdgeVersionHistory CF not found"))?;

        let name_hash = NameHash::from_name(&self.name);

        // Find current forward edge
        let (current_forward_key_bytes, current_forward) = find_current_forward_edge_version(txn, txn_db, self.src_id, self.dst_id, name_hash)?
            .ok_or_else(|| anyhow::anyhow!("Edge not found for update"))?;

        // Find current reverse edge
        let (current_reverse_key_bytes, current_reverse) = find_current_reverse_edge_version(txn, txn_db, self.dst_id, self.src_id, name_hash)?
            .ok_or_else(|| anyhow::anyhow!("ReverseEdge not found for update"))?;

        let now = crate::TimestampMilli::now();
        let new_version = current_forward.4 + 1;

        // 1. Mark old forward edge as superseded
        let old_forward_value = ForwardEdgeCfValue(
            Some(now),                  // ValidUntil = now
            current_forward.1.clone(),  // ActivePeriod
            current_forward.2,          // Weight (old)
            current_forward.3,          // SummaryHash
            current_forward.4,          // Version
            current_forward.5,          // Deleted
        );
        let old_forward_bytes = ForwardEdges::value_to_bytes(&old_forward_value)?;
        txn.put_cf(forward_cf, &current_forward_key_bytes, old_forward_bytes)?;

        // 2. Mark old reverse edge as superseded
        let old_reverse_value = ReverseEdgeCfValue(
            Some(now),                  // ValidUntil = now
            current_reverse.1.clone(),  // ActivePeriod
        );
        let old_reverse_bytes = ReverseEdges::value_to_bytes(&old_reverse_value)?;
        txn.put_cf(reverse_cf, &current_reverse_key_bytes, old_reverse_bytes)?;

        // 3. Create new forward edge with updated weight
        let new_forward_key = ForwardEdgeCfKey(self.src_id, self.dst_id, name_hash, now);
        let new_forward_key_bytes = ForwardEdges::key_to_bytes(&new_forward_key);
        let new_forward_value = ForwardEdgeCfValue(
            None,                       // ValidUntil = None (current)
            current_forward.1.clone(),  // ActivePeriod (unchanged)
            Some(self.weight),          // New weight
            current_forward.3,          // SummaryHash (unchanged)
            new_version,                // Incremented version
            current_forward.5,          // Deleted (unchanged)
        );
        let new_forward_bytes = ForwardEdges::value_to_bytes(&new_forward_value)?;
        txn.put_cf(forward_cf, new_forward_key_bytes, new_forward_bytes)?;

        // 4. Create new reverse edge
        let new_reverse_key = ReverseEdgeCfKey(self.dst_id, self.src_id, name_hash, now);
        let new_reverse_key_bytes = ReverseEdges::key_to_bytes(&new_reverse_key);
        let new_reverse_value = ReverseEdgeCfValue(
            None,                       // ValidUntil = None (current)
            current_forward.1.clone(),  // ActivePeriod (unchanged)
        );
        let new_reverse_bytes = ReverseEdges::value_to_bytes(&new_reverse_value)?;
        txn.put_cf(reverse_cf, new_reverse_key_bytes, new_reverse_bytes)?;

        // 5. Write EdgeVersionHistory snapshot
        let history_key = EdgeVersionHistoryCfKey(self.src_id, self.dst_id, name_hash, now, new_version);
        let history_key_bytes = EdgeVersionHistory::key_to_bytes(&history_key);
        let history_value = EdgeVersionHistoryCfValue(
            now,                        // UpdatedAt
            current_forward.3,          // SummaryHash (unchanged)
            Some(self.weight),          // New weight
            current_forward.1,          // ActivePeriod (unchanged)
        );
        let history_value_bytes = EdgeVersionHistory::value_to_bytes(&history_value)?;
        txn.put_cf(history_cf, history_key_bytes, history_value_bytes)?;

        tracing::info!(
            src = %self.src_id,
            dst = %self.dst_id,
            old_version = current_forward.4,
            new_version = new_version,
            "UpdateEdgeWeight completed"
        );

        Ok(())
    }
}

// ============================================================================
// CONTENT-ADDRESS: MutationExecutor Implementations for Update/Delete
// (claude, 2026-02-02, implementing)
// ============================================================================

impl MutationExecutor for UpdateNodeSummary {
    /// (claude, 2026-02-06, in-progress: VERSIONING prefix scan and new field indices)
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
            NodeCfValue, Nodes,
            NodeSummaryIndex, NodeSummaryIndexCfKey, NodeSummaryIndexCfValue, VERSION_MAX,
            NodeVersionHistory, NodeVersionHistoryCfKey, NodeVersionHistoryCfValue,
        };

        // 1. Find current node version via prefix scan
        let nodes_cf = txn_db
            .cf_handle(Nodes::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("Nodes CF not found"))?;

        let (node_key_bytes, current) = find_current_node_version(txn, txn_db, self.id)?
            .ok_or_else(|| anyhow::anyhow!("Node not found: {}", self.id))?;

        // Field indices (VERSIONING): 0=ValidUntil, 1=ActivePeriod, 2=NameHash, 3=SummaryHash, 4=Version, 5=Deleted
        let current_version = current.4;
        let old_hash = current.3;
        let is_deleted = current.5;

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

        // 6. Ensure new summary exists (idempotent, content-addressed)
        ensure_node_summary(txn, txn_db, new_hash, &self.new_summary)?;

        // 7. Mark old summary as orphan candidate (deferred to GC)
        if let Some(old_h) = old_hash {
            mark_node_summary_orphan_candidate(txn, txn_db, old_h)?;
        }

        // 8. Flip old index entry to STALE (if exists)
        if let Some(old_h) = old_hash {
            let index_cf = txn_db
                .cf_handle(NodeSummaryIndex::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!("NodeSummaryIndex CF not found"))?;
            let old_index_key = NodeSummaryIndexCfKey(old_h, self.id, current_version);
            let old_index_key_bytes = NodeSummaryIndex::key_to_bytes(&old_index_key);
            let stale_value_bytes = NodeSummaryIndex::value_to_bytes(&NodeSummaryIndexCfValue::stale())?;
            txn.put_cf(index_cf, old_index_key_bytes, stale_value_bytes)?;
        }

        // 9. Write new index entry with CURRENT marker
        let index_cf = txn_db
            .cf_handle(NodeSummaryIndex::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("NodeSummaryIndex CF not found"))?;
        let new_index_key = NodeSummaryIndexCfKey(new_hash, self.id, new_version);
        let new_index_key_bytes = NodeSummaryIndex::key_to_bytes(&new_index_key);
        let current_value_bytes = NodeSummaryIndex::value_to_bytes(&NodeSummaryIndexCfValue::current())?;
        txn.put_cf(index_cf, new_index_key_bytes, current_value_bytes)?;

        // 10. VERSIONING: Mark old version as superseded and create new version
        // The key includes ValidSince, so we need to:
        // a) Update old row: set ValidUntil = now
        // b) Create new row with ValidSince = now and ValidUntil = None
        let now = crate::TimestampMilli::now();

        // 10a. Update old row's ValidUntil to mark it as superseded
        let old_node_value = NodeCfValue(
            Some(now),          // ValidUntil = now (this version is no longer current)
            current.1,          // ActivePeriod
            current.2,          // NameHash
            current.3,          // SummaryHash (keep old hash)
            current.4,          // Version (keep old version)
            current.5,          // Deleted
        );
        let old_node_bytes = Nodes::value_to_bytes(&old_node_value)
            .map_err(|e| anyhow::anyhow!("Failed to serialize old node version: {}", e))?;
        txn.put_cf(nodes_cf, &node_key_bytes, old_node_bytes)?;

        // 10b. Create new row with new ValidSince
        use super::schema::NodeCfKey;
        let new_node_key = NodeCfKey(self.id, now);
        let new_node_key_bytes = Nodes::key_to_bytes(&new_node_key);
        let new_node_value = NodeCfValue(
            None,               // ValidUntil = None (current version)
            current.1,          // ActivePeriod (preserve)
            current.2,          // NameHash (preserve)
            Some(new_hash),     // SummaryHash (new)
            new_version,        // Version
            false,              // Deleted
        );
        let new_node_bytes = Nodes::value_to_bytes(&new_node_value)
            .map_err(|e| anyhow::anyhow!("Failed to serialize new node version: {}", e))?;
        txn.put_cf(nodes_cf, new_node_key_bytes, new_node_bytes)?;
        // (codex, 2026-02-07, decision: VERSIONING expects NodeVersionHistory snapshot for delete/tombstone version; missing here.)

        // (claude, 2026-02-07, FIXED: Added NodeVersionHistory snapshot - Codex Item 10)
        // Write version history entry for rollback support
        let history_cf = txn_db
            .cf_handle(NodeVersionHistory::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("NodeVersionHistory CF not found"))?;
        let history_key = NodeVersionHistoryCfKey(self.id, now, new_version);
        let history_key_bytes = NodeVersionHistory::key_to_bytes(&history_key);
        let history_value = NodeVersionHistoryCfValue(
            now,                // UpdatedAt
            Some(new_hash),     // SummaryHash (new)
            current.2,          // NameHash (preserved)
            current.1.clone(),  // ActivePeriod (preserved)
        );
        let history_value_bytes = NodeVersionHistory::value_to_bytes(&history_value)?;
        txn.put_cf(history_cf, history_key_bytes, history_value_bytes)?;

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
    /// (claude, 2026-02-06, in-progress: VERSIONING prefix scan and new field indices)
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
            ForwardEdgeCfValue, ForwardEdges,
            EdgeSummaryIndex, EdgeSummaryIndexCfKey, EdgeSummaryIndexCfValue, VERSION_MAX,
            EdgeVersionHistory, EdgeVersionHistoryCfKey, EdgeVersionHistoryCfValue,
        };

        let name_hash = NameHash::from_name(&self.name);

        // 1. Find current edge via prefix scan
        let forward_cf = txn_db
            .cf_handle(ForwardEdges::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("ForwardEdges CF not found"))?;

        let (edge_key_bytes, current) = find_current_forward_edge_version(txn, txn_db, self.src_id, self.dst_id, name_hash)?
            .ok_or_else(|| anyhow::anyhow!("Edge not found: {}→{}", self.src_id, self.dst_id))?;

        // Field indices (VERSIONING): 0=ValidUntil, 1=ActivePeriod, 2=Weight, 3=SummaryHash, 4=Version, 5=Deleted
        let current_version = current.4;
        let old_hash = current.3;
        let is_deleted = current.5;

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

        // 6. Ensure new summary exists (idempotent, content-addressed)
        ensure_edge_summary(txn, txn_db, new_hash, &self.new_summary)?;

        // 7. Mark old summary as orphan candidate (deferred to GC)
        if let Some(old_h) = old_hash {
            mark_edge_summary_orphan_candidate(txn, txn_db, old_h)?;
        }

        // 8. Flip old index entry to STALE (if exists)
        if let Some(old_h) = old_hash {
            let index_cf = txn_db
                .cf_handle(EdgeSummaryIndex::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!("EdgeSummaryIndex CF not found"))?;
            let old_index_key = EdgeSummaryIndexCfKey(old_h, self.src_id, self.dst_id, name_hash, current_version);
            let old_index_key_bytes = EdgeSummaryIndex::key_to_bytes(&old_index_key);
            let stale_value_bytes = EdgeSummaryIndex::value_to_bytes(&EdgeSummaryIndexCfValue::stale())?;
            txn.put_cf(index_cf, old_index_key_bytes, stale_value_bytes)?;
        }

        // 9. Write new index entry with CURRENT marker
        let index_cf = txn_db
            .cf_handle(EdgeSummaryIndex::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("EdgeSummaryIndex CF not found"))?;
        let new_index_key = EdgeSummaryIndexCfKey(new_hash, self.src_id, self.dst_id, name_hash, new_version);
        let new_index_key_bytes = EdgeSummaryIndex::key_to_bytes(&new_index_key);
        let current_value_bytes = EdgeSummaryIndex::value_to_bytes(&EdgeSummaryIndexCfValue::current())?;
        txn.put_cf(index_cf, new_index_key_bytes, current_value_bytes)?;

        // 10. VERSIONING: Mark old version as superseded and create new version
        // (Same pattern as UpdateNodeSummary for time-travel support)
        let now = crate::TimestampMilli::now();

        // 10a. Update old row's ValidUntil to mark it as superseded
        let old_edge_value = ForwardEdgeCfValue(
            Some(now),          // ValidUntil = now (this version is no longer current)
            current.1.clone(),  // ActivePeriod
            current.2,          // Weight
            current.3,          // SummaryHash (keep old hash)
            current.4,          // Version (keep old version)
            current.5,          // Deleted
        );
        let old_edge_bytes = ForwardEdges::value_to_bytes(&old_edge_value)
            .map_err(|e| anyhow::anyhow!("Failed to serialize old edge version: {}", e))?;
        txn.put_cf(forward_cf, &edge_key_bytes, old_edge_bytes)?;

        // 10b. Create new row with new ValidSince
        use super::schema::ForwardEdgeCfKey;
        let new_edge_key = ForwardEdgeCfKey(self.src_id, self.dst_id, name_hash, now);
        let new_edge_key_bytes = ForwardEdges::key_to_bytes(&new_edge_key);
        let new_edge_value = ForwardEdgeCfValue(
            None,               // ValidUntil = None (current version)
            current.1,          // ActivePeriod (preserve)
            current.2,          // Weight (preserve)
            Some(new_hash),     // SummaryHash (new)
            new_version,        // Version
            false,              // Deleted
        );
        let new_edge_bytes = ForwardEdges::value_to_bytes(&new_edge_value)
            .map_err(|e| anyhow::anyhow!("Failed to serialize new edge version: {}", e))?;
        txn.put_cf(forward_cf, new_edge_key_bytes, new_edge_bytes)?;
        // (codex, 2026-02-07, decision: VERSIONING expects EdgeVersionHistory snapshot for delete/tombstone version; missing here.)

        // (claude, 2026-02-07, FIXED: Added EdgeVersionHistory snapshot - Codex Item 11)
        // Write version history entry for rollback support
        let history_cf = txn_db
            .cf_handle(EdgeVersionHistory::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("EdgeVersionHistory CF not found"))?;
        let history_key = EdgeVersionHistoryCfKey(self.src_id, self.dst_id, name_hash, now, new_version);
        let history_key_bytes = EdgeVersionHistory::key_to_bytes(&history_key);
        let history_value = EdgeVersionHistoryCfValue(
            now,                // UpdatedAt
            Some(new_hash),     // SummaryHash (new)
            current.2,          // Weight (preserved)
            current.1.clone(),  // ActivePeriod (preserved)
        );
        let history_value_bytes = EdgeVersionHistory::value_to_bytes(&history_value)?;
        txn.put_cf(history_cf, history_key_bytes, history_value_bytes)?;

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
    /// (claude, 2026-02-06, in-progress: VERSIONING prefix scan and new field indices)
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
            NodeCfValue, Nodes,
            NodeSummaryIndex, NodeSummaryIndexCfKey, NodeSummaryIndexCfValue, VERSION_MAX,
        };

        // 1. Find current node version via prefix scan
        let nodes_cf = txn_db
            .cf_handle(Nodes::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("Nodes CF not found"))?;

        let (node_key_bytes, current) = find_current_node_version(txn, txn_db, self.id)?
            .ok_or_else(|| anyhow::anyhow!("Node not found: {}", self.id))?;

        // Field indices (VERSIONING): 0=ValidUntil, 1=ActivePeriod, 2=NameHash, 3=SummaryHash, 4=Version, 5=Deleted
        let current_version = current.4;
        let current_hash = current.3;
        let is_deleted = current.5;

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

        // 5. VERSIONING: Mark old version as superseded and create new tombstone version
        // (Same pattern as UpdateNodeSummary for time-travel support)
        let now = crate::TimestampMilli::now();
        let new_version = current_version + 1;

        // 5a. Update old row's ValidUntil to mark it as superseded
        let old_node_value = NodeCfValue(
            Some(now),          // ValidUntil = now (this version is no longer current)
            current.1.clone(),  // ActivePeriod
            current.2,          // NameHash
            current.3,          // SummaryHash (keep old hash)
            current.4,          // Version (keep old version)
            current.5,          // Deleted (keep old value = false)
        );
        let old_node_bytes = Nodes::value_to_bytes(&old_node_value)
            .map_err(|e| anyhow::anyhow!("Failed to serialize old node version: {}", e))?;
        txn.put_cf(nodes_cf, &node_key_bytes, old_node_bytes)?;

        // 5b. Create new row with new ValidSince and deleted=true
        use super::schema::NodeCfKey;
        let new_node_key = NodeCfKey(self.id, now);
        let new_node_key_bytes = Nodes::key_to_bytes(&new_node_key);
        let new_node_value = NodeCfValue(
            None,               // ValidUntil = None (current version)
            current.1,          // ActivePeriod (preserve)
            current.2,          // NameHash (preserve)
            current.3,          // SummaryHash (preserve)
            new_version,        // Version
            true,               // Deleted = true (tombstone)
        );
        let new_node_bytes = Nodes::value_to_bytes(&new_node_value)
            .map_err(|e| anyhow::anyhow!("Failed to serialize new node version: {}", e))?;
        txn.put_cf(nodes_cf, new_node_key_bytes, new_node_bytes)?;

        // 6. Mark summary as orphan candidate (deferred to GC)
        if let Some(hash) = current_hash {
            mark_node_summary_orphan_candidate(txn, txn_db, hash)?;
        }

        // 7. Flip current index entry to STALE
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
    /// (claude, 2026-02-06, in-progress: VERSIONING prefix scan and new field indices)
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
            ForwardEdgeCfValue, ForwardEdges,
            EdgeSummaryIndex, EdgeSummaryIndexCfKey, EdgeSummaryIndexCfValue, VERSION_MAX,
        };

        let name_hash = NameHash::from_name(&self.name);

        // 1. Find current edge via prefix scan
        let forward_cf = txn_db
            .cf_handle(ForwardEdges::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("ForwardEdges CF not found"))?;

        let (edge_key_bytes, current) = find_current_forward_edge_version(txn, txn_db, self.src_id, self.dst_id, name_hash)?
            .ok_or_else(|| anyhow::anyhow!("Edge not found: {}→{}", self.src_id, self.dst_id))?;

        // Field indices (VERSIONING): 0=ValidUntil, 1=ActivePeriod, 2=Weight, 3=SummaryHash, 4=Version, 5=Deleted
        let current_version = current.4;
        let current_hash = current.3;
        let is_deleted = current.5;

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

        // 5. VERSIONING: Mark old version as superseded and create new tombstone version
        // (Same pattern as DeleteNode for time-travel support)
        let now = crate::TimestampMilli::now();
        let new_version = current_version + 1;

        // 5a. Update old row's ValidUntil to mark it as superseded
        let old_edge_value = ForwardEdgeCfValue(
            Some(now),          // ValidUntil = now (this version is no longer current)
            current.1.clone(),  // ActivePeriod
            current.2,          // Weight
            current.3,          // SummaryHash
            current.4,          // Version (keep old version)
            current.5,          // Deleted (keep old value = false)
        );
        let old_edge_bytes = ForwardEdges::value_to_bytes(&old_edge_value)
            .map_err(|e| anyhow::anyhow!("Failed to serialize old edge version: {}", e))?;
        txn.put_cf(forward_cf, &edge_key_bytes, old_edge_bytes)?;

        // 5b. Create new row with new ValidSince and deleted=true
        use super::schema::ForwardEdgeCfKey;
        let new_edge_key = ForwardEdgeCfKey(self.src_id, self.dst_id, name_hash, now);
        let new_edge_key_bytes = ForwardEdges::key_to_bytes(&new_edge_key);
        let new_edge_value = ForwardEdgeCfValue(
            None,               // ValidUntil = None (current version)
            current.1,          // ActivePeriod (preserve)
            current.2,          // Weight (preserve)
            current.3,          // SummaryHash (preserve)
            new_version,        // Version
            true,               // Deleted = true (tombstone)
        );
        let new_edge_bytes = ForwardEdges::value_to_bytes(&new_edge_value)
            .map_err(|e| anyhow::anyhow!("Failed to serialize new edge version: {}", e))?;
        txn.put_cf(forward_cf, new_edge_key_bytes, new_edge_bytes)?;

        // 6. Mark summary as orphan candidate (deferred to GC)
        if let Some(hash) = current_hash {
            mark_edge_summary_orphan_candidate(txn, txn_db, hash)?;
        }

        // 7. Flip current index entry to STALE
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

// (claude, 2026-02-07, FIXED: Added MutationExecutor for RestoreNode/RestoreEdge - Codex Item 1)
// (claude, 2026-02-07, FIXED: Changed to as_of timestamp lookup per Codex review)
impl MutationExecutor for RestoreNode {
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
    ) -> Result<()> {
        tracing::debug!(
            id = %self.id,
            as_of = ?self.as_of,
            "Executing RestoreNode mutation"
        );

        use super::schema::{
            NodeCfKey, NodeCfValue, Nodes,
            NodeVersionHistory, NodeVersionHistoryCfKey, NodeVersionHistoryCfValue,
            NodeSummaryIndex, NodeSummaryIndexCfKey, NodeSummaryIndexCfValue,
        };

        // 1. Find the version that was active at as_of time in NodeVersionHistory
        let history_cf = txn_db
            .cf_handle(NodeVersionHistory::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("NodeVersionHistory CF not found"))?;

        // Scan all versions of this node to find the one active at as_of
        // Find version where UpdatedAt <= as_of (most recent version before as_of)
        let prefix = self.id.into_bytes().to_vec();
        let iter = txn.prefix_iterator_cf(history_cf, &prefix);

        let mut target_history: Option<(NodeVersionHistoryCfKey, NodeVersionHistoryCfValue)> = None;
        for item in iter {
            let (key_bytes, value_bytes) = item?;
            if !key_bytes.starts_with(&prefix) {
                break;
            }
            let key: NodeVersionHistoryCfKey = NodeVersionHistory::key_from_bytes(&key_bytes)?;
            let value: NodeVersionHistoryCfValue = NodeVersionHistory::value_from_bytes(&value_bytes)?;

            // Check if this version was created at or before as_of
            // value.0 is UpdatedAt
            if value.0 <= self.as_of {
                // Keep the most recent version that's still <= as_of
                if target_history.is_none() || value.0 > target_history.as_ref().unwrap().1.0 {
                    target_history = Some((key, value));
                }
            }
        }

        let (_history_key, history_value) = target_history
            .ok_or_else(|| anyhow::anyhow!(
                "No version found in NodeVersionHistory at or before {} for node {}",
                self.as_of.0,
                self.id
            ))?;

        // 1b. Verify referenced summary exists (not GC'd)
        // (claude, 2026-02-07, FIXED: Item 19 - Restore must verify summary exists before referencing)
        if let Some(hash) = history_value.1 {
            if !verify_node_summary_exists(txn, txn_db, hash)? {
                return Err(anyhow::anyhow!(
                    "Cannot restore node {}: summary hash {:?} was garbage collected",
                    self.id,
                    hash
                ));
            }
        }

        // 2. Find current node version (if exists) and mark as superseded
        let nodes_cf = txn_db
            .cf_handle(Nodes::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("Nodes CF not found"))?;

        let now = crate::TimestampMilli::now();
        let mut new_version = 1u32;

        // Track old summary hash for STALE marking
        let mut old_summary_hash: Option<SummaryHash> = None;
        let mut old_version: schema::Version = 0;

        if let Some((current_key_bytes, current)) = find_current_node_version(txn, txn_db, self.id)? {
            // Capture old summary info for STALE marking
            old_summary_hash = current.3;
            old_version = current.4;

            // Mark old row as superseded
            let old_node_value = NodeCfValue(
                Some(now),          // ValidUntil = now
                current.1.clone(),  // ActivePeriod
                current.2,          // NameHash
                current.3,          // SummaryHash
                current.4,          // Version
                current.5,          // Deleted
            );
            let old_node_bytes = Nodes::value_to_bytes(&old_node_value)?;
            txn.put_cf(nodes_cf, &current_key_bytes, old_node_bytes)?;
            new_version = current.4 + 1;
        }

        // 3. Create new node with restored state
        let new_node_key = NodeCfKey(self.id, now);
        let new_node_key_bytes = Nodes::key_to_bytes(&new_node_key);
        let new_node_value = NodeCfValue(
            None,               // ValidUntil = None (current)
            history_value.3,    // ActivePeriod from history
            history_value.2,    // NameHash from history
            history_value.1,    // SummaryHash from history
            new_version,        // New version
            false,              // Not deleted
        );
        let new_node_bytes = Nodes::value_to_bytes(&new_node_value)?;
        txn.put_cf(nodes_cf, new_node_key_bytes, new_node_bytes)?;

        // 4. Remove reused summary from OrphanSummaries (rollback safety)
        // (claude, 2026-02-07, FIXED: Rollback reuses old summary - remove from orphan index)
        if let Some(hash) = history_value.1 {
            remove_summary_from_orphans(txn, txn_db, hash)?;
        }

        // 5. Mark old summary index entry as STALE (if exists)
        // (claude, 2026-02-07, FIXED: Mark prior CURRENT entry as STALE per Codex Item 1)
        if let Some(old_hash) = old_summary_hash {
            let index_cf = txn_db
                .cf_handle(NodeSummaryIndex::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!("NodeSummaryIndex CF not found"))?;
            let old_index_key = NodeSummaryIndexCfKey(old_hash, self.id, old_version);
            let old_index_key_bytes = NodeSummaryIndex::key_to_bytes(&old_index_key);
            let stale_value_bytes = NodeSummaryIndex::value_to_bytes(&NodeSummaryIndexCfValue::stale())?;
            txn.put_cf(index_cf, old_index_key_bytes, stale_value_bytes)?;

            // Mark old summary as orphan candidate (may be shared, GC will verify)
            mark_node_summary_orphan_candidate(txn, txn_db, old_hash)?;
        }

        // 6. Write new summary index entry with CURRENT marker
        if let Some(hash) = history_value.1 {
            let index_cf = txn_db
                .cf_handle(NodeSummaryIndex::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!("NodeSummaryIndex CF not found"))?;
            let new_index_key = NodeSummaryIndexCfKey(hash, self.id, new_version);
            let new_index_key_bytes = NodeSummaryIndex::key_to_bytes(&new_index_key);
            let current_value_bytes = NodeSummaryIndex::value_to_bytes(&NodeSummaryIndexCfValue::current())?;
            txn.put_cf(index_cf, new_index_key_bytes, current_value_bytes)?;
        }

        // 7. Write NodeVersionHistory entry for the new version
        let new_history_key = NodeVersionHistoryCfKey(self.id, now, new_version);
        let new_history_key_bytes = NodeVersionHistory::key_to_bytes(&new_history_key);
        let new_history_value = NodeVersionHistoryCfValue(
            now,                // UpdatedAt
            history_value.1,    // SummaryHash (restored)
            history_value.2,    // NameHash (restored)
            history_value.3,    // ActivePeriod (restored)
        );
        let new_history_value_bytes = NodeVersionHistory::value_to_bytes(&new_history_value)?;
        txn.put_cf(history_cf, new_history_key_bytes, new_history_value_bytes)?;

        // (claude, 2026-02-07, FIXED: Item 1 - use as_of instead of target_version)
        tracing::info!(
            id = %self.id,
            as_of = ?self.as_of,
            new_version = new_version,
            "RestoreNode completed"
        );

        Ok(())
    }
}

impl MutationExecutor for RestoreEdge {
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
    ) -> Result<()> {
        // (claude, 2026-02-07, FIXED: Item 1 - use as_of instead of target_version)
        tracing::debug!(
            src = %self.src_id,
            dst = %self.dst_id,
            name = %self.name,
            as_of = ?self.as_of,
            "Executing RestoreEdge mutation"
        );

        use super::schema::{
            ForwardEdgeCfKey, ForwardEdgeCfValue, ForwardEdges,
            ReverseEdgeCfKey, ReverseEdgeCfValue, ReverseEdges,
            EdgeVersionHistory, EdgeVersionHistoryCfKey, EdgeVersionHistoryCfValue,
            EdgeSummaryIndex, EdgeSummaryIndexCfKey, EdgeSummaryIndexCfValue,
        };

        let name_hash = NameHash::from_name(&self.name);

        // 1. Find the version active at as_of in EdgeVersionHistory
        // (claude, 2026-02-07, FIXED: Item 1 - use as_of timestamp to find version)
        let history_cf = txn_db
            .cf_handle(EdgeVersionHistory::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("EdgeVersionHistory CF not found"))?;

        // Build prefix for this edge (src, dst, name_hash) = 40 bytes
        let mut prefix = Vec::with_capacity(40);
        prefix.extend_from_slice(&self.src_id.into_bytes());
        prefix.extend_from_slice(&self.dst_id.into_bytes());
        prefix.extend_from_slice(name_hash.as_bytes());

        let iter = txn.prefix_iterator_cf(history_cf, &prefix);

        let mut target_history: Option<(EdgeVersionHistoryCfKey, EdgeVersionHistoryCfValue)> = None;
        for item in iter {
            let (key_bytes, value_bytes) = item?;
            if !key_bytes.starts_with(&prefix) {
                break;
            }
            let key: EdgeVersionHistoryCfKey = EdgeVersionHistory::key_from_bytes(&key_bytes)?;
            let value: EdgeVersionHistoryCfValue = EdgeVersionHistory::value_from_bytes(&value_bytes)?;

            // Check if this version was created at or before as_of
            // value.0 is UpdatedAt
            if value.0 <= self.as_of {
                // Keep the most recent version that's still <= as_of
                if target_history.is_none() || value.0 > target_history.as_ref().unwrap().1.0 {
                    target_history = Some((key, value));
                }
            }
        }

        let (_history_key, history_value) = target_history
            .ok_or_else(|| anyhow::anyhow!(
                "No version found in EdgeVersionHistory at or before {} for edge {}→{}",
                self.as_of.0,
                self.src_id,
                self.dst_id
            ))?;

        // 1b. Verify referenced summary exists (not GC'd)
        // (claude, 2026-02-07, FIXED: Item 19 - Restore must verify summary exists before referencing)
        if let Some(hash) = history_value.1 {
            if !verify_edge_summary_exists(txn, txn_db, hash)? {
                return Err(anyhow::anyhow!(
                    "Cannot restore edge {}→{}: summary hash {:?} was garbage collected",
                    self.src_id,
                    self.dst_id,
                    hash
                ));
            }
        }

        // 2. Find current edge version (if exists) and mark as superseded
        let forward_cf = txn_db
            .cf_handle(ForwardEdges::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("ForwardEdges CF not found"))?;
        let reverse_cf = txn_db
            .cf_handle(ReverseEdges::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("ReverseEdges CF not found"))?;

        let now = crate::TimestampMilli::now();
        let mut new_version = 1u32;

        // Track old summary hash for STALE marking
        let mut old_summary_hash: Option<SummaryHash> = None;
        let mut old_version: schema::Version = 0;

        if let Some((current_key_bytes, current)) = find_current_forward_edge_version(txn, txn_db, self.src_id, self.dst_id, name_hash)? {
            // Capture old summary info for STALE marking
            old_summary_hash = current.3;
            old_version = current.4;

            // Mark old forward edge as superseded
            let old_edge_value = ForwardEdgeCfValue(
                Some(now),          // ValidUntil = now
                current.1.clone(),  // ActivePeriod
                current.2,          // Weight
                current.3,          // SummaryHash
                current.4,          // Version
                current.5,          // Deleted
            );
            let old_edge_bytes = ForwardEdges::value_to_bytes(&old_edge_value)?;
            txn.put_cf(forward_cf, &current_key_bytes, old_edge_bytes)?;
            new_version = current.4 + 1;

            // Mark old reverse edge as superseded
            if let Some((reverse_key_bytes, reverse)) = find_current_reverse_edge_version(txn, txn_db, self.dst_id, self.src_id, name_hash)? {
                let old_reverse_value = ReverseEdgeCfValue(
                    Some(now),          // ValidUntil = now
                    reverse.1.clone(),  // ActivePeriod
                );
                let old_reverse_bytes = ReverseEdges::value_to_bytes(&old_reverse_value)?;
                txn.put_cf(reverse_cf, &reverse_key_bytes, old_reverse_bytes)?;
            }
        }

        // 3. Create new forward edge with restored state
        let new_forward_key = ForwardEdgeCfKey(self.src_id, self.dst_id, name_hash, now);
        let new_forward_key_bytes = ForwardEdges::key_to_bytes(&new_forward_key);
        let new_forward_value = ForwardEdgeCfValue(
            None,               // ValidUntil = None (current)
            history_value.3.clone(), // ActivePeriod from history
            history_value.2,    // Weight from history
            history_value.1,    // SummaryHash from history
            new_version,        // New version
            false,              // Not deleted
        );
        let new_forward_bytes = ForwardEdges::value_to_bytes(&new_forward_value)?;
        txn.put_cf(forward_cf, new_forward_key_bytes, new_forward_bytes)?;

        // 4. Create new reverse edge
        let new_reverse_key = ReverseEdgeCfKey(self.dst_id, self.src_id, name_hash, now);
        let new_reverse_key_bytes = ReverseEdges::key_to_bytes(&new_reverse_key);
        let new_reverse_value = ReverseEdgeCfValue(
            None,               // ValidUntil = None (current)
            history_value.3.clone(), // ActivePeriod from history
        );
        let new_reverse_bytes = ReverseEdges::value_to_bytes(&new_reverse_value)?;
        txn.put_cf(reverse_cf, new_reverse_key_bytes, new_reverse_bytes)?;

        // 5. Remove reused summary from OrphanSummaries (rollback safety)
        // (claude, 2026-02-07, FIXED: Rollback reuses old summary - remove from orphan index)
        if let Some(hash) = history_value.1 {
            remove_summary_from_orphans(txn, txn_db, hash)?;
        }

        // 6. Mark old summary index entry as STALE (if exists)
        // (claude, 2026-02-07, FIXED: Mark prior CURRENT entry as STALE per Codex Item 1)
        if let Some(old_hash) = old_summary_hash {
            let index_cf = txn_db
                .cf_handle(EdgeSummaryIndex::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!("EdgeSummaryIndex CF not found"))?;
            let old_index_key = EdgeSummaryIndexCfKey(old_hash, self.src_id, self.dst_id, name_hash, old_version);
            let old_index_key_bytes = EdgeSummaryIndex::key_to_bytes(&old_index_key);
            let stale_value_bytes = EdgeSummaryIndex::value_to_bytes(&EdgeSummaryIndexCfValue::stale())?;
            txn.put_cf(index_cf, old_index_key_bytes, stale_value_bytes)?;

            // Mark old summary as orphan candidate (may be shared, GC will verify)
            mark_edge_summary_orphan_candidate(txn, txn_db, old_hash)?;
        }

        // 7. Write new summary index entry with CURRENT marker
        if let Some(hash) = history_value.1 {
            let index_cf = txn_db
                .cf_handle(EdgeSummaryIndex::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!("EdgeSummaryIndex CF not found"))?;
            let new_index_key = EdgeSummaryIndexCfKey(hash, self.src_id, self.dst_id, name_hash, new_version);
            let new_index_key_bytes = EdgeSummaryIndex::key_to_bytes(&new_index_key);
            let current_value_bytes = EdgeSummaryIndex::value_to_bytes(&EdgeSummaryIndexCfValue::current())?;
            txn.put_cf(index_cf, new_index_key_bytes, current_value_bytes)?;
        }

        // 8. Write EdgeVersionHistory entry for the new version
        let new_history_key = EdgeVersionHistoryCfKey(self.src_id, self.dst_id, name_hash, now, new_version);
        let new_history_key_bytes = EdgeVersionHistory::key_to_bytes(&new_history_key);
        let new_history_value = EdgeVersionHistoryCfValue(
            now,                // UpdatedAt
            history_value.1,    // SummaryHash (restored)
            history_value.2,    // Weight (restored)
            history_value.3,    // ActivePeriod (restored)
        );
        let new_history_value_bytes = EdgeVersionHistory::value_to_bytes(&new_history_value)?;
        txn.put_cf(history_cf, new_history_key_bytes, new_history_value_bytes)?;

        // (claude, 2026-02-07, FIXED: Item 1 - use as_of instead of target_version)
        tracing::info!(
            src = %self.src_id,
            dst = %self.dst_id,
            as_of = ?self.as_of,
            new_version = new_version,
            "RestoreEdge completed"
        );

        Ok(())
    }
}

impl MutationExecutor for RestoreEdges {
    /// Execute batch edge restore: restores all outgoing edges from src (optionally filtered by name)
    /// to their state at as_of time.
    /// (claude, 2026-02-07, FIXED: Added RestoreEdges batch API per VERSIONING.md)
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
    ) -> Result<()> {
        tracing::debug!(
            src = %self.src_id,
            name = ?self.name,
            as_of = ?self.as_of,
            "Executing RestoreEdges batch mutation"
        );

        use super::schema::{
            ForwardEdgeCfKey, ForwardEdgeCfValue, ForwardEdges,
            ReverseEdgeCfKey, ReverseEdgeCfValue, ReverseEdges,
            EdgeVersionHistory, EdgeVersionHistoryCfKey, EdgeVersionHistoryCfValue,
            EdgeSummaryIndex, EdgeSummaryIndexCfKey, EdgeSummaryIndexCfValue,
        };

        let forward_cf = txn_db
            .cf_handle(ForwardEdges::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("ForwardEdges CF not found"))?;
        let reverse_cf = txn_db
            .cf_handle(ReverseEdges::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("ReverseEdges CF not found"))?;
        let history_cf = txn_db
            .cf_handle(EdgeVersionHistory::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("EdgeVersionHistory CF not found"))?;
        let index_cf = txn_db
            .cf_handle(EdgeSummaryIndex::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("EdgeSummaryIndex CF not found"))?;

        // 1. Scan ForwardEdges to find all unique (dst, name_hash) combinations from src
        // We only need current versions (ValidUntil = None)
        let prefix = self.src_id.into_bytes().to_vec();
        let iter = txn.prefix_iterator_cf(forward_cf, &prefix);

        // Collect unique edges to restore: (dst_id, name_hash)
        let mut edges_to_restore: Vec<(Id, NameHash)> = Vec::new();
        let mut seen: std::collections::HashSet<(Id, NameHash)> = std::collections::HashSet::new();

        for item in iter {
            let (key_bytes, value_bytes) = item?;
            if !key_bytes.starts_with(&prefix) {
                break;
            }
            let key: ForwardEdgeCfKey = ForwardEdges::key_from_bytes(&key_bytes)?;
            let value: ForwardEdgeCfValue = ForwardEdges::value_from_bytes(&value_bytes)?;

            // Only consider current versions (ValidUntil = None)
            if value.0.is_some() {
                continue;
            }

            let dst_id = key.1;
            let name_hash = key.2;

            // Filter by name if specified
            if let Some(ref filter_name) = self.name {
                let filter_hash = NameHash::from_name(filter_name);
                if name_hash != filter_hash {
                    continue;
                }
            }

            // Deduplicate
            if seen.insert((dst_id, name_hash)) {
                edges_to_restore.push((dst_id, name_hash));
            }
        }

        let now = crate::TimestampMilli::now();
        let mut restored_count = 0u32;

        // 2. For each edge, find version at as_of and restore it
        for (dst_id, name_hash) in edges_to_restore {
            // Build history prefix for this edge
            let mut history_prefix = Vec::with_capacity(40);
            history_prefix.extend_from_slice(&self.src_id.into_bytes());
            history_prefix.extend_from_slice(&dst_id.into_bytes());
            history_prefix.extend_from_slice(name_hash.as_bytes());

            let history_iter = txn.prefix_iterator_cf(history_cf, &history_prefix);

            // Find version active at as_of
            let mut target_history: Option<(EdgeVersionHistoryCfKey, EdgeVersionHistoryCfValue)> = None;
            for item in history_iter {
                let (key_bytes, value_bytes) = item?;
                if !key_bytes.starts_with(&history_prefix) {
                    break;
                }
                let key: EdgeVersionHistoryCfKey = EdgeVersionHistory::key_from_bytes(&key_bytes)?;
                let value: EdgeVersionHistoryCfValue = EdgeVersionHistory::value_from_bytes(&value_bytes)?;

                // value.0 is UpdatedAt
                if value.0 <= self.as_of {
                    if target_history.is_none() || value.0 > target_history.as_ref().unwrap().1.0 {
                        target_history = Some((key, value));
                    }
                }
            }

            let Some((_history_key, history_value)) = target_history else {
                // No version found at as_of for this edge - skip
                tracing::debug!(
                    src = %self.src_id,
                    dst = %dst_id,
                    "No version found at as_of for edge, skipping"
                );
                continue;
            };

            // Verify referenced summary exists (not GC'd)
            // (claude, 2026-02-07, FIXED: Item 19 - Restore must verify summary exists before referencing)
            if let Some(hash) = history_value.1 {
                if !verify_edge_summary_exists(txn, txn_db, hash)? {
                    // Skip this edge if summary was GC'd (batch mode: warn and continue)
                    tracing::warn!(
                        src = %self.src_id,
                        dst = %dst_id,
                        hash = ?hash,
                        "Cannot restore edge: summary was garbage collected, skipping"
                    );
                    continue;
                }
            }

            // Find current version to supersede
            // (claude, 2026-02-07, FIXED: Codex Item 18 - track old summary for STALE marking)
            let mut new_version = 1u32;
            let mut old_summary_hash: Option<SummaryHash> = None;
            let mut old_version: schema::Version = 0;

            if let Some((current_key_bytes, current)) = find_current_forward_edge_version(txn, txn_db, self.src_id, dst_id, name_hash)? {
                // Capture old summary info for STALE marking
                old_summary_hash = current.3;
                old_version = current.4;

                // Mark old forward edge as superseded
                let old_edge_value = ForwardEdgeCfValue(
                    Some(now),
                    current.1.clone(),
                    current.2,
                    current.3,
                    current.4,
                    current.5,
                );
                let old_edge_bytes = ForwardEdges::value_to_bytes(&old_edge_value)?;
                txn.put_cf(forward_cf, &current_key_bytes, old_edge_bytes)?;
                new_version = current.4 + 1;

                // Mark old reverse edge as superseded
                if let Some((reverse_key_bytes, reverse)) = find_current_reverse_edge_version(txn, txn_db, dst_id, self.src_id, name_hash)? {
                    let old_reverse_value = ReverseEdgeCfValue(
                        Some(now),
                        reverse.1.clone(),
                    );
                    let old_reverse_bytes = ReverseEdges::value_to_bytes(&old_reverse_value)?;
                    txn.put_cf(reverse_cf, &reverse_key_bytes, old_reverse_bytes)?;
                }
            }

            // Create new forward edge with restored state
            let new_forward_key = ForwardEdgeCfKey(self.src_id, dst_id, name_hash, now);
            let new_forward_key_bytes = ForwardEdges::key_to_bytes(&new_forward_key);
            let new_forward_value = ForwardEdgeCfValue(
                None,
                history_value.3.clone(),
                history_value.2,
                history_value.1,
                new_version,
                false,
            );
            let new_forward_bytes = ForwardEdges::value_to_bytes(&new_forward_value)?;
            txn.put_cf(forward_cf, new_forward_key_bytes, new_forward_bytes)?;

            // Create new reverse edge
            let new_reverse_key = ReverseEdgeCfKey(dst_id, self.src_id, name_hash, now);
            let new_reverse_key_bytes = ReverseEdges::key_to_bytes(&new_reverse_key);
            let new_reverse_value = ReverseEdgeCfValue(
                None,
                history_value.3.clone(),
            );
            let new_reverse_bytes = ReverseEdges::value_to_bytes(&new_reverse_value)?;
            txn.put_cf(reverse_cf, new_reverse_key_bytes, new_reverse_bytes)?;

            // Remove reused summary from orphans
            if let Some(hash) = history_value.1 {
                remove_summary_from_orphans(txn, txn_db, hash)?;
            }

            // (claude, 2026-02-07, FIXED: Codex Item 18 - mark old summary index STALE and orphan candidate)
            // Mark old summary index entry as STALE (if exists)
            if let Some(old_hash) = old_summary_hash {
                let old_index_key = EdgeSummaryIndexCfKey(old_hash, self.src_id, dst_id, name_hash, old_version);
                let old_index_key_bytes = EdgeSummaryIndex::key_to_bytes(&old_index_key);
                let stale_value_bytes = EdgeSummaryIndex::value_to_bytes(&EdgeSummaryIndexCfValue::stale())?;
                txn.put_cf(index_cf, old_index_key_bytes, stale_value_bytes)?;

                // Mark old summary as orphan candidate (may be shared, GC will verify)
                mark_edge_summary_orphan_candidate(txn, txn_db, old_hash)?;
            }

            // Write summary index entry
            if let Some(hash) = history_value.1 {
                let new_index_key = EdgeSummaryIndexCfKey(hash, self.src_id, dst_id, name_hash, new_version);
                let new_index_key_bytes = EdgeSummaryIndex::key_to_bytes(&new_index_key);
                let current_value_bytes = EdgeSummaryIndex::value_to_bytes(&EdgeSummaryIndexCfValue::current())?;
                txn.put_cf(index_cf, new_index_key_bytes, current_value_bytes)?;
            }

            // Write history entry
            let new_history_key = EdgeVersionHistoryCfKey(self.src_id, dst_id, name_hash, now, new_version);
            let new_history_key_bytes = EdgeVersionHistory::key_to_bytes(&new_history_key);
            let new_history_value = EdgeVersionHistoryCfValue(
                now,
                history_value.1,
                history_value.2,
                history_value.3,
            );
            let new_history_value_bytes = EdgeVersionHistory::value_to_bytes(&new_history_value)?;
            txn.put_cf(history_cf, new_history_key_bytes, new_history_value_bytes)?;

            restored_count += 1;
        }

        tracing::info!(
            src = %self.src_id,
            name = ?self.name,
            as_of = ?self.as_of,
            restored_count = restored_count,
            "RestoreEdges batch completed"
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
            // (claude, 2026-02-07, FIXED: Added RestoreNode/RestoreEdge/RestoreEdges dispatch)
            Mutation::RestoreNode(m) => m.execute(txn, txn_db),
            Mutation::RestoreEdge(m) => m.execute(txn, txn_db),
            Mutation::RestoreEdges(m) => m.execute(txn, txn_db),
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
            // (claude, 2026-02-07, FIXED: Added RestoreNode/RestoreEdge/RestoreEdges dispatch)
            Mutation::RestoreNode(m) => m.execute(txn, txn_db),
            Mutation::RestoreEdge(m) => m.execute(txn, txn_db),
            Mutation::RestoreEdges(m) => m.execute(txn, txn_db),
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

// (claude, 2026-02-07, FIXED: Added Runnable impls for RestoreNode/RestoreEdge)
#[async_trait::async_trait]
impl Runnable for RestoreNode {
    async fn run(self, writer: &Writer) -> Result<()> {
        writer.send(vec![Mutation::RestoreNode(self)]).await
    }
}

#[async_trait::async_trait]
impl Runnable for RestoreEdge {
    async fn run(self, writer: &Writer) -> Result<()> {
        writer.send(vec![Mutation::RestoreEdge(self)]).await
    }
}

// (claude, 2026-02-07, FIXED: Added RestoreEdges Runnable impl per VERSIONING.md)
#[async_trait::async_trait]
impl Runnable for RestoreEdges {
    async fn run(self, writer: &Writer) -> Result<()> {
        writer.send(vec![Mutation::RestoreEdges(self)]).await
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

// (claude, 2026-02-07, FIXED: Added From impls for RestoreNode/RestoreEdge)
impl From<RestoreNode> for Mutation {
    fn from(m: RestoreNode) -> Self {
        Mutation::RestoreNode(m)
    }
}

impl From<RestoreEdge> for Mutation {
    fn from(m: RestoreEdge) -> Self {
        Mutation::RestoreEdge(m)
    }
}

// (claude, 2026-02-07, FIXED: Added RestoreEdges From impl per VERSIONING.md)
impl From<RestoreEdges> for Mutation {
    fn from(m: RestoreEdges) -> Self {
        Mutation::RestoreEdges(m)
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
