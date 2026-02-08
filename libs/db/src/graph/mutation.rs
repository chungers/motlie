//! Mutation module providing mutation types and dispatch glue.
//!
//! Business logic lives in `graph::ops`. MutationExecutor impls delegate to ops helpers.
//! Infrastructure (traits, Writer, Consumer, spawn functions) is in the `writer` module.

use std::sync::Mutex;

use anyhow::Result;
use tokio::sync::oneshot;

use crate::rocksdb::MutationCodec;

use super::name_hash::NameHash;
use super::ops;
use super::schema::{
    self, EdgeFragmentCfKey, EdgeFragmentCfValue, EdgeFragments, NodeFragmentCfKey,
    NodeFragmentCfValue, NodeFragments, Version,
};
use super::writer::{MutationExecutor, Writer};
use crate::writer::Runnable;
use crate::{Id, TimestampMilli};
use crate::request::{ReplyEnvelope, RequestMeta};

// ============================================================================
// Execution Options + Replies
// ============================================================================

/// Execution options applied at runtime (not part of mutation payload).
#[derive(Debug, Clone, Copy, Default)]
pub struct ExecOptions {
    pub dry_run: bool,
}

/// Mutation execution result used by run_with_result paths.
#[derive(Debug, Clone)]
pub enum MutationResult {
    AddNode { id: Id, version: Version },
    AddEdge { version: Version },
    AddNodeFragment,
    AddEdgeFragment,
    UpdateNode { id: Id, version: Version },
    UpdateEdge { version: Version },
    DeleteNode { id: Id, version: Version },
    DeleteEdge { version: Version },
    RestoreNode { id: Id, version: Version },
    RestoreEdge { version: Version },
    Flush,
}

impl MutationResult {
    fn as_update_node(self) -> Result<(Id, Version)> {
        match self {
            MutationResult::UpdateNode { id, version } => Ok((id, version)),
            other => Err(anyhow::anyhow!("Unexpected mutation result: {:?}", other)),
        }
    }

    fn as_add_node(self) -> Result<(Id, Version)> {
        match self {
            MutationResult::AddNode { id, version } => Ok((id, version)),
            other => Err(anyhow::anyhow!("Unexpected mutation result: {:?}", other)),
        }
    }

    fn as_restore_node(self) -> Result<(Id, Version)> {
        match self {
            MutationResult::RestoreNode { id, version } => Ok((id, version)),
            other => Err(anyhow::anyhow!("Unexpected mutation result: {:?}", other)),
        }
    }

    fn as_delete_node(self) -> Result<(Id, Version)> {
        match self {
            MutationResult::DeleteNode { id, version } => Ok((id, version)),
            other => Err(anyhow::anyhow!("Unexpected mutation result: {:?}", other)),
        }
    }

    fn as_add_edge(self) -> Result<Version> {
        match self {
            MutationResult::AddEdge { version } => Ok(version),
            other => Err(anyhow::anyhow!("Unexpected mutation result: {:?}", other)),
        }
    }

    fn as_update_edge(self) -> Result<Version> {
        match self {
            MutationResult::UpdateEdge { version } => Ok(version),
            other => Err(anyhow::anyhow!("Unexpected mutation result: {:?}", other)),
        }
    }

    fn as_restore_edge(self) -> Result<Version> {
        match self {
            MutationResult::RestoreEdge { version } => Ok(version),
            other => Err(anyhow::anyhow!("Unexpected mutation result: {:?}", other)),
        }
    }

    fn as_delete_edge(self) -> Result<Version> {
        match self {
            MutationResult::DeleteEdge { version } => Ok(version),
            other => Err(anyhow::anyhow!("Unexpected mutation result: {:?}", other)),
        }
    }
}

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
    // CONTENT-ADDRESS: Update/Delete with optimistic locking
    /// Consolidated node update: active_period and/or summary
    UpdateNode(UpdateNode),
    /// Consolidated edge update: weight, active_period, and/or summary
    UpdateEdge(UpdateEdge),
    DeleteNode(DeleteNode),
    DeleteEdge(DeleteEdge),
    // (codex, 2026-02-07, eval: VERSIONING plan includes RestoreNode/RestoreEdge and rollback mutations; missing here, so rollback API is incomplete.)
    // (claude, 2026-02-07, FIXED: Added RestoreNode/RestoreEdge mutations below)
    RestoreNode(RestoreNode),
    RestoreEdge(RestoreEdge),

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

impl RequestMeta for AddNode {
    type Reply = ReplyEnvelope<(Id, Version)>;
    type Options = ExecOptions;

    fn request_kind(&self) -> &'static str {
        "add_node"
    }
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

impl RequestMeta for AddEdge {
    type Reply = ReplyEnvelope<Version>;
    type Options = ExecOptions;

    fn request_kind(&self) -> &'static str {
        "add_edge"
    }
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

impl RequestMeta for AddNodeFragment {
    type Reply = ReplyEnvelope<()>;
    type Options = ExecOptions;

    fn request_kind(&self) -> &'static str {
        "add_node_fragment"
    }
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

impl RequestMeta for AddEdgeFragment {
    type Reply = ReplyEnvelope<()>;
    type Options = ExecOptions;

    fn request_kind(&self) -> &'static str {
        "add_edge_fragment"
    }
}

/// Consolidated node update mutation with optimistic locking.
///
/// All updates require node id and expected_version.
/// Mutable fields are optional - only non-None fields are updated.
/// At least one optional field must be Some for the mutation to have effect.
///
/// ## Version Semantics
/// - `expected_version`: Must match current node version (optimistic locking)
/// - On success, creates a new version
/// - On version mismatch, returns `Err(VersionMismatch)`
///
/// ## Example
/// ```rust,ignore
/// use motlie_db::graph::UpdateNode;
///
/// // Update summary only
/// let update = UpdateNode {
///     id: node_id,
///     expected_version: 1,
///     new_active_period: None,
///     new_summary: Some(NodeSummary::from_text("Updated description")),
/// };
/// update.run(&writer).await?;
/// ```
#[derive(Debug, Clone)]
pub struct UpdateNode {
    /// The UUID of the Node to update
    pub id: Id,

    /// Expected version for optimistic locking.
    /// If the current version doesn't match, the update fails.
    pub expected_version: schema::Version,

    /// Optional: Update active period (valid_since, valid_until)
    /// - `None` = no change
    /// - `Some(None)` = reset/clear active period
    /// - `Some(Some(period))` = set to specific period
    pub new_active_period: Option<Option<schema::ActivePeriod>>,

    /// Optional: Update node summary/description
    pub new_summary: Option<schema::NodeSummary>,
}

impl RequestMeta for UpdateNode {
    type Reply = ReplyEnvelope<(Id, Version)>;
    type Options = ExecOptions;

    fn request_kind(&self) -> &'static str {
        "update_node"
    }
}

/// Consolidated edge update mutation with optimistic locking.
///
/// All updates require edge identity (src_id, dst_id, name) and expected_version.
/// Mutable fields are optional - only non-None fields are updated.
/// At least one optional field must be Some for the mutation to have effect.
///
/// ## Version Semantics
/// - `expected_version`: Must match current edge version (optimistic locking)
/// - On success, creates a new version
/// - On version mismatch, returns `Err(VersionMismatch)`
///
/// ## Example
/// ```rust,ignore
/// use motlie_db::graph::UpdateEdge;
///
/// // Update weight only
/// let update = UpdateEdge {
///     src_id,
///     dst_id,
///     name: "knows".to_string(),
///     expected_version: 1,
///     new_weight: Some(Some(0.8)),
///     new_active_period: None,
///     new_summary: None,
/// };
/// update.run(&writer).await?;
/// ```
#[derive(Debug, Clone)]
pub struct UpdateEdge {
    /// The UUID of the source Node
    pub src_id: Id,

    /// The UUID of the destination Node
    pub dst_id: Id,

    /// The name of the Edge
    pub name: schema::EdgeName,

    /// Expected version for optimistic locking.
    /// If the current version doesn't match, the update fails.
    pub expected_version: schema::Version,

    /// Optional: Update edge weight
    /// - `None` = no change
    /// - `Some(None)` = reset/clear weight
    /// - `Some(Some(value))` = set to specific weight
    pub new_weight: Option<Option<schema::EdgeWeight>>,

    /// Optional: Update active period (valid_since, valid_until)
    /// - `None` = no change
    /// - `Some(None)` = reset/clear active period
    /// - `Some(Some(period))` = set to specific period
    pub new_active_period: Option<Option<schema::ActivePeriod>>,

    /// Optional: Update edge summary/description
    pub new_summary: Option<schema::EdgeSummary>,
}

impl RequestMeta for UpdateEdge {
    type Reply = ReplyEnvelope<schema::Version>;
    type Options = ExecOptions;

    fn request_kind(&self) -> &'static str {
        "update_edge"
    }
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

impl RequestMeta for DeleteNode {
    type Reply = ReplyEnvelope<(Id, Version)>;
    type Options = ExecOptions;

    fn request_kind(&self) -> &'static str {
        "delete_node"
    }
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

impl RequestMeta for DeleteEdge {
    type Reply = ReplyEnvelope<Version>;
    type Options = ExecOptions;

    fn request_kind(&self) -> &'static str {
        "delete_edge"
    }
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

    /// Optional optimistic check against current version (if present).
    pub expected_version: Option<Version>,
}

impl RequestMeta for RestoreNode {
    type Reply = ReplyEnvelope<(Id, Version)>;
    type Options = ExecOptions;

    fn request_kind(&self) -> &'static str {
        "restore_node"
    }
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

    /// Optional optimistic check against current version (if present).
    pub expected_version: Option<Version>,
}

impl RequestMeta for RestoreEdge {
    type Reply = ReplyEnvelope<Version>;
    type Options = ExecOptions;

    fn request_kind(&self) -> &'static str {
        "restore_edge"
    }
}

// ============================================================================
// MutationReply - typed reply mapping for each mutation type
// ============================================================================

impl MutationReply for AddNode {
    type Reply = (Id, Version);

    fn from_result(result: MutationResult) -> Result<Self::Reply> {
        result.as_add_node()
    }
}

impl MutationReply for AddEdge {
    type Reply = Version;

    fn from_result(result: MutationResult) -> Result<Self::Reply> {
        result.as_add_edge()
    }
}

impl MutationReply for AddNodeFragment {
    type Reply = ();

    fn from_result(result: MutationResult) -> Result<Self::Reply> {
        match result {
            MutationResult::AddNodeFragment => Ok(()),
            other => Err(anyhow::anyhow!("Unexpected mutation result: {:?}", other)),
        }
    }
}

impl MutationReply for AddEdgeFragment {
    type Reply = ();

    fn from_result(result: MutationResult) -> Result<Self::Reply> {
        match result {
            MutationResult::AddEdgeFragment => Ok(()),
            other => Err(anyhow::anyhow!("Unexpected mutation result: {:?}", other)),
        }
    }
}

impl MutationReply for UpdateNode {
    type Reply = (Id, Version);

    fn from_result(result: MutationResult) -> Result<Self::Reply> {
        result.as_update_node()
    }
}

impl MutationReply for UpdateEdge {
    type Reply = Version;

    fn from_result(result: MutationResult) -> Result<Self::Reply> {
        result.as_update_edge()
    }
}

impl MutationReply for DeleteNode {
    type Reply = (Id, Version);

    fn from_result(result: MutationResult) -> Result<Self::Reply> {
        result.as_delete_node()
    }
}

impl MutationReply for DeleteEdge {
    type Reply = Version;

    fn from_result(result: MutationResult) -> Result<Self::Reply> {
        result.as_delete_edge()
    }
}

impl MutationReply for RestoreNode {
    type Reply = (Id, Version);

    fn from_result(result: MutationResult) -> Result<Self::Reply> {
        result.as_restore_node()
    }
}

impl MutationReply for RestoreEdge {
    type Reply = Version;

    fn from_result(result: MutationResult) -> Result<Self::Reply> {
        result.as_restore_edge()
    }
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
// MutationExecutor Implementations
// ============================================================================

impl MutationExecutor for AddNode {
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
    ) -> Result<()> {
        ops::node::add_node(txn, txn_db, self, None).map(|_| ())
    }

    fn execute_with_cache(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        cache: &super::name_hash::NameCache,
    ) -> Result<()> {
        ops::node::add_node(txn, txn_db, self, Some(cache)).map(|_| ())
    }

    fn execute_with_cache_and_options(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        cache: &super::name_hash::NameCache,
        _options: ExecOptions,
    ) -> Result<MutationResult> {
        let (node_id, version) = ops::node::add_node(txn, txn_db, self, Some(cache))?;
        Ok(MutationResult::AddNode {
            id: node_id,
            version,
        })
    }
}

impl MutationExecutor for AddEdge {
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
    ) -> Result<()> {
        ops::edge::add_edge(txn, txn_db, self, None).map(|_| ())
    }

    fn execute_with_cache(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        cache: &super::name_hash::NameCache,
    ) -> Result<()> {
        ops::edge::add_edge(txn, txn_db, self, Some(cache)).map(|_| ())
    }

    fn execute_with_cache_and_options(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        cache: &super::name_hash::NameCache,
        _options: ExecOptions,
    ) -> Result<MutationResult> {
        let version = ops::edge::add_edge(txn, txn_db, self, Some(cache))?;
        Ok(MutationResult::AddEdge { version })
    }
}

impl MutationExecutor for AddNodeFragment {
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
    ) -> Result<()> {
        ops::fragment::add_node_fragment(txn, txn_db, self)
    }

    fn execute_with_options(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        _options: ExecOptions,
    ) -> Result<MutationResult> {
        self.execute(txn, txn_db)?;
        Ok(MutationResult::AddNodeFragment)
    }
}

impl MutationExecutor for AddEdgeFragment {
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
    ) -> Result<()> {
        ops::fragment::add_edge_fragment(txn, txn_db, self, None)
    }

    fn execute_with_cache(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        cache: &super::name_hash::NameCache,
    ) -> Result<()> {
        ops::fragment::add_edge_fragment(txn, txn_db, self, Some(cache))
    }

    fn execute_with_options(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        _options: ExecOptions,
    ) -> Result<MutationResult> {
        self.execute(txn, txn_db)?;
        Ok(MutationResult::AddEdgeFragment)
    }
}

impl MutationExecutor for UpdateNode {
    /// Consolidated node update with optimistic locking.
    /// Updates any combination of active_period and summary.
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
    ) -> Result<()> {
        ops::node::update_node(txn, txn_db, self).map(|_| ())
    }

    fn execute_with_options(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        _options: ExecOptions,
    ) -> Result<MutationResult> {
        let (node_id, version) = ops::node::update_node(txn, txn_db, self)?;
        Ok(MutationResult::UpdateNode {
            id: node_id,
            version,
        })
    }
}

impl MutationExecutor for UpdateEdge {
    /// Consolidated edge update with optimistic locking.
    /// Updates any combination of weight, active_period, and summary.
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
    ) -> Result<()> {
        ops::edge::update_edge(txn, txn_db, self).map(|_| ())
    }

    fn execute_with_options(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        _options: ExecOptions,
    ) -> Result<MutationResult> {
        let version = ops::edge::update_edge(txn, txn_db, self)?;
        Ok(MutationResult::UpdateEdge { version })
    }
}



impl MutationExecutor for DeleteNode {
    /// (claude, 2026-02-06, in-progress: VERSIONING prefix scan and new field indices)
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
    ) -> Result<()> {
        ops::node::delete_node(txn, txn_db, self)
    }

    fn execute_with_options(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        _options: ExecOptions,
    ) -> Result<MutationResult> {
        ops::node::delete_node(txn, txn_db, self)?;
        Ok(MutationResult::DeleteNode {
            id: self.id,
            version: self.expected_version + 1,
        })
    }
}

impl MutationExecutor for DeleteEdge {
    /// (claude, 2026-02-06, in-progress: VERSIONING prefix scan and new field indices)
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
    ) -> Result<()> {
        ops::edge::delete_edge(txn, txn_db, self)
    }

    fn execute_with_options(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        _options: ExecOptions,
    ) -> Result<MutationResult> {
        ops::edge::delete_edge(txn, txn_db, self)?;
        Ok(MutationResult::DeleteEdge {
            version: self.expected_version + 1,
        })
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
        ops::node::restore_node(txn, txn_db, self).map(|_| ())
    }

    fn execute_with_options(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        _options: ExecOptions,
    ) -> Result<MutationResult> {
        let (node_id, version) = ops::node::restore_node(txn, txn_db, self)?;
        Ok(MutationResult::RestoreNode {
            id: node_id,
            version,
        })
    }
}

impl MutationExecutor for RestoreEdge {
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
    ) -> Result<()> {
        ops::edge::restore_edge(txn, txn_db, self).map(|_| ())
    }

    fn execute_with_options(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        _options: ExecOptions,
    ) -> Result<MutationResult> {
        let version = ops::edge::restore_edge(txn, txn_db, self)?;
        Ok(MutationResult::RestoreEdge { version })
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
            // CONTENT-ADDRESS: Update/Delete with optimistic locking
            Mutation::UpdateNode(m) => m.execute(txn, txn_db),
            Mutation::UpdateEdge(m) => m.execute(txn, txn_db),
            Mutation::DeleteNode(m) => m.execute(txn, txn_db),
            Mutation::DeleteEdge(m) => m.execute(txn, txn_db),
            // (claude, 2026-02-07, FIXED: Added RestoreNode/RestoreEdge dispatch)
            Mutation::RestoreNode(m) => m.execute(txn, txn_db),
            Mutation::RestoreEdge(m) => m.execute(txn, txn_db),
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
            // CONTENT-ADDRESS: Update/Delete with optimistic locking (no new names)
            Mutation::UpdateNode(m) => m.execute(txn, txn_db), // No new names
            Mutation::UpdateEdge(m) => m.execute(txn, txn_db), // No new names
            Mutation::DeleteNode(m) => m.execute(txn, txn_db),
            Mutation::DeleteEdge(m) => m.execute(txn, txn_db),
            // (claude, 2026-02-07, FIXED: Added RestoreNode/RestoreEdge dispatch)
            Mutation::RestoreNode(m) => m.execute(txn, txn_db),
            Mutation::RestoreEdge(m) => m.execute(txn, txn_db),
            Mutation::Flush(_) => Ok(()),
        }
    }

    /// Execute this mutation with access to the name cache and runtime options.
    pub fn execute_with_cache_and_options(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        cache: &super::name_hash::NameCache,
        options: ExecOptions,
    ) -> Result<MutationResult> {
        match self {
            Mutation::AddNode(m) => m.execute_with_cache_and_options(txn, txn_db, cache, options),
            Mutation::AddEdge(m) => m.execute_with_cache_and_options(txn, txn_db, cache, options),
            Mutation::AddNodeFragment(m) => m.execute_with_options(txn, txn_db, options),
            Mutation::AddEdgeFragment(m) => m.execute_with_cache_and_options(txn, txn_db, cache, options),
            Mutation::UpdateNode(m) => m.execute_with_options(txn, txn_db, options),
            Mutation::UpdateEdge(m) => m.execute_with_options(txn, txn_db, options),
            Mutation::DeleteNode(m) => m.execute_with_options(txn, txn_db, options),
            Mutation::DeleteEdge(m) => m.execute_with_options(txn, txn_db, options),
            Mutation::RestoreNode(m) => m.execute_with_options(txn, txn_db, options),
            Mutation::RestoreEdge(m) => m.execute_with_options(txn, txn_db, options),
            Mutation::Flush(_) => Ok(MutationResult::Flush),
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

// CONTENT-ADDRESS: Runnable for Update/Delete mutations
#[async_trait::async_trait]
impl Runnable for UpdateNode {
    async fn run(self, writer: &Writer) -> Result<()> {
        writer.send(vec![Mutation::UpdateNode(self)]).await
    }
}

#[async_trait::async_trait]
impl Runnable for UpdateEdge {
    async fn run(self, writer: &Writer) -> Result<()> {
        writer.send(vec![Mutation::UpdateEdge(self)]).await
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

// ============================================================================
// RunnableWithResult - Typed replies for single mutations
// ============================================================================

#[async_trait::async_trait]
pub trait RunnableWithResult<R> {
    async fn run_with_result(self, writer: &Writer, options: ExecOptions) -> Result<ReplyEnvelope<R>>;
}

pub trait MutationReply: Into<Mutation> + Send {
    type Reply;
    fn from_result(result: MutationResult) -> Result<Self::Reply>;
}

fn single_result(
    reply: ReplyEnvelope<Vec<MutationResult>>,
) -> Result<ReplyEnvelope<MutationResult>> {
    let mut results = reply.payload;
    if results.len() != 1 {
        return Err(anyhow::anyhow!(
            "Expected exactly one mutation result, got {}",
            results.len()
        ));
    }
    Ok(ReplyEnvelope::new(
        reply.request_id,
        reply.elapsed_time,
        results.remove(0),
    ))
}

#[async_trait::async_trait]
impl<M> RunnableWithResult<M::Reply> for M
where
    M: MutationReply,
{
    async fn run_with_result(
        self,
        writer: &Writer,
        options: ExecOptions,
    ) -> Result<ReplyEnvelope<M::Reply>> {
        let reply = writer.send_with_result(vec![self.into()], options).await?;
        let single = single_result(reply)?;
        let payload = M::from_result(single.payload)?;
        Ok(ReplyEnvelope::new(
            single.request_id,
            single.elapsed_time,
            payload,
        ))
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

// CONTENT-ADDRESS: From for Update/Delete mutations
impl From<UpdateNode> for Mutation {
    fn from(m: UpdateNode) -> Self {
        Mutation::UpdateNode(m)
    }
}

impl From<UpdateEdge> for Mutation {
    fn from(m: UpdateEdge) -> Self {
        Mutation::UpdateEdge(m)
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


// ============================================================================
// Vec<Mutation> - Zero-overhead batching
// ============================================================================

impl RequestMeta for Vec<Mutation> {
    type Reply = ReplyEnvelope<Vec<MutationResult>>;
    type Options = ExecOptions;

    fn request_kind(&self) -> &'static str {
        "mutation_batch"
    }
}

#[async_trait::async_trait]
impl Runnable for Vec<Mutation> {
    async fn run(self, writer: &Writer) -> Result<()> {
        writer.send(self).await
    }
}

/// Extension trait for batch-only helpers on Vec<Mutation>.
#[async_trait::async_trait]
pub trait MutationBatchExt {
    async fn run_with_result(
        self,
        writer: &Writer,
        options: ExecOptions,
    ) -> Result<ReplyEnvelope<Vec<MutationResult>>>;
}

#[async_trait::async_trait]
impl MutationBatchExt for Vec<Mutation> {
    async fn run_with_result(
        self,
        writer: &Writer,
        options: ExecOptions,
    ) -> Result<ReplyEnvelope<Vec<MutationResult>>> {
        writer.send_with_result(self, options).await
    }
}

// ============================================================================
// mutations![] Macro - Ergonomic batch construction
// ============================================================================

/// Convenience macro for creating a Vec<Mutation> with automatic type conversion.
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
        Vec::<$crate::Mutation>::new()
    };
    ($($mutation:expr),+ $(,)?) => {
        vec![$($mutation.into()),+]
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
