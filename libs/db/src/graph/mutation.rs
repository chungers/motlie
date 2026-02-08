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
    NodeFragmentCfValue, NodeFragments,
};
use super::writer::{MutationExecutor, Writer};
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
    UpdateNodeActivePeriod(UpdateNodeActivePeriod),
    UpdateEdgeActivePeriod(UpdateEdgeActivePeriod),
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
pub struct UpdateNodeActivePeriod {
    /// The UUID of the Node
    pub id: Id,

    /// The temporal validity range for this fragment
    pub temporal_range: schema::ActivePeriod,

    /// The reason for invalidation
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct UpdateEdgeActivePeriod {
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
/// new current edges with that state. Strict by default (missing summaries
/// fail); set `dry_run` to validate and use `Transaction::validate_restore_edges`
/// for a structured report.
#[derive(Debug, Clone)]
pub struct RestoreEdges {
    /// The UUID of the source Node
    pub src_id: Id,

    /// Optional edge name filter (None = restore all edge names)
    pub name: Option<schema::EdgeName>,

    /// The timestamp to restore to (finds version active at this time)
    pub as_of: crate::TimestampMilli,

    /// Dry run: validate and report blockers without writing.
    pub dry_run: bool,
}

#[derive(Debug, Clone)]
pub struct RestoreEdgesReport {
    pub candidates: u32,
    pub restorable: u32,
    pub skipped_no_version: Vec<(Id, NameHash)>,
}

impl RestoreEdges {
    /// Create a RestoreEdges request (strict, writes enabled by default).
    pub fn new(src_id: Id, name: Option<schema::EdgeName>, as_of: crate::TimestampMilli) -> Self {
        Self {
            src_id,
            name,
            as_of,
            dry_run: false,
        }
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
        ops::node::add_node(txn, txn_db, self, None)
    }

    fn execute_with_cache(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        cache: &super::name_hash::NameCache,
    ) -> Result<()> {
        ops::node::add_node(txn, txn_db, self, Some(cache))
    }
}

impl MutationExecutor for AddEdge {
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
    ) -> Result<()> {
        ops::edge::add_edge(txn, txn_db, self, None)
    }

    fn execute_with_cache(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        cache: &super::name_hash::NameCache,
    ) -> Result<()> {
        ops::edge::add_edge(txn, txn_db, self, Some(cache))
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
}

impl MutationExecutor for UpdateNodeActivePeriod {
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
    ) -> Result<()> {
        ops::node::update_node_active_period(txn, txn_db, self)
    }
}

impl MutationExecutor for UpdateEdgeActivePeriod {
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
    ) -> Result<()> {
        ops::edge::update_edge_active_period(txn, txn_db, self)
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
        ops::edge::update_edge_weight(txn, txn_db, self)
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
        ops::node::update_node_summary(txn, txn_db, self)
    }
}

impl MutationExecutor for UpdateEdgeSummary {
    /// (claude, 2026-02-06, in-progress: VERSIONING prefix scan and new field indices)
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
    ) -> Result<()> {
        ops::edge::update_edge_summary(txn, txn_db, self)
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
}

// (claude, 2026-02-07, FIXED: Added MutationExecutor for RestoreNode/RestoreEdge - Codex Item 1)
// (claude, 2026-02-07, FIXED: Changed to as_of timestamp lookup per Codex review)
impl MutationExecutor for RestoreNode {
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
    ) -> Result<()> {
        ops::node::restore_node(txn, txn_db, self)
    }
}

impl MutationExecutor for RestoreEdge {
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
    ) -> Result<()> {
        ops::edge::restore_edge(txn, txn_db, self)
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
        ops::edge::restore_edges(txn, txn_db, self)
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
            Mutation::UpdateNodeActivePeriod(m) => m.execute(txn, txn_db),
            Mutation::UpdateEdgeActivePeriod(m) => m.execute(txn, txn_db),
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
            Mutation::UpdateNodeActivePeriod(m) => m.execute(txn, txn_db), // No new names
            Mutation::UpdateEdgeActivePeriod(m) => m.execute(txn, txn_db), // No new names
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
impl Runnable for UpdateNodeActivePeriod {
    async fn run(self, writer: &Writer) -> Result<()> {
        writer
            .send(vec![Mutation::UpdateNodeActivePeriod(self)])
            .await
    }
}

#[async_trait::async_trait]
impl Runnable for UpdateEdgeActivePeriod {
    async fn run(self, writer: &Writer) -> Result<()> {
        writer
            .send(vec![Mutation::UpdateEdgeActivePeriod(self)])
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

impl From<UpdateNodeActivePeriod> for Mutation {
    fn from(m: UpdateNodeActivePeriod) -> Self {
        Mutation::UpdateNodeActivePeriod(m)
    }
}

impl From<UpdateEdgeActivePeriod> for Mutation {
    fn from(m: UpdateEdgeActivePeriod) -> Self {
        Mutation::UpdateEdgeActivePeriod(m)
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
