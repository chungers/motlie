//! Mutation module providing mutation types and their business logic implementations.
//!
//! This module contains only business logic - mutation type definitions and their
//! MutationExecutor implementations. Infrastructure (traits, Writer, Consumer, spawn
//! functions) is in the `writer` module.

use anyhow::Result;

use super::schema;
use super::writer::{MutationExecutor, Writer};
use crate::{Id, TimestampMilli};

#[derive(Debug, Clone)]
pub enum Mutation {
    AddNode(AddNode),
    AddEdge(AddEdge),
    AddNodeFragment(AddNodeFragment),
    AddEdgeFragment(AddEdgeFragment),
    UpdateNodeValidSinceUntil(UpdateNodeValidSinceUntil),
    UpdateEdgeValidSinceUntil(UpdateEdgeValidSinceUntil),
    UpdateEdgeWeight(UpdateEdgeWeight),
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
// Helper Functions - Shared logic for mutation execution
// ============================================================================

/// Helper function to update TemporalRange for a single node.
/// Updates both Nodes CF and NodeNames CF.
fn update_node_valid_range(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    node_id: Id,
    new_range: schema::TemporalRange,
) -> Result<()> {
    use super::{ColumnFamilyRecord, ValidRangePatchable};
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
fn update_edge_valid_range(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    src_id: Id,
    dst_id: Id,
    edge_name: &schema::EdgeName,
    new_range: schema::TemporalRange,
) -> Result<()> {
    use super::{ColumnFamilyRecord, ValidRangePatchable};
    use super::schema::{ForwardEdgeCfKey, ForwardEdges, ReverseEdgeCfKey, ReverseEdges};

    // Patch ForwardEdges CF
    let forward_cf = txn_db
        .cf_handle(ForwardEdges::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("ForwardEdges CF not found"))?;

    let forward_key = ForwardEdgeCfKey(src_id, dst_id, edge_name.clone());
    let forward_key_bytes = ForwardEdges::key_to_bytes(&forward_key);

    let forward_value_bytes = txn.get_cf(forward_cf, &forward_key_bytes)?.ok_or_else(|| {
        anyhow::anyhow!(
            "ForwardEdge not found: src={}, dst={}, name={}",
            src_id,
            dst_id,
            edge_name
        )
    })?;

    let forward_edges = ForwardEdges;
    let patched_forward_bytes = forward_edges.patch_valid_range(&forward_value_bytes, new_range)?;
    txn.put_cf(forward_cf, &forward_key_bytes, patched_forward_bytes)?;

    // Patch ReverseEdges CF
    let reverse_cf = txn_db
        .cf_handle(ReverseEdges::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("ReverseEdges CF not found"))?;

    let reverse_key = ReverseEdgeCfKey(dst_id, src_id, edge_name.clone());
    let reverse_key_bytes = ReverseEdges::key_to_bytes(&reverse_key);

    let reverse_value_bytes = txn.get_cf(reverse_cf, &reverse_key_bytes)?.ok_or_else(|| {
        anyhow::anyhow!(
            "ReverseEdge not found: src={}, dst={}, name={}",
            src_id,
            dst_id,
            edge_name
        )
    })?;

    let reverse_edges = ReverseEdges;
    let patched_reverse_bytes = reverse_edges.patch_valid_range(&reverse_value_bytes, new_range)?;
    txn.put_cf(reverse_cf, &reverse_key_bytes, patched_reverse_bytes)?;

    Ok(())
}

/// Helper function to find all edges connected to a node.
/// Returns a deduplicated list of (src_id, dst_id, edge_name) tuples.
fn find_connected_edges(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    node_id: Id,
) -> Result<Vec<(Id, Id, schema::EdgeName)>> {
    use super::ColumnFamilyRecord;
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
        // ReverseEdgeCfKey is (dst_id, src_id, edge_name), extract as (src_id, dst_id, name)
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

        use super::ColumnFamilyRecord;
        use super::schema::Nodes;

        // Write to Nodes CF
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

        use super::ColumnFamilyRecord;
        use super::schema::{ForwardEdges, ReverseEdges};

        // Write to ForwardEdges CF
        let forward_cf = txn_db.cf_handle(ForwardEdges::CF_NAME).ok_or_else(|| {
            anyhow::anyhow!("Column family '{}' not found", ForwardEdges::CF_NAME)
        })?;
        let (forward_key, forward_value) = ForwardEdges::create_bytes(self)?;
        txn.put_cf(forward_cf, forward_key, forward_value)?;

        // Write to ReverseEdges CF
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

        use super::ColumnFamilyRecord;
        use super::schema::NodeFragments;

        let cf = txn_db.cf_handle(NodeFragments::CF_NAME).ok_or_else(|| {
            anyhow::anyhow!("Column family '{}' not found", NodeFragments::CF_NAME)
        })?;
        let (key, value) = NodeFragments::create_bytes(self)?;
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

        use super::ColumnFamilyRecord;
        use super::schema::EdgeFragments;

        let cf = txn_db.cf_handle(EdgeFragments::CF_NAME).ok_or_else(|| {
            anyhow::anyhow!("Column family '{}' not found", EdgeFragments::CF_NAME)
        })?;
        let (key, value) = EdgeFragments::create_bytes(self)?;
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

        // 2. Find all connected edges (N edges)
        let edges = find_connected_edges(txn, txn_db, node_id)?;

        tracing::info!(
            node_id = %node_id,
            edge_count = edges.len(),
            "[UpdateNodeValidSinceUntil] Updating node and connected edges"
        );

        // 3. Update each edge (N × 2 operations = 2N operations)
        for (src_id, dst_id, edge_name) in edges {
            update_edge_valid_range(txn, txn_db, src_id, dst_id, &edge_name, new_range)?;
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

        // Simply delegate to the helper
        update_edge_valid_range(
            txn,
            txn_db,
            self.src_id,
            self.dst_id,
            &self.name,
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

        use super::ColumnFamilyRecord;
        use super::schema::{ForwardEdgeCfKey, ForwardEdgeCfValue, ForwardEdges};

        let cf = txn_db.cf_handle(ForwardEdges::CF_NAME).ok_or_else(|| {
            anyhow::anyhow!("Column family '{}' not found", ForwardEdges::CF_NAME)
        })?;

        let key = ForwardEdgeCfKey(self.src_id, self.dst_id, self.name.clone());
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
        }
    }
}

// ============================================================================
// Runnable Trait - Execute mutations against a Writer
// ============================================================================

/// Trait for mutations that can be executed against a Writer.
///
/// This trait follows the same pattern as the Query API's Runnable trait,
/// enabling mutations to be constructed separately from execution.
///
/// # Examples
///
/// ```rust,ignore
/// use motlie_db::{AddNode, Id, TimestampMilli, Runnable};
///
/// // Construct mutation
/// let mutation = AddNode {
///     id: Id::new(),
///     name: "Alice".to_string(),
///     ts_millis: TimestampMilli::now(),
///     valid_range: None,
/// };
///
/// // Execute it
/// mutation.run(&writer).await?;
/// ```
pub trait Runnable {
    /// Execute this mutation against the writer
    async fn run(self, writer: &Writer) -> Result<()>;
}

// Implement Runnable for individual mutation types
impl Runnable for AddNode {
    async fn run(self, writer: &Writer) -> Result<()> {
        writer.send(vec![Mutation::AddNode(self)]).await
    }
}

impl Runnable for AddEdge {
    async fn run(self, writer: &Writer) -> Result<()> {
        writer.send(vec![Mutation::AddEdge(self)]).await
    }
}

impl Runnable for AddNodeFragment {
    async fn run(self, writer: &Writer) -> Result<()> {
        writer.send(vec![Mutation::AddNodeFragment(self)]).await
    }
}

impl Runnable for AddEdgeFragment {
    async fn run(self, writer: &Writer) -> Result<()> {
        writer.send(vec![Mutation::AddEdgeFragment(self)]).await
    }
}

impl Runnable for UpdateNodeValidSinceUntil {
    async fn run(self, writer: &Writer) -> Result<()> {
        writer
            .send(vec![Mutation::UpdateNodeValidSinceUntil(self)])
            .await
    }
}

impl Runnable for UpdateEdgeValidSinceUntil {
    async fn run(self, writer: &Writer) -> Result<()> {
        writer
            .send(vec![Mutation::UpdateEdgeValidSinceUntil(self)])
            .await
    }
}

impl Runnable for UpdateEdgeWeight {
    async fn run(self, writer: &Writer) -> Result<()> {
        writer.send(vec![Mutation::UpdateEdgeWeight(self)]).await
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
