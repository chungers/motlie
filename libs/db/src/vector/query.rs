//! Vector query types and executors.
//!
//! This module contains query type definitions and their execution logic
//! for vector storage operations. Following the pattern from `graph::query`,
//! queries are grouped in an enum for type-safe dispatch.
//!
//! # Phase 1 Queries (Point Lookups)
//!
//! - `GetVector` - Retrieve vector by external ID
//! - `GetInternalId` - Resolve external ID to internal vec_id
//! - `GetExternalId` - Resolve internal vec_id to external ID
//! - `ResolveIds` - Batch resolve vec_ids to external IDs
//!
//! # Phase 2 Queries (Search - requires HNSW)
//!
//! - `SearchKNN` - K-nearest neighbor search
//! - `SearchKNNFiltered` - Filtered KNN search
//!
//! # Phase 3+ Queries (Graph Introspection)
//!
//! - `GetGraphMeta` - HNSW graph metadata
//! - `GetNeighbors` - Get HNSW neighbors at a layer

use std::time::Duration;

use anyhow::Result;
use tokio::sync::oneshot;

use super::schema::{
    EmbeddingCode, IdForward, IdForwardCfKey, IdReverse, IdReverseCfKey, VecId, VectorCfKey,
    Vectors,
};
use super::Storage;
use crate::rocksdb::ColumnFamily;
use crate::Id;

// ============================================================================
// Query Enum
// ============================================================================

/// Query enum for vector storage operations.
///
/// All vector queries are variants of this enum, enabling type-safe
/// dispatch in query consumers.
#[derive(Debug)]
pub enum Query {
    // ─────────────────────────────────────────────────────────────
    // Point Lookups (Phase 1)
    // ─────────────────────────────────────────────────────────────
    /// Get vector by external ID
    GetVector(GetVectorDispatch),
    /// Get internal vec_id for external ID
    GetInternalId(GetInternalIdDispatch),
    /// Get external ID for internal vec_id
    GetExternalId(GetExternalIdDispatch),
    /// Batch resolve vec_ids to external IDs
    ResolveIds(ResolveIdsDispatch),

    // ─────────────────────────────────────────────────────────────
    // Search Operations (Phase 2 - requires HNSW)
    // ─────────────────────────────────────────────────────────────
    // SearchKNN(SearchKNNDispatch),
    // SearchKNNFiltered(SearchKNNFilteredDispatch),

    // ─────────────────────────────────────────────────────────────
    // Graph Introspection (Phase 3+)
    // ─────────────────────────────────────────────────────────────
    // GetGraphMeta(GetGraphMetaDispatch),
    // GetNeighbors(GetNeighborsDispatch),
}

impl std::fmt::Display for Query {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Query::GetVector(q) => {
                write!(f, "GetVector: embedding={}, id={}", q.params.embedding, q.params.id)
            }
            Query::GetInternalId(q) => {
                write!(f, "GetInternalId: embedding={}, id={}", q.params.embedding, q.params.id)
            }
            Query::GetExternalId(q) => write!(
                f,
                "GetExternalId: embedding={}, vec_id={}",
                q.params.embedding, q.params.vec_id
            ),
            Query::ResolveIds(q) => write!(
                f,
                "ResolveIds: embedding={}, count={}",
                q.params.embedding,
                q.params.vec_ids.len()
            ),
        }
    }
}

// ============================================================================
// QueryExecutor Trait
// ============================================================================

/// Trait for executing queries against vector storage.
///
/// Each query type implements this to define its execution logic.
/// This follows the same pattern as `graph::reader::QueryExecutor`.
#[async_trait::async_trait]
pub trait QueryExecutor: Send + Sync {
    /// The type of result this query produces
    type Output: Send;

    /// Execute this query against the storage layer
    async fn execute(&self, storage: &Storage) -> Result<Self::Output>;

    /// Get the timeout for this query
    fn timeout(&self) -> Duration;
}

// ============================================================================
// QueryWithTimeout Trait - Blanket implementation for QueryExecutor
// ============================================================================

/// Trait for queries that produce results with timeout handling.
/// Automatically implemented for all QueryExecutor types.
#[async_trait::async_trait]
pub trait QueryWithTimeout: Send + Sync {
    /// The type of result this query produces
    type ResultType: Send;

    /// Execute the query with timeout and return the result
    async fn result(&self, storage: &Storage) -> Result<Self::ResultType>;

    /// Get the timeout for this query
    fn timeout(&self) -> Duration;
}

/// Blanket implementation: any QueryExecutor automatically gets QueryWithTimeout
#[async_trait::async_trait]
impl<T: QueryExecutor> QueryWithTimeout for T {
    type ResultType = T::Output;

    async fn result(&self, storage: &Storage) -> Result<Self::ResultType> {
        let result = tokio::time::timeout(self.timeout(), self.execute(storage)).await;

        match result {
            Ok(r) => r,
            Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", self.timeout())),
        }
    }

    fn timeout(&self) -> Duration {
        QueryExecutor::timeout(self)
    }
}

// ============================================================================
// QueryProcessor Trait
// ============================================================================

/// Trait for processing queries without needing to know the result type.
/// Allows the Consumer to process queries polymorphically.
#[async_trait::async_trait]
pub trait QueryProcessor: Send {
    /// Process the query and send the result (consumes self)
    async fn process_and_send(self, storage: &Storage);
}

// ============================================================================
// GetVector - Retrieve vector by external ID
// ============================================================================

/// Get vector by external ID.
///
/// This query retrieves the raw vector data for a given external ID
/// within an embedding space.
///
/// # Example
///
/// ```rust,ignore
/// let query = GetVector::new(embedding_code, external_id);
/// let vector_opt = query.execute(&storage).await?;
/// ```
#[derive(Debug, Clone)]
pub struct GetVector {
    /// Embedding space code
    pub embedding: EmbeddingCode,
    /// External document ID
    pub id: Id,
}

impl GetVector {
    /// Create a new GetVector query.
    pub fn new(embedding: EmbeddingCode, id: Id) -> Self {
        Self { embedding, id }
    }
}

/// Dispatch wrapper with oneshot channel for GetVector.
#[derive(Debug)]
pub(crate) struct GetVectorDispatch {
    pub(crate) params: GetVector,
    pub(crate) timeout: Duration,
    pub(crate) result_tx: oneshot::Sender<Result<Option<Vec<f32>>>>,
}

impl GetVectorDispatch {
    /// Send the result back to the caller.
    pub fn send_result(self, result: Result<Option<Vec<f32>>>) {
        let _ = self.result_tx.send(result);
    }
}

impl From<GetVector> for Query {
    fn from(q: GetVector) -> Self {
        // This is used when we have a default timeout; caller should use dispatch directly
        let (tx, _rx) = oneshot::channel();
        Query::GetVector(GetVectorDispatch {
            params: q,
            timeout: Duration::from_secs(5),
            result_tx: tx,
        })
    }
}

#[async_trait::async_trait]
impl QueryExecutor for GetVector {
    type Output = Option<Vec<f32>>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        // 1. Look up the internal vec_id from the external ID
        let forward_key = IdForwardCfKey(self.embedding, self.id);
        let forward_key_bytes = IdForward::key_to_bytes(&forward_key);

        let txn_db = storage.transaction_db()?;
        let forward_cf = txn_db
            .cf_handle(IdForward::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("IdForward CF not found"))?;

        let vec_id = match txn_db.get_cf(&forward_cf, &forward_key_bytes)? {
            Some(bytes) => IdForward::value_from_bytes(&bytes)?.0,
            None => return Ok(None), // External ID not found
        };

        // 2. Look up the vector data
        let vec_key = VectorCfKey(self.embedding, vec_id);
        let vec_key_bytes = Vectors::key_to_bytes(&vec_key);

        let vectors_cf = txn_db
            .cf_handle(Vectors::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("Vectors CF not found"))?;

        match txn_db.get_cf(&vectors_cf, &vec_key_bytes)? {
            Some(bytes) => {
                let value = Vectors::value_from_bytes(&bytes)?;
                Ok(Some(value.0))
            }
            None => Ok(None), // Vector data not found (inconsistent state)
        }
    }

    fn timeout(&self) -> Duration {
        Duration::from_secs(5)
    }
}

#[async_trait::async_trait]
impl QueryProcessor for GetVectorDispatch {
    async fn process_and_send(self, storage: &Storage) {
        let result = self.params.result(storage).await;
        self.send_result(result);
    }
}

// ============================================================================
// GetInternalId - Resolve external ID to internal vec_id
// ============================================================================

/// Get internal vec_id for external ID.
///
/// This query resolves an external document ID to its internal u32 vector ID
/// used for HNSW graph operations.
#[derive(Debug, Clone)]
pub struct GetInternalId {
    /// Embedding space code
    pub embedding: EmbeddingCode,
    /// External document ID
    pub id: Id,
}

impl GetInternalId {
    /// Create a new GetInternalId query.
    pub fn new(embedding: EmbeddingCode, id: Id) -> Self {
        Self { embedding, id }
    }
}

/// Dispatch wrapper with oneshot channel for GetInternalId.
#[derive(Debug)]
pub(crate) struct GetInternalIdDispatch {
    pub(crate) params: GetInternalId,
    pub(crate) timeout: Duration,
    pub(crate) result_tx: oneshot::Sender<Result<Option<VecId>>>,
}

impl GetInternalIdDispatch {
    /// Send the result back to the caller.
    pub fn send_result(self, result: Result<Option<VecId>>) {
        let _ = self.result_tx.send(result);
    }
}

#[async_trait::async_trait]
impl QueryExecutor for GetInternalId {
    type Output = Option<VecId>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        let forward_key = IdForwardCfKey(self.embedding, self.id);
        let forward_key_bytes = IdForward::key_to_bytes(&forward_key);

        let txn_db = storage.transaction_db()?;
        let forward_cf = txn_db
            .cf_handle(IdForward::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("IdForward CF not found"))?;

        match txn_db.get_cf(&forward_cf, &forward_key_bytes)? {
            Some(bytes) => Ok(Some(IdForward::value_from_bytes(&bytes)?.0)),
            None => Ok(None),
        }
    }

    fn timeout(&self) -> Duration {
        Duration::from_secs(5)
    }
}

#[async_trait::async_trait]
impl QueryProcessor for GetInternalIdDispatch {
    async fn process_and_send(self, storage: &Storage) {
        let result = self.params.result(storage).await;
        self.send_result(result);
    }
}

// ============================================================================
// GetExternalId - Resolve internal vec_id to external ID
// ============================================================================

/// Get external ID for internal vec_id.
///
/// This query resolves an internal u32 vector ID back to its external
/// document ID.
#[derive(Debug, Clone)]
pub struct GetExternalId {
    /// Embedding space code
    pub embedding: EmbeddingCode,
    /// Internal vector ID
    pub vec_id: VecId,
}

impl GetExternalId {
    /// Create a new GetExternalId query.
    pub fn new(embedding: EmbeddingCode, vec_id: VecId) -> Self {
        Self { embedding, vec_id }
    }
}

/// Dispatch wrapper with oneshot channel for GetExternalId.
#[derive(Debug)]
pub(crate) struct GetExternalIdDispatch {
    pub(crate) params: GetExternalId,
    pub(crate) timeout: Duration,
    pub(crate) result_tx: oneshot::Sender<Result<Option<Id>>>,
}

impl GetExternalIdDispatch {
    /// Send the result back to the caller.
    pub fn send_result(self, result: Result<Option<Id>>) {
        let _ = self.result_tx.send(result);
    }
}

#[async_trait::async_trait]
impl QueryExecutor for GetExternalId {
    type Output = Option<Id>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        let reverse_key = IdReverseCfKey(self.embedding, self.vec_id);
        let reverse_key_bytes = IdReverse::key_to_bytes(&reverse_key);

        let txn_db = storage.transaction_db()?;
        let reverse_cf = txn_db
            .cf_handle(IdReverse::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("IdReverse CF not found"))?;

        match txn_db.get_cf(&reverse_cf, &reverse_key_bytes)? {
            Some(bytes) => Ok(Some(IdReverse::value_from_bytes(&bytes)?.0)),
            None => Ok(None),
        }
    }

    fn timeout(&self) -> Duration {
        Duration::from_secs(5)
    }
}

#[async_trait::async_trait]
impl QueryProcessor for GetExternalIdDispatch {
    async fn process_and_send(self, storage: &Storage) {
        let result = self.params.result(storage).await;
        self.send_result(result);
    }
}

// ============================================================================
// ResolveIds - Batch resolve vec_ids to external IDs
// ============================================================================

/// Batch resolve vec_ids to external IDs.
///
/// This query efficiently resolves multiple internal vector IDs to their
/// external document IDs in a single batch operation.
#[derive(Debug, Clone)]
pub struct ResolveIds {
    /// Embedding space code
    pub embedding: EmbeddingCode,
    /// Vector IDs to resolve
    pub vec_ids: Vec<VecId>,
}

impl ResolveIds {
    /// Create a new ResolveIds query.
    pub fn new(embedding: EmbeddingCode, vec_ids: Vec<VecId>) -> Self {
        Self { embedding, vec_ids }
    }
}

/// Dispatch wrapper with oneshot channel for ResolveIds.
#[derive(Debug)]
pub(crate) struct ResolveIdsDispatch {
    pub(crate) params: ResolveIds,
    pub(crate) timeout: Duration,
    pub(crate) result_tx: oneshot::Sender<Result<Vec<Option<Id>>>>,
}

impl ResolveIdsDispatch {
    /// Send the result back to the caller.
    pub fn send_result(self, result: Result<Vec<Option<Id>>>) {
        let _ = self.result_tx.send(result);
    }
}

#[async_trait::async_trait]
impl QueryExecutor for ResolveIds {
    type Output = Vec<Option<Id>>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        let txn_db = storage.transaction_db()?;
        let reverse_cf = txn_db
            .cf_handle(IdReverse::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("IdReverse CF not found"))?;

        // Build keys for multi_get
        let keys: Vec<Vec<u8>> = self
            .vec_ids
            .iter()
            .map(|&vec_id| {
                let key = IdReverseCfKey(self.embedding, vec_id);
                IdReverse::key_to_bytes(&key)
            })
            .collect();

        // Perform batch lookup
        let results: Vec<Option<Id>> = keys
            .iter()
            .map(|key_bytes| {
                match txn_db.get_cf(&reverse_cf, key_bytes) {
                    Ok(Some(bytes)) => IdReverse::value_from_bytes(&bytes).ok().map(|v| v.0),
                    _ => None,
                }
            })
            .collect();

        Ok(results)
    }

    fn timeout(&self) -> Duration {
        Duration::from_secs(10) // Longer timeout for batch operations
    }
}

#[async_trait::async_trait]
impl QueryProcessor for ResolveIdsDispatch {
    async fn process_and_send(self, storage: &Storage) {
        let result = self.params.result(storage).await;
        self.send_result(result);
    }
}

// ============================================================================
// Query Dispatch
// ============================================================================

impl Query {
    /// Process this query and send the result.
    pub async fn process(self, storage: &Storage) {
        match self {
            Query::GetVector(dispatch) => dispatch.process_and_send(storage).await,
            Query::GetInternalId(dispatch) => dispatch.process_and_send(storage).await,
            Query::GetExternalId(dispatch) => dispatch.process_and_send(storage).await,
            Query::ResolveIds(dispatch) => dispatch.process_and_send(storage).await,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_vector_query() {
        let id = Id::new();
        let query = GetVector::new(1, id);
        assert_eq!(query.embedding, 1);
        assert_eq!(query.id, id);
    }

    #[test]
    fn test_get_internal_id_query() {
        let id = Id::new();
        let query = GetInternalId::new(42, id);
        assert_eq!(query.embedding, 42);
        assert_eq!(query.id, id);
    }

    #[test]
    fn test_get_external_id_query() {
        let query = GetExternalId::new(1, 100);
        assert_eq!(query.embedding, 1);
        assert_eq!(query.vec_id, 100);
    }

    #[test]
    fn test_resolve_ids_query() {
        let vec_ids = vec![1, 2, 3, 4, 5];
        let query = ResolveIds::new(1, vec_ids.clone());
        assert_eq!(query.embedding, 1);
        assert_eq!(query.vec_ids, vec_ids);
    }

    #[test]
    fn test_query_display() {
        let id = Id::new();

        // Test GetVector display
        let (tx, _rx) = oneshot::channel();
        let query = Query::GetVector(GetVectorDispatch {
            params: GetVector::new(1, id),
            timeout: Duration::from_secs(5),
            result_tx: tx,
        });
        let display = format!("{}", query);
        assert!(display.contains("GetVector"));
        assert!(display.contains("embedding=1"));

        // Test GetInternalId display
        let (tx, _rx) = oneshot::channel();
        let query = Query::GetInternalId(GetInternalIdDispatch {
            params: GetInternalId::new(2, id),
            timeout: Duration::from_secs(5),
            result_tx: tx,
        });
        let display = format!("{}", query);
        assert!(display.contains("GetInternalId"));
        assert!(display.contains("embedding=2"));

        // Test GetExternalId display
        let (tx, _rx) = oneshot::channel();
        let query = Query::GetExternalId(GetExternalIdDispatch {
            params: GetExternalId::new(3, 100),
            timeout: Duration::from_secs(5),
            result_tx: tx,
        });
        let display = format!("{}", query);
        assert!(display.contains("GetExternalId"));
        assert!(display.contains("vec_id=100"));

        // Test ResolveIds display
        let (tx, _rx) = oneshot::channel();
        let query = Query::ResolveIds(ResolveIdsDispatch {
            params: ResolveIds::new(4, vec![1, 2, 3]),
            timeout: Duration::from_secs(10),
            result_tx: tx,
        });
        let display = format!("{}", query);
        assert!(display.contains("ResolveIds"));
        assert!(display.contains("count=3"));
    }

    #[test]
    fn test_query_timeout() {
        let query = GetVector::new(1, Id::new());
        assert_eq!(QueryExecutor::timeout(&query), Duration::from_secs(5));

        let query = GetInternalId::new(1, Id::new());
        assert_eq!(QueryExecutor::timeout(&query), Duration::from_secs(5));

        let query = GetExternalId::new(1, 0);
        assert_eq!(QueryExecutor::timeout(&query), Duration::from_secs(5));

        let query = ResolveIds::new(1, vec![1, 2, 3]);
        assert_eq!(QueryExecutor::timeout(&query), Duration::from_secs(10));
    }
}
