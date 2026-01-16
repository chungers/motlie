//! Vector query types and executors.
//!
//! This module contains query type definitions and their execution logic
//! for vector storage operations. Following the pattern from `graph::query`,
//! queries are grouped in an enum for type-safe dispatch.
//!
//! # Point Lookups
//!
//! - `GetVector` - Retrieve vector by external ID
//! - `GetInternalId` - Resolve external ID to internal vec_id
//! - `GetExternalId` - Resolve internal vec_id to external ID
//! - `ResolveIds` - Batch resolve vec_ids to external IDs
//!
//! # Search Operations
//!
//! - `SearchKNN` - K-nearest neighbor search via HNSW
//!
//! # Graph Introspection (Future)
//!
//! - `GetGraphMeta` - HNSW graph metadata
//! - `GetNeighbors` - Get HNSW neighbors at a layer

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::sync::oneshot;

use super::embedding::Embedding;
use super::processor::{Processor, SearchResult};
use super::schema::{
    EmbeddingCode, IdForward, IdForwardCfKey, IdReverse, IdReverseCfKey, VecId, VectorCfKey,
    Vectors,
};
use super::search::SearchConfig;
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
    // Point Lookups
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
    // Registry Queries
    // ─────────────────────────────────────────────────────────────
    /// List all registered embeddings
    ListEmbeddings(ListEmbeddingsDispatch),
    /// Find embeddings by filter criteria
    FindEmbeddings(FindEmbeddingsDispatch),

    // ─────────────────────────────────────────────────────────────
    // Search Operations
    // ─────────────────────────────────────────────────────────────
    /// K-nearest neighbor search via HNSW
    SearchKNN(SearchKNNDispatch),

    // ─────────────────────────────────────────────────────────────
    // Graph Introspection (Future)
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
            Query::ListEmbeddings(_) => write!(f, "ListEmbeddings"),
            Query::FindEmbeddings(q) => write!(
                f,
                "FindEmbeddings: model={:?}, dim={:?}, distance={:?}",
                q.params.filter.model,
                q.params.filter.dim,
                q.params.filter.distance
            ),
            Query::SearchKNN(q) => write!(
                f,
                "SearchKNN: embedding={}, k={}, ef={}, exact={}",
                q.params.embedding.code(),
                q.params.k,
                q.params.ef,
                q.params.exact
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
        if self.vec_ids.is_empty() {
            return Ok(Vec::new());
        }

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

        // Perform batch lookup using multi_get_cf (Phase 3 optimization)
        let results: Vec<std::result::Result<Option<Vec<u8>>, rocksdb::Error>> =
            txn_db.multi_get_cf(keys.iter().map(|k| (&reverse_cf, k.as_slice())));

        // Parse results, converting RocksDB errors to None
        let resolved: Vec<Option<Id>> = results
            .into_iter()
            .map(|result| {
                result
                    .ok()
                    .flatten()
                    .and_then(|bytes| IdReverse::value_from_bytes(&bytes).ok().map(|v| v.0))
            })
            .collect();

        Ok(resolved)
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
// ListEmbeddings - List all registered embeddings
// ============================================================================

/// List all registered embedding spaces.
///
/// Returns a list of all `Embedding` objects that have been registered
/// in the embedding registry.
///
/// # Example
///
/// ```rust,ignore
/// let embeddings = ListEmbeddings::new()
///     .run(&reader, timeout)
///     .await?;
/// for emb in embeddings {
///     println!("{}: {}D {:?}", emb.model(), emb.dim(), emb.distance());
/// }
/// ```
#[derive(Debug, Clone, Default)]
pub struct ListEmbeddings;

impl ListEmbeddings {
    /// Create a new ListEmbeddings query.
    pub fn new() -> Self {
        Self
    }
}

/// Dispatch wrapper with oneshot channel for ListEmbeddings.
#[derive(Debug)]
pub(crate) struct ListEmbeddingsDispatch {
    pub(crate) params: ListEmbeddings,
    pub(crate) timeout: Duration,
    pub(crate) result_tx: oneshot::Sender<Result<Vec<Embedding>>>,
}

impl ListEmbeddingsDispatch {
    /// Send the result back to the caller.
    pub fn send_result(self, result: Result<Vec<Embedding>>) {
        let _ = self.result_tx.send(result);
    }
}

#[async_trait::async_trait]
impl QueryExecutor for ListEmbeddings {
    type Output = Vec<Embedding>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        Ok(storage.cache().list_all())
    }

    fn timeout(&self) -> Duration {
        Duration::from_secs(5)
    }
}

#[async_trait::async_trait]
impl QueryProcessor for ListEmbeddingsDispatch {
    async fn process_and_send(self, storage: &Storage) {
        let result = self.params.result(storage).await;
        self.send_result(result);
    }
}

// ============================================================================
// FindEmbeddings - Find embeddings by filter criteria
// ============================================================================

/// Find embedding spaces matching filter criteria.
///
/// Searches the embedding registry for embeddings matching the specified
/// filter (model name, dimensionality, and/or distance metric).
///
/// # Example
///
/// ```rust,ignore
/// use motlie_db::vector::{FindEmbeddings, EmbeddingFilter, Distance};
///
/// // Find all Cosine embeddings
/// let embeddings = FindEmbeddings::new(
///     EmbeddingFilter::default().distance(Distance::Cosine)
/// ).run(&reader, timeout).await?;
///
/// // Find by model name
/// let embeddings = FindEmbeddings::new(
///     EmbeddingFilter::default().model("clip-vit-b32")
/// ).run(&reader, timeout).await?;
///
/// // Find by multiple criteria
/// let embeddings = FindEmbeddings::new(
///     EmbeddingFilter::default()
///         .model("clip-vit-b32")
///         .dim(512)
///         .distance(Distance::Cosine)
/// ).run(&reader, timeout).await?;
/// ```
#[derive(Debug, Clone)]
pub struct FindEmbeddings {
    /// Filter criteria
    pub filter: super::registry::EmbeddingFilter,
}

impl FindEmbeddings {
    /// Create a new FindEmbeddings query with the given filter.
    pub fn new(filter: super::registry::EmbeddingFilter) -> Self {
        Self { filter }
    }
}

/// Dispatch wrapper with oneshot channel for FindEmbeddings.
#[derive(Debug)]
pub(crate) struct FindEmbeddingsDispatch {
    pub(crate) params: FindEmbeddings,
    pub(crate) timeout: Duration,
    pub(crate) result_tx: oneshot::Sender<Result<Vec<Embedding>>>,
}

impl FindEmbeddingsDispatch {
    /// Send the result back to the caller.
    pub fn send_result(self, result: Result<Vec<Embedding>>) {
        let _ = self.result_tx.send(result);
    }
}

#[async_trait::async_trait]
impl QueryExecutor for FindEmbeddings {
    type Output = Vec<Embedding>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        Ok(storage.cache().find(&self.filter))
    }

    fn timeout(&self) -> Duration {
        Duration::from_secs(5)
    }
}

#[async_trait::async_trait]
impl QueryProcessor for FindEmbeddingsDispatch {
    async fn process_and_send(self, storage: &Storage) {
        let result = self.params.result(storage).await;
        self.send_result(result);
    }
}

// ============================================================================
// SearchKNN - K-nearest neighbor search via HNSW
// ============================================================================

/// K-nearest neighbor search via HNSW index.
///
/// This query performs approximate nearest neighbor search using the HNSW
/// graph index. Requires a Processor instance for execution.
///
/// # Strategy Selection
///
/// By default, the search strategy is auto-selected based on the embedding's
/// distance metric:
/// - Cosine → RaBitQ (ADC approximates angular distance)
/// - L2/DotProduct → Exact
///
/// Use `.exact()` to force exact distance computation regardless of metric.
///
/// # Example
///
/// ```rust,ignore
/// // Auto-select strategy based on distance metric
/// let results = SearchKNN::new(&embedding, query_vector, 10)
///     .with_ef(100)
///     .execute_with_processor(&processor)?;
///
/// // Force exact search (bypass RaBitQ)
/// let results = SearchKNN::new(&embedding, query_vector, 10)
///     .exact()
///     .execute_with_processor(&processor)?;
/// ```
#[derive(Debug, Clone)]
pub struct SearchKNN {
    /// Embedding space specification (owned, cloned from reference)
    pub embedding: Embedding,
    /// Query vector
    pub query: Vec<f32>,
    /// Number of results to return
    pub k: usize,
    /// Search expansion factor (higher = more accurate, slower)
    pub ef: usize,
    /// Force exact distance computation (bypass RaBitQ even for Cosine)
    pub exact: bool,
    /// Re-rank factor for RaBitQ mode (ignored when exact=true)
    pub rerank_factor: usize,
}

impl SearchKNN {
    /// Create a new SearchKNN query.
    ///
    /// Takes a reference to Embedding and clones it internally.
    /// This ensures the caller retains their reference while SearchKNN
    /// owns its copy (required for channel dispatch).
    pub fn new(embedding: &Embedding, query: Vec<f32>, k: usize) -> Self {
        Self {
            embedding: embedding.clone(),
            query,
            k,
            ef: 100,           // sensible default
            exact: false,      // auto-select strategy
            rerank_factor: 10, // 10x re-ranking for ~90% recall
        }
    }

    /// Set the search expansion factor.
    ///
    /// Higher ef = better recall but slower search. Typical values: 50-200.
    pub fn with_ef(mut self, ef: usize) -> Self {
        self.ef = ef;
        self
    }

    /// Force exact distance computation.
    ///
    /// When set, bypasses RaBitQ approximation even for Cosine distance.
    /// Use this when maximum accuracy is required at the cost of speed.
    pub fn exact(mut self) -> Self {
        self.exact = true;
        self
    }

    /// Set the re-rank factor for RaBitQ mode.
    ///
    /// Number of candidates to re-rank = k * rerank_factor.
    /// Higher factor = better recall but more I/O. Typical values: 4-20.
    /// Ignored when `exact=true`.
    pub fn with_rerank_factor(mut self, factor: usize) -> Self {
        self.rerank_factor = factor;
        self
    }

    /// Execute with a Processor (for direct execution without channel).
    ///
    /// Strategy selection:
    /// - If `exact=true`: Uses `processor.search()` with exact distances
    /// - If `exact=false`: Constructs `SearchConfig` for auto-strategy selection
    pub fn execute_with_processor(&self, processor: &Processor) -> Result<Vec<SearchResult>> {
        if self.exact {
            // Exact search - always use precise distance computation
            processor.search(&self.embedding, &self.query, self.k, self.ef)
        } else {
            // Auto-select strategy via SearchConfig
            let config = SearchConfig::new(self.embedding.clone(), self.k)
                .with_ef(self.ef)
                .with_rerank_factor(self.rerank_factor);
            processor.search_with_config(&config, &self.query)
        }
    }
}

/// Dispatch wrapper with oneshot channel for SearchKNN.
///
/// Unlike point lookups that only need Storage, SearchKNN requires a Processor
/// for HNSW index access. The dispatch wraps the query along with a reference
/// to the processor.
/// Internal dispatch wrapper - users should use `SearchKNN::run()` instead.
pub(crate) struct SearchKNNDispatch {
    pub(crate) params: SearchKNN,
    pub(crate) timeout: Duration,
    pub(crate) result_tx: oneshot::Sender<Result<Vec<SearchResult>>>,
    pub(crate) processor: Arc<Processor>,
}

impl std::fmt::Debug for SearchKNNDispatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchKNNDispatch")
            .field("params", &self.params)
            .field("timeout", &self.timeout)
            .field("result_tx", &"<oneshot::Sender>")
            .field("processor", &"<Arc<Processor>>")
            .finish()
    }
}

impl SearchKNNDispatch {
    /// Create a new dispatch wrapper.
    pub fn new(
        params: SearchKNN,
        timeout: Duration,
        processor: Arc<Processor>,
    ) -> (Self, oneshot::Receiver<Result<Vec<SearchResult>>>) {
        let (tx, rx) = oneshot::channel();
        (
            Self {
                params,
                timeout,
                result_tx: tx,
                processor,
            },
            rx,
        )
    }

    /// Send the result back to the caller.
    pub fn send_result(self, result: Result<Vec<SearchResult>>) {
        let _ = self.result_tx.send(result);
    }
}

#[async_trait::async_trait]
impl QueryProcessor for SearchKNNDispatch {
    async fn process_and_send(self, _storage: &Storage) {
        // Use processor directly instead of storage
        let result = self.params.execute_with_processor(&self.processor);
        self.send_result(result);
    }
}

// ============================================================================
// Runnable Implementation for SearchKNN
// ============================================================================

/// Runnable implementation for SearchKNN with SearchReader.
///
/// This allows the ergonomic pattern:
/// ```rust,ignore
/// let results = SearchKNN::new(&embedding, query, 10)
///     .with_ef(100)
///     .run(&search_reader, timeout)
///     .await?;
/// ```
#[async_trait::async_trait]
impl crate::reader::Runnable<super::reader::SearchReader> for SearchKNN {
    type Output = Vec<SearchResult>;

    async fn run(
        self,
        reader: &super::reader::SearchReader,
        timeout: Duration,
    ) -> Result<Self::Output> {
        let (dispatch, rx) =
            SearchKNNDispatch::new(self, timeout, reader.processor().clone());

        reader
            .reader()
            .send_query(Query::SearchKNN(dispatch))
            .await
            .context("Failed to send SearchKNN query")?;

        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(anyhow::anyhow!("SearchKNN query channel closed")),
            Err(_) => Err(anyhow::anyhow!("SearchKNN query timeout after {:?}", timeout)),
        }
    }
}

// ============================================================================
// Runnable Implementations for Point Lookups
// ============================================================================

/// Runnable implementation for GetVector.
///
/// # Example
/// ```rust,ignore
/// let vector = GetVector::new(embedding.code(), id)
///     .run(&reader, timeout)
///     .await?;
/// ```
#[async_trait::async_trait]
impl crate::reader::Runnable<super::reader::Reader> for GetVector {
    type Output = Option<Vec<f32>>;

    async fn run(
        self,
        reader: &super::reader::Reader,
        timeout: Duration,
    ) -> Result<Self::Output> {
        let (tx, rx) = oneshot::channel();
        let dispatch = GetVectorDispatch {
            params: self,
            timeout,
            result_tx: tx,
        };

        reader
            .send_query(Query::GetVector(dispatch))
            .await
            .context("Failed to send GetVector query")?;

        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(anyhow::anyhow!("GetVector query channel closed")),
            Err(_) => Err(anyhow::anyhow!("GetVector query timeout after {:?}", timeout)),
        }
    }
}

/// Runnable implementation for GetInternalId.
///
/// # Example
/// ```rust,ignore
/// let vec_id = GetInternalId::new(embedding.code(), external_id)
///     .run(&reader, timeout)
///     .await?;
/// ```
#[async_trait::async_trait]
impl crate::reader::Runnable<super::reader::Reader> for GetInternalId {
    type Output = Option<VecId>;

    async fn run(
        self,
        reader: &super::reader::Reader,
        timeout: Duration,
    ) -> Result<Self::Output> {
        let (tx, rx) = oneshot::channel();
        let dispatch = GetInternalIdDispatch {
            params: self,
            timeout,
            result_tx: tx,
        };

        reader
            .send_query(Query::GetInternalId(dispatch))
            .await
            .context("Failed to send GetInternalId query")?;

        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(anyhow::anyhow!("GetInternalId query channel closed")),
            Err(_) => Err(anyhow::anyhow!("GetInternalId query timeout after {:?}", timeout)),
        }
    }
}

/// Runnable implementation for GetExternalId.
///
/// # Example
/// ```rust,ignore
/// let external_id = GetExternalId::new(embedding.code(), vec_id)
///     .run(&reader, timeout)
///     .await?;
/// ```
#[async_trait::async_trait]
impl crate::reader::Runnable<super::reader::Reader> for GetExternalId {
    type Output = Option<Id>;

    async fn run(
        self,
        reader: &super::reader::Reader,
        timeout: Duration,
    ) -> Result<Self::Output> {
        let (tx, rx) = oneshot::channel();
        let dispatch = GetExternalIdDispatch {
            params: self,
            timeout,
            result_tx: tx,
        };

        reader
            .send_query(Query::GetExternalId(dispatch))
            .await
            .context("Failed to send GetExternalId query")?;

        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(anyhow::anyhow!("GetExternalId query channel closed")),
            Err(_) => Err(anyhow::anyhow!("GetExternalId query timeout after {:?}", timeout)),
        }
    }
}

/// Runnable implementation for ResolveIds.
///
/// # Example
/// ```rust,ignore
/// let external_ids = ResolveIds::new(embedding.code(), vec_ids)
///     .run(&reader, timeout)
///     .await?;
/// ```
#[async_trait::async_trait]
impl crate::reader::Runnable<super::reader::Reader> for ResolveIds {
    type Output = Vec<Option<Id>>;

    async fn run(
        self,
        reader: &super::reader::Reader,
        timeout: Duration,
    ) -> Result<Self::Output> {
        let (tx, rx) = oneshot::channel();
        let dispatch = ResolveIdsDispatch {
            params: self,
            timeout,
            result_tx: tx,
        };

        reader
            .send_query(Query::ResolveIds(dispatch))
            .await
            .context("Failed to send ResolveIds query")?;

        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(anyhow::anyhow!("ResolveIds query channel closed")),
            Err(_) => Err(anyhow::anyhow!("ResolveIds query timeout after {:?}", timeout)),
        }
    }
}

/// Runnable implementation for ListEmbeddings.
///
/// # Example
/// ```rust,ignore
/// let embeddings = ListEmbeddings::new()
///     .run(&reader, timeout)
///     .await?;
/// ```
#[async_trait::async_trait]
impl crate::reader::Runnable<super::reader::Reader> for ListEmbeddings {
    type Output = Vec<Embedding>;

    async fn run(
        self,
        reader: &super::reader::Reader,
        timeout: Duration,
    ) -> Result<Self::Output> {
        let (tx, rx) = oneshot::channel();
        let dispatch = ListEmbeddingsDispatch {
            params: self,
            timeout,
            result_tx: tx,
        };

        reader
            .send_query(Query::ListEmbeddings(dispatch))
            .await
            .context("Failed to send ListEmbeddings query")?;

        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(anyhow::anyhow!("ListEmbeddings query channel closed")),
            Err(_) => Err(anyhow::anyhow!("ListEmbeddings query timeout after {:?}", timeout)),
        }
    }
}

/// Runnable implementation for FindEmbeddings.
///
/// # Example
/// ```rust,ignore
/// let embeddings = FindEmbeddings::new(
///     EmbeddingFilter::default().distance(Distance::Cosine)
/// ).run(&reader, timeout).await?;
/// ```
#[async_trait::async_trait]
impl crate::reader::Runnable<super::reader::Reader> for FindEmbeddings {
    type Output = Vec<Embedding>;

    async fn run(
        self,
        reader: &super::reader::Reader,
        timeout: Duration,
    ) -> Result<Self::Output> {
        let (tx, rx) = oneshot::channel();
        let dispatch = FindEmbeddingsDispatch {
            params: self,
            timeout,
            result_tx: tx,
        };

        reader
            .send_query(Query::FindEmbeddings(dispatch))
            .await
            .context("Failed to send FindEmbeddings query")?;

        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(anyhow::anyhow!("FindEmbeddings query channel closed")),
            Err(_) => Err(anyhow::anyhow!("FindEmbeddings query timeout after {:?}", timeout)),
        }
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
            Query::ListEmbeddings(dispatch) => dispatch.process_and_send(storage).await,
            Query::FindEmbeddings(dispatch) => dispatch.process_and_send(storage).await,
            Query::SearchKNN(dispatch) => dispatch.process_and_send(storage).await,
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
