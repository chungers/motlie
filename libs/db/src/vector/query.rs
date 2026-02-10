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

use std::time::{Duration, Instant};

use anyhow::Result;

use super::embedding::Embedding;
use super::processor::{Processor, SearchResult};
use super::schema::{
    EmbeddingCode, ExternalKey, IdForward, IdForwardCfKey, IdReverse, IdReverseCfKey, VecId, VectorCfKey,
    Vectors,
};
use super::search::SearchConfig;
use crate::request::{new_request_id, RequestEnvelope, RequestMeta};
use crate::rocksdb::ColumnFamily;
use crate::Id;

// ============================================================================
// Query Enum
// ============================================================================

/// Query enum for vector storage operations.
///
/// All vector queries are variants of this enum, enabling type-safe
/// dispatch in query consumers. This is an internal dispatch mechanism;
/// the public API is through builder types like `GetVector`, `SearchKNN`, etc.
#[doc(hidden)]
#[allow(private_interfaces)]
#[derive(Debug)]
pub enum Query {
    // ─────────────────────────────────────────────────────────────
    // Point Lookups
    // ─────────────────────────────────────────────────────────────
    /// Get vector by external ID
    GetVector(GetVector),
    /// Get internal vec_id for external ID
    GetInternalId(GetInternalId),
    /// Get external ID for internal vec_id
    GetExternalId(GetExternalId),
    /// Batch resolve vec_ids to external IDs
    ResolveIds(ResolveIds),

    // ─────────────────────────────────────────────────────────────
    // Registry Queries
    // ─────────────────────────────────────────────────────────────
    /// List all registered embeddings
    ListEmbeddings(ListEmbeddings),
    /// Find embeddings by filter criteria
    FindEmbeddings(FindEmbeddings),

    // ─────────────────────────────────────────────────────────────
    // Search Operations
    // ─────────────────────────────────────────────────────────────
    /// K-nearest neighbor search via HNSW
    SearchKNN(SearchKNN),

    // ─────────────────────────────────────────────────────────────
    // Graph Introspection (Future)
    // ─────────────────────────────────────────────────────────────
    // GetGraphMeta(GetGraphMetaDispatch),
    // GetNeighbors(GetNeighborsDispatch),
}

impl std::fmt::Display for Query {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Query::GetVector(q) => write!(f, "GetVector: embedding={}, id={}", q.embedding, q.id),
            Query::GetInternalId(q) => {
                write!(f, "GetInternalId: embedding={}, key={:?}", q.embedding, q.external_key)
            }
            Query::GetExternalId(q) => write!(
                f,
                "GetExternalId: embedding={}, vec_id={}",
                q.embedding, q.vec_id
            ),
            Query::ResolveIds(q) => write!(
                f,
                "ResolveIds: embedding={}, count={}",
                q.embedding,
                q.vec_ids.len()
            ),
            Query::ListEmbeddings(_) => write!(f, "ListEmbeddings"),
            Query::FindEmbeddings(q) => write!(
                f,
                "FindEmbeddings: model={:?}, dim={:?}, distance={:?}",
                q.filter.model,
                q.filter.dim,
                q.filter.distance
            ),
            Query::SearchKNN(q) => write!(
                f,
                "SearchKNN: embedding={}, k={}, ef={}, exact={}",
                q.embedding.code(),
                q.k,
                q.ef,
                q.exact
            ),
        }
    }
}

#[derive(Debug)]
pub enum QueryResult {
    GetVector(Option<Vec<f32>>),
    GetInternalId(Option<VecId>),
    GetExternalId(Option<ExternalKey>),
    ResolveIds(Vec<Option<ExternalKey>>),
    ListEmbeddings(Vec<Embedding>),
    FindEmbeddings(Vec<Embedding>),
    SearchKNN(Vec<SearchResult>),
}

impl RequestMeta for Query {
    type Reply = QueryResult;
    type Options = ();

    fn request_kind(&self) -> &'static str {
        match self {
            Query::GetVector(_) => "get_vector",
            Query::GetInternalId(_) => "get_internal_id",
            Query::GetExternalId(_) => "get_external_id",
            Query::ResolveIds(_) => "resolve_ids",
            Query::ListEmbeddings(_) => "list_embeddings",
            Query::FindEmbeddings(_) => "find_embeddings",
            Query::SearchKNN(_) => "search_knn",
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

    /// Execute this query against the processor
    async fn execute(&self, processor: &Processor) -> Result<Self::Output>;
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

impl From<GetVector> for Query {
    fn from(q: GetVector) -> Self {
        Query::GetVector(q)
    }
}

#[async_trait::async_trait]
impl QueryExecutor for GetVector {
    type Output = Option<Vec<f32>>;

    async fn execute(&self, processor: &Processor) -> Result<Self::Output> {
        // 1. Look up the internal vec_id from the external ID
        let forward_key = IdForwardCfKey(self.embedding, ExternalKey::NodeId(self.id));
        let forward_key_bytes = IdForward::key_to_bytes(&forward_key);

        let txn_db = processor.storage().transaction_db()?;
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

}

// ============================================================================
// GetInternalId - Resolve external ID to internal vec_id
// ============================================================================

/// Get internal vec_id for external key.
///
/// This query resolves an external key to its internal u32 vector ID
/// used for HNSW graph operations.
#[derive(Debug, Clone)]
pub struct GetInternalId {
    /// Embedding space code
    pub embedding: EmbeddingCode,
    /// External key (node, edge, fragment, summary, etc.)
    pub external_key: ExternalKey,
}

impl GetInternalId {
    /// Create a new GetInternalId query.
    pub fn new(embedding: EmbeddingCode, external_key: ExternalKey) -> Self {
        Self { embedding, external_key }
    }
}


#[async_trait::async_trait]
impl QueryExecutor for GetInternalId {
    type Output = Option<VecId>;

    async fn execute(&self, processor: &Processor) -> Result<Self::Output> {
        let forward_key = IdForwardCfKey(self.embedding, self.external_key.clone());
        let forward_key_bytes = IdForward::key_to_bytes(&forward_key);

        let txn_db = processor.storage().transaction_db()?;
        let forward_cf = txn_db
            .cf_handle(IdForward::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("IdForward CF not found"))?;

        match txn_db.get_cf(&forward_cf, &forward_key_bytes)? {
            Some(bytes) => Ok(Some(IdForward::value_from_bytes(&bytes)?.0)),
            None => Ok(None),
        }
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


#[async_trait::async_trait]
impl QueryExecutor for GetExternalId {
    type Output = Option<ExternalKey>;

    async fn execute(&self, processor: &Processor) -> Result<Self::Output> {
        let reverse_key = IdReverseCfKey(self.embedding, self.vec_id);
        let reverse_key_bytes = IdReverse::key_to_bytes(&reverse_key);

        let txn_db = processor.storage().transaction_db()?;
        let reverse_cf = txn_db
            .cf_handle(IdReverse::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("IdReverse CF not found"))?;

        match txn_db.get_cf(&reverse_cf, &reverse_key_bytes)? {
            Some(bytes) => {
                let external_key = IdReverse::value_from_bytes(&bytes)?.0;
                Ok(Some(external_key))
            }
            None => Ok(None),
        }
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

#[async_trait::async_trait]
impl QueryExecutor for ResolveIds {
    type Output = Vec<Option<ExternalKey>>;

    async fn execute(&self, processor: &Processor) -> Result<Self::Output> {
        if self.vec_ids.is_empty() {
            return Ok(Vec::new());
        }

        let txn_db = processor.storage().transaction_db()?;
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
        let resolved: Vec<Option<ExternalKey>> = results
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

#[async_trait::async_trait]
impl QueryExecutor for ListEmbeddings {
    type Output = Vec<Embedding>;

    async fn execute(&self, processor: &Processor) -> Result<Self::Output> {
        Ok(processor.registry().list_all())
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

#[async_trait::async_trait]
impl QueryExecutor for FindEmbeddings {
    type Output = Vec<Embedding>;

    async fn execute(&self, processor: &Processor) -> Result<Self::Output> {
        Ok(processor.registry().find(&self.filter))
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
    pub(crate) fn execute_with_processor(&self, processor: &Processor) -> Result<Vec<SearchResult>> {
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

// ============================================================================
// QueryReply - typed replies for vector queries
// ============================================================================

pub type QueryRequest = RequestEnvelope<Query>;

pub trait QueryReply: Send {
    type Reply: Send + 'static;
    fn into_query(self) -> Query;
    fn from_result(result: QueryResult) -> Result<Self::Reply>;
}

#[async_trait::async_trait]
impl<Q> crate::reader::Runnable<super::reader::Reader> for Q
where
    Q: QueryReply,
{
    type Output = Q::Reply;

    async fn run(self, reader: &super::reader::Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let request = QueryRequest {
            payload: self.into_query(),
            options: (),
            reply: Some(result_tx),
            timeout: Some(timeout),
            request_id: new_request_id(),
            created_at: Instant::now(),
        };

        reader.send_query(request).await?;
        let result = result_rx.await??;
        Q::from_result(result)
    }
}

impl Query {
    /// Execute a query using the Processor.
    ///
    /// This is the primary execution method for all query types.
    /// All queries route through Processor for a unified execution path.
    pub(crate) async fn execute_with_processor(
        &self,
        processor: &Processor,
    ) -> Result<QueryResult> {
        match self {
            Query::GetVector(q) => q.execute(processor).await.map(QueryResult::GetVector),
            Query::GetInternalId(q) => q.execute(processor).await.map(QueryResult::GetInternalId),
            Query::GetExternalId(q) => q.execute(processor).await.map(QueryResult::GetExternalId),
            Query::ResolveIds(q) => q.execute(processor).await.map(QueryResult::ResolveIds),
            Query::ListEmbeddings(q) => q.execute(processor).await.map(QueryResult::ListEmbeddings),
            Query::FindEmbeddings(q) => q.execute(processor).await.map(QueryResult::FindEmbeddings),
            Query::SearchKNN(q) => q
                .execute_with_processor(processor)
                .map(QueryResult::SearchKNN),
        }
    }
}

impl QueryReply for GetVector {
    type Reply = Option<Vec<f32>>;

    fn into_query(self) -> Query {
        Query::GetVector(self)
    }

    fn from_result(result: QueryResult) -> Result<Self::Reply> {
        match result {
            QueryResult::GetVector(result) => Ok(result),
            other => Err(anyhow::anyhow!("Unexpected query result: {:?}", other)),
        }
    }
}

impl QueryReply for GetInternalId {
    type Reply = Option<VecId>;

    fn into_query(self) -> Query {
        Query::GetInternalId(self)
    }

    fn from_result(result: QueryResult) -> Result<Self::Reply> {
        match result {
            QueryResult::GetInternalId(result) => Ok(result),
            other => Err(anyhow::anyhow!("Unexpected query result: {:?}", other)),
        }
    }
}

impl QueryReply for GetExternalId {
    type Reply = Option<ExternalKey>;

    fn into_query(self) -> Query {
        Query::GetExternalId(self)
    }

    fn from_result(result: QueryResult) -> Result<Self::Reply> {
        match result {
            QueryResult::GetExternalId(result) => Ok(result),
            other => Err(anyhow::anyhow!("Unexpected query result: {:?}", other)),
        }
    }
}

impl QueryReply for ResolveIds {
    type Reply = Vec<Option<ExternalKey>>;

    fn into_query(self) -> Query {
        Query::ResolveIds(self)
    }

    fn from_result(result: QueryResult) -> Result<Self::Reply> {
        match result {
            QueryResult::ResolveIds(result) => Ok(result),
            other => Err(anyhow::anyhow!("Unexpected query result: {:?}", other)),
        }
    }
}

impl QueryReply for ListEmbeddings {
    type Reply = Vec<Embedding>;

    fn into_query(self) -> Query {
        Query::ListEmbeddings(self)
    }

    fn from_result(result: QueryResult) -> Result<Self::Reply> {
        match result {
            QueryResult::ListEmbeddings(result) => Ok(result),
            other => Err(anyhow::anyhow!("Unexpected query result: {:?}", other)),
        }
    }
}

impl QueryReply for FindEmbeddings {
    type Reply = Vec<Embedding>;

    fn into_query(self) -> Query {
        Query::FindEmbeddings(self)
    }

    fn from_result(result: QueryResult) -> Result<Self::Reply> {
        match result {
            QueryResult::FindEmbeddings(result) => Ok(result),
            other => Err(anyhow::anyhow!("Unexpected query result: {:?}", other)),
        }
    }
}

impl QueryReply for SearchKNN {
    type Reply = Vec<SearchResult>;

    fn into_query(self) -> Query {
        Query::SearchKNN(self)
    }

    fn from_result(result: QueryResult) -> Result<Self::Reply> {
        match result {
            QueryResult::SearchKNN(result) => Ok(result),
            other => Err(anyhow::anyhow!("Unexpected query result: {:?}", other)),
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
        let query = GetInternalId::new(42, ExternalKey::NodeId(id));
        assert_eq!(query.embedding, 42);
        assert_eq!(query.external_key, ExternalKey::NodeId(id));
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
        let query = Query::GetVector(GetVector::new(1, id));
        let display = format!("{}", query);
        assert!(display.contains("GetVector"));
        assert!(display.contains("embedding=1"));

        // Test GetInternalId display
        let query = Query::GetInternalId(GetInternalId::new(2, ExternalKey::NodeId(id)));
        let display = format!("{}", query);
        assert!(display.contains("GetInternalId"));
        assert!(display.contains("embedding=2"));

        // Test GetExternalId display
        let query = Query::GetExternalId(GetExternalId::new(3, 100));
        let display = format!("{}", query);
        assert!(display.contains("GetExternalId"));
        assert!(display.contains("vec_id=100"));

        // Test ResolveIds display
        let query = Query::ResolveIds(ResolveIds::new(4, vec![1, 2, 3]));
        let display = format!("{}", query);
        assert!(display.contains("ResolveIds"));
        assert!(display.contains("count=3"));
    }
}
