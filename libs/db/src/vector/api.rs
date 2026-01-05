//! Convenience API functions for common vector operations.
//!
//! These provide a simpler interface for basic operations without
//! requiring direct use of Reader/Writer infrastructure.
//!
//! # Phase 1 Functions (Available Now)
//!
//! - [`insert`] - Insert a vector into an embedding space
//! - [`get_vector`] - Retrieve a vector by external ID
//! - [`delete`] - Delete a vector by external ID
//! - [`get_internal_id`] - Resolve external ID to internal vec_id
//! - [`get_external_id`] - Resolve internal vec_id to external ID
//!
//! # Phase 2 Functions (Requires HNSW)
//!
//! - `search` - K-nearest neighbor search (not yet implemented)
//!
//! # Example
//!
//! ```rust,ignore
//! use motlie_db::vector::{api, Writer, Reader, EmbeddingCode};
//! use motlie_db::Id;
//!
//! // Insert a vector
//! let id = Id::new();
//! api::insert(&writer, embedding_code, id, vec![1.0, 2.0, 3.0]).await?;
//!
//! // Retrieve the vector
//! let vector = api::get_vector(&reader, embedding_code, id).await?;
//!
//! // Delete the vector
//! api::delete(&writer, embedding_code, id).await?;
//! ```

use anyhow::Result;

use super::mutation::{DeleteVector, InsertVector, Mutation};
use super::query::{GetExternalId, GetInternalId, GetVector, QueryExecutor};
use super::schema::{EmbeddingCode, VecId};
use super::writer::Writer;
use super::Storage;
use crate::Id;

// ============================================================================
// Insert Operations
// ============================================================================

/// Insert a vector into an embedding space.
///
/// This is a convenience wrapper that creates the mutation and sends it
/// through the writer. The vector is indexed asynchronously by default.
///
/// # Arguments
///
/// * `writer` - The mutation writer handle
/// * `embedding` - The embedding space code
/// * `id` - External document ID (ULID)
/// * `vector` - The vector data (must match embedding dimension)
///
/// # Example
///
/// ```rust,ignore
/// let id = Id::new();
/// api::insert(&writer, embedding_code, id, vec![1.0; 768]).await?;
/// ```
pub async fn insert(
    writer: &Writer,
    embedding: EmbeddingCode,
    id: Id,
    vector: Vec<f32>,
) -> Result<()> {
    let mutation = InsertVector::new(embedding, id, vector);
    writer.send_sync(vec![Mutation::InsertVector(mutation)]).await
}

/// Insert a vector with immediate indexing.
///
/// Like [`insert`], but the vector is indexed synchronously before returning.
/// This is slower but guarantees the vector is searchable immediately.
pub async fn insert_immediate(
    writer: &Writer,
    embedding: EmbeddingCode,
    id: Id,
    vector: Vec<f32>,
) -> Result<()> {
    let mutation = InsertVector::new(embedding, id, vector).immediate();
    writer.send_sync(vec![Mutation::InsertVector(mutation)]).await
}

// ============================================================================
// Query Operations
// ============================================================================

/// Get a vector by its external ID.
///
/// Returns `None` if the vector doesn't exist.
///
/// # Arguments
///
/// * `storage` - The vector storage
/// * `embedding` - The embedding space code
/// * `id` - External document ID (ULID)
///
/// # Example
///
/// ```rust,ignore
/// if let Some(vector) = api::get_vector(&storage, embedding_code, id).await? {
///     println!("Vector has {} dimensions", vector.len());
/// }
/// ```
pub async fn get_vector(
    storage: &Storage,
    embedding: EmbeddingCode,
    id: Id,
) -> Result<Option<Vec<f32>>> {
    GetVector::new(embedding, id).execute(storage).await
}

/// Get the internal vec_id for an external ID.
///
/// Returns `None` if the external ID doesn't exist.
///
/// # Arguments
///
/// * `storage` - The vector storage
/// * `embedding` - The embedding space code
/// * `id` - External document ID (ULID)
pub async fn get_internal_id(
    storage: &Storage,
    embedding: EmbeddingCode,
    id: Id,
) -> Result<Option<VecId>> {
    GetInternalId::new(embedding, id).execute(storage).await
}

/// Get the external ID for an internal vec_id.
///
/// Returns `None` if the vec_id doesn't exist.
///
/// # Arguments
///
/// * `storage` - The vector storage
/// * `embedding` - The embedding space code
/// * `vec_id` - Internal vector ID (u32)
pub async fn get_external_id(
    storage: &Storage,
    embedding: EmbeddingCode,
    vec_id: VecId,
) -> Result<Option<Id>> {
    GetExternalId::new(embedding, vec_id).execute(storage).await
}

// ============================================================================
// Delete Operations
// ============================================================================

/// Delete a vector by its external ID.
///
/// This removes the vector data and ID mappings. The internal vec_id
/// is returned to the free list for reuse.
///
/// # Arguments
///
/// * `writer` - The mutation writer handle
/// * `embedding` - The embedding space code
/// * `id` - External document ID (ULID)
///
/// # Example
///
/// ```rust,ignore
/// api::delete(&writer, embedding_code, id).await?;
/// ```
pub async fn delete(
    writer: &Writer,
    embedding: EmbeddingCode,
    id: Id,
) -> Result<()> {
    let mutation = DeleteVector::new(embedding, id);
    writer.send_sync(vec![Mutation::DeleteVector(mutation)]).await
}

// ============================================================================
// Search Operations (Phase 2 - requires HNSW)
// ============================================================================

// TODO: Phase 2 - Implement search when HNSW is available
//
// /// Search for k nearest neighbors.
// ///
// /// Convenience wrapper for SearchKNN query.
// pub async fn search(
//     reader: &Reader,
//     embedding: EmbeddingCode,
//     query: Vec<f32>,
//     k: usize,
// ) -> Result<Vec<SearchResult>> {
//     SearchKNN {
//         embedding,
//         query,
//         k,
//         ef: k * 10,  // Default ef = 10x k
//     }.run(reader, Duration::from_secs(30)).await
// }

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Full integration tests require RocksDB storage.
    // These tests verify the API signatures and basic construction.

    #[test]
    fn test_insert_vector_mutation_construction() {
        let id = Id::new();
        let mutation = InsertVector::new(1, id, vec![1.0, 2.0, 3.0]);
        assert_eq!(mutation.embedding, 1);
        assert_eq!(mutation.id, id);
        assert_eq!(mutation.vector.len(), 3);
        assert!(!mutation.immediate_index);
    }

    #[test]
    fn test_insert_immediate_mutation_construction() {
        let id = Id::new();
        let mutation = InsertVector::new(1, id, vec![1.0, 2.0, 3.0]).immediate();
        assert!(mutation.immediate_index);
    }

    #[test]
    fn test_delete_vector_mutation_construction() {
        let id = Id::new();
        let mutation = DeleteVector::new(42, id);
        assert_eq!(mutation.embedding, 42);
        assert_eq!(mutation.id, id);
    }

    #[test]
    fn test_query_construction() {
        let id = Id::new();

        let q = GetVector::new(1, id);
        assert_eq!(q.embedding, 1);
        assert_eq!(q.id, id);

        let q = GetInternalId::new(2, id);
        assert_eq!(q.embedding, 2);

        let q = GetExternalId::new(3, 100);
        assert_eq!(q.embedding, 3);
        assert_eq!(q.vec_id, 100);
    }
}
