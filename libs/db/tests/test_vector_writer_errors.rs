//! Integration tests for vector writer error handling.
//!
//! ## Migration Note (Task 5.9)
//!
//! This test has been migrated to use only public APIs:
//! - `spawn_mutation_consumer_with_storage_autoreg` instead of direct Consumer creation
//!
//! This enables making Processor `pub(crate)`.

use std::sync::Arc;

use motlie_db::vector;
use motlie_db::vector::mutation::{InsertVector, Mutation};
use motlie_db::vector::writer::{create_writer, WriterConfig};
use motlie_db::vector::spawn_mutation_consumer_with_storage_autoreg;
use motlie_db::Id;
use tempfile::TempDir;

#[tokio::test]
async fn test_insert_unknown_embedding_fails() {
    let temp_dir = TempDir::new().expect("temp dir");
    let db_path = temp_dir.path().join("vector_db");

    let mut storage = vector::Storage::readwrite(&db_path);
    storage.ready().expect("storage ready");
    let storage = Arc::new(storage);

    let (writer, receiver) = create_writer(WriterConfig::default());

    // Use public API to spawn consumer instead of directly creating Consumer with Processor
    let handle = spawn_mutation_consumer_with_storage_autoreg(
        receiver,
        WriterConfig::default(),
        storage.clone(),
    );

    // Construct InsertVector directly with invalid embedding code to test error handling
    // (Normal usage goes through InsertVector::new(&embedding, ...) which ensures validity)
    let mutation = InsertVector {
        embedding: 42, // Unknown embedding code
        external_key: motlie_db::vector::schema::ExternalKey::NodeId(Id::new()),
        vector: vec![1.0, 2.0, 3.0],
        immediate_index: false,
    };
    writer
        .send(vec![Mutation::InsertVector(mutation)])
        .await
        .expect("send mutation");
    drop(writer);

    let result = handle.await.expect("consumer join");
    let err = result.expect_err("expected consumer error");
    let matches = err
        .chain()
        .any(|cause| cause.to_string().contains("Unknown embedding code: 42"));
    assert!(matches, "unexpected error chain: {:?}", err);
}
