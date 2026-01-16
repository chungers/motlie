//! Integration tests for vector writer error handling.

use std::sync::Arc;

use motlie_db::vector;
use motlie_db::vector::mutation::{InsertVector, Mutation};
use motlie_db::vector::writer::{create_writer, Consumer, WriterConfig};
use motlie_db::Id;
use tempfile::TempDir;

#[tokio::test]
async fn test_insert_unknown_embedding_fails() {
    let temp_dir = TempDir::new().expect("temp dir");
    let db_path = temp_dir.path().join("vector_db");

    let mut storage = vector::Storage::readwrite(&db_path);
    storage.ready().expect("storage ready");
    let storage = Arc::new(storage);

    let registry = storage.cache().clone();
    let processor = Arc::new(vector::Processor::new(storage, registry));

    let (writer, receiver) = create_writer(WriterConfig::default());
    let consumer = Consumer::new(receiver, WriterConfig::default(), processor);
    let handle = tokio::spawn(async move { consumer.run().await });

    // Construct InsertVector directly with invalid embedding code to test error handling
    // (Normal usage goes through InsertVector::new(&embedding, ...) which ensures validity)
    let mutation = InsertVector {
        embedding: 42, // Unknown embedding code
        id: Id::new(),
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
