use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use motlie_db::vector::{
    create_reader_with_storage, create_writer, spawn_mutation_consumer_with_storage_autoreg,
    spawn_query_consumers_with_storage_autoreg, Distance, EmbeddingBuilder, ExternalKey,
    InsertVector, ListEmbeddings, MutationRunnable, ReaderConfig, Runnable, SearchKNN, Storage,
    WriterConfig,
};
use motlie_db::Id;

#[tokio::main]
async fn main() -> Result<()> {
    let base_path = std::env::temp_dir().join(format!(
        "motlie-db-vector-example-{}",
        Id::new()
    ));

    let mut storage = Storage::readwrite(&base_path);
    storage.ready()?;
    let storage = Arc::new(storage);

    let registry = storage.cache().clone();
    registry.set_storage(storage.clone())?;
    let embedding = registry.register(
        EmbeddingBuilder::new("demo-embedding", 3, Distance::Cosine),
    )?;

    let (writer, mutation_rx) = create_writer(WriterConfig::default());
    let mutation_handle = spawn_mutation_consumer_with_storage_autoreg(
        mutation_rx,
        WriterConfig::default(),
        storage.clone(),
    );

    let (reader, query_rx) = create_reader_with_storage(ReaderConfig::default());
    let query_handles = spawn_query_consumers_with_storage_autoreg(
        query_rx,
        ReaderConfig::default(),
        storage.clone(),
        2,
    );

    let id_a = Id::new();
    let id_b = Id::new();
    let id_c = Id::new();

    InsertVector::new(
        &embedding,
        ExternalKey::NodeId(id_a),
        vec![1.0, 0.0, 0.0],
    )
    .immediate()
    .run(&writer)
    .await?;
    InsertVector::new(
        &embedding,
        ExternalKey::NodeId(id_b),
        vec![0.9, 0.1, 0.0],
    )
    .immediate()
    .run(&writer)
    .await?;
    InsertVector::new(
        &embedding,
        ExternalKey::NodeId(id_c),
        vec![0.0, 1.0, 0.0],
    )
    .immediate()
    .run(&writer)
    .await?;

    writer.flush().await?;

    let timeout = Duration::from_secs(5);
    let embeddings = ListEmbeddings::new().run(&reader, timeout).await?;
    println!("registered embeddings: {}", embeddings.len());

    let results = SearchKNN::new(&embedding, vec![0.95, 0.05, 0.0], 2)
        .with_ef(50)
        .run(&reader, timeout)
        .await?;

    println!("SearchKNN returned {} results", results.len());
    for (idx, result) in results.iter().enumerate() {
        println!(
            "{}. key={:?}, vec_id={}, distance={:.4}",
            idx + 1,
            result.external_key,
            result.vec_id,
            result.distance
        );
    }

    drop(writer);
    drop(reader);

    mutation_handle.await??;
    for handle in query_handles {
        handle.await??;
    }

    if let Err(err) = std::fs::remove_dir_all(&base_path) {
        eprintln!(
            "warning: failed to remove temporary path {}: {}",
            base_path.display(),
            err
        );
    }

    Ok(())
}
