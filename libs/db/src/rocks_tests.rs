#[cfg(test)]
mod tests {
    use crate::rocks::{spawn_rocks_consumer, spawn_rocks_consumer_with_next};
    use crate::{
        create_mutation_writer, AddEdgeArgs, AddFragmentArgs, AddVertexArgs, Id, WriterConfig,
    };
    use tokio::sync::mpsc;
    use tokio::time::Duration;

    #[tokio::test]
    async fn test_rocks_consumer_basic_processing() {
        let config = WriterConfig {
            channel_buffer_size: 10,
        };

        let (writer, receiver) = create_mutation_writer(config.clone());

        // Spawn consumer
        let consumer_handle = spawn_rocks_consumer(receiver, config);

        // Send some mutations
        let vertex_args = AddVertexArgs {
            id: Id::new(),
            ts_millis: 1234567890,
            name: "test_vertex".to_string(),
        };
        writer.add_vertex(vertex_args).await.unwrap();

        let edge_args = AddEdgeArgs {
            source_vertex_id: Id::new(),
            target_vertex_id: Id::new(),
            ts_millis: 1234567890,
            name: "test_edge".to_string(),
        };
        writer.add_edge(edge_args).await.unwrap();

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Drop writer to close channel
        drop(writer);

        // Wait for consumer to finish
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_rocks_consumer_multiple_mutations() {
        let config = WriterConfig {
            channel_buffer_size: 100,
        };

        let (writer, receiver) = create_mutation_writer(config.clone());
        let consumer_handle = spawn_rocks_consumer(receiver, config);

        // Send 5 mutations rapidly
        for i in 0..5 {
            let vertex_args = AddVertexArgs {
                id: Id::new(),
                ts_millis: 1234567890 + i,
                name: format!("test_vertex_{}", i),
            };
            writer.add_vertex(vertex_args).await.unwrap();
        }

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Close and wait
        drop(writer);
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_rocks_consumer_all_mutation_types() {
        let config = WriterConfig::default();
        let (writer, receiver) = create_mutation_writer(config.clone());
        let consumer_handle = spawn_rocks_consumer(receiver, config);

        // Test all mutation types
        writer
            .add_vertex(AddVertexArgs {
                id: Id::new(),
                ts_millis: 1234567890,
                name: "vertex".to_string(),
            })
            .await
            .unwrap();

        writer
            .add_edge(AddEdgeArgs {
                source_vertex_id: Id::new(),
                target_vertex_id: Id::new(),
                ts_millis: 1234567890,
                name: "edge".to_string(),
            })
            .await
            .unwrap();

        writer
            .add_fragment(AddFragmentArgs {
                id: Id::new(),
                ts_millis: 1234567890,
                body: "fragment body".to_string(),
            })
            .await
            .unwrap();

        writer
            .invalidate(crate::InvalidateArgs {
                id: Id::new(),
                ts_millis: 1234567890,
                reason: "test invalidation".to_string(),
            })
            .await
            .unwrap();

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Close and wait
        drop(writer);
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_rocks_to_bm25_chaining() {
        let config = WriterConfig {
            channel_buffer_size: 100,
        };

        // Create the BM25 consumer (end of chain)
        let (bm25_sender, bm25_receiver) = mpsc::channel(config.channel_buffer_size);
        let bm25_handle = crate::spawn_bm25_consumer(bm25_receiver, config.clone());

        // Create the RocksDB consumer that forwards to BM25
        let (writer, rocks_receiver) = create_mutation_writer(config.clone());
        let rocks_handle =
            spawn_rocks_consumer_with_next(rocks_receiver, config.clone(), bm25_sender);

        // Send mutations - they should flow through RocksDB -> BM25
        for i in 0..3 {
            let vertex_args = AddVertexArgs {
                id: Id::new(),
                ts_millis: 1234567890 + i,
                name: format!("chained_vertex_{}", i),
            };
            let fragment_args = AddFragmentArgs {
                id: Id::new(),
                ts_millis: 1234567890 + i,
                body: format!("Chained fragment {} processed by both RocksDB and BM25", i),
            };

            writer.add_vertex(vertex_args).await.unwrap();
            writer.add_fragment(fragment_args).await.unwrap();
        }

        // Give both consumers time to process
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Shutdown the chain from the beginning
        drop(writer);

        // Wait for RocksDB consumer to complete (which will close BM25's channel)
        rocks_handle.await.unwrap().unwrap();

        // Wait for BM25 consumer to complete
        bm25_handle.await.unwrap().unwrap();
    }
}
