#[cfg(test)]
mod tests {
    use crate::graph::{spawn_graph_consumer, spawn_graph_consumer_with_next};
    use crate::{
        create_mutation_writer, AddEdgeArgs, AddFragmentArgs, AddVertexArgs, Id, WriterConfig,
    };
    use std::path::Path;
    use tokio::sync::mpsc;
    use tokio::time::Duration;

    #[tokio::test]
    async fn test_graph_consumer_basic_processing() {
        let config = WriterConfig {
            channel_buffer_size: 10,
        };

        let (writer, receiver) = create_mutation_writer(config.clone());

        // Spawn consumer
        let consumer_handle =
            spawn_graph_consumer(receiver, config, Path::new("/tmp/test_graph_db"));

        // Send some mutations
        let vertex_args = AddVertexArgs {
            id: Id::new(),
            ts_millis: 1234567890,
            name: "test_vertex".to_string(),
        };
        writer.add_vertex(vertex_args).await.unwrap();

        let edge_args = AddEdgeArgs {
            id: Id::new(),
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
    async fn test_graph_consumer_multiple_mutations() {
        let config = WriterConfig {
            channel_buffer_size: 100,
        };

        let (writer, receiver) = create_mutation_writer(config.clone());
        let consumer_handle =
            spawn_graph_consumer(receiver, config, Path::new("/tmp/test_graph_db"));

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
    async fn test_graph_consumer_all_mutation_types() {
        let config = WriterConfig::default();
        let (writer, receiver) = create_mutation_writer(config.clone());
        let consumer_handle =
            spawn_graph_consumer(receiver, config, Path::new("/tmp/test_graph_db"));

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
                id: Id::new(),
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
    async fn test_graph_to_fulltext_chaining() {
        let config = WriterConfig {
            channel_buffer_size: 100,
        };

        // Create the FullText consumer (end of chain)
        let (fulltext_sender, fulltext_receiver) = mpsc::channel(config.channel_buffer_size);
        let fulltext_handle = crate::spawn_fulltext_consumer(fulltext_receiver, config.clone());

        // Create the Graph consumer that forwards to FullText
        let (writer, graph_receiver) = create_mutation_writer(config.clone());
        let graph_handle = spawn_graph_consumer_with_next(
            graph_receiver,
            config.clone(),
            Path::new("/tmp/test_graph_db"),
            fulltext_sender,
        );

        // Send mutations - they should flow through Graph -> FullText
        for i in 0..3 {
            let vertex_args = AddVertexArgs {
                id: Id::new(),
                ts_millis: 1234567890 + i,
                name: format!("chained_vertex_{}", i),
            };
            let fragment_args = AddFragmentArgs {
                id: Id::new(),
                ts_millis: 1234567890 + i,
                body: format!(
                    "Chained fragment {} processed by both Graph and FullText",
                    i
                ),
            };

            writer.add_vertex(vertex_args).await.unwrap();
            writer.add_fragment(fragment_args).await.unwrap();
        }

        // Give both consumers time to process
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Shutdown the chain from the beginning
        drop(writer);

        // Wait for Graph consumer to complete (which will close FullText's channel)
        graph_handle.await.unwrap().unwrap();

        // Wait for FullText consumer to complete
        fulltext_handle.await.unwrap().unwrap();
    }
}
