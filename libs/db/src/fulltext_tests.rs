#[cfg(test)]
mod tests {
    use crate::fulltext::{
        spawn_fulltext_consumer, spawn_fulltext_consumer_with_params, FullTextProcessor,
    };
    use crate::{
        create_mutation_writer, AddEdgeArgs, AddFragmentArgs, AddVertexArgs, Id, WriterConfig,
    };
    use tokio::time::Duration;

    #[tokio::test]
    async fn test_fulltext_consumer_basic_processing() {
        let config = WriterConfig {
            channel_buffer_size: 10,
        };

        let (writer, receiver) = create_mutation_writer(config.clone());

        // Spawn consumer
        let consumer_handle = spawn_fulltext_consumer(receiver, config);

        // Send some mutations
        let vertex_args = AddVertexArgs {
            id: Id::new(),
            ts_millis: 1234567890,
            name: "test_vertex".to_string(),
        };
        writer.add_vertex(vertex_args).await.unwrap();

        let fragment_args = AddFragmentArgs {
            id: Id::new(),
            ts_millis: 1234567890,
            body: "This is a test fragment with some searchable content".to_string(),
        };
        writer.add_fragment(fragment_args).await.unwrap();

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Drop writer to close channel
        drop(writer);

        // Wait for consumer to finish
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_fulltext_consumer_with_custom_params() {
        let config = WriterConfig {
            channel_buffer_size: 10,
        };

        let k1 = 1.5;
        let b = 0.8;

        let (writer, receiver) = create_mutation_writer(config.clone());
        let consumer_handle = spawn_fulltext_consumer_with_params(receiver, config, k1, b);

        // Send a fragment with substantial content
        let fragment_args = AddFragmentArgs {
            id: Id::new(),
            ts_millis: 1234567890,
            body: "The quick brown fox jumps over the lazy dog. This is a longer text fragment that would benefit from BM25 scoring with custom parameters.".to_string(),
        };
        writer.add_fragment(fragment_args).await.unwrap();

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Close and wait
        drop(writer);
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_fulltext_consumer_all_mutation_types() {
        let config = WriterConfig::default();
        let (writer, receiver) = create_mutation_writer(config.clone());
        let consumer_handle = spawn_fulltext_consumer(receiver, config);

        // Test all mutation types with search-relevant content
        writer
            .add_vertex(AddVertexArgs {
                id: Id::new(),
                ts_millis: 1234567890,
                name: "search vertex".to_string(),
            })
            .await
            .unwrap();

        writer
            .add_edge(AddEdgeArgs {
                source_vertex_id: Id::new(),
                target_vertex_id: Id::new(),
                ts_millis: 1234567890,
                name: "connects to".to_string(),
            })
            .await
            .unwrap();

        writer
            .add_fragment(AddFragmentArgs {
                id: Id::new(),
                ts_millis: 1234567890,
                body: "This fragment contains searchable text that should be indexed using BM25 algorithm for effective information retrieval.".to_string(),
            })
            .await
            .unwrap();

        writer
            .invalidate(crate::InvalidateArgs {
                id: Id::new(),
                ts_millis: 1234567890,
                reason: "content removed from search index".to_string(),
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
    async fn test_fulltext_processor_creation() {
        // Test default processor
        let processor = FullTextProcessor::new();
        assert_eq!(processor.k1, 1.2);
        assert_eq!(processor.b, 0.75);

        // Test processor with custom params
        let processor = FullTextProcessor::with_params(2.0, 0.5);
        assert_eq!(processor.k1, 2.0);
        assert_eq!(processor.b, 0.5);

        // Test default trait
        let processor: FullTextProcessor = Default::default();
        assert_eq!(processor.k1, 1.2);
        assert_eq!(processor.b, 0.75);
    }

    #[tokio::test]
    async fn test_fulltext_multiple_mutations() {
        let config = WriterConfig {
            channel_buffer_size: 100,
        };

        let (writer, receiver) = create_mutation_writer(config.clone());
        let consumer_handle = spawn_fulltext_consumer(receiver, config);

        // Send 5 fragments rapidly
        for i in 0..5 {
            let fragment_args = AddFragmentArgs {
                id: Id::new(),
                ts_millis: 1234567890 + i,
                body: format!(
                    "Fragment {} with searchable content for full-text indexing",
                    i
                ),
            };
            writer.add_fragment(fragment_args).await.unwrap();
        }

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Close and wait
        drop(writer);
        consumer_handle.await.unwrap().unwrap();
    }
}
