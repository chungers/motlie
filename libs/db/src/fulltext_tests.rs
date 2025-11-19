#[cfg(test)]
mod tests {
    use crate::fulltext::{
        spawn_fulltext_consumer, spawn_fulltext_consumer_with_params, FullTextProcessor,
    };
    use crate::mutation::Runnable as MutRunnable;
    use crate::{
        create_mutation_writer, AddEdge, AddNode, AddNodeFragment, EdgeSummary, Id, TimestampMilli, WriterConfig,
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
        let node_args = AddNode {
            id: Id::new(),
            ts_millis: TimestampMilli::now(),
            name: "test_node".to_string(),
            temporal_range: None,
        };
        node_args.run(&writer).await.unwrap();

        let fragment_args = AddNodeFragment {
            id: Id::new(),
            ts_millis: TimestampMilli(1234567890),
            content: crate::DataUrl::from_text(
                "This is a test fragment with some searchable content",
            ),
            temporal_range: None,
        };
        fragment_args.run(&writer).await.unwrap();

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
        let fragment_args = AddNodeFragment {
            id: Id::new(),
            ts_millis: TimestampMilli(1234567890),
            content: crate::DataUrl::from_text("The quick brown fox jumps over the lazy dog. This is a longer text fragment that would benefit from BM25 scoring with custom parameters."),
            temporal_range: None,
        };
        fragment_args.run(&writer).await.unwrap();

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
        AddNode {
            id: Id::new(),
            ts_millis: TimestampMilli::now(),
            name: "search node".to_string(),
            temporal_range: None,
        }
        .run(&writer)
        .await
        .unwrap();

        AddEdge {
            source_node_id: Id::new(),
            target_node_id: Id::new(),
            ts_millis: TimestampMilli::now(),
            name: "connects to".to_string(),
            summary: EdgeSummary::from_text(""),
            weight: Some(1.0),
            temporal_range: None,
        }
        .run(&writer)
        .await
        .unwrap();

        AddNodeFragment {
            id: Id::new(),
            ts_millis: TimestampMilli::now(),
            content: crate::DataUrl::from_text("This fragment contains searchable text that should be indexed using BM25 algorithm for effective information retrieval."),
            temporal_range: None,
        }
        .run(&writer)
        .await
        .unwrap();

        let src_id = Id::new();
        let dst_id = Id::new();
        crate::UpdateEdgeValidSinceUntil {
            src_id,
            dst_id,
            name: "test_edge".to_string(),
            temporal_range: crate::schema::ValidTemporalRange(None, None),
            reason: "content removed from search index".to_string(),
        }
        .run(&writer)
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
            let fragment_args = AddNodeFragment {
                id: Id::new(),
                ts_millis: TimestampMilli::now(),
                content: crate::DataUrl::from_text(&format!(
                    "Fragment {} with searchable content for full-text indexing",
                    i
                )),
                temporal_range: None,
            };
            fragment_args.run(&writer).await.unwrap();
        }

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Close and wait
        drop(writer);
        consumer_handle.await.unwrap().unwrap();
    }
}
