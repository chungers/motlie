//! Integration test demonstrating consumer chaining: Graph -> FullText
//!
//! This test shows how to:
//! - Chain consumers so mutations flow through Graph storage first, then to FullText indexing
//! - Use spawn_mutation_consumer_with_next to forward mutations downstream
//! - Verify both consumers process the same data in sequence

use motlie_db::fulltext::spawn_mutation_consumer as spawn_fulltext_mutation_consumer;
use motlie_db::graph::mutation::{AddEdge, AddNode, AddNodeFragment, Runnable as MutationRunnable};
use motlie_db::graph::schema::{EdgeSummary, NodeSummary};
use motlie_db::graph::writer::{
    create_mutation_writer, spawn_mutation_consumer_with_next, WriterConfig,
};
use motlie_db::{DataUrl, Id, TimestampMilli};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::Value;
use tantivy::Index;
use tempfile::TempDir;
use tokio::sync::mpsc;
use tokio::time::Duration;

/// Test that mutations flow through Graph -> FullText consumer chain
#[tokio::test]
async fn test_fulltext_chaining_basic() {
    // Create temporary directories for storage
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");
    let index_path = temp_dir.path().join("fulltext_index");

    let config = WriterConfig {
        channel_buffer_size: 100,
    };

    // Create the FullText consumer (end of chain)
    let (fulltext_sender, fulltext_receiver) = mpsc::channel(config.channel_buffer_size);
    let fulltext_handle = spawn_fulltext_mutation_consumer(fulltext_receiver, config.clone(), &index_path);

    // Create the Graph consumer that forwards to FullText
    let (writer, graph_receiver) = create_mutation_writer(config.clone());
    let graph_handle = spawn_mutation_consumer_with_next(
        graph_receiver,
        config.clone(),
        &db_path,
        fulltext_sender,
    );

    println!("=== Testing Graph -> FullText Consumer Chaining ===");

    // Create nodes and fragments that will flow through the chain
    let node_ids: Vec<Id> = (0..3).map(|_| Id::new()).collect();

    // Add nodes
    for (i, node_id) in node_ids.iter().enumerate() {
        let node = AddNode {
            id: *node_id,
            ts_millis: TimestampMilli::now(),
            name: format!("ChainedNode_{}", i),
            valid_range: None,
            summary: NodeSummary::from_text(&format!("Summary for node {}", i)),
        };
        node.run(&writer).await.unwrap();
    }

    // Add fragments with searchable content
    let fragment_contents = [
        "Rust is a systems programming language focused on safety and performance",
        "Python is great for data science and machine learning applications",
        "JavaScript powers interactive web applications and Node.js servers",
    ];

    for (i, (node_id, content)) in node_ids.iter().zip(fragment_contents.iter()).enumerate() {
        let fragment = AddNodeFragment {
            id: *node_id,
            ts_millis: TimestampMilli::now(),
            content: DataUrl::from_markdown(content),
            valid_range: None,
        };
        fragment.run(&writer).await.unwrap();
        println!("Sent fragment {} through chain: {}", i, content);
    }

    // Add an edge to verify edge mutations also flow through
    let edge = AddEdge {
        source_node_id: node_ids[0],
        target_node_id: node_ids[1],
        ts_millis: TimestampMilli::now(),
        name: "related_to".to_string(),
        summary: EdgeSummary::from_text("Rust and Python are both popular programming languages"),
        weight: Some(0.8),
        valid_range: None,
    };
    edge.run(&writer).await.unwrap();

    // Give consumers time to process
    println!("Waiting for chain to process mutations...");
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Shutdown the chain from the beginning
    drop(writer);

    // Wait for Graph consumer to complete (which will close FullText's channel)
    graph_handle.await.unwrap().unwrap();
    println!("Graph consumer completed");

    // Wait for FullText consumer to complete
    fulltext_handle.await.unwrap().unwrap();
    println!("FullText consumer completed");

    // Verify FullText index received and indexed the data
    println!("\n=== Verifying FullText Index ===");

    let index = Index::open_in_dir(&index_path).expect("Failed to open index");
    let reader = index.reader().expect("Failed to create reader");
    let searcher = reader.searcher();

    let schema = index.schema();
    let content_field = schema.get_field("content").expect("content field not found");
    let node_name_field = schema
        .get_field("node_name")
        .expect("node_name field not found");

    // Search for "programming" - should find multiple results
    let query_parser = QueryParser::for_index(&index, vec![content_field, node_name_field]);
    let query = query_parser.parse_query("programming").unwrap();
    let top_docs = searcher.search(&query, &TopDocs::with_limit(10)).unwrap();

    println!("Search for 'programming' found {} results", top_docs.len());
    assert!(
        top_docs.len() >= 2,
        "Expected at least 2 documents about programming"
    );

    // Search for "Rust" - should find the Rust fragment
    let query = query_parser.parse_query("Rust").unwrap();
    let top_docs = searcher.search(&query, &TopDocs::with_limit(5)).unwrap();
    println!("Search for 'Rust' found {} results", top_docs.len());
    assert!(!top_docs.is_empty(), "Expected to find Rust-related content");

    // Verify the nodes were indexed
    for (score, doc_address) in &top_docs {
        let retrieved_doc = searcher
            .doc::<tantivy::TantivyDocument>(*doc_address)
            .unwrap();
        if let Some(name_value) = retrieved_doc.get_first(node_name_field) {
            if let Some(text) = name_value.as_str() {
                println!("  - Found: {} (score: {:.4})", text, score);
            }
        }
    }

    // Verify total document count
    let total_docs = searcher.num_docs();
    println!("\nTotal documents indexed: {}", total_docs);
    // 3 nodes + 3 fragments + 1 edge = 7 documents (at least)
    assert!(
        total_docs >= 6,
        "Expected at least 6 documents in the index"
    );

    println!("\n✅ Graph -> FullText chaining test passed!");
}

/// Test chaining with high-volume mutations
#[tokio::test]
async fn test_fulltext_chaining_high_volume() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");
    let index_path = temp_dir.path().join("fulltext_index");

    let config = WriterConfig {
        channel_buffer_size: 1000,
    };

    // Create the chain: Writer -> Graph -> FullText
    let (fulltext_sender, fulltext_receiver) = mpsc::channel(config.channel_buffer_size);
    let fulltext_handle = spawn_fulltext_mutation_consumer(fulltext_receiver, config.clone(), &index_path);

    let (writer, graph_receiver) = create_mutation_writer(config.clone());
    let graph_handle = spawn_mutation_consumer_with_next(
        graph_receiver,
        config.clone(),
        &db_path,
        fulltext_sender,
    );

    // Send many mutations rapidly
    let mutation_count = 100;
    println!(
        "Sending {} mutations through the chain...",
        mutation_count * 2
    );

    for i in 0..mutation_count {
        let node_id = Id::new();

        let node = AddNode {
            id: node_id,
            ts_millis: TimestampMilli::now(),
            name: format!("HighVolumeNode_{}", i),
            valid_range: None,
            summary: NodeSummary::from_text(&format!("High volume summary {}", i)),
        };

        let fragment = AddNodeFragment {
            id: node_id,
            ts_millis: TimestampMilli::now(),
            content: DataUrl::from_text(&format!(
                "High volume content {} with unique identifier {}",
                i,
                node_id.as_str()
            )),
            valid_range: None,
        };

        node.run(&writer).await.unwrap();
        fragment.run(&writer).await.unwrap();
    }

    // Give consumers time to process
    tokio::time::sleep(Duration::from_millis(1000)).await;

    // Shutdown
    drop(writer);
    graph_handle.await.unwrap().unwrap();
    fulltext_handle.await.unwrap().unwrap();

    // Verify
    let index = Index::open_in_dir(&index_path).expect("Failed to open index");
    let reader = index.reader().expect("Failed to create reader");
    let searcher = reader.searcher();

    let total_docs = searcher.num_docs();
    println!("Total documents indexed: {}", total_docs);

    // We expect at least the fragments to be indexed
    assert!(
        total_docs >= mutation_count as u64,
        "Expected at least {} documents, got {}",
        mutation_count,
        total_docs
    );

    println!("✅ High-volume chaining test passed!");
}

/// Test that chain handles shutdown gracefully
#[tokio::test]
async fn test_fulltext_chaining_graceful_shutdown() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");
    let index_path = temp_dir.path().join("fulltext_index");

    let config = WriterConfig {
        channel_buffer_size: 10,
    };

    // Create the chain
    let (fulltext_sender, fulltext_receiver) = mpsc::channel(config.channel_buffer_size);
    let fulltext_handle = spawn_fulltext_mutation_consumer(fulltext_receiver, config.clone(), &index_path);

    let (writer, graph_receiver) = create_mutation_writer(config.clone());
    let graph_handle = spawn_mutation_consumer_with_next(
        graph_receiver,
        config.clone(),
        &db_path,
        fulltext_sender,
    );

    // Send a few mutations
    for i in 0..5 {
        let node = AddNode {
            id: Id::new(),
            ts_millis: TimestampMilli::now(),
            name: format!("ShutdownTest_{}", i),
            valid_range: None,
            summary: NodeSummary::from_text(&format!("Shutdown test summary {}", i)),
        };
        node.run(&writer).await.unwrap();
    }

    // Immediate shutdown (don't wait for processing)
    drop(writer);

    // Both consumers should shutdown gracefully
    let graph_result = graph_handle.await.unwrap();
    assert!(graph_result.is_ok(), "Graph consumer should shutdown cleanly");

    let fulltext_result = fulltext_handle.await.unwrap();
    assert!(
        fulltext_result.is_ok(),
        "FullText consumer should shutdown cleanly"
    );

    println!("✅ Graceful shutdown test passed!");
}
