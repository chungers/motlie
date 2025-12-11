//! Integration test demonstrating fulltext search usage with Tantivy
//!
//! This test shows how to:
//! - Set up a fulltext indexing consumer alongside graph storage
//! - Index nodes, edges, and fragments with searchable content
//! - Perform searches using Tantivy's query parser
//! - Retrieve and rank results using BM25 scoring

use motlie_db::fulltext::spawn_mutation_consumer as spawn_fulltext_mutation_consumer;
use motlie_db::graph::mutation::{AddEdge, AddNode, AddNodeFragment, Runnable as MutationRunnable};
use motlie_db::graph::schema::{EdgeSummary, NodeSummary};
use motlie_db::graph::writer::{create_mutation_writer, spawn_mutation_consumer, WriterConfig};
use motlie_db::{DataUrl, Id, TimestampMilli};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::Value;
use tantivy::Index;
use tempfile::TempDir;
use tokio::time::Duration;

#[tokio::test]
async fn test_fulltext_search_integration() {
    // Create temporary directories for storage
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");
    let index_path = temp_dir.path().join("fulltext_index");

    let config = WriterConfig {
        channel_buffer_size: 100,
    };

    // Create separate writers for graph and fulltext to demonstrate fanout pattern
    let (graph_writer, graph_receiver) = create_mutation_writer(config.clone());
    let (fulltext_writer, fulltext_receiver) = create_mutation_writer(config.clone());

    // Spawn both consumers
    let graph_handle = spawn_mutation_consumer(graph_receiver, config.clone(), &db_path);
    let fulltext_handle = spawn_fulltext_mutation_consumer(fulltext_receiver, config.clone(), &index_path);

    // Scenario: Building a knowledge graph about programming languages
    println!("=== Building Knowledge Graph ===");

    // Add nodes for programming languages
    let rust_id = Id::new();
    let rust_node = AddNode {
        id: rust_id,
        ts_millis: TimestampMilli::now(),
        name: "Rust".to_string(),
        valid_range: None,
        summary: NodeSummary::from_text("Rust programming language"),
    };

    let python_id = Id::new();
    let python_node = AddNode {
        id: python_id,
        ts_millis: TimestampMilli::now(),
        name: "Python".to_string(),
        valid_range: None,
        summary: NodeSummary::from_text("Python programming language"),
    };

    let javascript_id = Id::new();
    let javascript_node = AddNode {
        id: javascript_id,
        ts_millis: TimestampMilli::now(),
        name: "JavaScript".to_string(),
        valid_range: None,
        summary: NodeSummary::from_text("JavaScript programming language"),
    };

    // Send to both graph and fulltext
    rust_node.clone().run(&graph_writer).await.unwrap();
    rust_node.run(&fulltext_writer).await.unwrap();

    python_node.clone().run(&graph_writer).await.unwrap();
    python_node.run(&fulltext_writer).await.unwrap();

    javascript_node
        .clone()
        .run(&graph_writer)
        .await
        .unwrap();
    javascript_node.run(&fulltext_writer).await.unwrap();

    // Add detailed content fragments (main searchable content)
    let rust_fragment = AddNodeFragment {
        id: rust_id,
        ts_millis: TimestampMilli::now(),
        content: DataUrl::from_markdown(
            r#"# Rust Programming Language

Rust is a systems programming language that focuses on **safety**, **concurrency**,
and **performance**. It provides memory safety without garbage collection through
its ownership system. Rust is ideal for building reliable and efficient systems,
including operating systems, game engines, and web servers.

Key features:
- Zero-cost abstractions
- Memory safety guarantees
- Fearless concurrency
- Powerful type system with traits
"#,
        ),
        valid_range: None,
    };

    let python_fragment = AddNodeFragment {
        id: python_id,
        ts_millis: TimestampMilli::now(),
        content: DataUrl::from_markdown(
            r#"# Python Programming Language

Python is a high-level, interpreted programming language known for its **simplicity**
and **readability**. It's widely used in data science, machine learning, web development,
and automation. Python's extensive ecosystem of libraries makes it versatile for
various applications.

Key features:
- Easy to learn and use
- Dynamic typing
- Extensive standard library
- Strong community support
- Great for rapid prototyping
"#,
        ),
        valid_range: None,
    };

    let javascript_fragment = AddNodeFragment {
        id: javascript_id,
        ts_millis: TimestampMilli::now(),
        content: DataUrl::from_markdown(
            r#"# JavaScript Programming Language

JavaScript is the language of the web, enabling **interactive** and **dynamic**
web applications. It runs in browsers and on servers (Node.js). JavaScript powers
modern web frameworks like React, Vue, and Angular.

Key features:
- Runs everywhere (browser, server, mobile)
- Event-driven and asynchronous
- Prototype-based object orientation
- Functional programming support
- Massive ecosystem (npm)
"#,
        ),
        valid_range: None,
    };

    rust_fragment.clone().run(&graph_writer).await.unwrap();
    rust_fragment.run(&fulltext_writer).await.unwrap();

    python_fragment
        .clone()
        .run(&graph_writer)
        .await
        .unwrap();
    python_fragment.run(&fulltext_writer).await.unwrap();

    javascript_fragment
        .clone()
        .run(&graph_writer)
        .await
        .unwrap();
    javascript_fragment
        .run(&fulltext_writer)
        .await
        .unwrap();

    // Add relationships
    let rust_to_systems = AddEdge {
        source_node_id: rust_id,
        target_node_id: Id::new(),
        ts_millis: TimestampMilli::now(),
        name: "used_for".to_string(),
        summary: EdgeSummary::from_text("Systems programming and performance-critical applications"),
        weight: Some(1.0),
        valid_range: None,
    };

    rust_to_systems
        .clone()
        .run(&graph_writer)
        .await
        .unwrap();
    rust_to_systems.run(&fulltext_writer).await.unwrap();

    // Give consumers time to process
    println!("Waiting for indexing to complete...");
    tokio::time::sleep(Duration::from_millis(500)).await;

    println!("=== Performing Fulltext Searches ===\n");

    // Open the Tantivy index for searching
    let index = Index::open_in_dir(&index_path).expect("Failed to open index");
    let reader = index.reader().expect("Failed to create reader");
    let searcher = reader.searcher();

    // Get the schema to know which fields to search
    let schema = index.schema();
    let content_field = schema.get_field("content").expect("content field not found");
    let node_name_field = schema
        .get_field("node_name")
        .expect("node_name field not found");

    // Search 1: Find languages good for "performance"
    println!("Search 1: Find content about 'performance'");
    let query_parser = QueryParser::for_index(&index, vec![content_field, node_name_field]);
    let query = query_parser.parse_query("performance").unwrap();
    let top_docs = searcher.search(&query, &TopDocs::with_limit(5)).unwrap();

    println!("Found {} results:", top_docs.len());
    for (_score, doc_address) in top_docs {
        let retrieved_doc = searcher.doc::<tantivy::TantivyDocument>(doc_address).unwrap();
        if let Some(name_value) = retrieved_doc.get_first(node_name_field) {
            if let Some(text) = name_value.as_str() {
                println!("  - {}", text);
            }
        }
    }
    println!();

    // Search 2: Find languages for "web" development
    println!("Search 2: Find content about 'web'");
    let query = query_parser.parse_query("web").unwrap();
    let top_docs = searcher.search(&query, &TopDocs::with_limit(5)).unwrap();

    println!("Found {} results:", top_docs.len());
    for (_score, doc_address) in top_docs {
        let retrieved_doc = searcher.doc::<tantivy::TantivyDocument>(doc_address).unwrap();
        if let Some(name_value) = retrieved_doc.get_first(node_name_field) {
            if let Some(text) = name_value.as_str() {
                println!("  - {}", text);
            }
        }
    }
    println!();

    // Search 3: Find content about "memory safety"
    println!("Search 3: Find content about 'memory safety'");
    let query = query_parser.parse_query("memory safety").unwrap();
    let top_docs = searcher.search(&query, &TopDocs::with_limit(5)).unwrap();

    println!("Found {} results:", top_docs.len());
    for (_score, doc_address) in top_docs {
        let retrieved_doc = searcher.doc::<tantivy::TantivyDocument>(doc_address).unwrap();
        if let Some(name_value) = retrieved_doc.get_first(node_name_field) {
            if let Some(text) = name_value.as_str() {
                println!("  - {}", text);
            }
        }
    }
    println!();

    // Search 4: Complex query with boolean operators
    println!("Search 4: Find 'programming AND (systems OR web)'");
    let query = query_parser
        .parse_query("programming AND (systems OR web)")
        .unwrap();
    let top_docs = searcher.search(&query, &TopDocs::with_limit(5)).unwrap();

    println!("Found {} results:", top_docs.len());
    for (_score, doc_address) in top_docs {
        let retrieved_doc = searcher.doc::<tantivy::TantivyDocument>(doc_address).unwrap();
        if let Some(name_value) = retrieved_doc.get_first(node_name_field) {
            if let Some(text) = name_value.as_str() {
                println!("  - {}", text);
            }
        }
    }
    println!();

    // Verify we indexed the right number of documents
    println!("=== Verification ===");
    let total_docs = searcher.num_docs();
    println!("Total documents indexed: {}", total_docs);
    assert!(
        total_docs >= 6,
        "Expected at least 6 documents (3 nodes + 3 fragments)"
    );

    // Shutdown
    drop(graph_writer);
    drop(fulltext_writer);

    graph_handle.await.unwrap().unwrap();
    fulltext_handle.await.unwrap().unwrap();

    println!("\n✅ Integration test completed successfully!");
}

#[tokio::test]
async fn test_fulltext_phrase_search() {
    // Demonstrate phrase search capabilities
    let temp_dir = TempDir::new().unwrap();
    let index_path = temp_dir.path().join("fulltext_index");

    let config = WriterConfig {
        channel_buffer_size: 10,
    };

    let (writer, receiver) = create_mutation_writer(config.clone());
    let fulltext_handle = spawn_fulltext_mutation_consumer(receiver, config, &index_path);

    // Add fragments with specific phrases
    AddNodeFragment {
        id: Id::new(),
        ts_millis: TimestampMilli::now(),
        content: DataUrl::from_text("The quick brown fox jumps over the lazy dog"),
        valid_range: None,
    }
    .run(&writer)
    .await
    .unwrap();

    AddNodeFragment {
        id: Id::new(),
        ts_millis: TimestampMilli::now(),
        content: DataUrl::from_text("A fast brown fox and a lazy cat"),
        valid_range: None,
    }
    .run(&writer)
    .await
    .unwrap();

    tokio::time::sleep(Duration::from_millis(200)).await;
    drop(writer);
    fulltext_handle.await.unwrap().unwrap();

    // Search for exact phrase
    let index = Index::open_in_dir(&index_path).unwrap();
    let reader = index.reader().unwrap();
    let searcher = reader.searcher();
    let schema = index.schema();
    let content_field = schema.get_field("content").unwrap();

    let query_parser = QueryParser::for_index(&index, vec![content_field]);

    // Phrase search: "quick brown fox"
    let query = query_parser.parse_query("\"quick brown fox\"").unwrap();
    let top_docs = searcher.search(&query, &TopDocs::with_limit(5)).unwrap();

    // Should find exactly 1 document with this phrase
    assert_eq!(
        top_docs.len(),
        1,
        "Phrase search should find exactly 1 document"
    );

    println!("✅ Phrase search test passed!");
}

#[tokio::test]
async fn test_fulltext_bm25_ranking() {
    // Demonstrate BM25 ranking by document relevance
    let temp_dir = TempDir::new().unwrap();
    let index_path = temp_dir.path().join("fulltext_index");

    let config = WriterConfig {
        channel_buffer_size: 10,
    };

    let (writer, receiver) = create_mutation_writer(config.clone());
    let fulltext_handle = spawn_fulltext_mutation_consumer(receiver, config, &index_path);

    // Add documents with varying term frequencies
    AddNodeFragment {
        id: Id::new(),
        ts_millis: TimestampMilli::now(),
        content: DataUrl::from_text(
            "Rust Rust Rust programming language for systems programming",
        ),
        valid_range: None,
    }
    .run(&writer)
    .await
    .unwrap();

    AddNodeFragment {
        id: Id::new(),
        ts_millis: TimestampMilli::now(),
        content: DataUrl::from_text("Rust is a programming language"),
        valid_range: None,
    }
    .run(&writer)
    .await
    .unwrap();

    AddNodeFragment {
        id: Id::new(),
        ts_millis: TimestampMilli::now(),
        content: DataUrl::from_text("Python is also a programming language"),
        valid_range: None,
    }
    .run(&writer)
    .await
    .unwrap();

    tokio::time::sleep(Duration::from_millis(200)).await;
    drop(writer);
    fulltext_handle.await.unwrap().unwrap();

    // Search for "Rust"
    let index = Index::open_in_dir(&index_path).unwrap();
    let reader = index.reader().unwrap();
    let searcher = reader.searcher();
    let schema = index.schema();
    let content_field = schema.get_field("content").unwrap();

    let query_parser = QueryParser::for_index(&index, vec![content_field]);
    let query = query_parser.parse_query("Rust").unwrap();
    let top_docs = searcher.search(&query, &TopDocs::with_limit(5)).unwrap();

    // Should find 2 documents with "Rust", ranked by BM25
    assert_eq!(
        top_docs.len(),
        2,
        "Should find 2 documents containing 'Rust'"
    );

    // The first document has higher term frequency, should rank higher
    let (first_score, _) = top_docs[0];
    let (second_score, _) = top_docs[1];
    assert!(
        first_score > second_score,
        "Document with higher term frequency should rank higher"
    );

    println!("✅ BM25 ranking test passed!");
}
