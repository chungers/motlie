//! Integration tests demonstrating the complete pipeline architecture:
//!
//! 1. Single mutation chain: Graph -> FullText
//! 2. Multiple query consumers sharing a single channel (MPMC pattern)
//! 3. Concurrent clients performing mixed graph and fulltext queries

use motlie_db::fulltext::{
    create_query_reader as create_fulltext_query_reader, Index as FulltextIndex,
    Nodes as FulltextNodes, ReaderConfig as FulltextReaderConfig,
    Runnable as FulltextQueryRunnable, spawn_mutation_consumer as spawn_fulltext_mutation_consumer,
    spawn_query_consumer_pool_shared as spawn_fulltext_query_consumer_pool_shared,
    Storage as FulltextStorage,
};
use motlie_db::graph::mutation::{AddEdge, AddNode, AddNodeFragment, Runnable as MutationRunnable};
use motlie_db::graph::query::{NodeById, OutgoingEdges, Runnable as QueryRunnable};
use motlie_db::graph::reader::{
    create_query_reader, spawn_query_consumer_pool_shared, ReaderConfig,
};
use motlie_db::graph::schema::{EdgeSummary, NodeSummary};
use motlie_db::graph::writer::{
    create_mutation_writer, spawn_mutation_consumer_with_next, WriterConfig,
};
use motlie_db::graph::{Graph, Storage};
use motlie_db::{DataUrl, Id, TimestampMilli};
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;
use tokio::sync::mpsc;

/// Test 1: Single pipeline chain - Graph -> FullText mutations
///
/// Demonstrates:
/// - Mutations flow through Graph consumer first
/// - Graph consumer forwards to FullText consumer via channel
/// - Both consumers process the same mutations in sequence
/// - Data is persisted in both RocksDB (graph) and Tantivy (fulltext)
#[tokio::test]
async fn test_single_mutation_pipeline_chain() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");
    let index_path = temp_dir.path().join("fulltext_index");

    let config = WriterConfig {
        channel_buffer_size: 100,
    };

    // Create the FullText consumer (end of chain)
    let (fulltext_sender, fulltext_receiver) = mpsc::channel(config.channel_buffer_size);
    let fulltext_handle =
        spawn_fulltext_mutation_consumer(fulltext_receiver, config.clone(), &index_path);

    // Create the Graph consumer that forwards to FullText (chained)
    let (writer, graph_receiver) = create_mutation_writer(config.clone());
    let graph_handle = spawn_mutation_consumer_with_next(
        graph_receiver,
        config.clone(),
        &db_path,
        fulltext_sender,
    );

    println!("=== Test 1: Single Pipeline Chain (Graph -> FullText) ===");

    // Create test data
    let node_ids: Vec<Id> = (0..5).map(|_| Id::new()).collect();
    let node_names = [
        "Alice",
        "Bob",
        "Charlie",
        "David",
        "Eve",
    ];
    let summaries = [
        "Software engineer specializing in Rust and systems programming",
        "Data scientist working on machine learning models",
        "DevOps engineer managing cloud infrastructure",
        "Product manager with technical background",
        "Security researcher focusing on cryptography",
    ];

    // Add nodes through the pipeline
    for (i, (node_id, (name, summary))) in node_ids
        .iter()
        .zip(node_names.iter().zip(summaries.iter()))
        .enumerate()
    {
        let node = AddNode {
            id: *node_id,
            ts_millis: TimestampMilli::now(),
            name: name.to_string(),
            valid_range: None,
            summary: NodeSummary::from_text(summary),
        };
        node.run(&writer).await.unwrap();
        println!("  Sent node {}: {} - {}", i, name, summary);
    }

    // Add some edges
    let edge = AddEdge {
        source_node_id: node_ids[0], // Alice
        target_node_id: node_ids[1], // Bob
        ts_millis: TimestampMilli::now(),
        name: "works_with".to_string(),
        summary: EdgeSummary::from_text("Alice and Bob collaborate on Rust projects"),
        weight: Some(0.9),
        valid_range: None,
    };
    edge.run(&writer).await.unwrap();
    println!("  Sent edge: Alice -> Bob (works_with)");

    // Add a node fragment with detailed content
    let fragment = AddNodeFragment {
        id: node_ids[0], // Alice
        ts_millis: TimestampMilli::now(),
        content: DataUrl::from_markdown(
            r#"# Alice's Profile

Alice is a senior software engineer with 10 years of experience.
She specializes in:
- Systems programming with Rust
- Database internals
- Distributed systems

Her recent projects include graph databases and fulltext search engines.
"#,
        ),
        valid_range: None,
    };
    fragment.run(&writer).await.unwrap();
    println!("  Sent fragment for Alice with detailed profile");

    // Allow time for processing
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Shutdown pipeline
    drop(writer);
    graph_handle.await.unwrap().unwrap();
    fulltext_handle.await.unwrap().unwrap();

    println!("  Pipeline shutdown complete");

    // Verify data in FullText index
    // Create readwrite backend for verification (follows graph::Graph pattern)
    let mut ft_storage = FulltextStorage::readwrite(&index_path);
    ft_storage.ready().unwrap();
    let backend = FulltextIndex::new(Arc::new(ft_storage));
    let reader = backend.tantivy_index().reader().unwrap();
    let searcher = reader.searcher();
    let total_docs = searcher.num_docs();
    println!("  FullText index contains {} documents", total_docs);

    assert!(
        total_docs >= 6, // 5 nodes + 1 edge + 1 fragment = 7 expected
        "Expected at least 6 documents in fulltext index, got {}",
        total_docs
    );

    println!("✅ Test 1 passed: Single pipeline chain works correctly\n");
}

/// Test 2: Single query channel with multiple consumers (MPMC pattern)
///
/// Demonstrates:
/// - One flume channel shared by multiple graph query consumers
/// - One flume channel shared by multiple fulltext query consumers
/// - Work distribution across consumers
#[tokio::test]
async fn test_multi_consumer_query_channels() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");
    let index_path = temp_dir.path().join("fulltext_index");

    let config = WriterConfig {
        channel_buffer_size: 100,
    };

    println!("=== Test 2: Multi-Consumer Query Channels ===");

    // First, populate the database
    let (fulltext_sender, fulltext_receiver) = mpsc::channel(config.channel_buffer_size);
    let fulltext_handle =
        spawn_fulltext_mutation_consumer(fulltext_receiver, config.clone(), &index_path);

    let (writer, graph_receiver) = create_mutation_writer(config.clone());
    let graph_handle =
        spawn_mutation_consumer_with_next(graph_receiver, config.clone(), &db_path, fulltext_sender);

    // Create test nodes
    let mut node_ids = Vec::new();
    for i in 0..10 {
        let node_id = Id::new();
        node_ids.push(node_id);

        let node = AddNode {
            id: node_id,
            ts_millis: TimestampMilli::now(),
            name: format!("TestNode_{}", i),
            valid_range: None,
            summary: NodeSummary::from_text(&format!(
                "This is test node {} with searchable content about topic {}",
                i,
                if i % 2 == 0 { "databases" } else { "networking" }
            )),
        };
        node.run(&writer).await.unwrap();
    }

    // Add edges between consecutive nodes
    for i in 0..9 {
        let edge = AddEdge {
            source_node_id: node_ids[i],
            target_node_id: node_ids[i + 1],
            ts_millis: TimestampMilli::now(),
            name: "connected_to".to_string(),
            summary: EdgeSummary::from_text("Sequential connection"),
            weight: Some(1.0),
            valid_range: None,
        };
        edge.run(&writer).await.unwrap();
    }

    tokio::time::sleep(Duration::from_millis(300)).await;

    drop(writer);
    graph_handle.await.unwrap().unwrap();
    fulltext_handle.await.unwrap().unwrap();

    println!("  Database populated with 10 nodes and 9 edges");

    // Now create query infrastructure with multiple consumers

    // Graph: Create shared storage and spawn 2 query consumers
    let mut storage = Storage::readwrite(&db_path);
    storage.ready().unwrap();
    let storage = Arc::new(storage);
    let graph = Arc::new(Graph::new(storage));

    let reader_config = ReaderConfig {
        channel_buffer_size: 100,
    };
    let (graph_reader, graph_query_receiver) = create_query_reader(reader_config.clone());

    // Spawn 2 graph query consumers sharing the same channel
    let graph_consumer_handles =
        spawn_query_consumer_pool_shared(graph_query_receiver, graph.clone(), 2);
    println!("  Spawned 2 graph query consumers");

    // FullText: Create 2 query consumers sharing the same channel
    // Using readonly Index allows multiple consumers to access the same index
    let fulltext_reader_config = FulltextReaderConfig {
        channel_buffer_size: 100,
    };
    let (fulltext_reader, fulltext_query_receiver) =
        create_fulltext_query_reader(fulltext_reader_config.clone());

    // Create a shared readonly Index (follows graph::Graph pattern)
    let mut ft_storage = FulltextStorage::readonly(&index_path);
    ft_storage.ready().unwrap();
    let fulltext_index = Arc::new(FulltextIndex::new(Arc::new(ft_storage)));

    // Spawn 2 fulltext query consumers sharing the same readonly Index
    let fulltext_consumer_handles = spawn_fulltext_query_consumer_pool_shared(
        fulltext_query_receiver,
        fulltext_index,
        2, // 2 fulltext query consumers
    );
    println!("  Spawned 2 fulltext query consumers (readonly mode)");

    // Send multiple queries - they should be distributed across consumers
    let timeout = Duration::from_secs(5);

    // Graph queries
    let mut graph_results = Vec::new();
    for node_id in &node_ids[0..5] {
        let query = NodeById::new(*node_id, None);
        let result = query.run(&graph_reader, timeout).await;
        graph_results.push(result.is_ok());
    }

    let graph_success_count = graph_results.iter().filter(|&&x| x).count();
    println!(
        "  Graph queries: {}/{} successful",
        graph_success_count,
        graph_results.len()
    );

    // FullText queries
    let mut fulltext_results = Vec::new();
    for query_term in &["databases", "networking", "test", "node", "content"] {
        let query = FulltextNodes::new(query_term.to_string(), 10);
        let result = query.run(&fulltext_reader, timeout).await;
        fulltext_results.push(result.is_ok());
    }

    let fulltext_success_count = fulltext_results.iter().filter(|&&x| x).count();
    println!(
        "  FullText queries: {}/{} successful",
        fulltext_success_count,
        fulltext_results.len()
    );

    // Shutdown
    drop(graph_reader);
    drop(fulltext_reader);

    for handle in graph_consumer_handles {
        handle.await.unwrap();
    }
    for handle in fulltext_consumer_handles {
        handle.await.unwrap();
    }

    assert!(
        graph_success_count >= 4,
        "Expected at least 4/5 graph queries to succeed"
    );
    assert!(
        fulltext_success_count >= 4,
        "Expected at least 4/5 fulltext queries to succeed"
    );

    println!("✅ Test 2 passed: Multi-consumer query channels work correctly\n");
}

/// Test 3: Concurrent clients with mixed queries
///
/// Demonstrates:
/// - Multiple client tasks running concurrently
/// - Some clients query graph, some query fulltext
/// - Queries are processed in parallel across consumers
#[tokio::test]
async fn test_concurrent_mixed_queries() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");
    let index_path = temp_dir.path().join("fulltext_index");

    let config = WriterConfig {
        channel_buffer_size: 100,
    };

    println!("=== Test 3: Concurrent Mixed Queries ===");

    // Populate database
    let (fulltext_sender, fulltext_receiver) = mpsc::channel(config.channel_buffer_size);
    let fulltext_handle =
        spawn_fulltext_mutation_consumer(fulltext_receiver, config.clone(), &index_path);

    let (writer, graph_receiver) = create_mutation_writer(config.clone());
    let graph_handle =
        spawn_mutation_consumer_with_next(graph_receiver, config.clone(), &db_path, fulltext_sender);

    // Create a more realistic dataset
    let categories = ["engineering", "science", "business", "design"];
    let mut all_node_ids = Vec::new();

    for (cat_idx, category) in categories.iter().enumerate() {
        for i in 0..5 {
            let node_id = Id::new();
            all_node_ids.push((node_id, category.to_string()));

            let node = AddNode {
                id: node_id,
                ts_millis: TimestampMilli::now(),
                name: format!("{}_{}", category, i),
                valid_range: None,
                summary: NodeSummary::from_text(&format!(
                    "Expert in {} with focus on area {}. Keywords: {}, professional, skilled.",
                    category, i, category
                )),
            };
            node.run(&writer).await.unwrap();
        }

        // Add edges within category
        let cat_nodes: Vec<_> = all_node_ids
            .iter()
            .filter(|(_, c)| c == *category)
            .map(|(id, _)| *id)
            .collect();

        for i in 0..cat_nodes.len().saturating_sub(1) {
            let edge = AddEdge {
                source_node_id: cat_nodes[i],
                target_node_id: cat_nodes[i + 1],
                ts_millis: TimestampMilli::now(),
                name: format!("{}_link", category),
                summary: EdgeSummary::from_text(&format!("{} collaboration", category)),
                weight: Some(0.8),
                valid_range: None,
            };
            edge.run(&writer).await.unwrap();
        }
    }

    tokio::time::sleep(Duration::from_millis(500)).await;
    drop(writer);
    graph_handle.await.unwrap().unwrap();
    fulltext_handle.await.unwrap().unwrap();

    println!("  Created {} nodes across {} categories", all_node_ids.len(), categories.len());

    // Setup query infrastructure
    let mut storage = Storage::readwrite(&db_path);
    storage.ready().unwrap();
    let storage = Arc::new(storage);
    let graph = Arc::new(Graph::new(storage));

    let reader_config = ReaderConfig {
        channel_buffer_size: 100,
    };
    let (graph_reader, graph_query_receiver) = create_query_reader(reader_config.clone());
    let graph_consumer_handles =
        spawn_query_consumer_pool_shared(graph_query_receiver, graph.clone(), 2);

    let fulltext_reader_config = FulltextReaderConfig {
        channel_buffer_size: 100,
    };
    let (fulltext_reader, fulltext_query_receiver) =
        create_fulltext_query_reader(fulltext_reader_config.clone());

    // Create a shared readonly Index for fulltext queries (follows graph::Graph pattern)
    let mut ft_storage = FulltextStorage::readonly(&index_path);
    ft_storage.ready().unwrap();
    let fulltext_index = Arc::new(FulltextIndex::new(Arc::new(ft_storage)));

    // Spawn 2 fulltext query consumers sharing the readonly Index
    let ft_consumer_handles = spawn_fulltext_query_consumer_pool_shared(
        fulltext_query_receiver,
        fulltext_index,
        2,
    );

    // Clone readers for concurrent clients
    let graph_reader_1 = graph_reader.clone();
    let graph_reader_2 = graph_reader.clone();
    let fulltext_reader_1 = fulltext_reader.clone();
    let fulltext_reader_2 = fulltext_reader.clone();

    let timeout = Duration::from_secs(5);

    // Client 1: Graph queries by node ID
    let node_ids_clone: Vec<_> = all_node_ids.iter().map(|(id, _)| *id).collect();
    let client1 = tokio::spawn(async move {
        let mut successes = 0;
        for node_id in &node_ids_clone[0..10] {
            let query = NodeById::new(*node_id, None);
            if query.run(&graph_reader_1, timeout).await.is_ok() {
                successes += 1;
            }
        }
        println!("  Client 1 (Graph NodeById): {}/10 successful", successes);
        successes
    });

    // Client 2: More Graph queries by node ID
    // (NodesByName was removed - name-based search is now via fulltext)
    let node_ids_clone_2: Vec<_> = all_node_ids.iter().map(|(id, _)| *id).collect();
    let client2 = tokio::spawn(async move {
        let mut successes = 0;
        for node_id in &node_ids_clone_2[10..15] {
            let query = NodeById::new(*node_id, None);
            if query.run(&graph_reader_2, timeout).await.is_ok() {
                successes += 1;
            }
        }
        println!("  Client 2 (Graph NodeById): {}/5 successful", successes);
        successes
    });

    // Client 3: FullText search queries
    let client3 = tokio::spawn(async move {
        let mut successes = 0;
        for term in &["engineering", "science", "professional", "expert"] {
            let query = FulltextNodes::new(term.to_string(), 5);
            if let Ok(results) = query.run(&fulltext_reader_1, timeout).await {
                if !results.is_empty() {
                    successes += 1;
                }
            }
        }
        println!("  Client 3 (FullText search): {}/4 with results", successes);
        successes
    });

    // Client 4: More FullText queries
    let client4 = tokio::spawn(async move {
        let mut successes = 0;
        for term in &["business", "design", "skilled", "keywords"] {
            let query = FulltextNodes::new(term.to_string(), 10);
            if let Ok(results) = query.run(&fulltext_reader_2, timeout).await {
                if !results.is_empty() {
                    successes += 1;
                }
            }
        }
        println!("  Client 4 (FullText search): {}/4 with results", successes);
        successes
    });

    // Wait for all clients to complete
    let (r1, r2, r3, r4) = tokio::join!(client1, client2, client3, client4);
    let total_successes = r1.unwrap() + r2.unwrap() + r3.unwrap() + r4.unwrap();

    // Shutdown
    drop(graph_reader);
    drop(fulltext_reader);

    for handle in graph_consumer_handles {
        handle.await.unwrap();
    }
    for handle in ft_consumer_handles {
        handle.await.unwrap();
    }

    println!(
        "  Total successful operations: {}/21",
        total_successes
    );

    assert!(
        total_successes >= 15,
        "Expected at least 15/21 operations to succeed, got {}",
        total_successes
    );

    println!("✅ Test 3 passed: Concurrent mixed queries work correctly\n");
}

/// Combined test demonstrating all three patterns together
#[tokio::test]
async fn test_complete_pipeline_architecture() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");
    let index_path = temp_dir.path().join("fulltext_index");

    println!("=== Complete Pipeline Architecture Test ===");
    println!("This test demonstrates:");
    println!("  1. Graph -> FullText mutation chain");
    println!("  2. Multiple query consumers (2 graph + 2 fulltext)");
    println!("  3. Concurrent clients (2 graph + 2 fulltext)\n");

    let config = WriterConfig {
        channel_buffer_size: 100,
    };

    // === SETUP: Mutation Pipeline ===
    let (fulltext_sender, fulltext_receiver) = mpsc::channel(config.channel_buffer_size);
    let fulltext_mut_handle =
        spawn_fulltext_mutation_consumer(fulltext_receiver, config.clone(), &index_path);

    let (writer, graph_receiver) = create_mutation_writer(config.clone());
    let graph_mut_handle =
        spawn_mutation_consumer_with_next(graph_receiver, config.clone(), &db_path, fulltext_sender);

    // Populate with test data
    let mut node_ids = Vec::new();
    for i in 0..20 {
        let node_id = Id::new();
        node_ids.push(node_id);

        let node = AddNode {
            id: node_id,
            ts_millis: TimestampMilli::now(),
            name: format!("Entity_{:02}", i),
            valid_range: None,
            summary: NodeSummary::from_text(&format!(
                "Entity {} description with searchable terms: alpha beta gamma",
                i
            )),
        };
        node.run(&writer).await.unwrap();
    }

    // Add some edges
    for i in 0..15 {
        let edge = AddEdge {
            source_node_id: node_ids[i],
            target_node_id: node_ids[i + 1],
            ts_millis: TimestampMilli::now(),
            name: "relates_to".to_string(),
            summary: EdgeSummary::from_text("Relationship description"),
            weight: Some(0.5 + (i as f64 * 0.03)),
            valid_range: None,
        };
        edge.run(&writer).await.unwrap();
    }

    tokio::time::sleep(Duration::from_millis(500)).await;
    drop(writer);
    graph_mut_handle.await.unwrap().unwrap();
    fulltext_mut_handle.await.unwrap().unwrap();

    println!("  [Mutations] 20 nodes + 15 edges written through pipeline");

    // === SETUP: Query Infrastructure ===

    // Graph query consumers (2)
    let mut storage = Storage::readwrite(&db_path);
    storage.ready().unwrap();
    let graph = Arc::new(Graph::new(Arc::new(storage)));

    let (graph_reader, graph_query_receiver) = create_query_reader(ReaderConfig {
        channel_buffer_size: 100,
    });
    let graph_query_handles = spawn_query_consumer_pool_shared(graph_query_receiver, graph, 2);

    // FullText query consumers (2 - using readonly Index)
    let (fulltext_reader, fulltext_query_receiver) =
        create_fulltext_query_reader(FulltextReaderConfig {
            channel_buffer_size: 100,
        });

    // Create shared readonly Index for fulltext queries (follows graph::Graph pattern)
    let mut ft_storage = FulltextStorage::readonly(&index_path);
    ft_storage.ready().unwrap();
    let fulltext_index = Arc::new(FulltextIndex::new(Arc::new(ft_storage)));

    // Spawn 2 fulltext query consumers sharing the readonly Index
    let ft_query_handles = spawn_fulltext_query_consumer_pool_shared(
        fulltext_query_receiver,
        fulltext_index,
        2, // 2 fulltext query consumers
    );

    println!("  [Query Consumers] 2 graph + 2 fulltext consumers ready");

    // === CONCURRENT QUERIES ===
    let timeout = Duration::from_secs(5);

    let gr1 = graph_reader.clone();
    let gr2 = graph_reader.clone();
    let fr1 = fulltext_reader.clone();
    let fr2 = fulltext_reader.clone();

    let node_ids_clone = node_ids.clone();

    // Graph client 1: NodeById queries
    let graph_client_1 = tokio::spawn(async move {
        let mut count = 0;
        for id in &node_ids_clone[0..8] {
            if NodeById::new(*id, None).run(&gr1, timeout).await.is_ok() {
                count += 1;
            }
        }
        count
    });

    // Graph client 2: OutgoingEdges queries
    let node_ids_clone2 = node_ids.clone();
    let graph_client_2 = tokio::spawn(async move {
        let mut count = 0;
        for id in &node_ids_clone2[0..5] {
            if OutgoingEdges::new(*id, None).run(&gr2, timeout).await.is_ok() {
                count += 1;
            }
        }
        count
    });

    // FullText client 1
    let fulltext_client_1 = tokio::spawn(async move {
        let mut count = 0;
        for term in &["alpha", "beta", "gamma", "Entity"] {
            if let Ok(r) = FulltextNodes::new(term.to_string(), 5).run(&fr1, timeout).await {
                if !r.is_empty() {
                    count += 1;
                }
            }
        }
        count
    });

    // FullText client 2
    let fulltext_client_2 = tokio::spawn(async move {
        let mut count = 0;
        for term in &["description", "searchable", "terms"] {
            if let Ok(r) = FulltextNodes::new(term.to_string(), 10).run(&fr2, timeout).await {
                if !r.is_empty() {
                    count += 1;
                }
            }
        }
        count
    });

    let (g1, g2, f1, f2) = tokio::join!(
        graph_client_1,
        graph_client_2,
        fulltext_client_1,
        fulltext_client_2
    );

    let graph_total = g1.unwrap() + g2.unwrap();
    let fulltext_total = f1.unwrap() + f2.unwrap();

    println!("  [Results] Graph queries: {}/13 successful", graph_total);
    println!("  [Results] FullText queries: {}/7 with results", fulltext_total);

    // Cleanup
    drop(graph_reader);
    drop(fulltext_reader);

    for h in graph_query_handles {
        h.await.unwrap();
    }
    for h in ft_query_handles {
        h.await.unwrap();
    }

    assert!(graph_total >= 10, "Expected at least 10/13 graph queries to succeed");
    assert!(fulltext_total >= 5, "Expected at least 5/7 fulltext queries with results");

    println!("\n✅ Complete pipeline architecture test passed!");
    println!("   - Mutation chain: Graph -> FullText ✓");
    println!("   - Multi-consumer query channels (2 graph + 2 fulltext) ✓");
    println!("   - Concurrent mixed queries ✓");
}
