//! End-to-end integration tests for the Version Playback API (HEAD~N).
//!
//! Tests the full async pipeline: Writer (mutations) → Reader (queries)
//! using the public `Runnable` trait for both.

use motlie_db::graph::writer::{spawn_mutation_consumer_with_storage, WriterConfig};
use motlie_db::graph::reader::{spawn_query_consumers_with_storage, ReaderConfig};
use motlie_db::graph::mutation::{AddEdge, AddNode, UpdateEdge, UpdateNode};
use motlie_db::graph::query::{
    EdgeAtVersion, EdgeVersions, NodeAtVersion, NodeVersions,
};
use motlie_db::graph::schema::{EdgeSummary, NodeSummary};
use motlie_db::graph::Storage;
use motlie_db::reader::Runnable as QueryRunnable;
use motlie_db::writer::Runnable as MutRunnable;
use motlie_db::{Id, TimestampMilli};
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;

fn setup_storage(db_path: &std::path::Path) -> Arc<Storage> {
    let mut storage = Storage::readwrite(db_path);
    storage.ready().unwrap();
    Arc::new(storage)
}

/// Test node version playback through the full async Writer → Reader pipeline.
///
/// Creates 3 versions of a node, then queries each version offset (HEAD, HEAD~1, HEAD~2).
#[tokio::test]
async fn test_node_version_playback_e2e() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");
    let storage = setup_storage(&db_path);

    let config = WriterConfig { channel_buffer_size: 100 };

    // === Mutation phase ===
    let (writer, writer_handle) = spawn_mutation_consumer_with_storage(storage.clone(), config);

    let node_id = Id::new();

    // v1
    AddNode {
        id: node_id,
        ts_millis: TimestampMilli::now(),
        name: "playback_node".to_string(),
        valid_range: None,
        summary: NodeSummary::from_text("Version 1"),
    }
    .run(&writer)
    .await
    .unwrap();
    writer.flush().await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;

    // v2
    UpdateNode {
        id: node_id,
        expected_version: 1,
        new_active_period: None,
        new_summary: Some(NodeSummary::from_text("Version 2")),
    }
    .run(&writer)
    .await
    .unwrap();
    writer.flush().await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;

    // v3
    UpdateNode {
        id: node_id,
        expected_version: 2,
        new_active_period: None,
        new_summary: Some(NodeSummary::from_text("Version 3")),
    }
    .run(&writer)
    .await
    .unwrap();
    writer.flush().await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Shutdown writer
    drop(writer);
    writer_handle.await.unwrap().unwrap();

    // === Query phase ===
    let (reader, reader_handles) = spawn_query_consumers_with_storage(
        storage,
        ReaderConfig::default(),
        2,
    );
    let timeout = Duration::from_secs(5);

    // HEAD (versions_back=0) → v3
    let snap = NodeAtVersion::new(node_id, 0)
        .run(&reader, timeout)
        .await
        .unwrap();
    assert_eq!(snap.version, 3);
    assert_eq!(snap.payload.1, NodeSummary::from_text("Version 3"));

    // HEAD~1 → v2
    let snap = NodeAtVersion::new(node_id, 1)
        .run(&reader, timeout)
        .await
        .unwrap();
    assert_eq!(snap.version, 2);
    assert_eq!(snap.payload.1, NodeSummary::from_text("Version 2"));

    // HEAD~2 → v1
    let snap = NodeAtVersion::new(node_id, 2)
        .run(&reader, timeout)
        .await
        .unwrap();
    assert_eq!(snap.version, 1);
    assert_eq!(snap.payload.1, NodeSummary::from_text("Version 1"));

    // HEAD~3 → error
    let result = NodeAtVersion::new(node_id, 3)
        .run(&reader, timeout)
        .await;
    assert!(result.is_err(), "HEAD~3 should fail for 3-version node");

    // Shutdown reader
    drop(reader);
    for h in reader_handles {
        h.await.ok();
    }
}

/// Test edge version playback through the full async pipeline.
#[tokio::test]
async fn test_edge_version_playback_e2e() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");
    let storage = setup_storage(&db_path);

    let config = WriterConfig { channel_buffer_size: 100 };
    let (writer, writer_handle) = spawn_mutation_consumer_with_storage(storage.clone(), config);

    let src_id = Id::new();
    let dst_id = Id::new();
    let edge_name = "relates_to".to_string();

    // v1
    AddEdge {
        source_node_id: src_id,
        target_node_id: dst_id,
        ts_millis: TimestampMilli::now(),
        name: edge_name.clone(),
        summary: EdgeSummary::from_text("Edge V1"),
        weight: Some(1.0),
        valid_range: None,
    }
    .run(&writer)
    .await
    .unwrap();
    writer.flush().await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;

    // v2
    UpdateEdge {
        src_id,
        dst_id,
        name: edge_name.clone(),
        expected_version: 1,
        new_weight: Some(Some(9.0)),
        new_active_period: None,
        new_summary: Some(EdgeSummary::from_text("Edge V2")),
    }
    .run(&writer)
    .await
    .unwrap();
    writer.flush().await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;

    drop(writer);
    writer_handle.await.unwrap().unwrap();

    // === Query phase ===
    let (reader, reader_handles) = spawn_query_consumers_with_storage(
        storage,
        ReaderConfig::default(),
        2,
    );
    let timeout = Duration::from_secs(5);

    // HEAD → v2
    let snap = EdgeAtVersion::new(src_id, dst_id, edge_name.clone(), 0)
        .run(&reader, timeout)
        .await
        .unwrap();
    assert_eq!(snap.version, 2);
    assert_eq!(snap.payload.0, EdgeSummary::from_text("Edge V2"));
    assert_eq!(snap.payload.1, Some(9.0));

    // HEAD~1 → v1
    let snap = EdgeAtVersion::new(src_id, dst_id, edge_name.clone(), 1)
        .run(&reader, timeout)
        .await
        .unwrap();
    assert_eq!(snap.version, 1);
    assert_eq!(snap.payload.0, EdgeSummary::from_text("Edge V1"));
    assert_eq!(snap.payload.1, Some(1.0));

    // HEAD~2 → error
    let result = EdgeAtVersion::new(src_id, dst_id, edge_name, 2)
        .run(&reader, timeout)
        .await;
    assert!(result.is_err(), "HEAD~2 should fail for 2-version edge");

    drop(reader);
    for h in reader_handles {
        h.await.ok();
    }
}

/// Test listing node version metadata through the async pipeline.
#[tokio::test]
async fn test_node_versions_listing_e2e() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");
    let storage = setup_storage(&db_path);

    let config = WriterConfig { channel_buffer_size: 100 };
    let (writer, writer_handle) = spawn_mutation_consumer_with_storage(storage.clone(), config);

    let node_id = Id::new();

    // Create 3 versions
    AddNode {
        id: node_id,
        ts_millis: TimestampMilli::now(),
        name: "list_test".to_string(),
        valid_range: None,
        summary: NodeSummary::from_text("V1"),
    }
    .run(&writer)
    .await
    .unwrap();
    writer.flush().await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;

    UpdateNode {
        id: node_id,
        expected_version: 1,
        new_active_period: None,
        new_summary: Some(NodeSummary::from_text("V2")),
    }
    .run(&writer)
    .await
    .unwrap();
    writer.flush().await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;

    UpdateNode {
        id: node_id,
        expected_version: 2,
        new_active_period: None,
        new_summary: Some(NodeSummary::from_text("V3")),
    }
    .run(&writer)
    .await
    .unwrap();
    writer.flush().await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;

    drop(writer);
    writer_handle.await.unwrap().unwrap();

    // === Query phase ===
    let (reader, reader_handles) = spawn_query_consumers_with_storage(
        storage,
        ReaderConfig::default(),
        2,
    );
    let timeout = Duration::from_secs(5);

    // List all versions
    let versions = NodeVersions::new(node_id, 10)
        .run(&reader, timeout)
        .await
        .unwrap();
    assert_eq!(versions.len(), 3, "Should list 3 versions");
    assert_eq!(versions[0].version, 3, "Newest first");
    assert_eq!(versions[1].version, 2);
    assert_eq!(versions[2].version, 1, "Oldest last");
    for v in &versions {
        assert!(v.summary_available);
    }

    // Pagination: limit=1
    let page = NodeVersions::new(node_id, 1)
        .run(&reader, timeout)
        .await
        .unwrap();
    assert_eq!(page.len(), 1);
    assert_eq!(page[0].version, 3, "Should return only HEAD");

    drop(reader);
    for h in reader_handles {
        h.await.ok();
    }
}

/// Test listing edge version metadata through the async pipeline.
#[tokio::test]
async fn test_edge_versions_listing_e2e() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");
    let storage = setup_storage(&db_path);

    let config = WriterConfig { channel_buffer_size: 100 };
    let (writer, writer_handle) = spawn_mutation_consumer_with_storage(storage.clone(), config);

    let src_id = Id::new();
    let dst_id = Id::new();
    let edge_name = "versioned_edge".to_string();

    AddEdge {
        source_node_id: src_id,
        target_node_id: dst_id,
        ts_millis: TimestampMilli::now(),
        name: edge_name.clone(),
        summary: EdgeSummary::from_text("EV1"),
        weight: Some(1.0),
        valid_range: None,
    }
    .run(&writer)
    .await
    .unwrap();
    writer.flush().await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;

    UpdateEdge {
        src_id,
        dst_id,
        name: edge_name.clone(),
        expected_version: 1,
        new_weight: Some(Some(2.0)),
        new_active_period: None,
        new_summary: Some(EdgeSummary::from_text("EV2")),
    }
    .run(&writer)
    .await
    .unwrap();
    writer.flush().await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;

    drop(writer);
    writer_handle.await.unwrap().unwrap();

    // === Query phase ===
    let (reader, reader_handles) = spawn_query_consumers_with_storage(
        storage,
        ReaderConfig::default(),
        2,
    );
    let timeout = Duration::from_secs(5);

    let versions = EdgeVersions::new(src_id, dst_id, edge_name, 10)
        .run(&reader, timeout)
        .await
        .unwrap();
    assert_eq!(versions.len(), 2, "Should list 2 versions");
    assert_eq!(versions[0].version, 2, "Newest first");
    assert_eq!(versions[1].version, 1, "Oldest last");
    assert!(versions[0].valid_since.0 >= versions[1].valid_since.0);

    drop(reader);
    for h in reader_handles {
        h.await.ok();
    }
}
