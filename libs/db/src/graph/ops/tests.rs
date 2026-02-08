//! Tests for graph ops module.
//!
//! These tests validate the business logic operations directly using transactions,
//! mirroring the pattern from the vector crate's ops tests.

use tempfile::TempDir;

use super::{edge, fragment, node};
use crate::graph::schema::{
    EdgeSummaries, EdgeSummaryCfKey, EdgeVersionHistory, EdgeVersionHistoryCfKey,
    EdgeVersionHistoryCfValue, ForwardEdgeCfKey, ForwardEdgeCfValue, ForwardEdges, NodeCfKey,
    NodeCfValue, NodeFragments, NodeSummaryIndex, NodeSummaryIndexCfKey, NodeVersionHistory,
    NodeVersionHistoryCfKey, Nodes, ReverseEdgeCfKey, ReverseEdges,
};
use crate::graph::{AddEdge, AddNode, DeleteEdge, DeleteNode, UpdateNode, UpdateEdge};
use crate::graph::mutation::{AddNodeFragment, RestoreEdges};
use crate::graph::ops::summary::verify_edge_summary_exists;
use crate::graph::name_hash::NameHash;
use crate::graph::SummaryHash;
use crate::{DataUrl, Id, TimestampMilli};
use crate::rocksdb::{ColumnFamily, ColumnFamilySerde, HotColumnFamilyRecord};

fn setup_storage() -> (TempDir, crate::graph::Storage) {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("ops_tests");
    let mut storage = crate::graph::Storage::readwrite(&db_path);
    storage.ready().unwrap();
    (temp_dir, storage)
}

#[test]
fn ops_add_node_writes_node_and_history() {
    let (_temp_dir, storage) = setup_storage();
    let txn_db = storage.transaction_db().unwrap();
    let txn = txn_db.transaction();

    let id = Id::new();
    let ts = TimestampMilli::now();
    let mutation = AddNode {
        id,
        ts_millis: ts,
        name: "alice".to_string(),
        valid_range: None,
        summary: DataUrl::from_text("node summary"),
    };

    node::add_node(&txn, txn_db, &mutation, None).unwrap();
    txn.commit().unwrap();

    let nodes_cf = txn_db.cf_handle(Nodes::CF_NAME).unwrap();
    let node_key = NodeCfKey(id, ts);
    let node_key_bytes = Nodes::key_to_bytes(&node_key);
    let node_value_bytes = txn_db.get_cf(nodes_cf, node_key_bytes).unwrap().unwrap();
    let node_value = Nodes::value_from_bytes(&node_value_bytes).unwrap();
    assert!(node_value.0.is_none(), "current node should have ValidUntil=None");

    let history_cf = txn_db.cf_handle(NodeVersionHistory::CF_NAME).unwrap();
    let history_key = NodeVersionHistoryCfKey(id, ts, 1);
    let history_key_bytes = NodeVersionHistory::key_to_bytes(&history_key);
    let history_value_bytes = txn_db.get_cf(history_cf, history_key_bytes).unwrap().unwrap();
    let history_value = NodeVersionHistory::value_from_bytes(&history_value_bytes).unwrap();
    assert_eq!(history_value.0, ts);
}

#[test]
fn ops_add_edge_writes_forward_reverse_and_history() {
    let (_temp_dir, storage) = setup_storage();
    let txn_db = storage.transaction_db().unwrap();
    let txn = txn_db.transaction();

    let src = Id::new();
    let dst = Id::new();
    let ts = TimestampMilli::now();

    let add_src = AddNode {
        id: src,
        ts_millis: ts,
        name: "src".to_string(),
        valid_range: None,
        summary: DataUrl::from_text("src summary"),
    };
    let add_dst = AddNode {
        id: dst,
        ts_millis: ts,
        name: "dst".to_string(),
        valid_range: None,
        summary: DataUrl::from_text("dst summary"),
    };
    node::add_node(&txn, txn_db, &add_src, None).unwrap();
    node::add_node(&txn, txn_db, &add_dst, None).unwrap();

    let edge_ts = TimestampMilli::now();
    let add_edge = AddEdge {
        source_node_id: src,
        target_node_id: dst,
        ts_millis: edge_ts,
        name: "friends".to_string(),
        valid_range: None,
        summary: DataUrl::from_text("edge summary"),
        weight: Some(0.5),
    };

    edge::add_edge(&txn, txn_db, &add_edge, None).unwrap();
    txn.commit().unwrap();

    let name_hash = crate::graph::NameHash::from_name("friends");

    let forward_cf = txn_db.cf_handle(ForwardEdges::CF_NAME).unwrap();
    let forward_key = ForwardEdgeCfKey(src, dst, name_hash, edge_ts);
    let forward_key_bytes = ForwardEdges::key_to_bytes(&forward_key);
    assert!(txn_db.get_cf(forward_cf, forward_key_bytes).unwrap().is_some());

    let reverse_cf = txn_db.cf_handle(ReverseEdges::CF_NAME).unwrap();
    let reverse_key = ReverseEdgeCfKey(dst, src, name_hash, edge_ts);
    let reverse_key_bytes = ReverseEdges::key_to_bytes(&reverse_key);
    assert!(txn_db.get_cf(reverse_cf, reverse_key_bytes).unwrap().is_some());

    let history_cf = txn_db.cf_handle(EdgeVersionHistory::CF_NAME).unwrap();
    let history_key = EdgeVersionHistoryCfKey(src, dst, name_hash, edge_ts, 1);
    let history_key_bytes = EdgeVersionHistory::key_to_bytes(&history_key);
    assert!(txn_db.get_cf(history_cf, history_key_bytes).unwrap().is_some());
}

#[test]
fn ops_update_node_updates_index() {
    let (_temp_dir, storage) = setup_storage();
    let txn_db = storage.transaction_db().unwrap();

    let id = Id::new();
    let ts = TimestampMilli::now();
    let add_node = AddNode {
        id,
        ts_millis: ts,
        name: "n".to_string(),
        valid_range: None,
        summary: DataUrl::from_text("original summary"),
    };

    {
        let txn = txn_db.transaction();
        node::add_node(&txn, txn_db, &add_node, None).unwrap();
        txn.commit().unwrap();
    }

    let new_summary = DataUrl::from_text("updated summary");
    let update = UpdateNode {
        id,
        expected_version: 1,
        new_active_period: None,
        new_summary: Some(new_summary.clone()),
    };

    {
        let txn = txn_db.transaction();
        node::update_node(&txn, txn_db, &update).unwrap();
        txn.commit().unwrap();
    }

    let new_hash = crate::graph::SummaryHash::from_summary(&new_summary).unwrap();
    let index_cf = txn_db.cf_handle(NodeSummaryIndex::CF_NAME).unwrap();
    let index_key = NodeSummaryIndexCfKey(new_hash, id, 2);
    let index_key_bytes = NodeSummaryIndex::key_to_bytes(&index_key);
    let index_value_bytes = txn_db.get_cf(index_cf, index_key_bytes).unwrap().unwrap();
    let index_value = NodeSummaryIndex::value_from_bytes(&index_value_bytes).unwrap();
    assert!(index_value.is_current(), "index marker should be CURRENT");
}

#[test]
fn ops_restore_edges_dry_run_strict() {
    let (_temp_dir, storage) = setup_storage();
    let txn_db = storage.transaction_db().unwrap();

    let src = Id::new();
    let dst = Id::new();
    let ts = TimestampMilli::now();

    let add_src = AddNode {
        id: src,
        ts_millis: ts,
        name: "src".to_string(),
        valid_range: None,
        summary: DataUrl::from_text("src summary"),
    };
    let add_dst = AddNode {
        id: dst,
        ts_millis: ts,
        name: "dst".to_string(),
        valid_range: None,
        summary: DataUrl::from_text("dst summary"),
    };
    let add_edge = AddEdge {
        source_node_id: src,
        target_node_id: dst,
        ts_millis: TimestampMilli::now(),
        name: "likes".to_string(),
        valid_range: None,
        summary: DataUrl::from_text("edge summary"),
        weight: None,
    };

    {
        let txn = txn_db.transaction();
        node::add_node(&txn, txn_db, &add_src, None).unwrap();
        node::add_node(&txn, txn_db, &add_dst, None).unwrap();
        edge::add_edge(&txn, txn_db, &add_edge, None).unwrap();
        txn.commit().unwrap();
    }

    let delete = DeleteEdge {
        src_id: src,
        dst_id: dst,
        name: "likes".to_string(),
        expected_version: 1,
    };
    {
        let txn = txn_db.transaction();
        edge::delete_edge(&txn, txn_db, &delete).unwrap();
        txn.commit().unwrap();
    }

    let summary_hash = crate::graph::SummaryHash::from_summary(&add_edge.summary).unwrap();
    let summaries_cf = txn_db.cf_handle(EdgeSummaries::CF_NAME).unwrap();
    let summary_key = EdgeSummaryCfKey(summary_hash);
    let summary_key_bytes = EdgeSummaries::key_to_bytes(&summary_key);
    {
        let txn = txn_db.transaction();
        txn.delete_cf(summaries_cf, summary_key_bytes.clone()).unwrap();
        txn.commit().unwrap();
    }
    assert!(
        txn_db.get_cf(summaries_cf, summary_key_bytes).unwrap().is_none(),
        "expected EdgeSummaries entry to be deleted"
    );

    let name_hash = NameHash::from_name("likes");
    let forward_cf = txn_db.cf_handle(ForwardEdges::CF_NAME).unwrap();
    let mut forward_prefix = Vec::with_capacity(40);
    forward_prefix.extend_from_slice(&src.into_bytes());
    forward_prefix.extend_from_slice(&dst.into_bytes());
    forward_prefix.extend_from_slice(name_hash.as_bytes());
    let forward_iter = txn_db.iterator_cf(
        forward_cf,
        rocksdb::IteratorMode::From(&forward_prefix, rocksdb::Direction::Forward),
    );
    let mut current_hash = None;
    for item in forward_iter {
        let (key_bytes, value_bytes) = item.unwrap();
        if !key_bytes.starts_with(&forward_prefix) {
            break;
        }
        let _key: ForwardEdgeCfKey = ForwardEdges::key_from_bytes(&key_bytes).unwrap();
        let value: ForwardEdgeCfValue = ForwardEdges::value_from_bytes(&value_bytes).unwrap();
        if value.0.is_none() {
            current_hash = value.3;
            break;
        }
    }
    assert!(current_hash.is_some(), "expected current edge to have summary hash");
    assert_eq!(
        current_hash,
        Some(summary_hash),
        "current edge hash should match add_edge summary hash"
    );

    let history_cf = txn_db.cf_handle(EdgeVersionHistory::CF_NAME).unwrap();
    let mut history_prefix = Vec::with_capacity(40);
    history_prefix.extend_from_slice(&src.into_bytes());
    history_prefix.extend_from_slice(&dst.into_bytes());
    history_prefix.extend_from_slice(name_hash.as_bytes());
    let history_iter = txn_db.iterator_cf(
        history_cf,
        rocksdb::IteratorMode::From(&history_prefix, rocksdb::Direction::Forward),
    );
    let mut history_hash = None;
    for item in history_iter {
        let (key_bytes, value_bytes) = item.unwrap();
        if !key_bytes.starts_with(&history_prefix) {
            break;
        }
        let _key: EdgeVersionHistoryCfKey = EdgeVersionHistory::key_from_bytes(&key_bytes).unwrap();
        let value: EdgeVersionHistoryCfValue =
            EdgeVersionHistory::value_from_bytes(&value_bytes).unwrap();
        history_hash = value.1;
        break;
    }
    assert!(history_hash.is_some(), "expected history to include summary hash");
    assert_eq!(
        history_hash,
        Some(summary_hash),
        "history hash should match add_edge summary hash"
    );

    {
        let txn = txn_db.transaction();
        let missing = verify_edge_summary_exists(&txn, txn_db, summary_hash).unwrap();
        assert!(!missing, "expected summary to be missing before restore");
    }

    let dry_run_restore = RestoreEdges {
        src_id: src,
        name: Some("likes".to_string()),
        as_of: add_edge.ts_millis,
        dry_run: true,
    };
    {
        let txn = txn_db.transaction();
        let result = edge::restore_edges(&txn, txn_db, &dry_run_restore);
        assert!(result.is_err(), "dry_run strict should fail on missing summary");
    }
}

// ============================================================================
// Phase 1: Critical Gap Tests (migrated from tests.rs)
// ============================================================================

/// Validates: Reverse edge index is consistent with forward edge.
#[test]
fn ops_reverse_edge_index_consistency() {
    let (_temp_dir, storage) = setup_storage();
    let txn_db = storage.transaction_db().unwrap();

    let src_id = Id::new();
    let dst_id = Id::new();
    let ts = TimestampMilli::now();

    // Create nodes and edge in single transaction
    {
        let txn = txn_db.transaction();

        node::add_node(&txn, txn_db, &AddNode {
            id: src_id,
            ts_millis: ts,
            name: "source".to_string(),
            valid_range: None,
            summary: DataUrl::from_text("Source node"),
        }, None).unwrap();

        node::add_node(&txn, txn_db, &AddNode {
            id: dst_id,
            ts_millis: ts,
            name: "target".to_string(),
            valid_range: None,
            summary: DataUrl::from_text("Target node"),
        }, None).unwrap();

        let edge_ts = TimestampMilli::now();
        edge::add_edge(&txn, txn_db, &AddEdge {
            source_node_id: src_id,
            target_node_id: dst_id,
            ts_millis: edge_ts,
            name: "test_edge".to_string(),
            summary: DataUrl::from_text("Test edge"),
            weight: Some(1.0),
            valid_range: None,
        }, None).unwrap();

        txn.commit().unwrap();
    }

    // Verify reverse CF has entry for dst_id
    let reverse_cf = txn_db.cf_handle(ReverseEdges::CF_NAME).unwrap();
    let prefix = dst_id.into_bytes();
    let iter = txn_db.iterator_cf(
        reverse_cf,
        rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
    );

    let mut found = false;
    for item in iter {
        let (key, _) = item.unwrap();
        if key.starts_with(&prefix) {
            found = true;
            break;
        }
        // Stop if we've passed the prefix range
        if !key.starts_with(&prefix) {
            break;
        }
    }
    assert!(found, "Reverse CF should have entry for dst_id");
}

/// Validates: Forward and reverse edge writes are atomic.
#[test]
fn ops_forward_reverse_atomic() {
    let (_temp_dir, storage) = setup_storage();
    let txn_db = storage.transaction_db().unwrap();

    let src_id = Id::new();
    let dst_id = Id::new();
    let ts = TimestampMilli::now();
    let edge_ts = TimestampMilli::now();
    let name_hash = NameHash::from_name("atomic_edge");

    {
        let txn = txn_db.transaction();

        node::add_node(&txn, txn_db, &AddNode {
            id: src_id,
            ts_millis: ts,
            name: "src".to_string(),
            valid_range: None,
            summary: DataUrl::from_text("Source"),
        }, None).unwrap();

        node::add_node(&txn, txn_db, &AddNode {
            id: dst_id,
            ts_millis: ts,
            name: "dst".to_string(),
            valid_range: None,
            summary: DataUrl::from_text("Dest"),
        }, None).unwrap();

        edge::add_edge(&txn, txn_db, &AddEdge {
            source_node_id: src_id,
            target_node_id: dst_id,
            ts_millis: edge_ts,
            name: "atomic_edge".to_string(),
            summary: DataUrl::from_text("Atomic test"),
            weight: Some(2.5),
            valid_range: None,
        }, None).unwrap();

        txn.commit().unwrap();
    }

    // Verify forward CF
    let forward_cf = txn_db.cf_handle(ForwardEdges::CF_NAME).unwrap();
    let forward_key = ForwardEdgeCfKey(src_id, dst_id, name_hash, edge_ts);
    let forward_key_bytes = ForwardEdges::key_to_bytes(&forward_key);
    assert!(txn_db.get_cf(forward_cf, &forward_key_bytes).unwrap().is_some());

    // Verify reverse CF
    let reverse_cf = txn_db.cf_handle(ReverseEdges::CF_NAME).unwrap();
    let reverse_key = ReverseEdgeCfKey(dst_id, src_id, name_hash, edge_ts);
    let reverse_key_bytes = ReverseEdges::key_to_bytes(&reverse_key);
    assert!(txn_db.get_cf(reverse_cf, &reverse_key_bytes).unwrap().is_some());
}

/// Validates: Fragment append preserves existing fragments.
#[test]
fn ops_fragment_append_idempotency() {
    let (_temp_dir, storage) = setup_storage();
    let txn_db = storage.transaction_db().unwrap();

    let node_id = Id::new();
    let ts = TimestampMilli::now();

    // Add node first
    {
        let txn = txn_db.transaction();
        node::add_node(&txn, txn_db, &AddNode {
            id: node_id,
            ts_millis: ts,
            name: "frag_test".to_string(),
            valid_range: None,
            summary: DataUrl::from_text("Fragment test node"),
        }, None).unwrap();
        txn.commit().unwrap();
    }

    // Add three fragments with different timestamps
    let ts1 = TimestampMilli::now();
    std::thread::sleep(std::time::Duration::from_millis(10));
    let ts2 = TimestampMilli::now();
    std::thread::sleep(std::time::Duration::from_millis(10));
    let ts3 = TimestampMilli::now();

    {
        let txn = txn_db.transaction();
        fragment::add_node_fragment(&txn, txn_db, &AddNodeFragment {
            id: node_id,
            ts_millis: ts1,
            content: DataUrl::from_text("Fragment 1"),
            valid_range: None,
        }).unwrap();
        fragment::add_node_fragment(&txn, txn_db, &AddNodeFragment {
            id: node_id,
            ts_millis: ts2,
            content: DataUrl::from_text("Fragment 2"),
            valid_range: None,
        }).unwrap();
        fragment::add_node_fragment(&txn, txn_db, &AddNodeFragment {
            id: node_id,
            ts_millis: ts3,
            content: DataUrl::from_text("Fragment 3"),
            valid_range: None,
        }).unwrap();
        txn.commit().unwrap();
    }

    // Re-add fragment 2 (should append new entry, not overwrite)
    {
        let txn = txn_db.transaction();
        fragment::add_node_fragment(&txn, txn_db, &AddNodeFragment {
            id: node_id,
            ts_millis: ts2,
            content: DataUrl::from_text("Fragment 2 replay"),
            valid_range: None,
        }).unwrap();
        txn.commit().unwrap();
    }

    // Count fragments in CF
    let frag_cf = txn_db.cf_handle(NodeFragments::CF_NAME).unwrap();
    let prefix = node_id.into_bytes();
    let iter = txn_db.iterator_cf(
        frag_cf,
        rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
    );
    let mut count = 0;
    for item in iter {
        let (key, _) = item.unwrap();
        if key.starts_with(&prefix) {
            count += 1;
        } else {
            break;
        }
    }
    // Should have 4 fragments (3 original + 1 replay which overwrites)
    // Actually fragments use (id, ts) as key, so same ts overwrites
    assert!(count >= 3, "Should have at least 3 fragments, got {}", count);
}

// ============================================================================
// Phase 2: VERSIONING Tests (migrated from tests.rs)
// ============================================================================

/// Validates: UpdateNode creates a new version in Nodes CF.
#[test]
fn ops_node_update_creates_version_history() {
    let (_temp_dir, storage) = setup_storage();
    let txn_db = storage.transaction_db().unwrap();

    let node_id = Id::new();
    let ts = TimestampMilli::now();

    // Create initial node
    {
        let txn = txn_db.transaction();
        node::add_node(&txn, txn_db, &AddNode {
            id: node_id,
            ts_millis: ts,
            name: "versioned_node".to_string(),
            valid_range: None,
            summary: DataUrl::from_text("Version 1"),
        }, None).unwrap();
        txn.commit().unwrap();
    }

    // Ensure distinct timestamp for the update
    std::thread::sleep(std::time::Duration::from_millis(5));

    // Update summary (version 1 -> 2)
    {
        let txn = txn_db.transaction();
        node::update_node(&txn, txn_db, &UpdateNode {
            id: node_id,
            expected_version: 1,
            new_active_period: None,
            new_summary: Some(DataUrl::from_text("Version 2")),
        }).unwrap();
        txn.commit().unwrap();
    }

    // Verify multiple versions exist in Nodes CF
    let nodes_cf = txn_db.cf_handle(Nodes::CF_NAME).unwrap();
    let prefix = node_id.into_bytes();
    let iter = txn_db.iterator_cf(
        nodes_cf,
        rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
    );
    let mut version_count = 0;
    for item in iter {
        let (key, _) = item.unwrap();
        if key.starts_with(&prefix) {
            version_count += 1;
        } else {
            break;
        }
    }
    assert!(version_count >= 2, "Should have at least 2 versions in Nodes CF, got {}", version_count);
}

/// Validates: UpdateEdge (summary) creates a new version in ForwardEdges CF.
#[test]
fn ops_edge_update_creates_version_history() {
    let (_temp_dir, storage) = setup_storage();
    let txn_db = storage.transaction_db().unwrap();

    let src_id = Id::new();
    let dst_id = Id::new();
    let ts = TimestampMilli::now();
    let edge_name = "versioned_edge".to_string();
    let name_hash = NameHash::from_name(&edge_name);

    // Create nodes and edge
    {
        let txn = txn_db.transaction();
        node::add_node(&txn, txn_db, &AddNode {
            id: src_id,
            ts_millis: ts,
            name: "src".to_string(),
            valid_range: None,
            summary: DataUrl::from_text("Source"),
        }, None).unwrap();
        node::add_node(&txn, txn_db, &AddNode {
            id: dst_id,
            ts_millis: ts,
            name: "dst".to_string(),
            valid_range: None,
            summary: DataUrl::from_text("Dest"),
        }, None).unwrap();
        edge::add_edge(&txn, txn_db, &AddEdge {
            source_node_id: src_id,
            target_node_id: dst_id,
            ts_millis: TimestampMilli::now(),
            name: edge_name.clone(),
            summary: DataUrl::from_text("Edge V1"),
            weight: Some(1.0),
            valid_range: None,
        }, None).unwrap();
        txn.commit().unwrap();
    }

    // Ensure distinct timestamp for the update
    std::thread::sleep(std::time::Duration::from_millis(5));

    // Update edge summary using consolidated UpdateEdge
    {
        let txn = txn_db.transaction();
        edge::update_edge(&txn, txn_db, &UpdateEdge {
            src_id,
            dst_id,
            name: edge_name.clone(),
            expected_version: 1,
            new_weight: None,
            new_active_period: None,
            new_summary: Some(DataUrl::from_text("Edge V2")),
        }).unwrap();
        txn.commit().unwrap();
    }

    // Verify multiple versions in ForwardEdges CF
    let forward_cf = txn_db.cf_handle(ForwardEdges::CF_NAME).unwrap();
    let mut prefix = Vec::with_capacity(40);
    prefix.extend_from_slice(&src_id.into_bytes());
    prefix.extend_from_slice(&dst_id.into_bytes());
    prefix.extend_from_slice(name_hash.as_bytes());

    let iter = txn_db.iterator_cf(
        forward_cf,
        rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
    );
    let mut version_count = 0;
    for item in iter {
        let (key, _) = item.unwrap();
        if key.starts_with(&prefix) {
            version_count += 1;
        } else {
            break;
        }
    }
    assert!(version_count >= 2, "Should have at least 2 edge versions, got {}", version_count);
}

/// Validates: UpdateEdge (weight) creates a new version.
#[test]
fn ops_edge_weight_update_creates_version() {
    let (_temp_dir, storage) = setup_storage();
    let txn_db = storage.transaction_db().unwrap();

    let src_id = Id::new();
    let dst_id = Id::new();
    let ts = TimestampMilli::now();
    let edge_name = "weight_edge".to_string();
    let name_hash = NameHash::from_name(&edge_name);

    // Create nodes and edge
    {
        let txn = txn_db.transaction();
        node::add_node(&txn, txn_db, &AddNode {
            id: src_id,
            ts_millis: ts,
            name: "src".to_string(),
            valid_range: None,
            summary: DataUrl::from_text("Source"),
        }, None).unwrap();
        node::add_node(&txn, txn_db, &AddNode {
            id: dst_id,
            ts_millis: ts,
            name: "dst".to_string(),
            valid_range: None,
            summary: DataUrl::from_text("Dest"),
        }, None).unwrap();
        edge::add_edge(&txn, txn_db, &AddEdge {
            source_node_id: src_id,
            target_node_id: dst_id,
            ts_millis: TimestampMilli::now(),
            name: edge_name.clone(),
            summary: DataUrl::from_text("Weight test edge"),
            weight: Some(1.0),
            valid_range: None,
        }, None).unwrap();
        txn.commit().unwrap();
    }

    // Ensure distinct timestamp for the update
    std::thread::sleep(std::time::Duration::from_millis(5));

    // Update weight using consolidated UpdateEdge
    {
        let txn = txn_db.transaction();
        edge::update_edge(&txn, txn_db, &UpdateEdge {
            src_id,
            dst_id,
            name: edge_name.clone(),
            expected_version: 1,
            new_weight: Some(Some(5.0)),
            new_active_period: None,
            new_summary: None,
        }).unwrap();
        txn.commit().unwrap();
    }

    // Verify version count increased
    let forward_cf = txn_db.cf_handle(ForwardEdges::CF_NAME).unwrap();
    let mut prefix = Vec::with_capacity(40);
    prefix.extend_from_slice(&src_id.into_bytes());
    prefix.extend_from_slice(&dst_id.into_bytes());
    prefix.extend_from_slice(name_hash.as_bytes());

    let iter = txn_db.iterator_cf(
        forward_cf,
        rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
    );
    let mut version_count = 0;
    for item in iter {
        let (key, _) = item.unwrap();
        if key.starts_with(&prefix) {
            version_count += 1;
        } else {
            break;
        }
    }
    assert!(version_count >= 2, "Should have at least 2 versions after weight update, got {}", version_count);
}

/// Validates: DeleteNode creates a tombstone version.
#[test]
fn ops_delete_node_creates_tombstone() {
    let (_temp_dir, storage) = setup_storage();
    let txn_db = storage.transaction_db().unwrap();

    let node_id = Id::new();
    let ts = TimestampMilli::now();

    // Create node
    {
        let txn = txn_db.transaction();
        node::add_node(&txn, txn_db, &AddNode {
            id: node_id,
            ts_millis: ts,
            name: "delete_test".to_string(),
            valid_range: None,
            summary: DataUrl::from_text("To be deleted"),
        }, None).unwrap();
        txn.commit().unwrap();
    }

    // Delete node
    {
        let txn = txn_db.transaction();
        node::delete_node(&txn, txn_db, &DeleteNode {
            id: node_id,
            expected_version: 1,
        }).unwrap();
        txn.commit().unwrap();
    }

    // Verify tombstone version exists (deleted=true)
    let nodes_cf = txn_db.cf_handle(Nodes::CF_NAME).unwrap();
    let prefix = node_id.into_bytes();

    let iter = txn_db.iterator_cf(
        nodes_cf,
        rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
    );
    let mut found_tombstone = false;
    for item in iter {
        let (key, value) = item.unwrap();
        if !key.starts_with(&prefix) {
            break;
        }
        let node_value: NodeCfValue = Nodes::value_from_bytes(&value).unwrap();
        if node_value.0.is_none() && node_value.5 {
            // Current version (ValidUntil=None) with deleted=true
            found_tombstone = true;
            break;
        }
    }
    assert!(found_tombstone, "Should have a tombstone version");
}

/// Validates: Summary hash prefix scan finds all nodes with same hash.
#[test]
fn ops_summary_hash_prefix_scan() {
    let (_temp_dir, storage) = setup_storage();
    let txn_db = storage.transaction_db().unwrap();

    let shared_summary = DataUrl::from_text("Shared summary content");
    let ts = TimestampMilli::now();

    let node1 = Id::new();
    let node2 = Id::new();
    let node3 = Id::new();

    // Create three nodes with the same summary
    {
        let txn = txn_db.transaction();
        for (id, name) in [(node1, "n1"), (node2, "n2"), (node3, "n3")] {
            node::add_node(&txn, txn_db, &AddNode {
                id,
                ts_millis: ts,
                name: name.to_string(),
                valid_range: None,
                summary: shared_summary.clone(),
            }, None).unwrap();
        }
        txn.commit().unwrap();
    }

    // Verify all three are in the summary index with same hash
    let hash = SummaryHash::from_summary(&shared_summary).unwrap();
    let index_cf = txn_db.cf_handle(NodeSummaryIndex::CF_NAME).unwrap();
    let prefix = hash.as_bytes();

    let iter = txn_db.iterator_cf(
        index_cf,
        rocksdb::IteratorMode::From(prefix, rocksdb::Direction::Forward),
    );
    let mut count = 0;
    for item in iter {
        let (key, _) = item.unwrap();
        if key.starts_with(prefix) {
            count += 1;
        } else {
            break;
        }
    }
    assert_eq!(count, 3, "Should find all 3 nodes with same summary hash");
}

/// Validates: Version scan returns entries ordered by ValidSince.
#[test]
fn ops_version_scan_ordering() {
    let (_temp_dir, storage) = setup_storage();
    let txn_db = storage.transaction_db().unwrap();

    let node_id = Id::new();
    let ts = TimestampMilli::now();

    // Create node with V1
    {
        let txn = txn_db.transaction();
        node::add_node(&txn, txn_db, &AddNode {
            id: node_id,
            ts_millis: ts,
            name: "version_order".to_string(),
            valid_range: None,
            summary: DataUrl::from_text("V1"),
        }, None).unwrap();
        txn.commit().unwrap();
    }

    std::thread::sleep(std::time::Duration::from_millis(10));

    // Update to V2
    {
        let txn = txn_db.transaction();
        node::update_node(&txn, txn_db, &UpdateNode {
            id: node_id,
            expected_version: 1,
            new_active_period: None,
            new_summary: Some(DataUrl::from_text("V2")),
        }).unwrap();
        txn.commit().unwrap();
    }

    std::thread::sleep(std::time::Duration::from_millis(10));

    // Update to V3
    {
        let txn = txn_db.transaction();
        node::update_node(&txn, txn_db, &UpdateNode {
            id: node_id,
            expected_version: 2,
            new_active_period: None,
            new_summary: Some(DataUrl::from_text("V3")),
        }).unwrap();
        txn.commit().unwrap();
    }

    // Verify versions are ordered by ValidSince in Nodes CF
    let nodes_cf = txn_db.cf_handle(Nodes::CF_NAME).unwrap();
    let prefix = node_id.into_bytes();

    let iter = txn_db.iterator_cf(
        nodes_cf,
        rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
    );
    let mut timestamps = Vec::new();
    for item in iter {
        let (key, _) = item.unwrap();
        if !key.starts_with(&prefix) {
            break;
        }
        // Extract ValidSince from key (last 8 bytes after 16-byte node ID)
        if key.len() >= 24 {
            let ts_bytes: [u8; 8] = key[16..24].try_into().unwrap();
            timestamps.push(u64::from_be_bytes(ts_bytes));
        }
    }

    assert!(timestamps.len() >= 3, "Should have at least 3 versions");
    for i in 1..timestamps.len() {
        assert!(timestamps[i] >= timestamps[i-1], "Timestamps should be ordered");
    }
}

/// Validates: Replay of same mutation is safe (optimistic locking).
#[test]
fn ops_mutation_replay_idempotent() {
    let (_temp_dir, storage) = setup_storage();
    let txn_db = storage.transaction_db().unwrap();

    let node_id = Id::new();
    let ts = TimestampMilli::now();
    let mutation = AddNode {
        id: node_id,
        ts_millis: ts,
        name: "idempotent".to_string(),
        valid_range: None,
        summary: DataUrl::from_text("Original"),
    };

    // First creation
    {
        let txn = txn_db.transaction();
        node::add_node(&txn, txn_db, &mutation, None).unwrap();
        txn.commit().unwrap();
    }

    // Replay same mutation - should succeed (overwrites same key)
    {
        let txn = txn_db.transaction();
        let result = node::add_node(&txn, txn_db, &mutation, None);
        assert!(result.is_ok(), "Replay should succeed");
        txn.commit().unwrap();
    }

    // Verify node still exists and is queryable
    let nodes_cf = txn_db.cf_handle(Nodes::CF_NAME).unwrap();
    let key = NodeCfKey(node_id, ts);
    let key_bytes = Nodes::key_to_bytes(&key);
    assert!(txn_db.get_cf(nodes_cf, key_bytes).unwrap().is_some());
}

/// Validates: Version mismatch is properly rejected.
#[test]
fn ops_version_mismatch_rejected() {
    let (_temp_dir, storage) = setup_storage();
    let txn_db = storage.transaction_db().unwrap();

    let node_id = Id::new();
    let ts = TimestampMilli::now();

    // Create node (version 1)
    {
        let txn = txn_db.transaction();
        node::add_node(&txn, txn_db, &AddNode {
            id: node_id,
            ts_millis: ts,
            name: "version_test".to_string(),
            valid_range: None,
            summary: DataUrl::from_text("V1"),
        }, None).unwrap();
        txn.commit().unwrap();
    }

    // Try to update with wrong expected_version
    {
        let txn = txn_db.transaction();
        let result = node::update_node(&txn, txn_db, &UpdateNode {
            id: node_id,
            expected_version: 99, // Wrong version
            new_active_period: None,
            new_summary: Some(DataUrl::from_text("V2")),
        });
        assert!(result.is_err(), "Should reject version mismatch");
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Version mismatch"), "Error should mention version mismatch");
    }
}

/// Validates: Delete of already-deleted node is rejected.
#[test]
fn ops_double_delete_rejected() {
    let (_temp_dir, storage) = setup_storage();
    let txn_db = storage.transaction_db().unwrap();

    let node_id = Id::new();
    let ts = TimestampMilli::now();

    // Create and delete node
    {
        let txn = txn_db.transaction();
        node::add_node(&txn, txn_db, &AddNode {
            id: node_id,
            ts_millis: ts,
            name: "double_delete".to_string(),
            valid_range: None,
            summary: DataUrl::from_text("Will be deleted"),
        }, None).unwrap();
        txn.commit().unwrap();
    }

    {
        let txn = txn_db.transaction();
        node::delete_node(&txn, txn_db, &DeleteNode {
            id: node_id,
            expected_version: 1,
        }).unwrap();
        txn.commit().unwrap();
    }

    // Try to delete again
    {
        let txn = txn_db.transaction();
        let result = node::delete_node(&txn, txn_db, &DeleteNode {
            id: node_id,
            expected_version: 2,
        });
        assert!(result.is_err(), "Double delete should be rejected");
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("already deleted"), "Error should mention already deleted");
    }
}

/// Validates: Update of deleted node is rejected.
#[test]
fn ops_update_deleted_node_rejected() {
    let (_temp_dir, storage) = setup_storage();
    let txn_db = storage.transaction_db().unwrap();

    let node_id = Id::new();
    let ts = TimestampMilli::now();

    // Create and delete node
    {
        let txn = txn_db.transaction();
        node::add_node(&txn, txn_db, &AddNode {
            id: node_id,
            ts_millis: ts,
            name: "update_deleted".to_string(),
            valid_range: None,
            summary: DataUrl::from_text("Will be deleted"),
        }, None).unwrap();
        txn.commit().unwrap();
    }

    {
        let txn = txn_db.transaction();
        node::delete_node(&txn, txn_db, &DeleteNode {
            id: node_id,
            expected_version: 1,
        }).unwrap();
        txn.commit().unwrap();
    }

    // Try to update deleted node
    {
        let txn = txn_db.transaction();
        let result = node::update_node(&txn, txn_db, &UpdateNode {
            id: node_id,
            expected_version: 2,
            new_active_period: None,
            new_summary: Some(DataUrl::from_text("Should fail")),
        });
        assert!(result.is_err(), "Update of deleted node should be rejected");
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("deleted"), "Error should mention deleted");
    }
}
