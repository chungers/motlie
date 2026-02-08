use tempfile::TempDir;

use super::{edge, node};
use crate::graph::schema::{
    EdgeSummaries, EdgeSummaryCfKey, EdgeVersionHistory, EdgeVersionHistoryCfKey,
    EdgeVersionHistoryCfValue, ForwardEdgeCfKey, ForwardEdgeCfValue, ForwardEdges, NodeCfKey,
    NodeSummaryIndex, NodeSummaryIndexCfKey, NodeVersionHistory, NodeVersionHistoryCfKey, Nodes,
    ReverseEdgeCfKey, ReverseEdges,
};
use crate::graph::{AddEdge, AddNode, DeleteEdge, UpdateNodeSummary};
use crate::graph::mutation::RestoreEdges;
use crate::graph::ops::summary::verify_edge_summary_exists;
use crate::graph::name_hash::NameHash;
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
    let (_temp_dir, mut storage) = setup_storage();
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
    let (_temp_dir, mut storage) = setup_storage();
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
fn ops_update_node_summary_updates_index() {
    let (_temp_dir, mut storage) = setup_storage();
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
    let update = UpdateNodeSummary {
        id,
        new_summary: new_summary.clone(),
        expected_version: 1,
    };

    {
        let txn = txn_db.transaction();
        node::update_node_summary(&txn, txn_db, &update).unwrap();
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
    let (_temp_dir, mut storage) = setup_storage();
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
