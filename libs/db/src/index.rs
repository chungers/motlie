use crate::{AddEdgeArgs, AddFragmentArgs, AddVertexArgs, InvalidateArgs};

pub trait Index {
    fn cf_name() -> &'static str;
}

/// Nodes column family.
/// Source of truth for nodes; optimized for key lookup and updates.
/// Normal case
/// Key = uuid + (invalidation timestamp)
/// Value = string summary of the node that contains the name and summarization of the fragments associated with the node.
/// Invalidating a node is done by
/// 1. Setting the value keyed by uuid to an empty string.
/// 2. Inserting the current summary (current value) with a new key (uuid + (invalidation timestamp))
#[derive(Debug, Clone, PartialEq)]
pub struct Nodes;
#[derive(Debug, Clone, PartialEq)]
pub struct Edges;
#[derive(Debug, Clone, PartialEq)]
pub struct Fragments;

impl Index for Nodes {
    fn cf_name() -> &'static str {
        "nodes(uuid + (invalidation timestamp)) -> summary"
    }
}
impl Index for Edges {
    fn cf_name() -> &'static str {
        "edges(src_uuid, dst_uuid, ts [,invalid_ts]) -> name"
    }
}
impl Index for Fragments {
    fn cf_name() -> &'static str {
        "fragments(node_or_edge_uuid, ts [,invalid_ts]) -> fragment"
    }
}

impl Nodes {
    fn key(args: &AddVertexArgs) -> Vec<u8> {
        format!("{}", args.id).into_bytes()
    }

    fn value(args: &AddVertexArgs) -> Vec<u8> {
        format!("{}", args.name).into_bytes()
    }

    fn keys(args: &InvalidateArgs) -> (Vec<u8>, Vec<u8>) {
        let invalidated = format!("{}", args.id).into_bytes();
        let tombstone = format!("{}{}", args.id, args.ts_millis).into_bytes();
        (invalidated, tombstone)
    }

    fn values(args: &InvalidateArgs, current_value: Vec<u8>) -> (Vec<u8>, Vec<u8>) {
        let invalidated = format!("").into_bytes();
        let tombstone = format!(
            "Invalidated on {}.\nReason: {}\nSummary: {}",
            args.id,
            args.ts_millis,
            String::from_utf8(current_value).unwrap()
        )
        .into_bytes();
        (invalidated, tombstone)
    }
}

impl Edges {
    fn key(args: &AddEdgeArgs) -> Vec<u8> {
        format!(
            "{}{}{}",
            args.source_vertex_id, args.target_vertex_id, args.name
        )
        .into_bytes()
    }

    fn value(args: &AddEdgeArgs) -> Vec<u8> {
        format!("{}", args.id).into_bytes()
    }

    fn keys(args: &InvalidateArgs) -> (Vec<u8>, Vec<u8>) {
        let invalidated = format!("{}", args.id).into_bytes();
        let tombstone = format!("{}{}", args.id, args.ts_millis).into_bytes();
        (invalidated, tombstone)
    }

    fn values(args: &InvalidateArgs, current_value: Vec<u8>) -> (Vec<u8>, Vec<u8>) {
        let invalidated = format!("").into_bytes();
        let tombstone = format!(
            "Invalidated on {}.\nReason: {}\nSummary: {}",
            args.id,
            args.ts_millis,
            String::from_utf8(current_value).unwrap()
        )
        .into_bytes();
        (invalidated, tombstone)
    }
}

impl Fragments {
    fn key(args: &AddFragmentArgs) -> Vec<u8> {
        format!("{}", args.id).into_bytes()
    }

    fn value(args: &AddFragmentArgs) -> Vec<u8> {
        format!("{}", args.body).into_bytes()
    }

    fn keys(args: &InvalidateArgs) -> (Vec<u8>, Vec<u8>) {
        let invalidated = format!("{}", args.id).into_bytes();
        let tombstone = format!("{}{}", args.id, args.ts_millis).into_bytes();
        (invalidated, tombstone)
    }

    fn values(args: &InvalidateArgs, current_value: Vec<u8>) -> (Vec<u8>, Vec<u8>) {
        let invalidated = format!("").into_bytes();
        let tombstone = format!(
            "Invalidated on {}.\nReason: {}\nSummary: {}",
            args.id,
            args.ts_millis,
            String::from_utf8(current_value).unwrap()
        )
        .into_bytes();
        (invalidated, tombstone)
    }
}
