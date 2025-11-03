use core::str;

use crate::{AddEdgeArgs, AddFragmentArgs, AddVertexArgs, Id};
use serde::{Deserialize, Serialize};

pub trait IsColumnFamily {
    const CF_NAME: &'static str;
}

#[derive(Serialize, Deserialize)]
struct TimestampMillisecond(u64);

/// Nodes column family.
pub struct Nodes;
#[derive(Serialize, Deserialize)]
struct NodeCfKey(Id);

#[derive(Serialize, Deserialize)]
struct NodeCfValue(NodeSummary);

#[derive(Serialize, Deserialize)]
struct NodeSummary(String);

pub struct Edges;

#[derive(Serialize, Deserialize)]
struct EdgeCfKey(Id);

#[derive(Serialize, Deserialize)]
struct EdgeCfValue(EdgeSummary);

#[derive(Serialize, Deserialize)]
struct EdgeSummary(String);

pub struct Fragments;

#[derive(Serialize, Deserialize)]
struct FragmentCfKey(Id, TimestampMillisecond);

#[derive(Serialize, Deserialize)]
struct FragmentCfValue(FragmentContent);

#[derive(Serialize, Deserialize)]
struct FragmentContent(String);

pub struct ForwardEdges;

#[derive(Serialize, Deserialize)]
struct ForwardEdgesCfKey(EdgeSourceId, EdgeDestinationId, EdgeName);

#[derive(Serialize, Deserialize)]
struct ForwardEdgesCfValue(EdgeSummary);

#[derive(Serialize, Deserialize)]
struct EdgeSourceId(Id);

#[derive(Serialize, Deserialize)]
struct EdgeDestinationId(Id);

#[derive(Serialize, Deserialize)]
struct EdgeName(String);

pub struct ReverseEdges;

#[derive(Serialize, Deserialize)]
struct ReverseEdgesCfKey(EdgeDestinationId, EdgeSourceId, EdgeName);

#[derive(Serialize, Deserialize)]
struct ReverseEdgesCfValue(EdgeSummary);

impl IsColumnFamily for Nodes {
    const CF_NAME: &'static str = "nodes";
}

impl IsColumnFamily for Edges {
    const CF_NAME: &'static str = "edges";
}

impl IsColumnFamily for Fragments {
    const CF_NAME: &'static str = "fragments";
}

impl IsColumnFamily for ForwardEdges {
    const CF_NAME: &'static str = "forward_edges";
}

impl IsColumnFamily for ReverseEdges {
    const CF_NAME: &'static str = "reverse_edges";
}

pub struct RowsToWrite(pub Vec<(Vec<u8>, Vec<u8>)>);

impl Nodes {
    pub fn plan(args: &AddVertexArgs) -> RowsToWrite {
        let key = Vec::from(args.id.into_bytes());
        let value = format!(
            // Markdown summary
            r#"[comment]:\#(id={})
# {}"#,
            args.id, args.name
        )
        .into_bytes();

        RowsToWrite(vec![(key, value)])
    }
}

impl Edges {
    pub fn plan(args: &AddEdgeArgs) -> RowsToWrite {
        let mut row1 = (Vec::new(), Vec::new());
        row1.0
            .extend_from_slice(&args.source_vertex_id.into_bytes());
        row1.0
            .extend_from_slice(&args.target_vertex_id.into_bytes());
        row1.0.extend_from_slice(args.name.as_bytes());
        row1.1.extend_from_slice(&args.id.into_bytes());

        let mut row2 = (Vec::new(), Vec::new());
        row2.0.extend_from_slice(&args.id.into_bytes());
        row2.1.extend_from_slice(
            format!(
                // Markdown summary
                r#"[comment]:\#(id={})
#{}"#,
                args.id, args.name
            )
            .as_bytes(),
        );

        RowsToWrite(vec![row1, row2])
    }
}

impl Fragments {
    pub fn plan(args: &AddFragmentArgs) -> RowsToWrite {
        let mut key = Vec::from(args.id.into_bytes());
        key.extend_from_slice(&args.ts_millis.to_be_bytes());
        let value = format!(
            // Markdown summary
            r#"[comment]:\#(id={})
# {}"#,
            args.id, args.content
        )
        .into_bytes();

        RowsToWrite(vec![(key, value)])
    }
}
