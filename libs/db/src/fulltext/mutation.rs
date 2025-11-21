//! Fulltext indexing mutation executor implementations
//!
//! This module defines how mutations index themselves into the Tantivy fulltext search index.
//! Mirrors the pattern in `crate::mutation::MutationExecutor` for graph storage.

use anyhow::{Context, Result};
use tantivy::schema::*;
use tantivy::{doc, IndexWriter};

use crate::mutation::{
    AddEdge, AddEdgeFragment, AddNode, AddNodeFragment, UpdateEdgeValidSinceUntil,
    UpdateEdgeWeight, UpdateNodeValidSinceUntil,
};

use super::{compute_time_bucket, extract_tags, weight_to_facet, FulltextFields};

/// Trait for mutations to index themselves into Tantivy.
///
/// This trait defines HOW to index the mutation into the fulltext search index.
/// Each mutation type knows how to extract and index its own searchable content.
///
/// Following the same pattern as MutationExecutor for graph storage.
pub trait FulltextIndexExecutor: Send + Sync {
    /// Index this mutation into the Tantivy index writer.
    /// Each mutation type knows how to extract and index its searchable content.
    fn index(&self, index_writer: &IndexWriter, fields: &FulltextFields) -> Result<()>;
}

// ============================================================================
// FulltextIndexExecutor Implementations
// ============================================================================

impl FulltextIndexExecutor for AddNode {
    fn index(&self, index_writer: &IndexWriter, fields: &FulltextFields) -> Result<()> {
        let mut doc = doc!(
            fields.id_field => self.id.as_bytes().to_vec(),
            fields.node_name_field => self.name.clone(),
            fields.doc_type_field => "node",
            fields.timestamp_field => self.ts_millis.0,
        );

        // Add facets
        doc.add_facet(fields.doc_type_facet, Facet::from("/type/node"));
        doc.add_facet(fields.time_bucket_facet, compute_time_bucket(self.ts_millis));

        index_writer
            .add_document(doc)
            .context("Failed to index AddNode")?;

        log::debug!("[FullText] Indexed node: id={}, name={}", self.id, self.name);
        Ok(())
    }
}

impl FulltextIndexExecutor for AddEdge {
    fn index(&self, index_writer: &IndexWriter, fields: &FulltextFields) -> Result<()> {
        // Decode edge summary content
        let summary_text = self
            .summary
            .decode_string()
            .unwrap_or_else(|_| String::new());

        // Extract tags from summary
        let tags = extract_tags(&summary_text);

        let mut doc = doc!(
            fields.src_id_field => self.source_node_id.as_bytes().to_vec(),
            fields.dst_id_field => self.target_node_id.as_bytes().to_vec(),
            fields.edge_name_field => self.name.clone(),
            fields.content_field => summary_text,
            fields.doc_type_field => "edge",
            fields.timestamp_field => self.ts_millis.0,
        );

        // Add facets
        doc.add_facet(fields.doc_type_facet, Facet::from("/type/edge"));
        doc.add_facet(fields.time_bucket_facet, compute_time_bucket(self.ts_millis));

        // Add weight if present
        if let Some(weight) = self.weight {
            doc.add_f64(fields.weight_field, weight);
            doc.add_facet(fields.weight_range_facet, weight_to_facet(weight));
        }

        // Add user-defined tags as facets
        for tag in tags {
            doc.add_facet(fields.tags_facet, Facet::from(&format!("/tag/{}", tag)));
        }

        index_writer
            .add_document(doc)
            .context("Failed to index AddEdge")?;

        log::debug!(
            "[FullText] Indexed edge: src={}, dst={}, name={}",
            self.source_node_id,
            self.target_node_id,
            self.name
        );
        Ok(())
    }
}

impl FulltextIndexExecutor for AddNodeFragment {
    fn index(&self, index_writer: &IndexWriter, fields: &FulltextFields) -> Result<()> {
        // Decode DataUrl content
        let content_text = self
            .content
            .decode_string()
            .context("Failed to decode fragment content")?;

        // Extract user-defined tags from content
        let tags = extract_tags(&content_text);

        let mut doc = doc!(
            fields.id_field => self.id.as_bytes().to_vec(),
            fields.content_field => content_text,
            fields.doc_type_field => "node_fragment",
            fields.timestamp_field => self.ts_millis.0,
        );

        // Add facets
        doc.add_facet(fields.doc_type_facet, Facet::from("/type/node_fragment"));
        doc.add_facet(fields.time_bucket_facet, compute_time_bucket(self.ts_millis));

        // Add user-defined tags as facets
        for tag in tags {
            doc.add_facet(fields.tags_facet, Facet::from(&format!("/tag/{}", tag)));
        }

        index_writer
            .add_document(doc)
            .context("Failed to index AddNodeFragment")?;

        log::debug!(
            "[FullText] Indexed node fragment: id={}, content_len={}",
            self.id,
            self.content.as_ref().len()
        );
        Ok(())
    }
}

impl FulltextIndexExecutor for AddEdgeFragment {
    fn index(&self, index_writer: &IndexWriter, fields: &FulltextFields) -> Result<()> {
        // Decode DataUrl content
        let content_text = self
            .content
            .decode_string()
            .context("Failed to decode edge fragment content")?;

        // Extract user-defined tags from content
        let tags = extract_tags(&content_text);

        let mut doc = doc!(
            fields.src_id_field => self.src_id.as_bytes().to_vec(),
            fields.dst_id_field => self.dst_id.as_bytes().to_vec(),
            fields.edge_name_field => self.edge_name.clone(),
            fields.content_field => content_text,
            fields.doc_type_field => "edge_fragment",
            fields.timestamp_field => self.ts_millis.0,
        );

        // Add facets
        doc.add_facet(fields.doc_type_facet, Facet::from("/type/edge_fragment"));
        doc.add_facet(fields.time_bucket_facet, compute_time_bucket(self.ts_millis));

        // Add user-defined tags as facets
        for tag in tags {
            doc.add_facet(fields.tags_facet, Facet::from(&format!("/tag/{}", tag)));
        }

        index_writer
            .add_document(doc)
            .context("Failed to index AddEdgeFragment")?;

        log::debug!(
            "[FullText] Indexed edge fragment: src={}, dst={}, name={}, content_len={}",
            self.src_id,
            self.dst_id,
            self.edge_name,
            self.content.as_ref().len()
        );
        Ok(())
    }
}

impl FulltextIndexExecutor for UpdateNodeValidSinceUntil {
    fn index(&self, index_writer: &IndexWriter, fields: &FulltextFields) -> Result<()> {
        // Delete existing documents for this node ID
        let id_term = tantivy::Term::from_field_bytes(fields.id_field, self.id.as_bytes());
        index_writer.delete_term(id_term);

        log::debug!(
            "[FullText] Deleted node documents for temporal update: id={}, reason={}",
            self.id,
            self.reason
        );
        Ok(())
    }
}

impl FulltextIndexExecutor for UpdateEdgeValidSinceUntil {
    fn index(&self, index_writer: &IndexWriter, fields: &FulltextFields) -> Result<()> {
        // Delete existing documents for this edge
        // We need to delete by composite key (src_id + dst_id + edge_name)
        // Tantivy doesn't support composite term deletion directly, so we delete by src_id
        // and let the search handle filtering
        let src_term = tantivy::Term::from_field_bytes(fields.src_id_field, self.src_id.as_bytes());
        index_writer.delete_term(src_term);

        log::debug!(
            "[FullText] Deleted edge documents for temporal update: src={}, dst={}, name={}, reason={}",
            self.src_id,
            self.dst_id,
            self.name,
            self.reason
        );
        Ok(())
    }
}

impl FulltextIndexExecutor for UpdateEdgeWeight {
    fn index(&self, _index_writer: &IndexWriter, _fields: &FulltextFields) -> Result<()> {
        // For weight updates, we'd need to delete and re-index
        // For now, just log as this is primarily a graph operation
        log::debug!(
            "[FullText] Edge weight updated (no index change needed): src={}, dst={}, name={}, weight={}",
            self.src_id,
            self.dst_id,
            self.name,
            self.weight
        );
        Ok(())
    }
}
