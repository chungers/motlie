//! Scan API for iterating over vector column families with pagination support.
//!
//! This module provides a visitor-based API for scanning vector column families
//! without exposing internal schema types, mirroring the `graph::scan` pattern.
//!
//! # Example
//!
//! ```ignore
//! use motlie_db::vector::scan::{AllEmbeddingSpecs, Visitable};
//!
//! let mut storage = vector::Storage::readonly(db_path);
//! storage.ready()?;
//!
//! let scan = AllEmbeddingSpecs { last: None, limit: 100, reverse: false };
//! scan.accept(&storage, &mut |record| {
//!     println!("{}\t{}\t{}", record.code, record.model, record.dim);
//!     true // continue scanning
//! })?;
//! ```

use anyhow::Result;
use rocksdb::{Direction, IteratorMode};

use super::schema::{
    BinaryCodeCfKey, BinaryCodes, EmbeddingCode, EmbeddingSpecCfKey, EmbeddingSpecs,
    Edges, EdgeCfKey,
    ExternalKey, GraphMeta, GraphMetaField,
    IdAlloc, IdAllocField,
    IdForward,
    IdReverse, IdReverseCfKey,
    LifecycleCounts, LifecycleCountsCfKey,
    Pending, PendingCfKey,
    VecId, VecMeta, VecMetaCfKey,
    Vectors, VectorCfKey,
};
use super::Storage;
use crate::rocksdb::ColumnFamily;

// ============================================================================
// Visitor Trait
// ============================================================================

/// Visitor trait for processing scanned records.
///
/// Return `true` to continue scanning, `false` to stop early.
pub trait Visitor<R> {
    fn visit(&mut self, record: &R) -> bool;
}

/// Blanket implementation for closures.
impl<R, F> Visitor<R> for F
where
    F: FnMut(&R) -> bool,
{
    fn visit(&mut self, record: &R) -> bool {
        self(record)
    }
}

// ============================================================================
// Visitable Trait
// ============================================================================

/// Trait for scan types that can accept visitors.
pub trait Visitable {
    type Record;

    fn accept<V: Visitor<Self::Record>>(
        &self,
        storage: &Storage,
        visitor: &mut V,
    ) -> Result<usize>;
}

// ============================================================================
// Public Record Types
// ============================================================================

/// Embedding specification record.
#[derive(Debug, Clone)]
pub struct EmbeddingSpecRecord {
    pub code: u64,
    pub model: String,
    pub dim: u32,
    pub distance: String,
    pub storage_type: String,
    pub hnsw_m: u16,
    pub hnsw_ef_construction: u16,
    pub rabitq_bits: u8,
}

/// Vector record (summary, not full data).
#[derive(Debug, Clone)]
pub struct VectorRecord {
    pub embedding_code: u64,
    pub vec_id: u32,
    pub dim: usize,
    pub byte_size: usize,
}

/// HNSW edge record.
#[derive(Debug, Clone)]
pub struct HnswEdgeRecord {
    pub embedding_code: u64,
    pub vec_id: u32,
    pub layer: u8,
    pub neighbor_bytes: usize,
}

/// Binary code record.
#[derive(Debug, Clone)]
pub struct BinaryCodeRecord {
    pub embedding_code: u64,
    pub vec_id: u32,
    pub code_len: usize,
    pub vector_norm: f32,
    pub quantization_error: f32,
}

/// Vector metadata record.
#[derive(Debug, Clone)]
pub struct VecMetaRecord {
    pub embedding_code: u64,
    pub vec_id: u32,
    pub max_layer: u8,
    pub lifecycle: String,
    pub created_at: u64,
}

/// Vector graph-level metadata record.
#[derive(Debug, Clone)]
pub struct VecGraphMetaRecord {
    pub embedding_code: u64,
    pub field: String,
    pub value: String,
}

/// Forward ID mapping record.
#[derive(Debug, Clone)]
pub struct IdForwardRecord {
    pub embedding_code: u64,
    pub external_key_type: String,
    pub external_key: String,
    pub vec_id: u32,
}

/// Reverse ID mapping record.
#[derive(Debug, Clone)]
pub struct IdReverseRecord {
    pub embedding_code: u64,
    pub vec_id: u32,
    pub external_key_type: String,
    pub external_key: String,
}

/// ID allocator record.
#[derive(Debug, Clone)]
pub struct IdAllocRecord {
    pub embedding_code: u64,
    pub field: String,
    pub value: String,
}

/// Pending queue record.
#[derive(Debug, Clone)]
pub struct PendingRecord {
    pub embedding_code: u64,
    pub timestamp: u64,
    pub vec_id: u32,
}

/// Lifecycle counts record.
#[derive(Debug, Clone)]
pub struct LifecycleCountsRecord {
    pub embedding_code: u64,
    pub indexed: u64,
    pub pending: u64,
    pub deleted: u64,
    pub pending_deleted: u64,
}

// ============================================================================
// Scan Types
// ============================================================================

/// Scan all embedding specifications.
#[derive(Debug, Clone, Default)]
pub struct AllEmbeddingSpecs {
    pub last: Option<EmbeddingCode>,
    pub limit: usize,
    pub reverse: bool,
}

/// Scan all vectors.
#[derive(Debug, Clone, Default)]
pub struct AllVectors {
    pub last: Option<(EmbeddingCode, VecId)>,
    pub limit: usize,
    pub reverse: bool,
}

/// Scan all HNSW edges.
#[derive(Debug, Clone, Default)]
pub struct AllHnswEdges {
    pub last: Option<(EmbeddingCode, VecId, u8)>,
    pub limit: usize,
    pub reverse: bool,
}

/// Scan all binary codes.
#[derive(Debug, Clone, Default)]
pub struct AllBinaryCodes {
    pub last: Option<(EmbeddingCode, VecId)>,
    pub limit: usize,
    pub reverse: bool,
}

/// Scan all vector metadata.
#[derive(Debug, Clone, Default)]
pub struct AllVecMeta {
    pub last: Option<(EmbeddingCode, VecId)>,
    pub limit: usize,
    pub reverse: bool,
}

/// Scan all vector graph-level metadata.
#[derive(Debug, Clone, Default)]
pub struct AllVecGraphMeta {
    pub last: Option<EmbeddingCode>,
    pub limit: usize,
    pub reverse: bool,
}

/// Scan all forward ID mappings.
#[derive(Debug, Clone, Default)]
pub struct AllIdForward {
    pub limit: usize,
    pub reverse: bool,
}

/// Scan all reverse ID mappings.
#[derive(Debug, Clone, Default)]
pub struct AllIdReverse {
    pub last: Option<(EmbeddingCode, VecId)>,
    pub limit: usize,
    pub reverse: bool,
}

/// Scan all ID allocator state.
#[derive(Debug, Clone, Default)]
pub struct AllIdAlloc {
    pub last: Option<EmbeddingCode>,
    pub limit: usize,
    pub reverse: bool,
}

/// Scan all pending queue entries.
#[derive(Debug, Clone, Default)]
pub struct AllPending {
    pub last: Option<(EmbeddingCode, u64, VecId)>,
    pub limit: usize,
    pub reverse: bool,
}

/// Scan all lifecycle counts.
#[derive(Debug, Clone, Default)]
pub struct AllLifecycleCounts {
    pub last: Option<EmbeddingCode>,
    pub limit: usize,
    pub reverse: bool,
}

// ============================================================================
// Internal Helpers
// ============================================================================

/// Format an ExternalKey for display.
fn format_external_key(key: &ExternalKey) -> String {
    match key {
        ExternalKey::NodeId(id) => format!("{}", id),
        ExternalKey::NodeFragment(id, ts) => format!("{}:{}", id, ts.0),
        ExternalKey::Edge(src, dst, name_hash) => {
            format!("{}:{}:{}", src, dst, bytes_to_hex(name_hash.as_bytes()))
        }
        ExternalKey::EdgeFragment(src, dst, name_hash, ts) => {
            format!(
                "{}:{}:{}:{}",
                src,
                dst,
                bytes_to_hex(name_hash.as_bytes()),
                ts.0
            )
        }
        ExternalKey::NodeSummary(hash) => bytes_to_hex(hash.as_bytes()),
        ExternalKey::EdgeSummary(hash) => bytes_to_hex(hash.as_bytes()),
    }
}

/// Convert bytes to hex string.
fn bytes_to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Internal helper to run iteration over a column family.
/// Simplified from graph::scan (no temporal filtering).
fn iterate_and_visit<CF, R, V, F>(
    storage: &Storage,
    seek_key: Vec<u8>,
    limit: usize,
    reverse: bool,
    cursor_matches: impl Fn(&[u8]) -> bool,
    transform: F,
    visitor: &mut V,
) -> Result<usize>
where
    CF: ColumnFamily,
    V: Visitor<R>,
    F: Fn(&[u8], &[u8]) -> Result<R>,
{
    let direction = if reverse {
        Direction::Reverse
    } else {
        Direction::Forward
    };

    let mode = if seek_key.is_empty() {
        if reverse {
            IteratorMode::End
        } else {
            IteratorMode::Start
        }
    } else {
        IteratorMode::From(&seek_key, direction)
    };

    let mut count = 0;

    if let Ok(db) = storage.db() {
        let cf = db.cf_handle(CF::CF_NAME).ok_or_else(|| {
            anyhow::anyhow!("Column family '{}' not found", CF::CF_NAME)
        })?;

        for item in db.iterator_cf(cf, mode) {
            if count >= limit {
                break;
            }
            let (key_bytes, value_bytes) = item?;
            if cursor_matches(&key_bytes) {
                continue;
            }
            let record = match transform(&key_bytes, &value_bytes) {
                Ok(r) => r,
                Err(e) => {
                    let msg = e.to_string();
                    if msg.starts_with("skip_") {
                        continue;
                    }
                    return Err(e);
                }
            };
            count += 1;
            if !visitor.visit(&record) {
                break;
            }
        }
    } else {
        let txn_db = storage.transaction_db()?;
        let cf = txn_db.cf_handle(CF::CF_NAME).ok_or_else(|| {
            anyhow::anyhow!("Column family '{}' not found", CF::CF_NAME)
        })?;

        for item in txn_db.iterator_cf(cf, mode) {
            if count >= limit {
                break;
            }
            let (key_bytes, value_bytes) = item?;
            if cursor_matches(&key_bytes) {
                continue;
            }
            let record = match transform(&key_bytes, &value_bytes) {
                Ok(r) => r,
                Err(e) => {
                    let msg = e.to_string();
                    if msg.starts_with("skip_") {
                        continue;
                    }
                    return Err(e);
                }
            };
            count += 1;
            if !visitor.visit(&record) {
                break;
            }
        }
    }

    Ok(count)
}

// ============================================================================
// Visitable Implementations
// ============================================================================

impl Visitable for AllEmbeddingSpecs {
    type Record = EmbeddingSpecRecord;

    fn accept<V: Visitor<Self::Record>>(
        &self,
        storage: &Storage,
        visitor: &mut V,
    ) -> Result<usize> {
        use crate::rocksdb::ColumnFamilySerde;

        let seek_key = self
            .last
            .map(|code| EmbeddingSpecs::key_to_bytes(&EmbeddingSpecCfKey(code)))
            .unwrap_or_default();

        let cursor_key = self
            .last
            .map(|code| EmbeddingSpecs::key_to_bytes(&EmbeddingSpecCfKey(code)));

        iterate_and_visit::<EmbeddingSpecs, _, _, _>(
            storage,
            seek_key,
            self.limit,
            self.reverse,
            |key_bytes| cursor_key.as_ref().map_or(false, |ck| key_bytes == ck.as_slice()),
            |key_bytes, value_bytes| {
                let key = EmbeddingSpecs::key_from_bytes(key_bytes)?;
                let value = EmbeddingSpecs::value_from_bytes(value_bytes)?;
                let spec = value.0;
                Ok(EmbeddingSpecRecord {
                    code: key.0,
                    model: spec.model,
                    dim: spec.dim,
                    distance: format!("{:?}", spec.distance),
                    storage_type: format!("{:?}", spec.storage_type),
                    hnsw_m: spec.hnsw_m,
                    hnsw_ef_construction: spec.hnsw_ef_construction,
                    rabitq_bits: spec.rabitq_bits,
                })
            },
            visitor,
        )
    }
}

impl Visitable for AllVectors {
    type Record = VectorRecord;

    fn accept<V: Visitor<Self::Record>>(
        &self,
        storage: &Storage,
        visitor: &mut V,
    ) -> Result<usize> {
        let seek_key = self
            .last
            .map(|(ec, vid)| Vectors::key_to_bytes(&VectorCfKey(ec, vid)))
            .unwrap_or_default();

        let cursor_key = self
            .last
            .map(|(ec, vid)| Vectors::key_to_bytes(&VectorCfKey(ec, vid)));

        iterate_and_visit::<Vectors, _, _, _>(
            storage,
            seek_key,
            self.limit,
            self.reverse,
            |key_bytes| cursor_key.as_ref().map_or(false, |ck| key_bytes == ck.as_slice()),
            |key_bytes, value_bytes| {
                let key = Vectors::key_from_bytes(key_bytes)?;
                // Don't deserialize full vector — just report dimensions
                let byte_size = value_bytes.len();
                let dim = byte_size / 4; // Assume f32 by default
                Ok(VectorRecord {
                    embedding_code: key.0,
                    vec_id: key.1,
                    dim,
                    byte_size,
                })
            },
            visitor,
        )
    }
}

impl Visitable for AllHnswEdges {
    type Record = HnswEdgeRecord;

    fn accept<V: Visitor<Self::Record>>(
        &self,
        storage: &Storage,
        visitor: &mut V,
    ) -> Result<usize> {
        let seek_key = self
            .last
            .map(|(ec, vid, layer)| Edges::key_to_bytes(&EdgeCfKey(ec, vid, layer)))
            .unwrap_or_default();

        let cursor_key = self
            .last
            .map(|(ec, vid, layer)| Edges::key_to_bytes(&EdgeCfKey(ec, vid, layer)));

        iterate_and_visit::<Edges, _, _, _>(
            storage,
            seek_key,
            self.limit,
            self.reverse,
            |key_bytes| cursor_key.as_ref().map_or(false, |ck| key_bytes == ck.as_slice()),
            |key_bytes, value_bytes| {
                let key = Edges::key_from_bytes(key_bytes)?;
                Ok(HnswEdgeRecord {
                    embedding_code: key.0,
                    vec_id: key.1,
                    layer: key.2,
                    neighbor_bytes: value_bytes.len(),
                })
            },
            visitor,
        )
    }
}

impl Visitable for AllBinaryCodes {
    type Record = BinaryCodeRecord;

    fn accept<V: Visitor<Self::Record>>(
        &self,
        storage: &Storage,
        visitor: &mut V,
    ) -> Result<usize> {
        let seek_key = self
            .last
            .map(|(ec, vid)| BinaryCodes::key_to_bytes(&BinaryCodeCfKey(ec, vid)))
            .unwrap_or_default();

        let cursor_key = self
            .last
            .map(|(ec, vid)| BinaryCodes::key_to_bytes(&BinaryCodeCfKey(ec, vid)));

        iterate_and_visit::<BinaryCodes, _, _, _>(
            storage,
            seek_key,
            self.limit,
            self.reverse,
            |key_bytes| cursor_key.as_ref().map_or(false, |ck| key_bytes == ck.as_slice()),
            |key_bytes, value_bytes| {
                let key = BinaryCodes::key_from_bytes(key_bytes)?;
                let value = BinaryCodes::value_from_bytes(value_bytes)?;
                Ok(BinaryCodeRecord {
                    embedding_code: key.0,
                    vec_id: key.1,
                    code_len: value.code.len(),
                    vector_norm: value.correction.vector_norm,
                    quantization_error: value.correction.quantization_error,
                })
            },
            visitor,
        )
    }
}

impl Visitable for AllVecMeta {
    type Record = VecMetaRecord;

    fn accept<V: Visitor<Self::Record>>(
        &self,
        storage: &Storage,
        visitor: &mut V,
    ) -> Result<usize> {
        use crate::rocksdb::HotColumnFamilyRecord;

        let seek_key = self
            .last
            .map(|(ec, vid)| VecMeta::key_to_bytes(&VecMetaCfKey(ec, vid)))
            .unwrap_or_default();

        let cursor_key = self
            .last
            .map(|(ec, vid)| VecMeta::key_to_bytes(&VecMetaCfKey(ec, vid)));

        iterate_and_visit::<VecMeta, _, _, _>(
            storage,
            seek_key,
            self.limit,
            self.reverse,
            |key_bytes| cursor_key.as_ref().map_or(false, |ck| key_bytes == ck.as_slice()),
            |key_bytes, value_bytes| {
                let key = VecMeta::key_from_bytes(key_bytes)?;
                // Use rkyv zero-copy deserialization
                let value = VecMeta::value_from_bytes(value_bytes)?;
                let meta = value.0;
                let lifecycle = format!("{:?}", meta.lifecycle());
                Ok(VecMetaRecord {
                    embedding_code: key.0,
                    vec_id: key.1,
                    max_layer: meta.max_layer,
                    lifecycle,
                    created_at: meta.created_at,
                })
            },
            visitor,
        )
    }
}

impl Visitable for AllVecGraphMeta {
    type Record = VecGraphMetaRecord;

    fn accept<V: Visitor<Self::Record>>(
        &self,
        storage: &Storage,
        visitor: &mut V,
    ) -> Result<usize> {
        // GraphMeta keys are 9 bytes: [embedding: u64][discriminant: u8]
        // Seek by embedding_code prefix
        let seek_key = self
            .last
            .map(|ec| ec.to_be_bytes().to_vec())
            .unwrap_or_default();

        let last_ec = self.last;

        iterate_and_visit::<GraphMeta, _, _, _>(
            storage,
            seek_key,
            self.limit,
            self.reverse,
            |_key_bytes| {
                // Skip all entries from the cursor embedding_code
                if let Some(ec) = last_ec {
                    _key_bytes.len() >= 8 && _key_bytes[..8] == ec.to_be_bytes()
                } else {
                    false
                }
            },
            |key_bytes, value_bytes| {
                let key = GraphMeta::key_from_bytes(key_bytes)?;
                let value = GraphMeta::value_from_bytes(&key, value_bytes)?;
                let (field_name, value_str) = match &value.0 {
                    GraphMetaField::EntryPoint(v) => ("EntryPoint".to_string(), v.to_string()),
                    GraphMetaField::MaxLevel(v) => ("MaxLevel".to_string(), v.to_string()),
                    GraphMetaField::Count(v) => ("Count".to_string(), v.to_string()),
                    GraphMetaField::SpecHash(v) => ("SpecHash".to_string(), format!("{:016x}", v)),
                };
                Ok(VecGraphMetaRecord {
                    embedding_code: key.0,
                    field: field_name,
                    value: value_str,
                })
            },
            visitor,
        )
    }
}

impl Visitable for AllIdForward {
    type Record = IdForwardRecord;

    fn accept<V: Visitor<Self::Record>>(
        &self,
        storage: &Storage,
        visitor: &mut V,
    ) -> Result<usize> {
        // IdForward has variable-length keys, no simple cursor
        iterate_and_visit::<IdForward, _, _, _>(
            storage,
            Vec::new(),
            self.limit,
            self.reverse,
            |_| false,
            |key_bytes, value_bytes| {
                let key = IdForward::key_from_bytes(key_bytes)?;
                let value = IdForward::value_from_bytes(value_bytes)?;
                Ok(IdForwardRecord {
                    embedding_code: key.0,
                    external_key_type: key.1.variant_name().to_string(),
                    external_key: format_external_key(&key.1),
                    vec_id: value.0,
                })
            },
            visitor,
        )
    }
}

impl Visitable for AllIdReverse {
    type Record = IdReverseRecord;

    fn accept<V: Visitor<Self::Record>>(
        &self,
        storage: &Storage,
        visitor: &mut V,
    ) -> Result<usize> {
        let seek_key = self
            .last
            .map(|(ec, vid)| IdReverse::key_to_bytes(&IdReverseCfKey(ec, vid)))
            .unwrap_or_default();

        let cursor_key = self
            .last
            .map(|(ec, vid)| IdReverse::key_to_bytes(&IdReverseCfKey(ec, vid)));

        iterate_and_visit::<IdReverse, _, _, _>(
            storage,
            seek_key,
            self.limit,
            self.reverse,
            |key_bytes| cursor_key.as_ref().map_or(false, |ck| key_bytes == ck.as_slice()),
            |key_bytes, value_bytes| {
                let key = IdReverse::key_from_bytes(key_bytes)?;
                let value = IdReverse::value_from_bytes(value_bytes)?;
                Ok(IdReverseRecord {
                    embedding_code: key.0,
                    vec_id: key.1,
                    external_key_type: value.0.variant_name().to_string(),
                    external_key: format_external_key(&value.0),
                })
            },
            visitor,
        )
    }
}

impl Visitable for AllIdAlloc {
    type Record = IdAllocRecord;

    fn accept<V: Visitor<Self::Record>>(
        &self,
        storage: &Storage,
        visitor: &mut V,
    ) -> Result<usize> {
        let seek_key = self
            .last
            .map(|ec| ec.to_be_bytes().to_vec())
            .unwrap_or_default();

        let last_ec = self.last;

        iterate_and_visit::<IdAlloc, _, _, _>(
            storage,
            seek_key,
            self.limit,
            self.reverse,
            |key_bytes| {
                if let Some(ec) = last_ec {
                    key_bytes.len() >= 8 && key_bytes[..8] == ec.to_be_bytes()
                } else {
                    false
                }
            },
            |key_bytes, value_bytes| {
                let key = IdAlloc::key_from_bytes(key_bytes)?;
                let value = IdAlloc::value_from_bytes(&key, value_bytes)?;
                let (field_name, value_str) = match &value.0 {
                    IdAllocField::NextId(v) => ("NextId".to_string(), v.to_string()),
                    IdAllocField::FreeBitmap(v) => {
                        ("FreeBitmap".to_string(), format!("{} bytes", v.len()))
                    }
                };
                Ok(IdAllocRecord {
                    embedding_code: key.0,
                    field: field_name,
                    value: value_str,
                })
            },
            visitor,
        )
    }
}

impl Visitable for AllPending {
    type Record = PendingRecord;

    fn accept<V: Visitor<Self::Record>>(
        &self,
        storage: &Storage,
        visitor: &mut V,
    ) -> Result<usize> {
        use crate::TimestampMilli;

        let seek_key = self
            .last
            .map(|(ec, ts, vid)| {
                Pending::key_to_bytes(&PendingCfKey(ec, TimestampMilli(ts), vid))
            })
            .unwrap_or_default();

        let cursor_key = self.last.map(|(ec, ts, vid)| {
            Pending::key_to_bytes(&PendingCfKey(ec, TimestampMilli(ts), vid))
        });

        iterate_and_visit::<Pending, _, _, _>(
            storage,
            seek_key,
            self.limit,
            self.reverse,
            |key_bytes| cursor_key.as_ref().map_or(false, |ck| key_bytes == ck.as_slice()),
            |key_bytes, _value_bytes| {
                let key = Pending::key_from_bytes(key_bytes)?;
                Ok(PendingRecord {
                    embedding_code: key.0,
                    timestamp: key.1 .0,
                    vec_id: key.2,
                })
            },
            visitor,
        )
    }
}

impl Visitable for AllLifecycleCounts {
    type Record = LifecycleCountsRecord;

    fn accept<V: Visitor<Self::Record>>(
        &self,
        storage: &Storage,
        visitor: &mut V,
    ) -> Result<usize> {
        let seek_key = self
            .last
            .map(|ec| LifecycleCounts::key_to_bytes(&LifecycleCountsCfKey(ec)))
            .unwrap_or_default();

        let cursor_key = self
            .last
            .map(|ec| LifecycleCounts::key_to_bytes(&LifecycleCountsCfKey(ec)));

        iterate_and_visit::<LifecycleCounts, _, _, _>(
            storage,
            seek_key,
            self.limit,
            self.reverse,
            |key_bytes| cursor_key.as_ref().map_or(false, |ck| key_bytes == ck.as_slice()),
            |key_bytes, value_bytes| {
                let key = LifecycleCounts::key_from_bytes(key_bytes)?;
                let value = LifecycleCounts::value_from_bytes(value_bytes)?;
                let counts = value.0;
                Ok(LifecycleCountsRecord {
                    embedding_code: key.0,
                    indexed: counts.indexed,
                    pending: counts.pending,
                    deleted: counts.deleted,
                    pending_deleted: counts.pending_deleted,
                })
            },
            visitor,
        )
    }
}
