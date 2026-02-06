//! Common utilities for vector search examples
//!
//! This module provides:
//! - Synthetic vector data generation (1024-dim, quantized to 1000 intervals)
//! - Distance functions (Euclidean, Cosine, Inner Product) with SIMD acceleration
//! - Vector storage/retrieval using NodeFragments
//! - Graph construction helpers

#![allow(dead_code)]

// Re-export from motlie-core
pub use motlie_core::distance;
pub use motlie_core::telemetry::{BuildInfo, log_build_info, print_build_info};

use anyhow::Result;
use motlie_db::mutation::{
    AddEdge, AddNode, AddNodeFragment, EdgeSummary, NodeSummary,
    Runnable as MutationRunnable, UpdateEdgeValidSinceUntil,
};
use motlie_db::query::{NodeFragments, OutgoingEdges, Runnable as QueryRunnable};
use motlie_db::reader::Reader;
use motlie_db::writer::Writer;
use motlie_db::{ValidRange, DataUrl, Id, ReadWriteHandles, Storage, StorageConfig, TimestampMilli};
use rand::Rng;
use serde_json;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::ops::Bound;
use std::path::Path;
use std::time::Duration;

// ============================================================================
// Constants
// ============================================================================

/// Vector dimensionality
pub const DIMENSIONS: usize = 1024;

/// Number of quantization intervals [0, 0.001, 0.002, ..., 0.999]
pub const QUANTIZATION_INTERVALS: u32 = 1000;

/// Default query timeout
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

// ============================================================================
// Distance Metrics (SIMD-accelerated)
// ============================================================================

// Re-export simd_level for checking the active SIMD implementation
pub use distance::simd_level;

/// Compute Euclidean (L2) distance between two vectors
///
/// Uses SIMD acceleration when available (AVX-512, AVX2+FMA, or NEON).
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    distance::euclidean(a, b)
}

/// Compute squared Euclidean distance (faster, no sqrt)
///
/// Uses SIMD acceleration when available (AVX-512, AVX2+FMA, or NEON).
#[inline]
pub fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    distance::euclidean_squared(a, b)
}

/// Compute cosine distance: 1 - cosine_similarity
///
/// Uses SIMD acceleration when available (AVX-512, AVX2+FMA, or NEON).
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    distance::cosine(a, b)
}

/// Compute negative inner product (for max inner product search)
#[inline]
pub fn negative_inner_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");
    -distance::dot(a, b)
}

/// Distance function type
pub type DistanceFn = fn(&[f32], &[f32]) -> f32;

/// Get distance function by name
pub fn get_distance_fn(name: &str) -> Result<DistanceFn> {
    match name.to_lowercase().as_str() {
        "euclidean" | "l2" => Ok(euclidean_distance),
        "euclidean_squared" | "l2_squared" => Ok(euclidean_distance_squared),
        "cosine" => Ok(cosine_distance),
        "inner_product" | "ip" => Ok(negative_inner_product),
        _ => anyhow::bail!("Unknown distance function: {}", name),
    }
}

// ============================================================================
// Synthetic Data Generation
// ============================================================================

/// Generate a single dimension value quantized to 1000 intervals
/// Returns values in [0.0, 0.999] with step 0.001
#[inline]
pub fn generate_dimension_value(rng: &mut impl Rng) -> f32 {
    let interval = rng.gen_range(0..QUANTIZATION_INTERVALS);
    interval as f32 / QUANTIZATION_INTERVALS as f32
}

/// Generate a random 1024-dimensional vector
pub fn generate_vector(rng: &mut impl Rng) -> Vec<f32> {
    (0..DIMENSIONS)
        .map(|_| generate_dimension_value(rng))
        .collect()
}

/// Generate multiple random vectors
pub fn generate_vectors(rng: &mut impl Rng, count: usize) -> Vec<Vec<f32>> {
    (0..count).map(|_| generate_vector(rng)).collect()
}

/// Serialize vector to bytes using MessagePack
pub fn serialize_vector(vector: &[f32]) -> Result<Vec<u8>> {
    rmp_serde::to_vec(vector).map_err(|e| anyhow::anyhow!("Failed to serialize vector: {}", e))
}

/// Deserialize vector from bytes
pub fn deserialize_vector(bytes: &[u8]) -> Result<Vec<f32>> {
    rmp_serde::from_slice(bytes).map_err(|e| anyhow::anyhow!("Failed to deserialize vector: {}", e))
}

// ============================================================================
// Priority Queue Helpers (for nearest neighbor search)
// ============================================================================

/// Entry in a max-heap for nearest neighbor search
/// We use negative distance for max-heap behavior (smallest distance = largest negative)
#[derive(Clone)]
pub struct DistanceEntry {
    pub distance: f32,
    pub id: Id,
}

impl PartialEq for DistanceEntry {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.id == other.id
    }
}

impl Eq for DistanceEntry {}

impl PartialOrd for DistanceEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DistanceEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // For a max-heap that keeps the k SMALLEST distances:
        // We want the LARGEST distance at the top so we can pop it.
        // So we use normal ordering (self vs other), not reversed.
        // When heap.len() > k, we pop the largest distance.
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Min-heap entry (smallest distance at top)
#[derive(Clone)]
pub struct MinDistanceEntry {
    pub distance: f32,
    pub id: Id,
}

impl PartialEq for MinDistanceEntry {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.id == other.id
    }
}

impl Eq for MinDistanceEntry {}

impl PartialOrd for MinDistanceEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinDistanceEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Normal order for min-heap
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ============================================================================
// Vector Storage Operations
// ============================================================================

/// Create a DataUrl from vector data as JSON
/// We use JSON instead of binary msgpack because the fulltext indexer
/// processes fragment content and expects UTF-8 text
pub fn vector_to_data_url(vector: &[f32]) -> Result<DataUrl> {
    // Serialize as JSON array for UTF-8 compatibility with fulltext indexer
    let json = serde_json::to_string(vector)
        .map_err(|e| anyhow::anyhow!("Failed to serialize vector to JSON: {}", e))?;
    Ok(DataUrl::from_json(&json))
}

/// Extract vector from a NodeFragment's content
pub fn extract_vector_from_fragment(content: &DataUrl) -> Result<Vec<f32>> {
    // DataUrl stores the data, we need to decode it and parse JSON
    let json_str = content.decode_string()
        .map_err(|e| anyhow::anyhow!("Failed to decode DataUrl: {}", e))?;
    serde_json::from_str(&json_str)
        .map_err(|e| anyhow::anyhow!("Failed to parse vector JSON: {}", e))
}

/// Store a vector as a node with its vector data in a fragment
pub async fn store_vector(
    writer: &Writer,
    node_id: Id,
    name: &str,
    vector: &[f32],
) -> Result<()> {
    let ts = TimestampMilli::now();

    // Create the node
    AddNode {
        id: node_id,
        ts_millis: ts,
        name: name.to_string(),
        valid_range: None,
        summary: NodeSummary::from_text(&format!("Vector node: {}", name)),
    }
    .run(writer)
    .await?;

    // Store vector data as a fragment
    let vector_data = vector_to_data_url(vector)?;
    AddNodeFragment {
        id: node_id,
        ts_millis: ts,
        content: vector_data,
        valid_range: None,
    }
    .run(writer)
    .await?;

    Ok(())
}

/// Retrieve vector data for a node
pub async fn get_vector(reader: &Reader, node_id: Id) -> Result<Vec<f32>> {
    // Use unbounded time range to get all fragments
    let time_range = (Bound::Unbounded, Bound::Unbounded);
    let fragments = NodeFragments::new(node_id, time_range, None)
        .run(reader, DEFAULT_TIMEOUT)
        .await?;

    if fragments.is_empty() {
        anyhow::bail!("No vector data found for node {}", node_id);
    }

    // Get the most recent fragment (last one, since they're sorted by timestamp)
    let (_ts, content) = fragments.last().unwrap();
    extract_vector_from_fragment(content)
}

/// Batch retrieve vectors for multiple nodes
pub async fn get_vectors_batch(
    reader: &Reader,
    node_ids: &[Id],
) -> Result<HashMap<Id, Vec<f32>>> {
    let mut result = HashMap::new();

    // TODO: Could optimize with parallel queries
    for &id in node_ids {
        match get_vector(reader, id).await {
            Ok(vector) => {
                result.insert(id, vector);
            }
            Err(e) => {
                tracing::warn!("Failed to get vector for node {}: {}", id, e);
            }
        }
    }

    Ok(result)
}

// ============================================================================
// Graph Edge Operations
// ============================================================================

/// Add an edge between two vector nodes with distance as weight
pub async fn add_vector_edge(
    writer: &Writer,
    source: Id,
    target: Id,
    edge_name: &str,
    distance: f32,
) -> Result<()> {
    AddEdge {
        source_node_id: source,
        target_node_id: target,
        ts_millis: TimestampMilli::now(),
        name: edge_name.to_string(),
        summary: EdgeSummary::from_text("vector neighbor"),
        weight: Some(distance as f64),
        valid_range: None,
    }
    .run(writer)
    .await
}

/// Prune an edge by setting its valid_until timestamp
pub async fn prune_edge(
    writer: &Writer,
    source: Id,
    target: Id,
    edge_name: &str,
) -> Result<()> {
    // ValidRange::valid_until returns Option<ValidRange>, we need to unwrap
    // since the mutation field expects ValidRange directly
    let now = TimestampMilli::now();
    UpdateEdgeValidSinceUntil {
        src_id: source,
        dst_id: target,
        name: edge_name.to_string(),
        temporal_range: ValidRange(None, Some(now)),
        reason: "pruned".to_string(),
    }
    .run(writer)
    .await
}

/// Get neighbors of a node at a specific layer (for HNSW)
pub async fn get_neighbors(
    reader: &Reader,
    node_id: Id,
    layer_prefix: Option<&str>,
) -> Result<Vec<(f64, Id)>> {
    let edges = OutgoingEdges::new(node_id, None)
        .run(reader, DEFAULT_TIMEOUT)
        .await?;

    let mut neighbors = Vec::new();
    for (weight, _src, dst, name) in edges {
        // Filter by layer prefix if specified
        if let Some(prefix) = layer_prefix {
            if !name.starts_with(prefix) {
                continue;
            }
        }

        let distance = weight.unwrap_or(f64::MAX);
        neighbors.push((distance, dst));
    }

    // Sort by distance
    neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(neighbors)
}

// ============================================================================
// Vector Index Base Structure
// ============================================================================

/// In-memory vector cache for fast distance computations
pub struct VectorCache {
    pub vectors: HashMap<Id, Vec<f32>>,
}

impl VectorCache {
    pub fn new() -> Self {
        Self {
            vectors: HashMap::new(),
        }
    }

    pub fn insert(&mut self, id: Id, vector: Vec<f32>) {
        self.vectors.insert(id, vector);
    }

    pub fn get(&self, id: &Id) -> Option<&Vec<f32>> {
        self.vectors.get(id)
    }

    pub fn contains(&self, id: &Id) -> bool {
        self.vectors.contains_key(id)
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Compute distance between a query and a cached vector
    pub fn distance_to(&self, query: &[f32], id: &Id, distance_fn: DistanceFn) -> Option<f32> {
        self.vectors.get(id).map(|v| distance_fn(query, v))
    }
}

impl Default for VectorCache {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Storage Initialization
// ============================================================================

/// Initialize storage for vector index
pub fn init_storage(db_path: &Path) -> Result<ReadWriteHandles> {
    // Clean up existing database
    if db_path.exists() {
        std::fs::remove_dir_all(db_path)?;
    }

    let storage = Storage::readwrite(db_path);
    storage.ready(StorageConfig::default())
}

// ============================================================================
// Metrics and Benchmarking
// ============================================================================

/// Compute recall@k between retrieved and ground truth neighbors
pub fn compute_recall(retrieved: &[Id], ground_truth: &[Id], k: usize) -> f64 {
    let k = k.min(ground_truth.len()).min(retrieved.len());
    if k == 0 {
        return 0.0;
    }

    let gt_set: HashSet<_> = ground_truth.iter().take(k).collect();
    let retrieved_set: HashSet<_> = retrieved.iter().take(k).collect();

    let intersection = gt_set.intersection(&retrieved_set).count();
    intersection as f64 / k as f64
}

/// Brute-force k-NN search (ground truth)
pub fn brute_force_knn(
    query: &[f32],
    vectors: &HashMap<Id, Vec<f32>>,
    k: usize,
    distance_fn: DistanceFn,
) -> Vec<(f32, Id)> {
    let mut heap: BinaryHeap<DistanceEntry> = BinaryHeap::new();

    for (id, vector) in vectors {
        let distance = distance_fn(query, vector);
        heap.push(DistanceEntry { distance, id: *id });

        // Keep only top k
        if heap.len() > k {
            heap.pop();
        }
    }

    // Convert to sorted vec (smallest distance first)
    let mut result: Vec<_> = heap
        .into_iter()
        .map(|e| (e.distance, e.id))
        .collect();
    result.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    result
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_generate_dimension_value() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        for _ in 0..1000 {
            let value = generate_dimension_value(&mut rng);
            assert!(value >= 0.0 && value < 1.0);
            // Check quantization: value * 1000 should be approximately an integer
            // Use a larger tolerance due to floating point precision
            let scaled = value * QUANTIZATION_INTERVALS as f32;
            assert!(
                (scaled - scaled.round()).abs() < 1e-4,
                "scaled={} round={} diff={}",
                scaled,
                scaled.round(),
                (scaled - scaled.round()).abs()
            );
        }
    }

    #[test]
    fn test_generate_vector() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let vector = generate_vector(&mut rng);

        assert_eq!(vector.len(), DIMENSIONS);
        for &v in &vector {
            assert!(v >= 0.0 && v < 1.0);
        }
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((euclidean_distance(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![1.0, 1.0, 1.0];
        let expected = (3.0_f32).sqrt();
        assert!((euclidean_distance(&a, &c) - expected).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        assert!(cosine_distance(&a, &b).abs() < 1e-6); // Same direction = 0 distance

        let c = vec![0.0, 1.0];
        assert!((cosine_distance(&a, &c) - 1.0).abs() < 1e-6); // Orthogonal = 1 distance
    }

    #[test]
    fn test_serialize_deserialize_vector() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let original = generate_vector(&mut rng);

        let serialized = serialize_vector(&original).unwrap();
        let deserialized = deserialize_vector(&serialized).unwrap();

        assert_eq!(original.len(), deserialized.len());
        for (a, b) in original.iter().zip(deserialized.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_distance_entry_ordering() {
        let mut heap = BinaryHeap::new();
        heap.push(DistanceEntry {
            distance: 3.0,
            id: Id::new(),
        });
        heap.push(DistanceEntry {
            distance: 1.0,
            id: Id::new(),
        });
        heap.push(DistanceEntry {
            distance: 2.0,
            id: Id::new(),
        });

        // Should pop in order: 3.0, 2.0, 1.0 (max first for max-heap with reversed Ord)
        assert_eq!(heap.pop().unwrap().distance, 3.0);
        assert_eq!(heap.pop().unwrap().distance, 2.0);
        assert_eq!(heap.pop().unwrap().distance, 1.0);
    }

    #[test]
    fn test_compute_recall() {
        let id1 = Id::new();
        let id2 = Id::new();
        let id3 = Id::new();
        let id4 = Id::new();

        let retrieved = vec![id1, id2, id3];
        let ground_truth = vec![id1, id2, id4];

        // 2 out of 3 match
        let recall = compute_recall(&retrieved, &ground_truth, 3);
        assert!((recall - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_brute_force_knn() {
        let mut vectors = HashMap::new();
        let query = vec![0.5; DIMENSIONS];

        // Create some test vectors
        let id1 = Id::new();
        let id2 = Id::new();
        let id3 = Id::new();

        vectors.insert(id1, vec![0.5; DIMENSIONS]); // Distance 0
        vectors.insert(id2, vec![0.6; DIMENSIONS]); // Small distance
        vectors.insert(id3, vec![0.0; DIMENSIONS]); // Larger distance

        let result = brute_force_knn(&query, &vectors, 2, euclidean_distance);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].1, id1); // Closest
        assert!(result[0].0 < result[1].0); // Sorted by distance
    }
}
