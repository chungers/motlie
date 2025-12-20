//! HNSW (Hierarchical Navigable Small World) Implementation
//!
//! This module implements the HNSW algorithm for approximate nearest neighbor search
//! using motlie_db as the underlying graph storage.
//!
//! # Key Design Decisions
//!
//! - **Layers as edge names**: Edges are named "hnsw_L0", "hnsw_L1", etc.
//! - **Distance as edge weight**: Edge weights store the distance between vectors
//! - **Temporal pruning**: Pruned edges use valid_until for soft deletion
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --example hnsw <db_path> <num_vectors> <num_queries> <k>
//! ```

#[path = "common.rs"]
mod common;

use anyhow::Result;
use common::{
    add_vector_edge, brute_force_knn, compute_recall, euclidean_distance, generate_vector,
    get_neighbors, get_vector, init_storage, prune_edge, store_vector, DistanceFn, VectorCache,
    DEFAULT_TIMEOUT,
};
use motlie_db::query::{AllNodes, Runnable as QueryRunnable};
use motlie_db::reader::Reader;
use motlie_db::writer::Writer;
use motlie_db::Id;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::env;
use std::path::Path;
use std::time::Instant;

// ============================================================================
// HNSW Parameters
// ============================================================================

/// HNSW index parameters
#[derive(Clone, Debug)]
pub struct HnswParams {
    /// Max connections per node per layer (except layer 0)
    pub m: usize,
    /// Max connections in layer 0
    pub m_max0: usize,
    /// Search width during construction
    pub ef_construction: usize,
    /// Level multiplier: 1/ln(M)
    pub ml: f64,
}

impl Default for HnswParams {
    fn default() -> Self {
        let m = 16;
        Self {
            m,
            m_max0: m * 2,
            ef_construction: 200,
            ml: 1.0 / (m as f64).ln(),
        }
    }
}

impl HnswParams {
    /// Generate random level for a new node
    pub fn random_level(&self, rng: &mut impl Rng) -> usize {
        let r: f64 = rng.gen();
        (-r.ln() * self.ml).floor() as usize
    }

    /// Get max connections for a layer
    pub fn max_connections(&self, layer: usize) -> usize {
        if layer == 0 {
            self.m_max0
        } else {
            self.m
        }
    }

    /// Format layer name for edges
    pub fn layer_edge_name(layer: usize) -> String {
        format!("hnsw_L{}", layer)
    }

    /// Parse layer from edge name
    pub fn parse_layer(edge_name: &str) -> Option<usize> {
        if edge_name.starts_with("hnsw_L") {
            edge_name[6..].parse().ok()
        } else {
            None
        }
    }
}

// ============================================================================
// Priority Queue Entry for Search
// ============================================================================

/// Entry for search candidate queue
#[derive(Clone)]
struct SearchCandidate {
    distance: f32,
    id: Id,
}

impl PartialEq for SearchCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for SearchCandidate {}

impl PartialOrd for SearchCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Min-heap: smallest distance first
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ============================================================================
// HNSW Index
// ============================================================================

/// HNSW index built on motlie_db
pub struct HnswIndex {
    /// Index parameters
    pub params: HnswParams,
    /// Entry point node ID (highest level node)
    pub entry_point: Option<Id>,
    /// Maximum level in the index
    pub max_level: usize,
    /// In-memory vector cache for fast distance computation
    pub cache: VectorCache,
    /// Node levels (which layers each node belongs to)
    pub node_levels: HashMap<Id, usize>,
    /// Distance function
    pub distance_fn: DistanceFn,
}

impl HnswIndex {
    /// Create a new HNSW index
    pub fn new(params: HnswParams) -> Self {
        Self {
            params,
            entry_point: None,
            max_level: 0,
            cache: VectorCache::new(),
            node_levels: HashMap::new(),
            distance_fn: euclidean_distance,
        }
    }

    /// Insert a vector into the index
    pub async fn insert(
        &mut self,
        writer: &Writer,
        reader: &Reader,
        node_id: Id,
        vector: Vec<f32>,
        rng: &mut impl Rng,
    ) -> Result<()> {
        // Determine the level for this node
        let level = self.params.random_level(rng);

        // Cache the vector
        self.cache.insert(node_id, vector.clone());
        self.node_levels.insert(node_id, level);

        // Store the vector in the database
        let name = format!("vec_{}", self.cache.len() - 1);
        store_vector(writer, node_id, &name, &vector).await?;

        // If this is the first node, set it as entry point
        if self.entry_point.is_none() {
            self.entry_point = Some(node_id);
            self.max_level = level;
            return Ok(());
        }

        let entry_point = self.entry_point.unwrap();

        // Phase 1: Greedy search from top to level+1
        let mut current_best = entry_point;

        for l in (level + 1..=self.max_level).rev() {
            let (best, _dist) = self
                .greedy_search_layer(reader, &vector, current_best, l)
                .await?;
            current_best = best;
        }

        // Phase 2: Insert at each layer from level down to 0
        for l in (0..=level.min(self.max_level)).rev() {
            // Find ef_construction nearest neighbors at this layer
            let neighbors = self
                .search_layer(
                    reader,
                    &vector,
                    vec![current_best],
                    self.params.ef_construction,
                    l,
                )
                .await?;

            // Select M best neighbors
            let m_max = self.params.max_connections(l);
            let selected = self.select_neighbors(&vector, neighbors, m_max);

            // Add bidirectional edges
            let edge_name = HnswParams::layer_edge_name(l);
            for (dist, neighbor_id) in &selected {
                // Forward edge: node -> neighbor
                add_vector_edge(writer, node_id, *neighbor_id, &edge_name, *dist).await?;

                // Reverse edge: neighbor -> node
                add_vector_edge(writer, *neighbor_id, node_id, &edge_name, *dist).await?;

                // Check if neighbor needs pruning
                self.prune_connections(writer, reader, *neighbor_id, l)
                    .await?;
            }

            // Update current_best for next layer
            if !selected.is_empty() {
                current_best = selected[0].1;
            }
        }

        // Update entry point if this node has higher level
        if level > self.max_level {
            self.entry_point = Some(node_id);
            self.max_level = level;
        }

        Ok(())
    }

    /// Greedy search within a single layer (finds single best)
    /// Uses async distance computation with DB fallback for robustness
    async fn greedy_search_layer(
        &mut self,
        reader: &Reader,
        query: &[f32],
        start: Id,
        layer: usize,
    ) -> Result<(Id, f32)> {
        let mut current = start;
        let mut current_dist = self.compute_distance_async(reader, query, &current).await;

        loop {
            let layer_prefix = HnswParams::layer_edge_name(layer);
            let neighbors = get_neighbors(reader, current, Some(&layer_prefix)).await?;

            let mut improved = false;
            for (_weight, neighbor_id) in neighbors {
                let dist = self.compute_distance_async(reader, query, &neighbor_id).await;
                if dist < current_dist {
                    current = neighbor_id;
                    current_dist = dist;
                    improved = true;
                    break;
                }
            }

            if !improved {
                break;
            }
        }

        Ok((current, current_dist))
    }

    /// Search within a layer with ef candidates
    /// Uses async distance computation with DB fallback for robustness
    async fn search_layer(
        &mut self,
        reader: &Reader,
        query: &[f32],
        entry_points: Vec<Id>,
        ef: usize,
        layer: usize,
    ) -> Result<Vec<(f32, Id)>> {
        let mut visited: HashSet<Id> = HashSet::new();
        let mut candidates: BinaryHeap<SearchCandidate> = BinaryHeap::new();
        let mut results: Vec<(f32, Id)> = Vec::new();

        // Initialize with entry points
        for ep in entry_points {
            let dist = self.compute_distance_async(reader, query, &ep).await;
            visited.insert(ep);
            candidates.push(SearchCandidate {
                distance: dist,
                id: ep,
            });
            results.push((dist, ep));
        }

        let layer_prefix = HnswParams::layer_edge_name(layer);

        while let Some(candidate) = candidates.pop() {
            // Get furthest result
            results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            // Stop if candidate is further than furthest result and we have enough
            if results.len() >= ef && candidate.distance > results.last().unwrap().0 {
                break;
            }

            // Expand candidate
            let neighbors = get_neighbors(reader, candidate.id, Some(&layer_prefix)).await?;

            for (_weight, neighbor_id) in neighbors {
                if visited.contains(&neighbor_id) {
                    continue;
                }
                visited.insert(neighbor_id);

                let dist = self.compute_distance_async(reader, query, &neighbor_id).await;

                // Add to candidates if better than worst result or we need more
                if results.len() < ef || dist < results.last().unwrap().0 {
                    candidates.push(SearchCandidate {
                        distance: dist,
                        id: neighbor_id,
                    });
                    results.push((dist, neighbor_id));

                    // Keep only top ef
                    results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
    }

    /// Select best neighbors using simple heuristic
    fn select_neighbors(
        &self,
        _query: &[f32],
        mut candidates: Vec<(f32, Id)>,
        m: usize,
    ) -> Vec<(f32, Id)> {
        // Sort by distance and take top M
        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(m);
        candidates
    }

    /// Prune connections if a node exceeds max neighbors
    /// Uses cached distance computation (called during build when all vectors are cached)
    async fn prune_connections(
        &mut self,
        writer: &Writer,
        reader: &Reader,
        node_id: Id,
        layer: usize,
    ) -> Result<()> {
        let layer_prefix = HnswParams::layer_edge_name(layer);
        let neighbors = get_neighbors(reader, node_id, Some(&layer_prefix)).await?;

        let m_max = self.params.max_connections(layer);
        if neighbors.len() <= m_max {
            return Ok(());
        }

        // Get the node's vector
        let node_vector = self.cache.get(&node_id).cloned();
        let node_vector = match node_vector {
            Some(v) => v,
            None => {
                // Load from database
                get_vector(reader, node_id).await?
            }
        };

        // Compute distances and sort (use cached version during build)
        let mut with_distances: Vec<(f32, Id)> = neighbors
            .iter()
            .map(|(_, id)| {
                let dist = self.compute_distance_cached(&node_vector, id);
                (dist, *id)
            })
            .collect();

        with_distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Prune edges beyond m_max
        for (_, neighbor_id) in with_distances.iter().skip(m_max) {
            prune_edge(writer, node_id, *neighbor_id, &layer_prefix).await?;
        }

        Ok(())
    }

    /// Search for k nearest neighbors
    /// Uses async distance computation with DB fallback for robustness
    pub async fn search(
        &mut self,
        reader: &Reader,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<(f32, Id)>> {
        let entry_point = match self.entry_point {
            Some(ep) => ep,
            None => return Ok(Vec::new()),
        };

        // Phase 1: Greedy descent from top layer to layer 1
        let mut current = entry_point;
        for l in (1..=self.max_level).rev() {
            let (best, _) = self.greedy_search_layer(reader, query, current, l).await?;
            current = best;
        }

        // Phase 2: Search at layer 0 with ef_search candidates
        let results = self
            .search_layer(reader, query, vec![current], ef_search, 0)
            .await?;

        // Return top k
        Ok(results.into_iter().take(k).collect())
    }

    /// Compute distance between query and a node (uses cache only - for build phase)
    fn compute_distance_cached(&self, query: &[f32], node_id: &Id) -> f32 {
        match self.cache.get(node_id) {
            Some(vector) => (self.distance_fn)(query, vector),
            None => {
                tracing::warn!("Vector not in cache for node {}, returning MAX distance", node_id);
                f32::MAX
            }
        }
    }

    /// Compute distance between query and a node with DB fallback
    /// This is the primary method for search - loads from DB if not in cache
    async fn compute_distance_async(
        &mut self,
        reader: &Reader,
        query: &[f32],
        node_id: &Id,
    ) -> f32 {
        // Check cache first
        if let Some(vector) = self.cache.get(node_id) {
            return (self.distance_fn)(query, vector);
        }

        // Load from database on cache miss
        match get_vector(reader, *node_id).await {
            Ok(vector) => {
                let dist = (self.distance_fn)(query, &vector);
                // Cache for future use
                self.cache.insert(*node_id, vector);
                dist
            }
            Err(e) => {
                tracing::warn!("Failed to load vector for node {}: {}", node_id, e);
                f32::MAX
            }
        }
    }

    /// Get all cached vectors (for ground truth comparison)
    pub fn get_all_vectors(&self) -> &HashMap<Id, Vec<f32>> {
        &self.cache.vectors
    }

    /// Get number of cached vectors
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }
}

// ============================================================================
// Main Entry Point
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    let args: Vec<String> = env::args().collect();
    if args.len() != 5 {
        eprintln!(
            "Usage: {} <db_path> <num_vectors> <num_queries> <k>",
            args[0]
        );
        eprintln!();
        eprintln!("Arguments:");
        eprintln!("  db_path      - Path to database directory");
        eprintln!("  num_vectors  - Number of vectors to index");
        eprintln!("  num_queries  - Number of queries to run");
        eprintln!("  k            - Number of nearest neighbors to find");
        eprintln!();
        eprintln!("Example:");
        eprintln!("  {} /tmp/hnsw_test 1000 100 10", args[0]);
        std::process::exit(1);
    }

    let db_path = Path::new(&args[1]);
    let num_vectors: usize = args[2].parse()?;
    let num_queries: usize = args[3].parse()?;
    let k: usize = args[4].parse()?;

    println!("=== HNSW Vector Search Example ===");
    println!("Database path: {:?}", db_path);
    println!("Vectors: {}", num_vectors);
    println!("Queries: {}", num_queries);
    println!("K: {}", k);
    println!();

    // Initialize storage
    println!("Initializing storage...");
    let handles = init_storage(db_path)?;
    let writer = handles.writer();
    let reader = handles.reader_clone();

    // Create HNSW index
    let params = HnswParams::default();
    println!("HNSW Parameters: {:?}", params);

    let mut index = HnswIndex::new(params.clone());
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Generate and index vectors
    println!("\nIndexing {} vectors...", num_vectors);
    let start = Instant::now();

    for i in 0..num_vectors {
        let node_id = Id::new();
        let vector = generate_vector(&mut rng);

        index.insert(writer, &reader, node_id, vector, &mut rng).await?;

        // Allow writes to propagate before next iteration
        // This is needed because the writer is async and edges need to be visible
        // for subsequent greedy searches
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        if (i + 1) % 100 == 0 {
            println!("  Indexed {}/{} vectors", i + 1, num_vectors);
        }
    }

    // Final sync delay to ensure all writes are visible
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let index_time = start.elapsed();
    println!(
        "Indexing completed in {:.2}s ({:.2} vectors/sec)",
        index_time.as_secs_f64(),
        num_vectors as f64 / index_time.as_secs_f64()
    );
    println!("Vectors in cache: {}", index.cache_size());

    // Debug: Check edges from entry point
    if let Some(ep) = index.entry_point {
        let layer0_edges = get_neighbors(&reader, ep, Some("hnsw_L0")).await?;
        println!("Entry point {} has {} layer 0 edges", ep, layer0_edges.len());
        for (weight, neighbor) in layer0_edges.iter().take(5) {
            println!("  -> {} (weight: {:?})", neighbor, weight);
        }
    }

    // Clone vectors for ground truth computation (avoids borrow conflict with mutable search)
    let all_vectors: HashMap<Id, Vec<f32>> = index.get_all_vectors().clone();
    println!("Vectors in all_vectors HashMap: {}", all_vectors.len());

    // Run queries
    println!("\nRunning {} queries...", num_queries);
    let mut total_recall = 0.0;
    let mut total_search_time = std::time::Duration::ZERO;

    for i in 0..num_queries {
        let query = generate_vector(&mut rng);

        // Ground truth with brute force using the same vectors as the index
        let gt_start = Instant::now();
        let ground_truth = brute_force_knn(&query, &all_vectors, k, euclidean_distance);
        let gt_time = gt_start.elapsed();

        // HNSW search
        let search_start = Instant::now();
        let ef_search = params.ef_construction.max(k * 2);
        let results = index.search(&reader, &query, k, ef_search).await?;
        let search_time = search_start.elapsed();
        total_search_time += search_time;

        // Compute recall
        let gt_ids: Vec<Id> = ground_truth.iter().map(|(_, id)| *id).collect();
        let result_ids: Vec<Id> = results.iter().map(|(_, id)| *id).collect();
        let recall = compute_recall(&result_ids, &gt_ids, k);
        total_recall += recall;

        // Debug output for first query
        if i == 0 {
            println!("\n  DEBUG Query 0:");
            println!("    Ground truth ({} results):", ground_truth.len());
            for (j, (dist, id)) in ground_truth.iter().take(5).enumerate() {
                let in_cache = index.cache.contains(id);
                println!("      {}: id={}, dist={:.4}, in_cache={}", j, id, dist, in_cache);
            }
            println!("    HNSW results ({} results):", results.len());
            for (j, (dist, id)) in results.iter().take(5).enumerate() {
                let in_cache = index.cache.contains(id);
                println!("      {}: id={}, dist={:.4}, in_cache={}", j, id, dist, in_cache);
            }
            println!();
        }

        if (i + 1) % 10 == 0 {
            println!(
                "  Query {}/{}: recall@{}={:.3}, search={:.2}ms, brute_force={:.2}ms",
                i + 1,
                num_queries,
                k,
                recall,
                search_time.as_secs_f64() * 1000.0,
                gt_time.as_secs_f64() * 1000.0
            );
        }
    }

    let avg_recall = total_recall / num_queries as f64;
    let avg_search_ms = total_search_time.as_secs_f64() * 1000.0 / num_queries as f64;

    println!("\n=== Results ===");
    println!("Average Recall@{}: {:.4}", k, avg_recall);
    println!("Average Search Time: {:.3} ms", avg_search_ms);
    println!(
        "Queries Per Second: {:.1}",
        num_queries as f64 / total_search_time.as_secs_f64()
    );
    println!("Max Level: {}", index.max_level);
    println!(
        "Entry Point: {:?}",
        index.entry_point.map(|id| format!("{}", id))
    );

    // Verify node count
    let nodes = AllNodes::new(num_vectors + 1000)
        .run(&reader, DEFAULT_TIMEOUT)
        .await?;
    println!("Total nodes in database: {}", nodes.len());

    // Shutdown
    handles.shutdown().await?;
    println!("\nDone!");

    Ok(())
}
