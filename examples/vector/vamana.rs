//! Vamana (DiskANN) Implementation
//!
//! This module implements the Vamana algorithm for approximate nearest neighbor search
//! using motlie_db as the underlying graph storage. Vamana is the core graph construction
//! algorithm behind Microsoft's DiskANN system.
//!
//! # Key Features
//!
//! - Single-layer graph (unlike HNSW's multi-layer)
//! - RNG pruning for removing redundant edges
//! - Medoid-based entry point for better navigation
//! - Designed for disk-based operation
//!
//! # Design Decisions
//!
//! - **Edge names**: All edges use "vamana" as the name
//! - **Distance as edge weight**: Edge weights store the distance between vectors
//! - **Temporal pruning**: Pruned edges use valid_until for soft deletion
//!
//! # Usage
//!
//! ```bash
//! # Synthetic data (default)
//! cargo run --release --example vamana <db_path> <num_vectors> <num_queries> <k>
//!
//! # Benchmark dataset (SIFT1M, SIFT10K)
//! cargo run --release --example vamana <db_path> <num_vectors> <num_queries> <k> --dataset sift10k
//! ```

#[path = "common.rs"]
mod common;

#[path = "benchmark.rs"]
mod benchmark;

use anyhow::Result;
use benchmark::{BenchmarkDataset, BenchmarkMetrics, DatasetConfig, DatasetName};
use common::{
    add_vector_edge, brute_force_knn, compute_recall, euclidean_distance, generate_vector,
    get_neighbors, get_vector, init_storage, prune_edge, store_vector, DistanceFn, VectorCache,
    DEFAULT_TIMEOUT, DIMENSIONS,
};
use motlie_db::query::{AllNodes, Runnable as QueryRunnable};
use motlie_db::reader::Reader;
use motlie_db::writer::Writer;
use motlie_db::Id;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::env;
use std::path::{Path, PathBuf};
use std::time::Instant;

// ============================================================================
// Vamana Parameters
// ============================================================================

/// Vamana index parameters
#[derive(Clone, Debug)]
pub struct VamanaParams {
    /// Maximum out-degree (max neighbors per node)
    pub r: usize,
    /// Search list size during construction
    pub l: usize,
    /// RNG pruning threshold (alpha >= 1.0)
    pub alpha: f32,
}

impl Default for VamanaParams {
    fn default() -> Self {
        Self {
            r: 64,
            l: 100,
            alpha: 1.2,
        }
    }
}

impl VamanaParams {
    /// Edge name for Vamana graph
    pub fn edge_name() -> &'static str {
        "vamana"
    }
}

// ============================================================================
// Priority Queue Entry for Search
// ============================================================================

/// Entry for search candidate queue (min-heap by distance)
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
        // Min-heap: smallest distance first (reverse comparison)
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ============================================================================
// Vamana Index
// ============================================================================

/// Vamana index built on motlie_db
pub struct VamanaIndex {
    /// Index parameters
    pub params: VamanaParams,
    /// Medoid (centroid) node ID - entry point for searches
    pub medoid: Option<Id>,
    /// In-memory vector cache for fast distance computation
    pub cache: VectorCache,
    /// Distance function
    pub distance_fn: DistanceFn,
    /// All node IDs in insertion order
    pub node_ids: Vec<Id>,
}

impl VamanaIndex {
    /// Create a new Vamana index
    pub fn new(params: VamanaParams) -> Self {
        Self {
            params,
            medoid: None,
            cache: VectorCache::new(),
            distance_fn: euclidean_distance,
            node_ids: Vec::new(),
        }
    }

    /// Build index from a batch of vectors
    ///
    /// Vamana typically builds in batch rather than incrementally.
    /// This allows for better medoid selection and multiple passes.
    pub async fn build(
        &mut self,
        writer: &Writer,
        reader: &Reader,
        vectors: Vec<(Id, Vec<f32>)>,
    ) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }

        println!("  Phase 1: Storing vectors and computing medoid...");

        // Store all vectors
        for (i, (node_id, vector)) in vectors.iter().enumerate() {
            let name = format!("vec_{}", i);
            store_vector(writer, *node_id, &name, vector).await?;
            self.cache.insert(*node_id, vector.clone());
            self.node_ids.push(*node_id);
        }

        // Flush to ensure writes are visible
        writer.flush().await?;

        // Compute medoid (node closest to centroid)
        self.medoid = Some(self.compute_medoid());
        println!("    Medoid: {:?}", self.medoid);

        println!("  Phase 2: Building graph with greedy search + RNG pruning...");

        // Initialize with random edges (optional, helps with connectivity)
        self.initialize_random_edges(writer, &mut ChaCha8Rng::seed_from_u64(42))
            .await?;

        // Multiple passes for better graph quality
        let num_passes = 2;
        for pass in 0..num_passes {
            println!("    Pass {}/{}...", pass + 1, num_passes);

            // Shuffle order for each pass
            let mut order: Vec<usize> = (0..self.node_ids.len()).collect();
            order.shuffle(&mut ChaCha8Rng::seed_from_u64((pass + 1) as u64));

            for (progress, &idx) in order.iter().enumerate() {
                let node_id = self.node_ids[idx];
                self.insert_node(writer, reader, node_id).await?;

                if (progress + 1) % 100 == 0 {
                    println!(
                        "      Processed {}/{} nodes",
                        progress + 1,
                        self.node_ids.len()
                    );
                }
            }
        }

        Ok(())
    }

    /// Initialize with random edges for better connectivity
    async fn initialize_random_edges(
        &self,
        writer: &Writer,
        rng: &mut impl rand::Rng,
    ) -> Result<()> {
        let edge_name = VamanaParams::edge_name();
        let num_initial = self.params.r.min(self.node_ids.len());

        for &node_id in &self.node_ids {
            // Pick random neighbors
            let mut candidates: Vec<_> = self
                .node_ids
                .iter()
                .filter(|&&id| id != node_id)
                .cloned()
                .collect();
            candidates.shuffle(rng);

            for neighbor_id in candidates.into_iter().take(num_initial) {
                // During build phase, all vectors are in cache
                let dist = self.compute_distance_by_id_cached(&node_id, &neighbor_id);
                add_vector_edge(writer, node_id, neighbor_id, edge_name, dist).await?;
            }
        }

        Ok(())
    }

    /// Insert/update a single node in the graph
    async fn insert_node(
        &mut self,
        writer: &Writer,
        reader: &Reader,
        node_id: Id,
    ) -> Result<()> {
        let vector = match self.cache.get(&node_id) {
            Some(v) => v.clone(),
            None => return Ok(()), // Node not in cache
        };

        // Greedy search to find L nearest candidates
        let start = self.medoid.unwrap_or(node_id);
        let candidates = self.greedy_search(reader, &vector, start, self.params.l).await?;

        // RNG-prune candidates to get R neighbors
        let pruned = self.rng_prune(&vector, candidates);

        // Update edges from node to pruned neighbors
        let edge_name = VamanaParams::edge_name();

        for (dist, neighbor_id) in &pruned {
            // Add forward edge
            add_vector_edge(writer, node_id, *neighbor_id, edge_name, *dist).await?;

            // Add reverse edge and potentially prune neighbor
            add_vector_edge(writer, *neighbor_id, node_id, edge_name, *dist).await?;

            // Check if neighbor exceeds R neighbors
            self.maybe_prune_node(writer, reader, *neighbor_id).await?;
        }

        Ok(())
    }

    /// Greedy search returning L nearest candidates
    /// Uses async distance computation with DB fallback for robustness
    async fn greedy_search(
        &mut self,
        reader: &Reader,
        query: &[f32],
        start: Id,
        l: usize,
    ) -> Result<Vec<(f32, Id)>> {
        let mut visited: HashSet<Id> = HashSet::new();
        let mut candidates: BinaryHeap<SearchCandidate> = BinaryHeap::new();
        let mut results: Vec<(f32, Id)> = Vec::new();

        // Initialize with start node (use async to ensure vector is loaded)
        let start_dist = self.compute_distance_async(reader, query, &start).await;
        visited.insert(start);
        candidates.push(SearchCandidate {
            distance: start_dist,
            id: start,
        });
        results.push((start_dist, start));

        let edge_name = VamanaParams::edge_name();

        while let Some(candidate) = candidates.pop() {
            // Get neighbors
            let neighbors = get_neighbors(reader, candidate.id, Some(edge_name)).await?;

            for (_, neighbor_id) in neighbors {
                if visited.contains(&neighbor_id) {
                    continue;
                }
                visited.insert(neighbor_id);

                // Use async distance computation with DB fallback
                let dist = self.compute_distance_async(reader, query, &neighbor_id).await;

                // Check if we should add this candidate
                let should_add = results.len() < l || {
                    results.sort_by(|a, b| {
                        a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    dist < results.last().map(|(d, _)| *d).unwrap_or(f32::MAX)
                };

                if should_add {
                    candidates.push(SearchCandidate {
                        distance: dist,
                        id: neighbor_id,
                    });
                    results.push((dist, neighbor_id));

                    // Keep only top L
                    results.sort_by(|a, b| {
                        a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    if results.len() > l {
                        results.pop();
                    }
                }
            }

            // Early termination: if candidate is worse than our L-th best, stop
            results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            if results.len() >= l && candidate.distance > results[l - 1].0 {
                break;
            }
        }

        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
    }

    /// RNG pruning: removes redundant edges
    ///
    /// An edge (v -> u) is pruned if there exists w such that:
    /// dist(v, w) < dist(v, u) AND dist(w, u) < dist(v, u) * alpha
    ///
    /// Note: Uses cached distance computation. Caller must ensure all candidate
    /// vectors are in cache before calling (done during build phase).
    fn rng_prune(&self, _query: &[f32], mut candidates: Vec<(f32, Id)>) -> Vec<(f32, Id)> {
        // Sort by distance
        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut result: Vec<(f32, Id)> = Vec::new();

        for (dist_vu, u) in candidates {
            if result.len() >= self.params.r {
                break;
            }

            // Check if u is dominated by any node already in result
            let mut is_dominated = false;

            for (dist_vw, w) in &result {
                // Condition 1: dist(v, w) < dist(v, u)
                if *dist_vw >= dist_vu {
                    continue;
                }

                // Condition 2: dist(w, u) < dist(v, u) * alpha
                let dist_wu = self.compute_distance_by_id_cached(w, &u);
                if dist_wu < dist_vu * self.params.alpha {
                    is_dominated = true;
                    break;
                }
            }

            if !is_dominated {
                result.push((dist_vu, u));
            }
        }

        result
    }

    /// Check if a node exceeds R neighbors and prune if necessary
    async fn maybe_prune_node(
        &self,
        writer: &Writer,
        reader: &Reader,
        node_id: Id,
    ) -> Result<()> {
        let edge_name = VamanaParams::edge_name();
        let neighbors = get_neighbors(reader, node_id, Some(edge_name)).await?;

        if neighbors.len() <= self.params.r {
            return Ok(());
        }

        // Get node vector
        let node_vector = match self.cache.get(&node_id) {
            Some(v) => v.clone(),
            None => get_vector(reader, node_id).await?,
        };

        // Convert to (distance, id) format using cached distances
        // During build phase, all neighbors should already be in cache
        let candidates: Vec<(f32, Id)> = neighbors
            .iter()
            .map(|(_, id)| {
                let dist = self.compute_distance_cached(&node_vector, id);
                (dist, *id)
            })
            .collect();

        // RNG prune
        let pruned = self.rng_prune(&node_vector, candidates.clone());
        let pruned_set: HashSet<_> = pruned.iter().map(|(_, id)| *id).collect();

        // Remove edges not in pruned set
        for (_, neighbor_id) in &candidates {
            if !pruned_set.contains(neighbor_id) {
                prune_edge(writer, node_id, *neighbor_id, edge_name).await?;
            }
        }

        Ok(())
    }

    /// Compute medoid (node closest to centroid of all vectors)
    fn compute_medoid(&self) -> Id {
        if self.node_ids.is_empty() {
            panic!("Cannot compute medoid of empty index");
        }

        // Compute centroid
        let mut centroid = vec![0.0_f32; DIMENSIONS];
        let n = self.node_ids.len() as f32;

        for id in &self.node_ids {
            if let Some(vector) = self.cache.get(id) {
                for (i, v) in vector.iter().enumerate() {
                    centroid[i] += v / n;
                }
            }
        }

        // Find node closest to centroid
        let mut best_id = self.node_ids[0];
        let mut best_dist = f32::MAX;

        for id in &self.node_ids {
            if let Some(vector) = self.cache.get(id) {
                let dist = (self.distance_fn)(&centroid, vector);
                if dist < best_dist {
                    best_dist = dist;
                    best_id = *id;
                }
            }
        }

        best_id
    }

    /// Search for k nearest neighbors
    /// Uses async distance computation with DB fallback for robustness
    pub async fn search(
        &mut self,
        reader: &Reader,
        query: &[f32],
        k: usize,
        l_search: usize,
    ) -> Result<Vec<(f32, Id)>> {
        let start = match self.medoid {
            Some(m) => m,
            None => return Ok(Vec::new()),
        };

        let results = self.greedy_search(reader, query, start, l_search).await?;
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

    /// Compute distance between two nodes by ID (cache only - for build phase)
    fn compute_distance_by_id_cached(&self, a: &Id, b: &Id) -> f32 {
        match (self.cache.get(a), self.cache.get(b)) {
            (Some(va), Some(vb)) => (self.distance_fn)(va, vb),
            _ => f32::MAX,
        }
    }

    /// Compute distance between two nodes by ID with DB fallback
    async fn compute_distance_by_id_async(
        &mut self,
        reader: &Reader,
        a: &Id,
        b: &Id,
    ) -> f32 {
        // Ensure both vectors are in cache
        self.ensure_vector_cached(reader, a).await;
        self.ensure_vector_cached(reader, b).await;

        match (self.cache.get(a), self.cache.get(b)) {
            (Some(va), Some(vb)) => (self.distance_fn)(va, vb),
            _ => f32::MAX,
        }
    }

    /// Ensure a vector is loaded into cache
    async fn ensure_vector_cached(&mut self, reader: &Reader, node_id: &Id) {
        if self.cache.contains(node_id) {
            return;
        }

        if let Ok(vector) = get_vector(reader, *node_id).await {
            self.cache.insert(*node_id, vector);
        }
    }

    /// Ensure multiple vectors are loaded into cache
    async fn ensure_vectors_cached(&mut self, reader: &Reader, node_ids: &[Id]) {
        for id in node_ids {
            self.ensure_vector_cached(reader, id).await;
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

/// Parse command line arguments
fn parse_args() -> Result<(PathBuf, usize, usize, usize, Option<DatasetName>, usize)> {
    let args: Vec<String> = env::args().collect();

    // Check for optional flags
    let mut dataset_name: Option<DatasetName> = None;
    let mut l_param: usize = 100; // Default L parameter
    let mut filtered_args: Vec<String> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        if args[i] == "--dataset" {
            if i + 1 < args.len() {
                dataset_name = DatasetName::from_str(&args[i + 1]);
                if dataset_name.is_none() {
                    eprintln!("Unknown dataset: {}. Valid options: sift1m, sift10k, random", args[i + 1]);
                    std::process::exit(1);
                }
                i += 2;
                continue;
            }
        }
        if args[i] == "--l" || args[i] == "-l" {
            if i + 1 < args.len() {
                l_param = args[i + 1].parse().unwrap_or_else(|_| {
                    eprintln!("Invalid L parameter: {}", args[i + 1]);
                    std::process::exit(1);
                });
                i += 2;
                continue;
            }
        }
        filtered_args.push(args[i].clone());
        i += 1;
    }

    if filtered_args.len() != 5 {
        eprintln!(
            "Usage: {} <db_path> <num_vectors> <num_queries> <k> [--dataset <name>] [--l <value>]",
            filtered_args.get(0).unwrap_or(&"vamana".to_string())
        );
        eprintln!();
        eprintln!("Arguments:");
        eprintln!("  db_path      - Path to database directory");
        eprintln!("  num_vectors  - Number of vectors to index");
        eprintln!("  num_queries  - Number of queries to run");
        eprintln!("  k            - Number of nearest neighbors to find");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --dataset <name>  Use benchmark dataset: sift1m, sift10k, random (default: random)");
        eprintln!("  --l <value>       Search list size L (default: 100, try 200 for better recall)");
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  {} /tmp/vamana_test 1000 100 10", filtered_args.get(0).unwrap_or(&"vamana".to_string()));
        eprintln!("  {} /tmp/vamana_sift 10000 100 10 --dataset sift10k", filtered_args.get(0).unwrap_or(&"vamana".to_string()));
        eprintln!("  {} /tmp/vamana_sift 10000 100 10 --dataset sift1m --l 200", filtered_args.get(0).unwrap_or(&"vamana".to_string()));
        std::process::exit(1);
    }

    let db_path = PathBuf::from(&filtered_args[1]);
    let num_vectors: usize = filtered_args[2].parse()?;
    let num_queries: usize = filtered_args[3].parse()?;
    let k: usize = filtered_args[4].parse()?;

    Ok((db_path, num_vectors, num_queries, k, dataset_name, l_param))
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    let (db_path, num_vectors, num_queries, k, dataset_name, l_param) = parse_args()?;

    // Load benchmark dataset if specified
    let benchmark_data: Option<BenchmarkDataset> = if let Some(ds_name) = dataset_name {
        println!("Loading benchmark dataset: {}...", ds_name);
        let config = DatasetConfig {
            name: ds_name,
            num_base: Some(num_vectors),
            num_queries: Some(num_queries),
            ground_truth_k: 100,
            ..Default::default()
        };
        Some(benchmark::load_dataset(&config).await?)
    } else {
        None
    };

    let dataset_str = dataset_name.map(|d| d.to_string()).unwrap_or("random".to_string());

    println!("=== Vamana (DiskANN) Vector Search Example ===");
    println!("Database path: {:?}", db_path);
    println!("Dataset: {}", dataset_str);
    println!("Vectors: {}", num_vectors);
    println!("Queries: {}", num_queries);
    println!("K: {}", k);
    println!();

    // Initialize storage
    println!("Initializing storage...");
    let handles = init_storage(&db_path)?;
    let writer = handles.writer();
    let reader = handles.reader_clone();

    // Create Vamana index with custom L parameter
    let params = VamanaParams {
        l: l_param,
        ..Default::default()
    };
    println!("Vamana Parameters: {:?}", params);

    let mut index = VamanaIndex::new(params.clone());
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut metrics = BenchmarkMetrics::new(&dataset_str, "Vamana");

    // Build mapping from vector index to node ID (for benchmark validation)
    let mut index_to_id: Vec<Id> = Vec::with_capacity(num_vectors);

    // Generate vectors
    println!("\nGenerating {} vectors...", num_vectors);
    let mut vectors: Vec<(Id, Vec<f32>)> = Vec::new();

    for i in 0..num_vectors {
        let node_id = Id::new();
        let vector = if let Some(ref data) = benchmark_data {
            data.base_vectors[i].clone()
        } else {
            generate_vector(&mut rng)
        };
        index_to_id.push(node_id);
        vectors.push((node_id, vector));
    }

    // Build index
    println!("\nBuilding Vamana index...");
    let start = Instant::now();
    index.build(writer, &reader, vectors).await?;
    let index_time = start.elapsed();

    metrics.num_vectors = num_vectors;
    metrics.build_time_secs = index_time.as_secs_f64();
    metrics.build_throughput = num_vectors as f64 / index_time.as_secs_f64();

    println!(
        "Indexing completed in {:.2}s ({:.2} vectors/sec)",
        index_time.as_secs_f64(),
        num_vectors as f64 / index_time.as_secs_f64()
    );
    println!("Vectors in cache: {}", index.cache_size());

    // Create reverse mapping for benchmark validation
    let id_to_index: HashMap<Id, usize> = index_to_id.iter()
        .enumerate()
        .map(|(idx, id)| (*id, idx))
        .collect();

    // Clone vectors for ground truth computation (for synthetic data)
    let all_vectors: HashMap<Id, Vec<f32>> = index.get_all_vectors().clone();

    // Run queries
    let actual_queries = if let Some(ref data) = benchmark_data {
        data.num_queries().min(num_queries)
    } else {
        num_queries
    };

    println!("\nRunning {} queries...", actual_queries);
    let mut total_recall = 0.0;
    let mut search_latencies: Vec<f64> = Vec::with_capacity(actual_queries);

    for i in 0..actual_queries {
        let query = if let Some(ref data) = benchmark_data {
            data.query_vectors[i].clone()
        } else {
            generate_vector(&mut rng)
        };

        // Vamana search
        let search_start = Instant::now();
        let l_search = params.l.max(k * 2);
        let results = index.search(&reader, &query, k, l_search).await?;
        let search_time = search_start.elapsed();
        search_latencies.push(search_time.as_secs_f64() * 1000.0);

        // Compute recall
        let recall = if let Some(ref data) = benchmark_data {
            // Use benchmark ground truth
            let result_indices: Vec<(f32, usize)> = results.iter()
                .filter_map(|(dist, id)| {
                    id_to_index.get(id).map(|idx| (*dist, *idx))
                })
                .collect();
            data.calculate_recall(i, &result_indices, k)
        } else {
            // Use brute force ground truth
            let ground_truth = brute_force_knn(&query, &all_vectors, k, euclidean_distance);
            let gt_ids: Vec<Id> = ground_truth.iter().map(|(_, id)| *id).collect();
            let result_ids: Vec<Id> = results.iter().map(|(_, id)| *id).collect();
            compute_recall(&result_ids, &gt_ids, k) as f32
        };
        total_recall += recall as f64;

        // Debug output for first query
        if i == 0 {
            println!("\n  DEBUG Query 0:");
            if let Some(ref data) = benchmark_data {
                println!("    Ground truth (first 5 of {}):", data.ground_truth[0].len());
                for j in 0..5.min(data.ground_truth[0].len()) {
                    let gt_idx = data.ground_truth[0][j];
                    println!("      {}: index={}", j, gt_idx);
                }
            }
            println!("    Vamana results ({} results):", results.len());
            for (j, (dist, id)) in results.iter().take(5).enumerate() {
                let idx = id_to_index.get(id).map(|i| format!("{}", i)).unwrap_or("?".to_string());
                println!("      {}: index={}, dist={:.4}", j, idx, dist);
            }
            println!();
        }

        if (i + 1) % 10 == 0 {
            println!(
                "  Query {}/{}: recall@{}={:.3}, search={:.2}ms",
                i + 1,
                actual_queries,
                k,
                recall,
                search_time.as_secs_f64() * 1000.0
            );
        }
    }

    let avg_recall = total_recall / actual_queries as f64;

    // Calculate metrics
    metrics.num_queries = actual_queries;
    metrics.k = k;
    metrics.recall_at_k = avg_recall;
    metrics.set_latency_percentiles(search_latencies);

    // Get disk usage
    if let Ok(entries) = std::fs::read_dir(&db_path) {
        let mut total_size = 0u64;
        for entry in entries.flatten() {
            if let Ok(meta) = entry.metadata() {
                total_size += meta.len();
            }
        }
        metrics.disk_usage_bytes = total_size;
    }

    println!("\n=== Results ===");
    println!("Dataset: {}", dataset_str);
    println!("Average Recall@{}: {:.4}", k, avg_recall);
    println!("Average Search Time: {:.3} ms", metrics.avg_search_latency_ms);
    println!("P50 Latency: {:.3} ms", metrics.p50_latency_ms);
    println!("P99 Latency: {:.3} ms", metrics.p99_latency_ms);
    println!("Queries Per Second: {:.1}", metrics.qps);
    println!("Medoid: {:?}", index.medoid.map(|id| format!("{}", id)));

    // Verify node count
    let nodes = AllNodes::new(num_vectors + 1000)
        .run(&reader, DEFAULT_TIMEOUT)
        .await?;
    println!("Total nodes in database: {}", nodes.len());

    // Print full metrics
    metrics.print();

    // Shutdown
    handles.shutdown().await?;
    println!("\nDone!");

    Ok(())
}
