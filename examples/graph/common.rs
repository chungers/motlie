/// Common utilities for graph algorithm examples
use anyhow::Result;
use motlie_db::graph::mutation::{AddEdge, AddNode, Runnable as MutationRunnable};
use motlie_db::graph::reader::{Reader, ReaderConfig};
use motlie_db::graph::schema::{EdgeSummary, NodeSummary};
use motlie_db::graph::writer::{create_mutation_writer, spawn_mutation_consumer, WriterConfig};
use motlie_db::{Id, TimestampMilli};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::time::Instant;
use tokio::time::Duration;

/// Parse scale factor from command line argument
pub fn parse_scale_factor(s: &str) -> Result<usize> {
    let scale = s.parse::<usize>()
        .map_err(|_| anyhow::anyhow!("Invalid scale factor. Must be a positive integer (e.g., 1, 10, 100, 1000)"))?;

    if scale == 0 {
        anyhow::bail!("Scale factor must be at least 1");
    }

    Ok(scale)
}

/// Compute hash of a result for correctness comparison
pub fn compute_hash<T: Hash>(data: &T) -> String {
    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Compute hash of f64 result for Dijkstra reference (path, cost_u32) -> consistent f64 hash
pub fn compute_hash_f64(data: &Option<(Vec<String>, u32)>) -> String {
    match data {
        Some((path, cost)) => {
            let mut hasher = DefaultHasher::new();
            path.hash(&mut hasher);
            // Convert u32 cost to f64 and hash as integer bits to avoid floating point issues
            let cost_f64 = (*cost as f64) / 10.0; // Reverse the *10.0 conversion from dijkstra
            cost_f64.to_bits().hash(&mut hasher);
            format!("{:016x}", hasher.finish())
        }
        None => "none".to_string(),
    }
}

/// Compute hash of f64 result for Dijkstra motlie (cost, path) -> consistent f64 hash
pub fn compute_hash_f64_motlie(data: &Option<(f64, Vec<String>)>) -> String {
    match data {
        Some((cost, path)) => {
            let mut hasher = DefaultHasher::new();
            path.hash(&mut hasher);
            // Hash cost as integer bits to avoid floating point issues
            cost.to_bits().hash(&mut hasher);
            format!("{:016x}", hasher.finish())
        }
        None => "none".to_string(),
    }
}

/// Compute hash of PageRank results (HashMap<String, f64>)
pub fn compute_hash_pagerank(data: &HashMap<String, f64>) -> String {
    // Sort by key for deterministic hashing
    let mut sorted: Vec<_> = data.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(b.0));

    let mut hasher = DefaultHasher::new();
    for (key, value) in sorted {
        key.hash(&mut hasher);
        value.to_bits().hash(&mut hasher);
    }
    format!("{:016x}", hasher.finish())
}

/// Implementation type for graph algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Implementation {
    Reference,
    MotlieDb,
}

impl Implementation {
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "reference" => Ok(Implementation::Reference),
            "motlie_db" | "motlie-db" | "motliedb" => Ok(Implementation::MotlieDb),
            _ => anyhow::bail!("Invalid implementation type. Must be 'reference' or 'motlie_db'"),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Implementation::Reference => "reference",
            Implementation::MotlieDb => "motlie_db",
        }
    }
}

pub struct GraphMetrics {
    pub algorithm_name: String,
    pub implementation: Implementation,
    pub scale: usize,
    pub num_nodes: usize,
    pub num_edges: usize,
    pub execution_time_ms: f64,
    pub memory_usage_bytes: Option<usize>,
    pub result_hash: Option<String>,  // For comparing correctness across implementations
    pub disk_files: Option<usize>,     // Number of files in RocksDB directory
    pub disk_size_bytes: Option<usize>, // Total size of RocksDB directory in bytes
}

impl GraphMetrics {
    pub fn print_comparison(motlie: &GraphMetrics, reference: &GraphMetrics) {
        println!("\n{:=<80}", "");
        println!("📊 Performance Comparison: {}", motlie.algorithm_name);
        println!("{:=<80}", "");

        println!("\n🔹 Graph Size:");
        println!("  Nodes: {}", motlie.num_nodes);
        println!("  Edges: {}", motlie.num_edges);

        println!("\n⏱️  Execution Time:");
        println!("  motlie_db:  {:.2} ms", motlie.execution_time_ms);
        println!("  {}:  {:.2} ms", reference.algorithm_name, reference.execution_time_ms);

        let speedup = reference.execution_time_ms / motlie.execution_time_ms;
        if speedup > 1.0 {
            println!("  Speedup:    {:.2}x (motlie_db is faster)", speedup);
        } else {
            println!("  Slowdown:   {:.2}x (reference is faster)", 1.0 / speedup);
        }

        if let (Some(motlie_mem), Some(ref_mem)) = (motlie.memory_usage_bytes, reference.memory_usage_bytes) {
            println!("\n💾 Memory Usage (delta):");

            // Format memory in appropriate units
            fn format_bytes(bytes: usize) -> String {
                if bytes >= 1024 * 1024 {
                    format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
                } else if bytes >= 1024 {
                    format!("{:.2} KB", bytes as f64 / 1024.0)
                } else {
                    format!("{} bytes", bytes)
                }
            }

            println!("  motlie_db:  {}", format_bytes(motlie_mem));
            println!("  {}:  {}", reference.algorithm_name, format_bytes(ref_mem));

            if ref_mem > 0 {
                let ratio = motlie_mem as f64 / ref_mem as f64;
                if ratio < 1.0 {
                    println!("  Savings:    {:.2}x less memory used by motlie_db", 1.0 / ratio);
                } else {
                    println!("  Overhead:   {:.2}x more memory used by motlie_db", ratio);
                }
            }
        }

        println!("\n{:=<80}", "");
    }

    /// Print metrics as CSV row (for data collection)
    pub fn print_csv(&self) {
        let memory_kb = self.memory_usage_bytes
            .map(|b| (b as f64 / 1024.0).to_string())
            .unwrap_or_else(|| "N/A".to_string());

        let result_hash = self.result_hash
            .clone()
            .unwrap_or_else(|| "N/A".to_string());

        let disk_files = self.disk_files
            .map(|n| n.to_string())
            .unwrap_or_else(|| "N/A".to_string());

        let disk_kb = self.disk_size_bytes
            .map(|b| (b as f64 / 1024.0).to_string())
            .unwrap_or_else(|| "N/A".to_string());

        println!("{},{},{},{},{},{:.4},{},{},{},{}",
            self.algorithm_name,
            self.implementation.as_str(),
            self.scale,
            self.num_nodes,
            self.num_edges,
            self.execution_time_ms,
            memory_kb,
            result_hash,
            disk_files,
            disk_kb
        );
    }

    /// Print CSV header
    pub fn print_csv_header() {
        println!("algorithm,implementation,scale,nodes,edges,time_ms,memory_kb,result_hash,disk_files,disk_kb");
    }
}

/// Node and edge structures for graph construction
#[derive(Clone)]
pub struct GraphNode {
    pub id: Id,
    pub name: String,
}

pub struct GraphEdge {
    pub source: Id,
    pub target: Id,
    pub name: String,
    pub weight: Option<f64>,
}

/// Build a graph in motlie_db and return a Reader for querying
pub async fn build_graph(
    db_path: &Path,
    nodes: Vec<GraphNode>,
    edges: Vec<GraphEdge>,
) -> Result<(Reader, HashMap<String, Id>, tokio::task::JoinHandle<Result<()>>)> {
    use motlie_db::graph::{Graph, Storage};
    use std::sync::Arc;

    // Clean up any existing database
    if db_path.exists() {
        std::fs::remove_dir_all(db_path)?;
    }

    let config = WriterConfig {
        channel_buffer_size: 1000,
    };

    let (writer, receiver) = create_mutation_writer(config.clone());
    let handle = spawn_mutation_consumer(receiver, config.clone(), db_path);

    // Create a name to ID mapping
    let mut name_to_id = HashMap::new();

    // Add nodes
    for node in nodes {
        name_to_id.insert(node.name.clone(), node.id);

        AddNode {
            id: node.id,
            ts_millis: TimestampMilli::now(),
            name: node.name.clone(),
            valid_range: None,
            summary: NodeSummary::from_text(&node.name),
        }
        .run(&writer)
        .await?;
    }

    // Add edges
    for edge in edges {
        AddEdge {
            source_node_id: edge.source,
            target_node_id: edge.target,
            ts_millis: TimestampMilli::now(),
            name: edge.name,
            summary: EdgeSummary::from_text(""),
            weight: edge.weight,
            valid_range: None,
        }
        .run(&writer)
        .await?;
    }

    // Shutdown writer and wait for processing to complete
    drop(writer);
    handle.await??;

    // Give the database a moment to finish flushing
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Create reader for querying
    let reader_config = ReaderConfig {
        channel_buffer_size: 100,
    };

    // Open storage for reading
    let mut storage = Storage::readonly(db_path);
    storage.ready()?;
    let storage = Arc::new(storage);
    let graph = Arc::new(Graph::new(storage));

    // Create query reader and spawn consumer
    let (reader, receiver) = motlie_db::graph::reader::create_query_reader(reader_config.clone());
    let query_handle =
        motlie_db::graph::reader::spawn_query_consumer_with_graph(receiver, reader_config, graph);

    Ok((reader, name_to_id, query_handle))
}

/// Measure execution time of a closure
pub fn measure_time<F, R>(f: F) -> (R, f64)
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = f();
    let elapsed = start.elapsed();
    (result, elapsed.as_secs_f64() * 1000.0)
}

/// Measure execution time of an async closure
pub async fn measure_time_async<F, Fut, R>(f: F) -> (R, f64)
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = R>,
{
    let start = Instant::now();
    let result = f().await;
    let elapsed = start.elapsed();
    (result, elapsed.as_secs_f64() * 1000.0)
}

/// Get current memory usage in bytes (RSS - Resident Set Size)
#[cfg(target_os = "linux")]
pub fn get_memory_usage() -> Option<usize> {
    use std::fs;

    // Read /proc/self/status for VmRSS (Resident Set Size)
    if let Ok(status) = fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    if let Ok(kb) = parts[1].parse::<usize>() {
                        return Some(kb * 1024); // Convert KB to bytes
                    }
                }
            }
        }
    }
    None
}

#[cfg(target_os = "macos")]
pub fn get_memory_usage() -> Option<usize> {
    use std::process::Command;

    // Use ps to get RSS (in KB)
    let output = Command::new("ps")
        .args(&["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()?;

    let rss_str = String::from_utf8(output.stdout).ok()?;
    let rss_kb = rss_str.trim().parse::<usize>().ok()?;
    Some(rss_kb * 1024) // Convert KB to bytes
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub fn get_memory_usage() -> Option<usize> {
    None
}

/// Measure execution time and memory usage of a closure
pub fn measure_time_and_memory<F, R>(f: F) -> (R, f64, Option<usize>)
where
    F: FnOnce() -> R,
{
    let mem_before = get_memory_usage();
    let start = Instant::now();
    let result = f();
    let elapsed = start.elapsed();
    let mem_after = get_memory_usage();

    let mem_delta = match (mem_before, mem_after) {
        (Some(before), Some(after)) => {
            // Memory might decrease due to GC, so take max of 0
            Some(after.saturating_sub(before))
        }
        _ => None,
    };

    (result, elapsed.as_secs_f64() * 1000.0, mem_delta)
}

/// Measure execution time and memory usage of an async closure
pub async fn measure_time_and_memory_async<F, Fut, R>(f: F) -> (R, f64, Option<usize>)
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = R>,
{
    let mem_before = get_memory_usage();
    let start = Instant::now();
    let result = f().await;
    let elapsed = start.elapsed();
    let mem_after = get_memory_usage();

    let mem_delta = match (mem_before, mem_after) {
        (Some(before), Some(after)) => {
            Some(after.saturating_sub(before))
        }
        _ => None,
    };

    (result, elapsed.as_secs_f64() * 1000.0, mem_delta)
}

/// Get RocksDB disk usage metrics (file count and total size in bytes)
pub fn get_disk_metrics(db_path: &Path) -> Result<(usize, usize)> {
    use std::fs;

    if !db_path.exists() {
        return Ok((0, 0));
    }

    let mut file_count = 0;
    let mut total_size = 0;

    // Recursively walk the directory tree
    fn walk_dir(path: &Path, file_count: &mut usize, total_size: &mut usize) -> Result<()> {
        if path.is_dir() {
            for entry in fs::read_dir(path)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_file() {
                    *file_count += 1;
                    *total_size += entry.metadata()?.len() as usize;
                } else if path.is_dir() {
                    walk_dir(&path, file_count, total_size)?;
                }
            }
        }
        Ok(())
    }

    walk_dir(db_path, &mut file_count, &mut total_size)?;
    Ok((file_count, total_size))
}
