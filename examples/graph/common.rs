/// Common utilities for graph algorithm examples
use anyhow::Result;
use motlie_db::{create_mutation_writer, spawn_graph_consumer, AddEdge, AddNode, DataUrl, EdgeSummary, Id, MutationRunnable, Reader, ReaderConfig, TimestampMilli, WriterConfig};
use std::collections::HashMap;
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

pub struct GraphMetrics {
    pub algorithm_name: String,
    pub num_nodes: usize,
    pub num_edges: usize,
    pub execution_time_ms: f64,
    pub memory_usage_bytes: Option<usize>,
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
            println!("\n💾 Memory Usage:");
            println!("  motlie_db:  {:.2} KB", motlie_mem as f64 / 1024.0);
            println!("  {}:  {:.2} KB", reference.algorithm_name, ref_mem as f64 / 1024.0);

            let ratio = motlie_mem as f64 / ref_mem as f64;
            if ratio > 1.0 {
                println!("  Overhead:   {:.2}x", ratio);
            } else {
                println!("  Savings:    {:.2}x", 1.0 / ratio);
            }
        }

        println!("\n{:=<80}", "");
    }
}

/// Node and edge structures for graph construction
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
    use motlie_db::{Storage, Graph};
    use std::sync::Arc;

    // Clean up any existing database
    if db_path.exists() {
        std::fs::remove_dir_all(db_path)?;
    }

    let config = WriterConfig {
        channel_buffer_size: 1000,
    };

    let (writer, receiver) = create_mutation_writer(config.clone());
    let handle = spawn_graph_consumer(receiver, config.clone(), db_path);

    // Create a name to ID mapping
    let mut name_to_id = HashMap::new();

    // Add nodes
    for node in nodes {
        name_to_id.insert(node.name.clone(), node.id);

        AddNode {
            id: node.id,
            ts_millis: TimestampMilli::now(),
            name: node.name,
            temporal_range: None,
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
            temporal_range: None,
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
    let graph = Graph::new(storage);

    // Create query reader and spawn consumer
    let (reader, receiver) = motlie_db::create_query_reader(reader_config.clone());
    let consumer = motlie_db::QueryConsumer::new(receiver, reader_config, graph);
    let query_handle = motlie_db::spawn_query_consumer(consumer);

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
