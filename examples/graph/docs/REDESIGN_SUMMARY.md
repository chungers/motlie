# Graph Demo Redesign Summary

## Overview

This document summarizes the complete redesign of the graph algorithm demos to support comprehensive performance data collection and analysis.

## Changes Made

### 1. Standardized Metrics Output (CSV Format)

**File**: `common.rs`

Added standardized CSV output format with the following fields:
- `algorithm`: Algorithm name (DFS, BFS, Topological Sort, Dijkstra, PageRank)
- `implementation`: Implementation type (reference, motlie_db)
- `scale`: Scale factor (1, 10, 100, 1000, 10000)
- `nodes`: Number of nodes in the graph
- `edges`: Number of edges in the graph
- `time_ms`: Execution time in milliseconds
- `memory_kb`: Memory usage in kilobytes
- `result_hash`: Hash of algorithm result for correctness verification
- `disk_files`: Number of RocksDB files (motlie_db only, N/A for reference)
- `disk_kb`: RocksDB disk size in kilobytes (motlie_db only, N/A for reference)

### 2. Command Line Interface Redesign

**All demo files updated**: `dfs.rs`, `bfs.rs`, `toposort.rs`, `dijkstra.rs`, `pagerank.rs`

**Previous Usage:**
```bash
./dfs <db_path> <scale_factor>
```

**New Usage:**
```bash
./dfs <implementation> <db_path> <scale_factor>
```

Where `implementation` is either:
- `reference`: Run only the reference implementation (petgraph, pathfinding, or custom)
- `motlie_db`: Run only the motlie_db implementation

**Benefits:**
- Clean separation of implementations
- Easy to run at scale
- Scriptable data collection
- CSV output for analysis

### 3. Result Hashing for Correctness Verification

**Added hash functions** to verify that both implementations produce identical results:

- **`compute_hash<T>`**: Generic hash for Vec and simple types
- **`compute_hash_f64`**: Specialized for Dijkstra reference (path, u32 cost)
- **`compute_hash_f64_motlie`**: Specialized for Dijkstra motlie_db (f64 cost, path)
- **`compute_hash_pagerank`**: Specialized for PageRank HashMap<String, f64>

All results are hashed and output in the CSV for automatic correctness comparison.

### 4. Base Graph Size Change

Changed from **8 nodes** to **10 nodes** per scale factor for easier mental math:

| Scale | Nodes    | Previous (8x) | New (10x) |
|-------|----------|---------------|-----------|
| 1     | 10       | 8             | 10        |
| 10    | 100      | 80            | 100       |
| 100   | 1,000    | 800           | 1,000     |
| 1000  | 10,000   | 8,000         | 10,000    |
| 10000 | 100,000  | 80,000        | 100,000   |

### 5. Data Collection Infrastructure

**Script**: `scripts/collect_all_metrics.sh`

Automated data collection script that:
- Runs all 5 algorithms
- Tests both implementations (reference, motlie_db)
- At all 5 scales (1, 10, 100, 1000, 10000)
- Total: 50 test runs
- Output: CSV file with all metrics

**Usage:**
```bash
./scripts/collect_all_metrics.sh
```

**Output**: `data/performance_metrics.csv`

### 6. Analysis Script

**Script**: `scripts/analyze_metrics.py`

Python script that analyzes the collected data and produces:

1. **Correctness Verification**: Compares result hashes between implementations
2. **Performance Summary**: Tables comparing time and memory for each algorithm
3. **Crossover Analysis**: Identifies memory crossover points where motlie_db becomes more efficient

**Usage:**
```bash
python3 scripts/analyze_metrics.py
# or
python3 scripts/analyze_metrics.py path/to/metrics.csv
```

## Implementation Details

### Reference Implementations Used

| Algorithm          | Reference Library    | Version |
|--------------------|---------------------|---------|
| DFS                | petgraph            | 0.6     |
| BFS                | petgraph            | 0.6     |
| Topological Sort   | petgraph            | 0.6     |
| Dijkstra           | pathfinding         | 4.0     |
| PageRank           | Custom (bespoke)    | N/A     |

### Memory Measurement

Memory is measured using RSS (Resident Set Size) delta:
- **macOS**: `ps -o rss=` command
- **Linux**: `/proc/self/status` VmRSS field

Memory delta captures the incremental memory cost of:
- Graph data structures
- Algorithm state (visited sets, queues, etc.)
- Intermediate results

### Graph Structure

Each algorithm uses a domain-appropriate graph structure:

**DFS**: Connected clusters with inter-cluster bridges
- Base: 10 nodes (A-J), 11 edges
- Pattern: Tree-like structure with cycles

**BFS**: Level-based tree structure
- Base: 10 nodes (Root, L1_A/B, L2_A/B/C/D, L3_A/B/C), breadth-first layout
- Pattern: Perfect for demonstrating level-order traversal

**Topological Sort**: DAG pipeline
- Base: 10 nodes (Start, Task_A-H, End), task dependency structure
- Pattern: Linear pipeline with some parallel paths

**Dijkstra**: Hub-and-spoke network
- Base: 10 nodes (Hub, N/S/E/W/NE/NW/SE/SW, Center), weighted edges
- Pattern: Shortest path from Hub to Center

**PageRank**: Web page link structure
- Base: 10 nodes (Page_A-J), directed links simulating web pages
- Pattern: Dense link structure with authority/hub pages

## Data Collection Process

### Step 1: Build
```bash
cd /Users/dchung/projects/github.com/chungers/motlie
cargo build --release --examples
```

### Step 2: Run Data Collection
```bash
cd examples/graph
./scripts/collect_all_metrics.sh
```

### Step 3: Analyze Results
```bash
python3 scripts/analyze_metrics.py
```

## Expected Outcomes

### Correctness
All result hashes should match between reference and motlie_db implementations for each algorithm at each scale.

### Performance Trends
Based on previous analysis:

1. **Memory Crossover Points** (where motlie_db uses ≤ memory than reference):
   - **Dijkstra**: ~1,000 nodes (scale=100-1000)
   - **DFS/BFS**: ~10,000-100,000 nodes (scale=1000-10000)
   - **Topological Sort**: Projected at ~15,000-25,000 nodes
   - **PageRank**: To be determined

2. **Time Performance**:
   - Reference implementations generally faster for small graphs
   - motlie_db shows consistent performance across scales
   - Trade-off: Speed vs. Memory efficiency

### Analysis Output

The analysis script will produce:

1. **Correctness table**: ✓/✗ for each algorithm at each scale
2. **Performance tables**: Side-by-side comparison of time and memory
3. **Crossover analysis**: Identification of efficiency breakpoints
4. **Summary findings**: Key insights and trends

## Future Work

- Add visualization charts for memory trends
- Extend to larger scales (100K, 1M nodes) where disk space permits
- Add persistence benchmarks (write vs. read performance)
- Compare with other graph databases
- Add concurrent query performance testing

## Files Changed

### Core Library
- `examples/graph/common.rs`: Added Implementation enum, hash functions, CSV output

### Demo Programs (all refactored similarly)
- `examples/graph/dfs.rs`
- `examples/graph/bfs.rs`
- `examples/graph/toposort.rs`
- `examples/graph/dijkstra.rs`
- `examples/graph/pagerank.rs`

### Scripts
- `examples/graph/scripts/collect_all_metrics.sh` (new)
- `examples/graph/scripts/analyze_metrics.py` (new)

### Documentation
- `examples/graph/REDESIGN_SUMMARY.md` (this file)

## Compilation Status

✅ All examples compile successfully with no errors (only warnings for unused helper functions)

```
Finished `release` profile [optimized] target(s) in 1.51s
```

## Testing

To test a single algorithm manually:

```bash
# Test DFS with reference implementation at scale 10
./target/release/examples/dfs reference /tmp/test_db 10

# Test DFS with motlie_db implementation at scale 10
./target/release/examples/dfs motlie_db /tmp/test_db 10
```

Output will be a single CSV line:
```
DFS,reference,10,100,110,1.2345,156.78,abc123def456
```

## Contact

For questions or issues, refer to the main project README or open an issue.
