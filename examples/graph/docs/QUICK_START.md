# Quick Start Guide

## Running Individual Tests

Each demo now accepts 3 arguments for independent testing:

```bash
./target/release/examples/<algorithm> <implementation> <db_path> <scale>
```

**Parameters:**
- `implementation`: `reference` or `motlie_db`
- `db_path`: Path to database directory (e.g., `/tmp/test_db`)
- `scale`: Scale factor (1, 10, 100, 1000, 10000)

**Examples:**

```bash
# Run DFS with reference implementation at scale 100
./target/release/examples/dfs reference /tmp/dfs_test 100

# Run PageRank with motlie_db at scale 1000
./target/release/examples/pagerank motlie_db /tmp/pr_test 1000

# Run Dijkstra with reference at scale 10
./target/release/examples/dijkstra reference /tmp/dijk_test 10
```

**Output:** Single CSV line with metrics
```
DFS,reference,100,1000,1298,0.0213,144,cf6b627af12bbda5
```

## Running Complete Data Collection

Automated script runs all 50 test combinations:

```bash
cd examples/graph
./scripts/collect_all_metrics.sh
```

**Output:** `data/performance_metrics.csv` with all results

**Duration:** ~3-4 minutes for all 50 tests

## Analyzing Results

### View Raw Data

```bash
# View all data
cat data/performance_metrics.csv

# View specific algorithm
grep "^PageRank," data/performance_metrics.csv

# Compare at specific scale
grep ",1000," data/performance_metrics.csv
```

### Run Analysis Script

```bash
# Requires: pip install pandas
python3 scripts/analyze_metrics.py
```

**Generates:**
- Correctness verification (hash comparison)
- Performance summary tables
- Memory crossover analysis

## Understanding the Results

### CSV Columns

| Column          | Description                           | Example        |
|-----------------|---------------------------------------|----------------|
| algorithm       | Algorithm name                        | DFS            |
| implementation  | reference or motlie_db                | reference      |
| scale           | Scale factor                          | 100            |
| nodes           | Number of nodes                       | 1000           |
| edges           | Number of edges                       | 1298           |
| time_ms         | Execution time (milliseconds)         | 0.0213         |
| memory_kb       | Memory usage (kilobytes, RSS delta)   | 144            |
| result_hash     | Result hash for correctness check     | cf6b627a...    |

### Key Metrics

**Memory Ratio** = motlie_db_memory / reference_memory
- **< 1.0**: motlie_db uses less memory (crossover achieved!)
- **= 1.0**: Equal memory usage
- **> 1.0**: motlie_db uses more memory

**Example:**
```
PageRank at scale 10,000:
- Reference: 37,072 KB
- motlie_db: 8,432 KB
- Ratio: 0.23x (77% memory savings!)
```

## Quick Comparisons

### Compare Memory at Scale

```bash
# Extract memory data for scale 10,000
awk -F',' '$3==10000 {printf "%-15s %-12s %8s KB\n", $1, $2, $7}' data/performance_metrics.csv | sort
```

### Find Crossover Points

```bash
# Show where motlie_db memory ≤ reference memory (by algorithm)
awk -F',' 'NR>1 {
    key=$1","$3; 
    if ($2=="reference") ref[key]=$7; 
    if ($2=="motlie_db") motlie[key]=$7;
} 
END {
    for (k in ref) {
        if (motlie[k] <= ref[k]) {
            split(k,a,",");
            printf "%s at scale %s: motlie=%s KB, ref=%s KB (%.2fx)\n", 
                a[1], a[2], motlie[k], ref[k], motlie[k]/ref[k]
        }
    }
}' data/performance_metrics.csv
```

## Documentation

- **[README.md](../README.md)** - Overview and usage instructions
- **[DETAILED_ANALYSIS.md](DETAILED_ANALYSIS.md)** - Complete performance analysis
- **[PERFORMANCE_SUMMARY.md](PERFORMANCE_SUMMARY.md)** - Quick test results summary
- **[REDESIGN_SUMMARY.md](REDESIGN_SUMMARY.md)** - Technical redesign documentation

## Common Tasks

### Test Single Algorithm at Multiple Scales

```bash
for scale in 1 10 100 1000 10000; do
    echo "Testing DFS at scale $scale..."
    ./target/release/examples/dfs reference /tmp/dfs_$scale $scale
done
```

### Compare Specific Implementation

```bash
# Run both implementations at same scale
./target/release/examples/bfs reference /tmp/bfs_ref 1000
./target/release/examples/bfs motlie_db /tmp/bfs_motlie 1000
```

### Verify Correctness

```bash
# Compare result hashes for same algorithm/scale
grep "^Dijkstra,.*,1," data/performance_metrics.csv | awk -F',' '{print $2, $8}'
```

Output:
```
reference a8a96f68511f8d14
motlie_db a8a96f68511f8d14
```
✓ Hashes match = correctness verified!

## Troubleshooting

### Memory Usage Shows 0 KB

Some reference implementations may show 0 KB memory due to:
- Compiler optimizations
- Result caching
- Measurement timing issues

Example: Dijkstra at scale 10,000 shows 0 KB (likely cached path result)

### Different Result Hashes

Expected for some algorithms due to:
- **DFS/BFS**: Different edge iteration order (both valid traversals)
- **Topological Sort**: Multiple valid orderings exist
- **PageRank**: Floating point precision differences

Verify by comparing actual results, not just hashes.

### Build Errors

Rebuild all examples:
```bash
cd /Users/dchung/projects/github.com/chungers/motlie
cargo build --release --examples
```

## Next Steps

1. Review [DETAILED_ANALYSIS.md](DETAILED_ANALYSIS.md) for insights
2. Run analysis script with pandas for detailed reports
3. Extend testing to larger scales (20K, 50K, 100K+)
4. Implement optimization suggestions from analysis
5. Test on different graph topologies (sparse, dense, random)

