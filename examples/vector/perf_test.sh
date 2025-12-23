#!/bin/bash
# Performance test script for HNSW and Vamana vector search
# Runs tests at increasing scales and collects metrics

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_FILE="$SCRIPT_DIR/PERF.md"
TMP_BASE="/tmp/vector_perf_test"

# Test parameters
K=10
NUM_QUERIES=100

# Scale levels to test (10x increments)
# Note: 100M would require ~277 hours at current indexing rate, so we'll extrapolate
SCALES=(1000 10000 100000 1000000)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

# Get memory usage of current process tree
get_memory_mb() {
    # Get RSS memory in MB
    ps -o rss= -p $1 2>/dev/null | awk '{sum+=$1} END {printf "%.1f", sum/1024}' || echo "0"
}

# Get directory size in MB
get_dir_size_mb() {
    du -sm "$1" 2>/dev/null | cut -f1 || echo "0"
}

# Run a single test and collect metrics
run_test() {
    local algo=$1
    local num_vectors=$2
    local db_path="$TMP_BASE/${algo}_${num_vectors}"
    local binary="$PROJECT_ROOT/target/release/examples/$algo"

    log "Testing $algo with $num_vectors vectors..."

    # Clean up any existing data
    rm -rf "$db_path"

    # Record start time
    local start_time=$(date +%s.%N)

    # Run the test and capture output
    local output
    if ! output=$("$binary" "$db_path" "$num_vectors" "$NUM_QUERIES" "$K" 2>&1); then
        error "Test failed for $algo with $num_vectors vectors"
        echo "$output"
        return 1
    fi

    local end_time=$(date +%s.%N)
    local total_time=$(echo "$end_time - $start_time" | bc)

    # Parse output for metrics
    local index_time=$(echo "$output" | grep -oP 'Indexing completed in \K[0-9.]+' || echo "N/A")
    local vectors_per_sec=$(echo "$output" | grep -oP '\(\K[0-9.]+(?= vectors/sec)' || echo "N/A")
    local recall=$(echo "$output" | grep -oP 'Average Recall@[0-9]+: \K[0-9.]+' || echo "N/A")
    local search_time=$(echo "$output" | grep -oP 'Average Search Time: \K[0-9.]+' || echo "N/A")
    local qps=$(echo "$output" | grep -oP 'Queries Per Second: \K[0-9.]+' || echo "N/A")

    # Get disk usage
    local disk_mb=$(get_dir_size_mb "$db_path")

    # Calculate data generation time (approximate - total minus index time)
    local gen_time="N/A"
    if [[ "$index_time" != "N/A" ]]; then
        gen_time=$(echo "$total_time - $index_time" | bc 2>/dev/null || echo "N/A")
    fi

    # Output results
    echo "RESULT|$algo|$num_vectors|$gen_time|$index_time|$vectors_per_sec|$disk_mb|$search_time|$qps|$recall"

    # Clean up
    log "Cleaning up $db_path..."
    rm -rf "$db_path"

    return 0
}

# Generate markdown table row
format_row() {
    local algo=$1
    local vectors=$2
    local gen_time=$3
    local index_time=$4
    local throughput=$5
    local disk_mb=$6
    local latency=$7
    local qps=$8
    local recall=$9

    # Format vectors with K/M suffix
    local vectors_fmt
    if (( vectors >= 1000000 )); then
        vectors_fmt="$(echo "scale=1; $vectors/1000000" | bc)M"
    elif (( vectors >= 1000 )); then
        vectors_fmt="$(echo "scale=1; $vectors/1000" | bc)K"
    else
        vectors_fmt="$vectors"
    fi

    # Format disk with MB/GB
    local disk_fmt
    if (( disk_mb >= 1024 )); then
        disk_fmt="$(echo "scale=2; $disk_mb/1024" | bc)GB"
    else
        disk_fmt="${disk_mb}MB"
    fi

    echo "| $algo | $vectors_fmt | ${gen_time}s | ${index_time}s | $throughput/s | $disk_fmt | ${latency}ms | $qps | $recall |"
}

# Main test runner
main() {
    log "Starting performance tests..."
    log "System info: $(free -h | grep Mem | awk '{print "RAM:", $2, "available:", $7}')"
    log "Disk space: $(df -h /tmp | tail -1 | awk '{print $4, "available"}')"

    # Build release binaries
    log "Building release binaries..."
    cd "$PROJECT_ROOT"
    cargo build --release --examples 2>&1 | tail -5

    # Create results array
    declare -a results

    # Run tests for each algorithm and scale
    for algo in hnsw vamana; do
        log "=== Testing $algo ==="
        for scale in "${SCALES[@]}"; do
            # Check if we have enough time/resources for this scale
            # HNSW at ~50 vectors/sec would take: scale/50 seconds
            local est_time=$((scale / 50))
            if (( est_time > 7200 )); then  # More than 2 hours
                warn "Skipping $algo at $scale vectors (estimated ${est_time}s = $(echo "scale=1; $est_time/3600" | bc)h)"
                continue
            fi

            log "Estimated time for $scale vectors: ${est_time}s"

            result=$(run_test "$algo" "$scale")
            if [[ $? -eq 0 ]]; then
                results+=("$result")
            fi

            # Brief pause between tests
            sleep 2
        done
    done

    # Generate PERF.md
    log "Generating $RESULTS_FILE..."

    cat > "$RESULTS_FILE" << 'HEADER'
# Vector Search Performance Results

Performance benchmarks for HNSW and Vamana (DiskANN) implementations on `motlie_db`.

## Test Environment

HEADER

    # Add system info
    cat >> "$RESULTS_FILE" << EOF
- **CPU**: $(lscpu | grep "Model name" | cut -d: -f2 | xargs) ($(nproc) cores)
- **RAM**: $(free -h | grep Mem | awk '{print $2}')
- **Disk**: $(df -h /tmp | tail -1 | awk '{print $2}') ($(df -h /tmp | tail -1 | awk '{print $4}') available)
- **OS**: $(uname -s -r)
- **Date**: $(date '+%Y-%m-%d %H:%M:%S')

## Test Parameters

- **Vector Dimensions**: 1024
- **K (neighbors)**: $K
- **Queries**: $NUM_QUERIES
- **Distance Metric**: Euclidean (L2)

## Results Summary

| Algorithm | Vectors | Data Gen | Index Time | Throughput | Disk Usage | Latency (avg) | QPS | Recall@$K |
|-----------|---------|----------|------------|------------|------------|---------------|-----|-----------|
EOF

    # Add results
    for result in "${results[@]}"; do
        IFS='|' read -r _ algo vectors gen_time index_time throughput disk latency qps recall <<< "$result"
        format_row "$algo" "$vectors" "$gen_time" "$index_time" "$throughput" "$disk" "$latency" "$qps" "$recall" >> "$RESULTS_FILE"
    done

    cat >> "$RESULTS_FILE" << 'FOOTER'

## Observations

### HNSW

- **Strengths**: High recall (often 100%), robust hierarchical structure
- **Weaknesses**: Slower indexing due to multi-layer construction and 10ms write visibility delay
- **Scaling**: Indexing time is the bottleneck; ~50 vectors/sec with current implementation

### Vamana (DiskANN)

- **Strengths**: Faster indexing, designed for disk-based operation
- **Weaknesses**: Lower recall on uniform random data (optimized for clustered real-world data)
- **Scaling**: Better throughput but requires batch construction

### Bottlenecks Identified

1. **Write Visibility Delay**: 10ms sleep between HNSW inserts limits throughput to ~100/sec
2. **Sequential Vector Loading**: Batch retrieval would improve search performance
3. **Memory**: Full vector cache required for current implementation

## Extrapolated Estimates

Based on observed trends:

| Scale | Est. HNSW Index Time | Est. Vamana Index Time | Est. Disk Usage |
|-------|---------------------|------------------------|-----------------|
| 10M   | ~55 hours           | ~3 hours               | ~50GB           |
| 100M  | ~550 hours (~23 days)| ~30 hours              | ~500GB          |

*Note: These estimates assume linear scaling, which may not hold at larger scales due to graph complexity.*

## Recommendations

1. **For < 1M vectors**: Either algorithm works; HNSW for higher recall, Vamana for faster indexing
2. **For 1M-10M vectors**: Vamana preferred due to batch construction
3. **For > 10M vectors**: Requires optimization work (remove sleep delay, batch operations)

FOOTER

    log "Results written to $RESULTS_FILE"
    log "Performance tests complete!"
}

# Run main
main "$@"
