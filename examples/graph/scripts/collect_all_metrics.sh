#!/bin/bash

# Comprehensive data collection script for all graph algorithm demos
# Runs all algorithms (DFS, BFS, Topological Sort, Dijkstra, PageRank)
# with both implementations (reference, motlie_db) at all scales (1, 10, 100, 1000, 10000)

set -e

# Configuration
PROJECT_ROOT="/Users/dchung/projects/github.com/chungers/motlie"
EXAMPLES_DIR="$PROJECT_ROOT/examples/graph"
OUTPUT_DIR="$EXAMPLES_DIR/data"
BINARY_DIR="$PROJECT_ROOT/target/release/examples"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Output file
OUTPUT_FILE="$OUTPUT_DIR/performance_metrics.csv"

# Algorithms to test
ALGORITHMS=("dfs" "bfs" "toposort" "dijkstra" "pagerank")

# Implementations to test
IMPLEMENTATIONS=("reference" "motlie_db")

# Scales to test
SCALES=(1 10 100 1000 10000 100000)

# Initialize CSV file with header
echo "algorithm,implementation,scale,nodes,edges,time_ms,memory_kb,result_hash" > "$OUTPUT_FILE"

echo "=========================================="
echo "Graph Algorithm Performance Data Collection"
echo "=========================================="
echo ""
echo "Algorithms: ${ALGORITHMS[*]}"
echo "Implementations: ${IMPLEMENTATIONS[*]}"
echo "Scales: ${SCALES[*]}"
echo "Output: $OUTPUT_FILE"
echo ""
echo "Starting data collection..."
echo ""

# Track progress
total_runs=$((${#ALGORITHMS[@]} * ${#IMPLEMENTATIONS[@]} * ${#SCALES[@]}))
current_run=0

# Run all combinations
for algo in "${ALGORITHMS[@]}"; do
    for impl in "${IMPLEMENTATIONS[@]}"; do
        for scale in "${SCALES[@]}"; do
            current_run=$((current_run + 1))

            # Create unique database path for each run
            db_path="/tmp/graph_${algo}_${impl}_${scale}"

            # Remove existing database
            rm -rf "$db_path"

            echo "[$current_run/$total_runs] Running: $algo | $impl | scale=$scale"

            # Set timeout (5 minutes = 300 seconds)
            TIMEOUT=300

            # Run the algorithm with timeout and append output to CSV
            if timeout $TIMEOUT "$BINARY_DIR/$algo" "$impl" "$db_path" "$scale" >> "$OUTPUT_FILE" 2>/dev/null; then
                echo "  ✓ Success"
            else
                exit_code=$?
                if [ $exit_code -eq 124 ]; then
                    echo "  ⏱ Timeout (>5 minutes)"
                    echo "$algo,$impl,$scale,TIMEOUT,TIMEOUT,TIMEOUT,TIMEOUT,TIMEOUT" >> "$OUTPUT_FILE"
                else
                    echo "  ✗ Failed (error code: $exit_code)"
                    echo "$algo,$impl,$scale,ERROR,ERROR,ERROR,ERROR,ERROR" >> "$OUTPUT_FILE"
                fi
            fi

            # Clean up database to save disk space
            rm -rf "$db_path"
        done
    done
    echo ""
done

echo "=========================================="
echo "Data collection complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_FILE"
echo "Total runs: $current_run"
echo ""
echo "Next steps:"
echo "1. Review the CSV file: cat $OUTPUT_FILE"
echo "2. Run analysis: python3 scripts/analyze_metrics.py"
echo ""
