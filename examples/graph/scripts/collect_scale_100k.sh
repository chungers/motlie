#!/bin/bash

# Collect data for scale=100000 only
# With 5-minute timeout per test

set -e

PROJECT_ROOT="/Users/dchung/projects/github.com/chungers/motlie"
EXAMPLES_DIR="$PROJECT_ROOT/examples/graph"
OUTPUT_DIR="$EXAMPLES_DIR/data"
BINARY_DIR="$PROJECT_ROOT/target/release/examples"

# Output file - append to existing
OUTPUT_FILE="$OUTPUT_DIR/performance_metrics.csv"

# Algorithms to test
ALGORITHMS=("dfs" "bfs" "toposort" "dijkstra" "pagerank")

# Implementations to test
IMPLEMENTATIONS=("reference" "motlie_db")

# Scale to test
SCALE=100000

echo "=========================================="
echo "Scale 100000 Data Collection"
echo "=========================================="
echo ""
echo "Algorithms: ${ALGORITHMS[*]}"
echo "Implementations: ${IMPLEMENTATIONS[*]}"
echo "Scale: $SCALE"
echo "Output: $OUTPUT_FILE"
echo ""
echo "Starting data collection with 5-minute timeout per test..."
echo ""

# Track progress
total_runs=$((${#ALGORITHMS[@]} * ${#IMPLEMENTATIONS[@]}))
current_run=0

# Run all combinations for scale=100000
for algo in "${ALGORITHMS[@]}"; do
    for impl in "${IMPLEMENTATIONS[@]}"; do
        current_run=$((current_run + 1))

        # Create unique database path for each run
        db_path="/tmp/graph_${algo}_${impl}_${SCALE}"

        # Remove existing database
        rm -rf "$db_path"

        echo "[$current_run/$total_runs] Running: $algo | $impl | scale=$SCALE"

        # Set timeout (5 minutes = 300 seconds)
        TIMEOUT=300

        # Run the algorithm with timeout and append output to CSV
        if gtimeout $TIMEOUT "$BINARY_DIR/$algo" "$impl" "$db_path" "$SCALE" >> "$OUTPUT_FILE" 2>/dev/null; then
            echo "  ✓ Success"
        else
            exit_code=$?
            if [ $exit_code -eq 124 ]; then
                echo "  ⏱ Timeout (>5 minutes)"
                echo "$algo,$impl,$SCALE,TIMEOUT,TIMEOUT,TIMEOUT,TIMEOUT,TIMEOUT" >> "$OUTPUT_FILE"
            else
                echo "  ✗ Failed (error code: $exit_code)"
                echo "$algo,$impl,$SCALE,ERROR,ERROR,ERROR,ERROR,ERROR" >> "$OUTPUT_FILE"
            fi
        fi

        # Clean up database to save disk space
        rm -rf "$db_path"
    done
    echo ""
done

echo "=========================================="
echo "Data collection complete!"
echo "=========================================="
echo ""
echo "Results appended to: $OUTPUT_FILE"
echo "Total runs: $current_run"
echo ""
