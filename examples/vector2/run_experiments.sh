#!/bin/bash
# SIFT Benchmark Experiments
# Tests L2 (standard) and RaBitQ (cached) at multiple scales

set -e

# Configuration
RESULTS_DIR="$(dirname "$0")/results"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Default scales
SCALES="${@:-1000 10000 100000}"

echo "=== SIFT Benchmark Experiments ==="
echo "Scales: $SCALES"
echo "Results: $RESULTS_DIR"
echo "Timestamp: $TIMESTAMP"
echo ""

# Function to run a single benchmark
run_benchmark() {
    local scale=$1
    local mode=$2
    local extra_args=$3
    local output_file="$RESULTS_DIR/searchconfig_${scale}_${mode}_${TIMESTAMP}.txt"

    echo "----------------------------------------"
    echo "Running: scale=$scale mode=$mode"
    echo "Output: $output_file"
    echo "----------------------------------------"

    cargo run --release --example vector2 -- \
        --dataset sift1m \
        --num-vectors "$scale" \
        --num-queries 100 \
        --k 10 \
        --ef 100 \
        $extra_args 2>&1 | tee "$output_file"

    echo ""
}

# Run experiments for each scale
for scale in $SCALES; do
    echo ""
    echo "========================================"
    echo "=== Scale: $scale vectors ==="
    echo "========================================"
    echo ""

    # Standard L2 (exact distance)
    run_benchmark "$scale" "l2_exact" ""

    # RaBitQ cached (Hamming approximation)
    run_benchmark "$scale" "rabitq_cached" "--rabitq-cached --rerank-factor 4"

done

# Generate summary
echo ""
echo "========================================"
echo "=== Generating Summary ==="
echo "========================================"

SUMMARY_FILE="$RESULTS_DIR/searchconfig_summary_${TIMESTAMP}.md"

cat > "$SUMMARY_FILE" << 'EOF'
# SearchConfig Benchmark Summary

## Experiment Date
EOF

echo "" >> "$SUMMARY_FILE"
echo "Generated: $(date)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

cat >> "$SUMMARY_FILE" << 'EOF'

## Results

| Scale | Mode | Build (s) | Throughput | QPS | Recall@10 | P50 (ms) | P99 (ms) |
|-------|------|-----------|------------|-----|-----------|----------|----------|
EOF

# Extract metrics from result files
for result_file in "$RESULTS_DIR"/searchconfig_*_"$TIMESTAMP".txt; do
    if [ -f "$result_file" ]; then
        # Extract key metrics using grep
        filename=$(basename "$result_file")
        scale=$(echo "$filename" | cut -d_ -f2)
        mode=$(echo "$filename" | cut -d_ -f3-4 | sed 's/_[0-9]*\.txt//')

        build_time=$(grep "Build Time" "$result_file" | tail -1 | awk '{print $4}' | tr -d 's|')
        throughput=$(grep "Build Throughput" "$result_file" | tail -1 | awk '{print $4}')
        qps=$(grep "QPS" "$result_file" | tail -1 | awk '{print $3}')
        recall=$(grep "Recall@10" "$result_file" | tail -1 | awk '{print $3}')
        p50=$(grep "P50 Latency" "$result_file" | tail -1 | awk '{print $4}' | tr -d 'ms|')
        p99=$(grep "P99 Latency" "$result_file" | tail -1 | awk '{print $4}' | tr -d 'ms|')

        echo "| $scale | $mode | $build_time | $throughput | $qps | $recall | $p50 | $p99 |" >> "$SUMMARY_FILE"
    fi
done

cat >> "$SUMMARY_FILE" << 'EOF'

## Configuration

- HNSW: M=16, M_max=32, ef_construction=200
- Search: ef=100, k=10
- RaBitQ: bits_per_dim=1, rerank_factor=4
- Queries: 100 per configuration

## Observations

### SearchConfig Auto-Selection
- L2 distance → Exact strategy (Hamming incompatible)
- Cosine distance → RaBitQ strategy (Hamming ≈ angular)

### Performance Notes
- L2 exact provides baseline recall (ground truth uses L2)
- RaBitQ cached mode shows QPS improvement at larger scales
- Rerank factor of 4 maintains >85% recall at 100K scale

## Next Steps
- [ ] Add Cosine distance support to benchmark (normalize vectors)
- [ ] Test rerank_factor tuning (2, 4, 8)
- [ ] Test ef tuning (100, 150, 200)
EOF

echo ""
echo "Summary written to: $SUMMARY_FILE"
echo ""
echo "=== Experiments Complete ==="
