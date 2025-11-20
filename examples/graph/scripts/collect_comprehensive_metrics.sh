#!/bin/bash

cd /Users/dchung/projects/github.com/chungers/motlie

echo "# Memory Usage Analysis: motlie_db vs In-Memory Implementations"
echo ""
echo "| Algorithm | Scale | Nodes | Edges | Reference Memory | motlie_db Memory | Ratio |"
echo "|-----------|-------|-------|-------|------------------|------------------|-------|"

for scale in 1 10 100; do
  for algo in dfs bfs toposort dijkstra pagerank; do
    output=$(./ target/release/examples/$algo /tmp/mem_${algo}_${scale} $scale 2>&1)

    nodes=$(echo "$output" | grep -E "Nodes:|Nodes \(tasks\):|Locations:|Pages:" | head -1 | awk '{print $NF}')
    edges=$(echo "$output" | grep -E "Edges:|Edges \(dependencies\):|Roads:|Links:" | head -1 | awk '{print $NF}')

    ref_mem=$(echo "$output" | grep -A1 "Memory Usage" | grep -E "petgraph:|pathfinding:|Reference:" | awk '{print $2, $3}')
    motlie_mem=$(echo "$output" | grep -A2 "Memory Usage" | grep "motlie_db:" | awk '{print $2, $3}')
    ratio=$(echo "$output" | grep -E "Overhead:|Savings:" | awk '{print $2}')

    if [ -n "$nodes" ] && [ -n "$ref_mem" ]; then
      echo "| $algo | $scale | $nodes | $edges | $ref_mem | $motlie_mem | $ratio |"
    fi
  done
done
