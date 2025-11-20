#!/bin/bash

cd /Users/dchung/projects/github.com/chungers/motlie

for scale in 1 10 100; do
  echo "========== Scale: $scale =========="
  for algo in dfs bfs toposort dijkstra pagerank; do
    echo "--- $algo at scale=$scale ---"
    ./target/release/examples/$algo /tmp/mem_${algo}_${scale} $scale 2>&1 | grep -A 10 "Memory Usage"
    echo ""
  done
done
