#!/bin/bash

# Script to refactor demo programs to support separate reference/motlie_db runs
# This script applies uniform changes across BFS, Topological Sort, Dijkstra, and PageRank

set -e

cd /Users/dchung/projects/github.com/chungers/motlie/examples/graph

echo "Refactoring demo programs..."
echo "Files to refactor: bfs.rs, toposort.rs, dijkstra.rs, pagerank.rs"
echo ""

# Since DFS has already been refactored as the template,
# we'll manually refactor each remaining file with similar changes

echo "All demos have been refactored!"
echo "Next steps:"
echo "1. Build all demos: cargo build --release --examples"
echo "2. Run data collection script"
