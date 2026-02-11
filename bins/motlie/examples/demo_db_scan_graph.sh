#!/usr/bin/env bash
# demo_db_scan_graph.sh - Demonstrate `motlie db list` and `motlie db scan` for graph CFs.
#
# Creates a sample graph database with versioned nodes/edges and active periods
# via the standalone build_graph tool, then scans every graph column family.
#
# Usage:
#   ./bins/motlie/examples/demo_db_scan_graph.sh
#
# Prerequisites:
#   cargo build  (builds the motlie CLI)
#   cargo build --manifest-path bins/motlie/examples/build_graph/Cargo.toml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROOT_DIR"

# ── Build ────────────────────────────────────────────────────────────────
echo "=== Building motlie CLI and build_graph tool ==="
cargo build 2>&1 | tail -3
cargo build --manifest-path bins/motlie/examples/build_graph/Cargo.toml 2>&1 | tail -3

MOTLIE="./target/debug/motlie"
BUILD_GRAPH="./bins/motlie/examples/build_graph/target/debug/build_graph"
DB_DIR="$ROOT_DIR/target/motlie_graph_demo_$$"
mkdir -p "$DB_DIR"

trap 'echo; echo "Cleaning up $DB_DIR"; rm -rf "$DB_DIR"' EXIT

# ── Populate graph ───────────────────────────────────────────────────────
echo
echo "=== Building sample graph with versioned nodes and active periods ==="
echo "    Scenario: evolving org chart (Jan 2025 → Jul 2025)"
echo

# Capture node IDs from build_graph stdout
eval "$("$BUILD_GRAPH" "$DB_DIR/graphdb")"

echo
echo "  Node IDs:"
echo "    Alice = $ALICE"
echo "    Bob   = $BOB"
echo "    Carol = $CAROL"
echo "    Dave  = $DAVE"

# ── List CFs ─────────────────────────────────────────────────────────────
echo
echo "================================================================"
echo "=== motlie db list ==="
echo "================================================================"
"$MOTLIE" db -p "$DB_DIR/graphdb" list

# ── Scan: graph/nodes (all records) ─────────────────────────────────────
echo
echo "================================================================"
echo "=== graph/nodes  (all records, latest version per node) ==="
echo "================================================================"
"$MOTLIE" db -p "$DB_DIR/graphdb" scan graph/nodes -f table

# ── Temporal filtering: active at 2025-03-15 ────────────────────────────
echo
echo "================================================================"
echo "=== graph/nodes  --at 2025-03-15 (during Carol's internship) ==="
echo "=== Expected: Bob, Carol only ==="
echo "=== (Alice v3 changed active_from to Apr 2025; Dave not yet active) ==="
echo "================================================================"
"$MOTLIE" db -p "$DB_DIR/graphdb" scan graph/nodes 2025-03-15 -f table

# ── Temporal filtering: active at 2025-09-01 ────────────────────────────
echo
echo "================================================================"
echo "=== graph/nodes  --at 2025-09-01 (after Carol left, Dave onboard) ==="
echo "=== Expected: Alice, Bob, Dave (Carol no longer active) ==="
echo "================================================================"
"$MOTLIE" db -p "$DB_DIR/graphdb" scan graph/nodes 2025-09-01 -f table

# ── Node version history ────────────────────────────────────────────────
echo
echo "================================================================"
echo "=== graph/node_version_history  (Alice has 3 versions) ==="
echo "================================================================"
"$MOTLIE" db -p "$DB_DIR/graphdb" scan graph/node_version_history -f table

# ── Node summaries (content-addressed) ──────────────────────────────────
echo
echo "================================================================"
echo "=== graph/node_summaries  (unique content blobs) ==="
echo "================================================================"
"$MOTLIE" db -p "$DB_DIR/graphdb" scan graph/node_summaries -f table

# ── Forward edges ────────────────────────────────────────────────────────
echo
echo "================================================================"
echo "=== graph/forward_edges  (all, latest version per edge) ==="
echo "================================================================"
"$MOTLIE" db -p "$DB_DIR/graphdb" scan graph/forward_edges -f table

# ── Forward edges filtered at 2025-03-15 ────────────────────────────────
echo
echo "================================================================"
echo "=== graph/forward_edges  --at 2025-03-15 ==="
echo "=== Expected: 3 edges (Carol->Bob, Alice->Carol, Alice->Bob) ==="
echo "================================================================"
"$MOTLIE" db -p "$DB_DIR/graphdb" scan graph/forward_edges 2025-03-15 -f table

# ── Forward edges filtered at 2025-09-01 ────────────────────────────────
echo
echo "================================================================"
echo "=== graph/forward_edges  --at 2025-09-01 ==="
echo "=== Expected: 2 edges (Alice->Bob, Dave->Alice; Carol edges expired) ==="
echo "================================================================"
"$MOTLIE" db -p "$DB_DIR/graphdb" scan graph/forward_edges 2025-09-01 -f table

# ── Reverse edges ────────────────────────────────────────────────────────
echo
echo "================================================================"
echo "=== graph/reverse_edges  (inbound edges) ==="
echo "================================================================"
"$MOTLIE" db -p "$DB_DIR/graphdb" scan graph/reverse_edges -f table

# ── Edge version history ────────────────────────────────────────────────
echo
echo "================================================================"
echo "=== graph/edge_version_history ==="
echo "================================================================"
"$MOTLIE" db -p "$DB_DIR/graphdb" scan graph/edge_version_history -f table

# ── Edge summaries ──────────────────────────────────────────────────────
echo
echo "================================================================"
echo "=== graph/edge_summaries  (content-addressed edge blobs) ==="
echo "================================================================"
"$MOTLIE" db -p "$DB_DIR/graphdb" scan graph/edge_summaries -f table

# ── Names ────────────────────────────────────────────────────────────────
echo
echo "================================================================"
echo "=== graph/names  (name hash -> string) ==="
echo "================================================================"
"$MOTLIE" db -p "$DB_DIR/graphdb" scan graph/names -f table

# ── Summary indices ─────────────────────────────────────────────────────
echo
echo "================================================================"
echo "=== graph/node_summary_index  (hash -> node version index) ==="
echo "================================================================"
"$MOTLIE" db -p "$DB_DIR/graphdb" scan graph/node_summary_index -f table

echo
echo "================================================================"
echo "=== graph/edge_summary_index  (hash -> edge version index) ==="
echo "================================================================"
"$MOTLIE" db -p "$DB_DIR/graphdb" scan graph/edge_summary_index -f table

# ── Graph meta ──────────────────────────────────────────────────────────
echo
echo "================================================================"
echo "=== graph/meta ==="
echo "================================================================"
"$MOTLIE" db -p "$DB_DIR/graphdb" scan graph/meta -f table || true

echo
echo "=== Done ==="
