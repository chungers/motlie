#!/usr/bin/env bash
# demo_db_scan.sh - Demonstrate `motlie db list` and `motlie db scan` for vector CFs.
#
# Creates a temporary RocksDB with 100 random vectors via bench_vector,
# then scans every vector column family with `motlie db`.
#
# Usage:
#   ./bins/motlie/examples/demo_db_scan.sh
#
# Prerequisites:
#   cargo build --features benchmark

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROOT_DIR"

# ── Build ──────────────────────────────────────────────────────────────
echo "=== Building bench_vector and motlie ==="
cargo build --features benchmark 2>&1 | tail -3

BENCH="./target/debug/bench_vector"
MOTLIE="./target/debug/motlie"
# RocksDB needs file-level locking which may not work on tmpfs/overlayfs.
# Use target/ which is always on a real filesystem.
DB_DIR="$ROOT_DIR/target/motlie_demo_$$"
mkdir -p "$DB_DIR"

trap 'echo; echo "Cleaning up $DB_DIR"; rm -rf "$DB_DIR"' EXIT

# ── Index 100 random vectors ──────────────────────────────────────────
echo
echo "=== Indexing 100 random vectors (dim=128, cosine) ==="
RUST_LOG=warn "$BENCH" index \
  --dataset random \
  --num-vectors 100 \
  --db-path "$DB_DIR/vecdb" \
  --dim 128 \
  --m 16 \
  --ef-construction 64 \
  --batch-size 50 \
  --stream \
  --fresh

# ── List CFs ──────────────────────────────────────────────────────────
echo
echo "================================================================"
echo "=== motlie db list ==="
echo "================================================================"
"$MOTLIE" db -p "$DB_DIR/vecdb" list

# ── Scan each vector CF ───────────────────────────────────────────────
VECTOR_CFS=(
  "vector/embedding_specs"
  "vector/vectors"
  "vector/edges"
  "vector/binary_codes"
  "vector/vec_meta"
  "vector/graph_meta"
  "vector/id_forward"
  "vector/id_reverse"
  "vector/id_alloc"
  "vector/pending"
  "vector/lifecycle_counts"
)

for cf in "${VECTOR_CFS[@]}"; do
  echo
  echo "================================================================"
  echo "=== motlie db scan $cf  (limit 5) ==="
  echo "================================================================"
  "$MOTLIE" db -p "$DB_DIR/vecdb" scan "$cf" --limit 5 -f table || true
done

# ── Demonstrate pagination ────────────────────────────────────────────
echo
echo "================================================================"
echo "=== Pagination demo: vector/vectors page 1 (limit 3) ==="
echo "================================================================"
"$MOTLIE" db -p "$DB_DIR/vecdb" scan vector/vectors --limit 3 -f table

echo
echo "=== Pagination demo: vector/vectors page 2 (--last from page 1) ==="
# Grab the last row's EMBEDDING:VEC_ID from TSV output
LAST=$("$MOTLIE" db -p "$DB_DIR/vecdb" scan vector/vectors --limit 3 -f tsv \
  | tail -1 | awk -F'\t' '{print $1 ":" $2}')
echo "(cursor: $LAST)"
"$MOTLIE" db -p "$DB_DIR/vecdb" scan vector/vectors --limit 3 --last "$LAST" -f table

# ── Reverse scan ──────────────────────────────────────────────────────
echo
echo "================================================================"
echo "=== Reverse scan: vector/vec_meta (last 5) ==="
echo "================================================================"
"$MOTLIE" db -p "$DB_DIR/vecdb" scan vector/vec_meta --limit 5 --reverse -f table

echo
echo "=== Done ==="
