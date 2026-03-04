#!/usr/bin/env bash
set -euo pipefail

FILES=(
  "libs/db/src/storage.rs"
  "libs/db/src/reader.rs"
  "libs/db/src/writer.rs"
  "libs/db/src/query.rs"
  "libs/db/src/mutation.rs"
  "libs/db/src/vector/reader.rs"
  "libs/db/src/vector/writer.rs"
  "libs/db/src/vector/subsystem.rs"
  "libs/db/README.md"
)

PATTERNS=(
  "handles\\.writer\\(\\)\\.unwrap\\("
  "graph::Graph"
  "create_reader_with_storage\\([^\\)]*,[[:space:]]*storage\\.clone\\("
  "search_reader\\.search_knn\\("
  "InsertVector::new\\(embedding_code"
)

echo "Checking motlie-db docs/examples for stale API snippets..."

if command -v rg >/dev/null 2>&1; then
  SEARCH_TOOL="rg"
elif command -v grep >/dev/null 2>&1; then
  SEARCH_TOOL="grep"
else
  echo "ERROR: neither 'rg' nor 'grep' is available on PATH." >&2
  exit 1
fi

echo "Using search tool: ${SEARCH_TOOL}"

failed=0
for pattern in "${PATTERNS[@]}"; do
  case "${SEARCH_TOOL}" in
    rg)
      if rg -n --no-heading "${pattern}" "${FILES[@]}"; then
        echo "ERROR: stale documentation pattern found: ${pattern}"
        failed=1
      fi
      ;;
    grep)
      if grep -nE "${pattern}" "${FILES[@]}"; then
        echo "ERROR: stale documentation pattern found: ${pattern}"
        failed=1
      fi
      ;;
  esac
done

if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi

echo "No stale documentation patterns detected."
