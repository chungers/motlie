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
  "create_reader_with_storage\\([^\\)]*,\\s*storage\\.clone\\("
  "search_reader\\.search_knn\\("
  "InsertVector::new\\(embedding_code"
)

echo "Checking motlie-db docs/examples for stale API snippets..."

failed=0
for pattern in "${PATTERNS[@]}"; do
  if rg -n --no-heading "${pattern}" "${FILES[@]}"; then
    echo "ERROR: stale documentation pattern found: ${pattern}"
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi

echo "No stale documentation patterns detected."
