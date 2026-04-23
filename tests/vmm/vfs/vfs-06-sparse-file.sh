#!/usr/bin/env bash
# Truncate to a large size, then write a small payload at offset 0. Assert the
# resulting file's apparent size matches the truncate target but the allocated
# block count is far smaller (i.e. holes are preserved).
set -u
TEST_NAME=vfs-06-sparse-file
. "$(dirname "$0")/../shared/result.sh"

f=/workspace/.vmm-sparse-$$
target_bytes=$((1024 * 1024))   # 1 MiB apparent
payload="hi"
truncate -s "$target_bytes" "$f"
printf '%s' "$payload" | dd of="$f" conv=notrunc status=none
apparent=$(stat -c '%s' "$f")
allocated_blocks=$(stat -c '%b' "$f")
block_size=$(stat -c '%B' "$f")
allocated=$(( allocated_blocks * block_size ))
rm -f "$f"
if [[ "$apparent" -ne "$target_bytes" ]]; then
  fail "apparent-size-wrong got=$apparent want=$target_bytes"
fi
# Allocated must be strictly less than apparent for the file to be sparse.
if [[ "$allocated" -ge "$apparent" ]]; then
  fail "not-sparse apparent=$apparent allocated=$allocated"
fi
pass "apparent=$apparent allocated=$allocated"
