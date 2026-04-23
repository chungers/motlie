#!/usr/bin/env bash
# Symlink resolves correctly via cat; readlink returns the original target
# string verbatim.
set -u
TEST_NAME=vfs-08-symlink
. "$(dirname "$0")/../shared/result.sh"

target=/workspace/.vmm-symtarget-$$
link=/workspace/.vmm-symlink-$$
echo "via-symlink" > "$target"
if ! ln -s "$target" "$link" 2>/dev/null; then
  rm -f "$target"
  skip "symlink-not-supported"
fi
read_via_link=$(cat "$link" 2>/dev/null || true)
got_target=$(readlink "$link" 2>/dev/null || true)
rm -f "$target" "$link"
if [[ "$read_via_link" != "via-symlink" ]]; then
  fail "follow-failed got='$read_via_link'"
fi
if [[ "$got_target" != "$target" ]]; then
  fail "readlink-mismatch got='$got_target' want='$target'"
fi
pass "ok"
