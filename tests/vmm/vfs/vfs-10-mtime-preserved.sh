#!/usr/bin/env bash
# Set a specific mtime via `touch -m -d`, then read it back via `stat -c %Y`
# and assert it round-trips. Build systems (cargo, ninja, make) all rely on
# this; if mtime is silently rounded or clobbered, incremental builds break.
set -u
TEST_NAME=vfs-10-mtime-preserved
declare -F pass >/dev/null 2>&1 || . "$(dirname "$0")/../shared/result.sh"

f=/workspace/.vmm-mtime-$$
: > "$f"
# A fixed UTC timestamp — far enough in the past that no clock skew can shadow it.
target_iso="2024-06-15T12:34:56Z"
target_epoch=$(date -u -d "$target_iso" +%s 2>/dev/null || true)
if [[ -z "$target_epoch" ]]; then
  rm -f "$f"
  skip "date-d-not-supported"
fi
touch -m -d "$target_iso" "$f"
got_epoch=$(stat -c '%Y' "$f")
rm -f "$f"
if [[ "$got_epoch" != "$target_epoch" ]]; then
  fail "mtime-drifted want=$target_epoch got=$got_epoch"
fi
pass "epoch=$got_epoch iso=$target_iso"
