#!/usr/bin/env bash
# Two concurrent processes append a unique line each. With O_APPEND semantics
# preserved through the filesystem, both lines must be present and intact
# after both writers finish.
set -u
TEST_NAME=vfs-09-concurrent-writes
declare -F pass >/dev/null 2>&1 || . "$(dirname "$0")/../shared/result.sh"

f=/workspace/.vmm-concurrent-$$
: > "$f"
( echo "writer-A-line" >> "$f" ) &
pid_a=$!
( echo "writer-B-line" >> "$f" ) &
pid_b=$!
wait "$pid_a" "$pid_b"
got_a=$(grep -c '^writer-A-line$' "$f" || true)
got_b=$(grep -c '^writer-B-line$' "$f" || true)
total_lines=$(wc -l < "$f" | tr -d ' ')
rm -f "$f"
if [[ "$got_a" -ne 1 || "$got_b" -ne 1 ]]; then
  fail "missing-or-duplicate a=$got_a b=$got_b lines=$total_lines"
fi
pass "a=$got_a b=$got_b lines=$total_lines"
