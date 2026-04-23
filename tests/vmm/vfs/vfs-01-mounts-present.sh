#!/usr/bin/env bash
# Both /workspace and /home/<current-user> are mounted and writable.
set -u
TEST_NAME=vfs-01-mounts-present
. "$(dirname "$0")/../shared/result.sh"

mounts=(/workspace "/home/$USER")
for mp in "${mounts[@]}"; do
  if ! findmnt -n "$mp" >/dev/null 2>&1; then
    fail "not-mounted mp=$mp"
  fi
  probe="$mp/.vmm-test-mount-$$"
  if ! ( touch "$probe" && rm "$probe" ) >/dev/null 2>&1; then
    fail "not-writable mp=$mp"
  fi
done
pass "mounts=${mounts[*]}"
