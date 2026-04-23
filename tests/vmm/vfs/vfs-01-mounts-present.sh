#!/usr/bin/env bash
# Both /workspace and /home/<current-user> are mounted and writable.
set -u
TEST_NAME=vfs-01-mounts-present
declare -F pass >/dev/null 2>&1 || . "$(dirname "$0")/../shared/result.sh"

# Resolve the current user via id(1) rather than $USER so the test does not
# leak the exec-environment shape (env-var inheritance) into its surface.
me=$(id -un)
mounts=(/workspace "/home/$me")
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
