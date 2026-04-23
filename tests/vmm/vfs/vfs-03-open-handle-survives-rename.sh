#!/usr/bin/env bash
# A read fd opened on a file should remain valid after the file is renamed
# in place. This is the property the implementer's
# `overlay_rename_disk_to_overlay_keeps_source_open_handle_working` test
# pins on the Rust side; this script asserts the same from the guest's POV.
set -u
TEST_NAME=vfs-03-open-handle-survives-rename
declare -F pass >/dev/null 2>&1 || . "$(dirname "$0")/../shared/result.sh"

src=/workspace/.vmm-rename-src-$$
dst=/workspace/.vmm-rename-dst-$$
echo "hello-handle" > "$src"
exec 7<"$src"
mv "$src" "$dst"
out=$(cat <&7)
exec 7<&-
rm -f "$dst"
if [[ "$out" != "hello-handle" ]]; then
  fail "stale-handle got='$out'"
fi
pass "fd-survives-rename"
