#!/usr/bin/env bash
# Hardlink yields a shared inode; both paths resolve to the same data; nlink
# increments correctly.
set -u
TEST_NAME=vfs-07-hardlink
. "$(dirname "$0")/../shared/result.sh"

src=/workspace/.vmm-link-src-$$
dst=/workspace/.vmm-link-dst-$$
echo "shared-content" > "$src"
if ! ln "$src" "$dst" 2>/dev/null; then
  rm -f "$src"
  skip "hardlink-not-supported"
fi
src_inode=$(stat -c '%i' "$src")
dst_inode=$(stat -c '%i' "$dst")
nlink=$(stat -c '%h' "$src")
src_content=$(cat "$src")
dst_content=$(cat "$dst")
rm -f "$src" "$dst"
if [[ "$src_inode" != "$dst_inode" ]]; then
  fail "different-inodes src=$src_inode dst=$dst_inode"
fi
if [[ "$nlink" -lt 2 ]]; then
  fail "nlink-not-bumped nlink=$nlink"
fi
if [[ "$src_content" != "$dst_content" ]]; then
  fail "content-mismatch"
fi
pass "inode=$src_inode nlink=$nlink"
