#!/usr/bin/env bash
# Pin the documented xattr behavior. virtio-fs currently returns
# "Operation not supported" for setfattr; this test asserts that contract.
# When/if xattr support lands, flip the assertion (don't delete the test).
set -u
TEST_NAME=vfs-04-xattr-contract
. "$(dirname "$0")/../shared/result.sh"

if ! command -v setfattr >/dev/null 2>&1; then
  skip "setfattr-missing"
fi
f=/workspace/.vmm-xattr-$$
echo a > "$f"
err=$(setfattr -n user.test -v y "$f" 2>&1 || true)
rm -f "$f"
if echo "$err" | grep -qE 'Operation not supported|not permitted'; then
  pass "expected-eopnotsupp"
fi
# If we got here, the contract changed (xattr now works). Surface it as a fail
# so somebody re-evaluates the assertion intentionally.
fail "contract-changed err='$err'"
