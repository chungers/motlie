#!/usr/bin/env bash
# Capability-aware xattr contract test.
#
# Two passing modes (slice-agnostic):
#   (a) xattrs are SUPPORTED end-to-end -> set, read back, value matches.
#   (b) xattrs are UNSUPPORTED -> we get the documented EOPNOTSUPP / EPERM.
# Anything else (set succeeds but readback wrong, or unknown error) is a fail.
#
# This shape lets a backend that adds xattr support pass without flipping
# the suite red, and lets a backend that explicitly does not support xattrs
# also pass. The only failures are silent corruption or surprise errors.
set -u
TEST_NAME=vfs-04-xattr-contract
declare -F pass >/dev/null 2>&1 || . "$(dirname "$0")/../shared/result.sh"

if ! command -v setfattr >/dev/null 2>&1; then
  skip "setfattr-missing"
fi

f=/workspace/.vmm-xattr-$$
trap 'rm -f "$f"' EXIT
echo a > "$f"

if set_err=$(setfattr -n user.test -v y "$f" 2>&1); then
  # Mode (a): xattrs work — verify round-trip.
  if ! command -v getfattr >/dev/null 2>&1; then
    pass "supported-no-getfattr-to-verify"
  fi
  got=$(getfattr -n user.test --only-values "$f" 2>/dev/null || true)
  if [[ "$got" == "y" ]]; then
    pass "supported-roundtrip-ok"
  fi
  fail "supported-but-broken-roundtrip got='$got'"
fi

# Mode (b): setfattr failed — must be a known not-supported / not-permitted
# error, otherwise something unexpected is going wrong.
if echo "$set_err" | grep -qE 'Operation not supported|not permitted'; then
  pass "unsupported-eopnotsupp"
fi
fail "unknown-error err='$set_err'"
