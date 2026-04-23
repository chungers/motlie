#!/usr/bin/env bash
# Shared pass/fail contract for tests/vmm/ scripts.
#
# Usage:
#   . "$(dirname "$0")/../shared/result.sh"
#   TEST_NAME=vnet-01-dns-resolves
#   pass "ip=$ip"
#   fail "no ipv4 returned"
#
# Output contract (parsed by tests/vmm/driver.sh):
#   TEST=<name> RESULT=pass|fail [DETAIL=...]
#
# Exit codes:
#   pass -> exit 0
#   fail -> exit 1

set -u

: "${TEST_NAME:?TEST_NAME must be set by the caller}"

pass() {
  local detail="${1:-}"
  if [[ -n "$detail" ]]; then
    echo "TEST=${TEST_NAME} RESULT=pass DETAIL=${detail}"
  else
    echo "TEST=${TEST_NAME} RESULT=pass"
  fi
  exit 0
}

fail() {
  local detail="${1:-}"
  if [[ -n "$detail" ]]; then
    echo "TEST=${TEST_NAME} RESULT=fail DETAIL=${detail}"
  else
    echo "TEST=${TEST_NAME} RESULT=fail"
  fi
  exit 1
}

# Convenience: skip a test (e.g. when a precondition isn't met).
# Counted as neither pass nor fail; driver tallies separately.
skip() {
  local detail="${1:-}"
  if [[ -n "$detail" ]]; then
    echo "TEST=${TEST_NAME} RESULT=skip DETAIL=${detail}"
  else
    echo "TEST=${TEST_NAME} RESULT=skip"
  fi
  exit 0
}
