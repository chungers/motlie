#!/usr/bin/env bash
# Shared pass/fail/skip contract for tests/vmm/ scripts.
#
# Usage modes:
#
#   1. Bundled by driver.sh (the supported invocation):
#      The driver concatenates this file in front of each test script and
#      pipes the bundle to bash via the configured exec-cmd. In this mode,
#      pass/fail/skip are already defined when the test body runs.
#
#   2. Standalone for local debugging:
#      The test scripts source this file defensively when pass/fail/skip
#      are not already defined, e.g.:
#        declare -F pass >/dev/null 2>&1 || . "$(dirname "$0")/../shared/result.sh"
#      That works for `bash tests/vmm/vnet/vnet-01-...sh` (because $0 is the
#      script path) but does NOT work for `bash -s < tests/vmm/.../foo.sh`
#      (because $0 is "bash" and `dirname $0` won't find this file). For the
#      latter case use the driver, which bundles instead of sourcing.
#
# Output contract (parsed by tests/vmm/driver.sh):
#   TEST=<name> RESULT=pass|fail|skip [DETAIL=...]
#
# Exit codes:
#   pass / skip -> 0
#   fail        -> 1
#
# TEST_NAME is checked when pass/fail/skip is called, not at source time, so
# bundling order is independent of when the test sets TEST_NAME.

# Intentionally not setting `set -u` here. Each test script sets its own
# shell options. Sourcing this helper must not change the caller's options.

pass() {
  local name="${TEST_NAME:?TEST_NAME must be set before calling pass}"
  local detail="${1:-}"
  if [[ -n "$detail" ]]; then
    echo "TEST=${name} RESULT=pass DETAIL=${detail}"
  else
    echo "TEST=${name} RESULT=pass"
  fi
  exit 0
}

fail() {
  local name="${TEST_NAME:?TEST_NAME must be set before calling fail}"
  local detail="${1:-}"
  if [[ -n "$detail" ]]; then
    echo "TEST=${name} RESULT=fail DETAIL=${detail}"
  else
    echo "TEST=${name} RESULT=fail"
  fi
  exit 1
}

# Skip when a precondition isn't met (missing tool, wrong env, etc.). Counted
# separately by the driver and does not fail the suite.
skip() {
  local name="${TEST_NAME:?TEST_NAME must be set before calling skip}"
  local detail="${1:-}"
  if [[ -n "$detail" ]]; then
    echo "TEST=${name} RESULT=skip DETAIL=${detail}"
  else
    echo "TEST=${name} RESULT=skip"
  fi
  exit 0
}
