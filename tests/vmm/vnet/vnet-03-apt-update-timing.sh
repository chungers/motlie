#!/usr/bin/env bash
# `apt-get update` completes within a reasonable window. Catches the
# slow-egress regression class (e.g. helper-side throughput collapse, mirror
# routing breakage). Threshold is configurable via APT_UPDATE_TIMEOUT_SECONDS.
set -u
TEST_NAME=vnet-03-apt-update-timing
declare -F pass >/dev/null 2>&1 || . "$(dirname "$0")/../shared/result.sh"

if ! command -v apt-get >/dev/null 2>&1; then
  skip "apt-get-missing"
fi

# Resolve a sudo prefix that works in the current exec environment:
#   - root: no sudo needed
#   - non-root with passwordless sudo: use `sudo -n`
#   - non-root without passwordless sudo: skip (this is a slice-shape thing,
#     not an egress thing, so we shouldn't fail the test for it)
SUDO=""
if [[ "$(id -u)" -ne 0 ]]; then
  if sudo -n true >/dev/null 2>&1; then
    SUDO="sudo -n"
  else
    skip "no-passwordless-sudo and not root"
  fi
fi

threshold=${APT_UPDATE_TIMEOUT_SECONDS:-60}
log=/tmp/${TEST_NAME}.log
trap 'rm -f "$log"' EXIT

t0=$(date +%s)
if $SUDO apt-get update -qq >"$log" 2>&1; then
  rc=0
else
  rc=$?
fi
t1=$(date +%s)
elapsed=$((t1 - t0))

if [[ $rc -ne 0 ]]; then
  fail "apt-get-failed rc=$rc elapsed=${elapsed}s"
fi
if [[ $elapsed -ge $threshold ]]; then
  fail "too-slow elapsed=${elapsed}s threshold=${threshold}s"
fi
pass "elapsed=${elapsed}s"
