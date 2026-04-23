#!/usr/bin/env bash
set -u
TEST_NAME=vnet-03-apt-update-timing
. "$(dirname "$0")/../shared/result.sh"

if ! command -v apt-get >/dev/null 2>&1; then
  skip "apt-get-missing"
fi
threshold=${APT_UPDATE_TIMEOUT_SECONDS:-60}
log=/tmp/${TEST_NAME}.log
t0=$(date +%s)
if sudo -n apt-get update -qq >"$log" 2>&1; then
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
