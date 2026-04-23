#!/usr/bin/env bash
# First-resolution latency for a fresh hostname. Catches DNS forwarder hangs
# (e.g. a misconfigured upstream that retries 5s before falling through).
set -u
TEST_NAME=vnet-06-dns-latency
declare -F pass >/dev/null 2>&1 || . "$(dirname "$0")/../shared/result.sh"

threshold_ms=${DNS_LATENCY_MS:-2000}
host="cdn.example.com"  # rarely-cached hostname; resolves at NXDOMAIN level still exercises path
t0=$(date +%s%N)
getent ahostsv4 "$host" >/dev/null 2>&1 || true
t1=$(date +%s%N)
elapsed_ms=$(( (t1 - t0) / 1000000 ))
if [[ $elapsed_ms -ge $threshold_ms ]]; then
  fail "too-slow elapsed_ms=$elapsed_ms threshold_ms=$threshold_ms"
fi
pass "elapsed_ms=$elapsed_ms"
