#!/usr/bin/env bash
# 5 sequential HTTPS requests; assert (a) all succeed and (b) latency does
# not grow unboundedly across iterations (catches helper-side memory leaks /
# socket-table exhaustion).
set -u
TEST_NAME=vnet-10-egress-stability
declare -F pass >/dev/null 2>&1 || . "$(dirname "$0")/../shared/result.sh"

if ! command -v curl >/dev/null 2>&1; then
  skip "curl-missing"
fi
url="https://1.1.1.1/cdn-cgi/trace"
n=5
times=()
for i in $(seq 1 $n); do
  t0=$(date +%s%N)
  if ! curl -fsS --max-time 10 -o /dev/null "$url"; then
    fail "iteration=$i request-failed"
  fi
  t1=$(date +%s%N)
  times+=( $(( (t1 - t0) / 1000000 )) )
done
first=${times[0]}
last=${times[$((n-1))]}
# Allow last to be at most 3x first, but always under 5s in absolute terms.
if (( last > 5000 )); then
  fail "absolute-too-slow last_ms=$last times_ms=${times[*]}"
fi
if (( first > 0 && last > first * 3 )); then
  fail "growing-latency first_ms=$first last_ms=$last times_ms=${times[*]}"
fi
pass "times_ms=${times[*]}"
