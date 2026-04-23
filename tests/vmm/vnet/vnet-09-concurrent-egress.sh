#!/usr/bin/env bash
# 10 parallel HTTPS requests. Catches egress-helper queue starvation, single-
# threaded SOCKS-style serialization in the userspace path, and per-flow rate
# limits that only show up under concurrency.
set -u
TEST_NAME=vnet-09-concurrent-egress
. "$(dirname "$0")/../shared/result.sh"

if ! command -v curl >/dev/null 2>&1; then
  skip "curl-missing"
fi
n=10
url="https://1.1.1.1/cdn-cgi/trace"
status_dir=/tmp/${TEST_NAME}.d
rm -rf "$status_dir"; mkdir -p "$status_dir"
for i in $(seq 1 $n); do
  ( curl -fsS --max-time 10 -o /dev/null "$url" \
      && echo ok > "$status_dir/$i" \
      || echo fail > "$status_dir/$i" ) &
done
wait
ok=$(grep -l '^ok$'   "$status_dir"/* 2>/dev/null | wc -l | tr -d ' ')
bad=$(grep -l '^fail$' "$status_dir"/* 2>/dev/null | wc -l | tr -d ' ')
rm -rf "$status_dir"
if [[ "$ok" -ne "$n" ]]; then
  fail "ok=$ok fail=$bad of=$n"
fi
pass "ok=$ok/$n"
