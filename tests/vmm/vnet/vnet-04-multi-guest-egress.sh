#!/usr/bin/env bash
# Per-guest egress probe. Driver runs this in each guest independently;
# multi-guest verification is the absence of fail across all dispatches.
set -u
TEST_NAME=vnet-04-multi-guest-egress
declare -F pass >/dev/null 2>&1 || . "$(dirname "$0")/../shared/result.sh"

if ! command -v curl >/dev/null 2>&1; then
  skip "curl-missing"
fi
out=$(curl -fsS --max-time 8 https://1.1.1.1/cdn-cgi/trace 2>/dev/null || true)
ip_line=$(echo "$out" | grep -E '^ip=' | head -1)
if [[ -z "$ip_line" ]]; then
  fail "no-ip-trace got='$out'"
fi
pass "$(hostname -s):$ip_line"
