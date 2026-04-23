#!/usr/bin/env bash
set -u
TEST_NAME=vnet-02-https-egress
declare -F pass >/dev/null 2>&1 || . "$(dirname "$0")/../shared/result.sh"

if ! command -v curl >/dev/null 2>&1; then
  skip "curl-missing"
fi
out=$(curl -fsSL --max-time 10 https://detectportal.firefox.com/success.txt 2>/dev/null || true)
if [[ "$out" == "success" ]]; then
  pass "got=success"
fi
fail "got='$out'"
