#!/usr/bin/env bash
# Endpoints commonly used during guest provisioning. If any of these stop
# resolving or routing, downstream apt/cargo/npm/pip flows will fail in
# confusing ways.
set -u
TEST_NAME=vnet-07-cloud-endpoints
. "$(dirname "$0")/../shared/result.sh"

if ! command -v curl >/dev/null 2>&1; then
  skip "curl-missing"
fi
endpoints=(
  "https://api.github.com/zen"
  "https://registry.npmjs.org/-/ping"
  "https://pypi.org/simple/"
)
fails=()
for url in "${endpoints[@]}"; do
  if ! curl -fsS --max-time 8 -o /dev/null "$url"; then
    fails+=("$url")
  fi
done
if [[ ${#fails[@]} -gt 0 ]]; then
  IFS=,; fail "unreachable=${fails[*]}"
fi
pass "ok=${#endpoints[@]}"
