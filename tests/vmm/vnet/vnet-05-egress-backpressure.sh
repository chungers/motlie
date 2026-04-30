#!/usr/bin/env bash
# Sustained UDP send-and-receive: positive evidence that the egress backend
# does NOT silently drop packets under sustained load. Fires N small DNS
# queries at a public resolver in a tight loop and asserts the response rate
# stays above a high-water mark. This catches the silent-helper-drop failure
# mode (where send_to() failures inside the egress helper are swallowed),
# because lost packets show up as missing responses regardless of whether
# the guest-side write call returned success.
#
# Tuning:
#   - VNET05_QUERIES   total queries to fire (default 100)
#   - VNET05_MIN_OK    minimum responses required (default 90)  i.e. ≥ 90% delivery
#   - VNET05_RESOLVER  resolver IP (default 1.1.1.1)
set -u
TEST_NAME=vnet-05-egress-backpressure
declare -F pass >/dev/null 2>&1 || . "$(dirname "$0")/../shared/result.sh"

if ! command -v dig >/dev/null 2>&1; then
  skip "dig-missing"
fi

queries=${VNET05_QUERIES:-100}
min_ok=${VNET05_MIN_OK:-90}
resolver=${VNET05_RESOLVER:-1.1.1.1}

ok=0
for i in $(seq 1 "$queries"); do
  # +time=2 +tries=1 ensures each query is one UDP send and one (or zero)
  # UDP receive — no built-in retry that would hide drops.
  if dig +short +time=2 +tries=1 "@${resolver}" example.com >/dev/null 2>&1; then
    ok=$((ok + 1))
  fi
done

if [[ "$ok" -lt "$min_ok" ]]; then
  fail "delivery-rate-low ok=${ok}/${queries} min=${min_ok}"
fi
pass "ok=${ok}/${queries}"
