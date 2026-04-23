#!/usr/bin/env bash
# Sustained UDP datagram send: assert the egress backend doesn't silently
# drop packets on ENOBUFS / EAGAIN. Catches the regression where the egress
# helper used default datagram buffers and dropped failed send_to() calls.
set -u
TEST_NAME=vnet-05-egress-backpressure
. "$(dirname "$0")/../shared/result.sh"

if ! command -v socat >/dev/null 2>&1; then
  skip "socat-missing"
fi
err=/tmp/${TEST_NAME}.err
: > "$err"
yes "x" | head -c 5M | socat -u - UDP-DATAGRAM:1.1.1.1:9 2>>"$err" || true
if grep -qE 'ENOBUFS|EAGAIN|Resource temporarily unavailable' "$err"; then
  detail=$(head -3 "$err" | tr '\n' '|')
  fail "buffer-error $detail"
fi
pass "no-enobufs"
