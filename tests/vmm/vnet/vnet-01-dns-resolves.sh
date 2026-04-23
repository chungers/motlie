#!/usr/bin/env bash
set -u
TEST_NAME=vnet-01-dns-resolves
. "$(dirname "$0")/../shared/result.sh"

ip=$(getent ahostsv4 deb.debian.org 2>/dev/null | awk '{print $1; exit}')
if [[ -z "$ip" ]]; then
  fail "no-ipv4-returned"
fi
case "$ip" in
  127.*|0.*) fail "loopback-or-zero ip=$ip" ;;
esac
pass "ip=$ip"
