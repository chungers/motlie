#!/usr/bin/env bash
# Per-guest isolation: the *other* guest's home should not be visible from
# this guest. The driver runs this in each guest; "this" guest's $USER
# defines its own home, and we check that the *other* guest's expected
# home path is absent (or, if present as a directory, that we cannot read
# the sentinel file an out-of-band setup might have written there).
#
# By default, we hardcode the well-known guest pair (alice, bob) used
# throughout motlie examples. Override via PEER_USER if your guest layout
# differs.
set -u
TEST_NAME=vfs-05-per-guest-isolation
. "$(dirname "$0")/../shared/result.sh"

me="$USER"
peer="${PEER_USER:-}"
if [[ -z "$peer" ]]; then
  case "$me" in
    alice) peer=bob ;;
    bob)   peer=alice ;;
    *)     skip "unknown-pairing user=$me" ;;
  esac
fi
peer_home="/home/$peer"
sentinel="$peer_home/.vmm-iso-sentinel"
if [[ ! -d "$peer_home" ]]; then
  pass "peer-home-absent peer=$peer"
fi
if [[ -r "$sentinel" ]]; then
  fail "peer-sentinel-readable peer=$peer"
fi
pass "peer-home-not-readable peer=$peer"
