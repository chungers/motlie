#!/usr/bin/env bash
# Per-guest isolation: from this guest, the *peer* guest's home directory
# must not be enumerable. The assertion is about visibility/listability,
# not about an optional sentinel file — a peer-home that is listable but
# happens to be empty would otherwise pass falsely.
#
# By default this hardcodes the well-known motlie pair (alice, bob). Override
# the peer name via PEER_USER if your guest layout differs.
set -u
TEST_NAME=vfs-05-per-guest-isolation
declare -F pass >/dev/null 2>&1 || . "$(dirname "$0")/../shared/result.sh"

# Resolve the current user via id(1) rather than $USER so this test does not
# leak the exec-environment shape into its surface.
me=$(id -un)
peer="${PEER_USER:-}"
if [[ -z "$peer" ]]; then
  case "$me" in
    alice) peer=bob ;;
    bob)   peer=alice ;;
    *)     skip "unknown-pairing user=$me set PEER_USER to override" ;;
  esac
fi
peer_home="/home/$peer"

# Strongest pass: peer home does not exist at all on this guest.
if [[ ! -e "$peer_home" ]]; then
  pass "peer-home-absent peer=$peer"
fi

# Otherwise the peer home must be unlistable from this guest. If we can
# enumerate its contents (even if empty), isolation is broken.
if ls -A "$peer_home" >/dev/null 2>&1; then
  fail "peer-home-listable peer=$peer"
fi

# Peer home exists but listing was denied — isolation holds.
pass "peer-home-not-listable peer=$peer"
