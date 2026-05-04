#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNNER_BIN="${MOTLIE_VZ_RUNNER_BIN:-$SCRIPT_DIR/artifacts/build/vz-vsock-runner}"
ENTITLEMENTS="${MOTLIE_VZ_ENTITLEMENTS_FILE:-$SCRIPT_DIR/vz.entitlements}"
IDENTITY="${MOTLIE_VZ_CODESIGN_IDENTITY:--}"

if [[ ! -f "$RUNNER_BIN" ]]; then
  echo "runner binary not found: $RUNNER_BIN" >&2
  exit 1
fi
if [[ ! -f "$ENTITLEMENTS" ]]; then
  echo "entitlements file not found: $ENTITLEMENTS" >&2
  exit 1
fi

codesign \
  --force \
  --sign "$IDENTITY" \
  --entitlements "$ENTITLEMENTS" \
  --timestamp=none \
  "$RUNNER_BIN"

codesign -d --entitlements :- "$RUNNER_BIN"
