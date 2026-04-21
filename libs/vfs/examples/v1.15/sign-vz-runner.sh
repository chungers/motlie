#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNNER_BIN="${MOTLIE_VZ_RUNNER_BIN:-$SCRIPT_DIR/artifacts/build/vz-vsock-runner}"
IDENTITY="${MOTLIE_VZ_CODESIGN_IDENTITY:-}"
ENTITLEMENTS="${MOTLIE_VZ_ENTITLEMENTS_FILE:-$SCRIPT_DIR/vz.entitlements}"

if [[ -z "$IDENTITY" ]]; then
  cat >&2 <<EOF
MOTLIE_VZ_CODESIGN_IDENTITY is required

Example:
  export MOTLIE_VZ_CODESIGN_IDENTITY='Apple Development: Your Name (TEAMID)'
EOF
  exit 1
fi

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
