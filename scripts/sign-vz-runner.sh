#!/bin/zsh
set -euo pipefail

if [[ $# -ne 2 ]]; then
  cat >&2 <<'EOF'
usage: sign-vz-runner.sh <runner-bin> <entitlements>
EOF
  exit 1
fi

RUNNER_BIN="$1"
ENTITLEMENTS="$2"
IDENTITY="${MOTLIE_VZ_CODESIGN_IDENTITY:-}"

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
