#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
RUNNER_BIN="${MOTLIE_VZ_RUNNER_BIN:-$SCRIPT_DIR/artifacts/build/vz-vsock-runner}"
ENTITLEMENTS="${MOTLIE_VZ_ENTITLEMENTS_FILE:-$SCRIPT_DIR/vz.entitlements}"
exec "$REPO_ROOT/scripts/sign-vz-runner.sh" "$RUNNER_BIN" "$ENTITLEMENTS"
