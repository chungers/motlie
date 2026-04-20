#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/artifacts/build"
SRC="$SCRIPT_DIR/vz-vsock-runner.m"
OUT="$BUILD_DIR/vz-vsock-runner"
IDENTITY="${MOTLIE_VZ_CODESIGN_IDENTITY:-}"
ENTITLEMENTS="${MOTLIE_VZ_ENTITLEMENTS_FILE:-}"

mkdir -p "$BUILD_DIR"

SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"

clang \
  -fobjc-arc \
  -framework Foundation \
  -framework Virtualization \
  -isysroot "$SDKROOT" \
  -O2 \
  -Wall \
  -Wextra \
  -o "$OUT" \
  "$SRC"

if [[ -n "$IDENTITY" ]]; then
  if [[ -z "$ENTITLEMENTS" ]]; then
    echo "MOTLIE_VZ_CODESIGN_IDENTITY is set but MOTLIE_VZ_ENTITLEMENTS_FILE is empty" >&2
    exit 1
  fi
  codesign \
    --force \
    --sign "$IDENTITY" \
    --entitlements "$ENTITLEMENTS" \
    --timestamp=none \
    "$OUT"
fi

echo "$OUT"
