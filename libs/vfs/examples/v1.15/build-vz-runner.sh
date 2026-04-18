#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/artifacts/build"
SRC="$SCRIPT_DIR/vz-vsock-runner.m"
OUT="$BUILD_DIR/vz-vsock-runner"

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

echo "$OUT"
