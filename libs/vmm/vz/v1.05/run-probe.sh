#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$ROOT/build"
MODULE_CACHE="$ROOT/.swift-module-cache"
BIN="$BUILD_DIR/minivz"

mkdir -p "$BUILD_DIR" "$MODULE_CACHE"

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <kernel> <initrd> <serial.log> [timeout-seconds]" >&2
  exit 64
fi

if [[ "${MOTLIE_SKIP_SWIFT:-0}" != "1" ]]; then
  swiftc \
    -module-cache-path "$MODULE_CACHE" \
    -framework Virtualization \
    "$ROOT/minivz.swift" \
    -o "$BIN" || {
      echo "swift build failed; falling back to Objective-C host probe for local feasibility" >&2
      clang \
        -fobjc-arc \
        -framework Foundation \
        -framework Virtualization \
        "$ROOT/minivz_objc.m" \
        -o "$BIN"
    }
else
  clang \
    -fobjc-arc \
    -framework Foundation \
    -framework Virtualization \
    "$ROOT/minivz_objc.m" \
    -o "$BIN"
fi

codesign --force --sign - --entitlements "$ROOT/minivz.entitlements" "$BIN"

exec "$BIN" "$@"
