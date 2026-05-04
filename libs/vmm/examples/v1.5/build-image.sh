#!/usr/bin/env bash
# build-image.sh -- common v1.5 image-builder entrypoint.
#
# This wrapper is the temporary shell CLI for the future Rust image builder.
# It dispatches to backend emitters, but the contract remains common:
# guest binaries are built during image assembly and launch/boot paths do not
# run package installs, cargo, rustup, npm, signing, or runtime repair.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./build-image.sh [--backend auto|vz|ch] [backend args...]

Defaults:
  auto selects vz on macOS and ch on Linux.

Examples:
  ./build-image.sh --backend vz
  ./build-image.sh --backend ch
  ./build-image.sh --backend ch --kernel skip
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="${MOTLIE_V15_BUILD_BACKEND:-auto}"
ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            [ $# -ge 2 ] || die "--backend requires a value"
            BACKEND="$2"
            shift 2
            ;;
        --backend=*)
            BACKEND="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

if [ "$BACKEND" = "auto" ]; then
    case "$(uname -s)" in
        Darwin) BACKEND="vz" ;;
        Linux) BACKEND="ch" ;;
        *) die "cannot auto-select backend on $(uname -s); pass --backend vz or --backend ch" ;;
    esac
fi

case "$BACKEND" in
    vz|apple-vz|apple_virtualization)
        exec /bin/zsh "$SCRIPT_DIR/build-guest.sh" "${ARGS[@]}"
        ;;
    ch|cloud-hypervisor|cloud_hypervisor)
        exec /usr/bin/env bash "$SCRIPT_DIR/build-ch-artifacts.sh" "${ARGS[@]}"
        ;;
    *)
        die "unsupported backend '$BACKEND' (expected auto, vz, or ch)"
        ;;
esac
