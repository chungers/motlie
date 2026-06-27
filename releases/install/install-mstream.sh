#!/usr/bin/env sh
# install-mstream.sh - mstream 0.1.0 (release 2026-06-bright-beacon)
#
# Downloads the platform-matched archive from the GitHub Release, verifies its
# SHA256, extracts bin/mstream, installs to /usr/local/bin/mstream, and re-signs
# on macOS so the installed-path execution invariant holds.

set -eu

RELEASE_TAG="${RELEASE_TAG:-2026-06-bright-beacon}"
BIN_VERSION="${BIN_VERSION:-0.1.0}"
PREFIX="${PREFIX:-/usr/local}"
REPO="chungers/motlie"
BIN_NAME="mstream"
ASSET_PREFIX="motlie-${BIN_NAME}-v${BIN_VERSION}"

while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2 ;;
    --version) RELEASE_TAG="$2"; shift 2 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "install-mstream.sh: unknown option: $1" >&2; exit 2 ;;
  esac
done

detect_os() {
  case "$(uname -s)" in
    Linux) echo linux ;;
    Darwin) echo darwin ;;
    *) echo "install-mstream.sh: unsupported OS: $(uname -s)" >&2; exit 1 ;;
  esac
}

detect_arch() {
  case "$(uname -m)" in
    x86_64|amd64) echo x64 ;;
    aarch64|arm64) echo arm64 ;;
    *) echo "install-mstream.sh: unsupported arch: $(uname -m)" >&2; exit 1 ;;
  esac
}

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "install-mstream.sh: missing required command: $1" >&2
    exit 1
  }
}

OS="$(detect_os)"
ARCH="$(detect_arch)"
LIBC_SUFFIX=""
if [ "$OS" = "linux" ]; then
  LIBC_SUFFIX="-musl"
fi

if [ "$OS" = "darwin" ] && [ "$ARCH" != "arm64" ]; then
  echo "install-mstream.sh: unsupported Darwin arch: ${ARCH}; this release ships Apple Silicon (darwin-arm64) only" >&2
  exit 1
fi

TARGET="${OS}-${ARCH}${LIBC_SUFFIX}"
ASSET="${ASSET_PREFIX}-${TARGET}.tar.gz"
URL="https://github.com/${REPO}/releases/download/${RELEASE_TAG}/${ASSET}"

need curl
need tar
need install
if [ "$OS" = "darwin" ]; then
  need codesign
fi

if command -v shasum >/dev/null 2>&1; then
  SHA256_CMD="shasum -a 256"
elif command -v sha256sum >/dev/null 2>&1; then
  SHA256_CMD="sha256sum"
else
  echo "install-mstream.sh: missing sha256 tool (shasum or sha256sum)" >&2
  exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo ">> Downloading ${ASSET}"
curl -fsSL -o "${TMP}/${ASSET}" "${URL}"

echo ">> Downloading SHA256SUMS"
curl -fsSL -o "${TMP}/SHA256SUMS" "https://github.com/${REPO}/releases/download/${RELEASE_TAG}/SHA256SUMS"

echo ">> Verifying checksum"
( cd "$TMP" && grep "  ${ASSET}\$" SHA256SUMS > expected.sum && $SHA256_CMD -c expected.sum >/dev/null ) || {
  echo "install-mstream.sh: checksum mismatch for ${ASSET}" >&2
  exit 1
}

echo ">> Extracting"
tar -xzf "${TMP}/${ASSET}" -C "$TMP"
[ -x "${TMP}/bin/${BIN_NAME}" ] || {
  echo "install-mstream.sh: archive missing bin/${BIN_NAME}" >&2
  exit 1
}

DEST_DIR="${PREFIX}/bin"
DEST="${DEST_DIR}/${BIN_NAME}"

echo ">> Installing to ${DEST}"
SUDO=""
if [ -w "$DEST_DIR" ] || { [ ! -e "$DEST_DIR" ] && [ -w "$(dirname "$DEST_DIR")" ]; }; then
  :
elif command -v sudo >/dev/null 2>&1; then
  SUDO="sudo"
else
  echo "install-mstream.sh: ${DEST_DIR} not writable and sudo unavailable" >&2
  exit 1
fi

$SUDO install -d "$DEST_DIR"
$SUDO install -m 755 "${TMP}/bin/${BIN_NAME}" "$DEST"

if [ "$OS" = "darwin" ]; then
  echo ">> Re-signing installed binary (ad-hoc)"
  $SUDO codesign --force --sign - "$DEST"
  codesign --verify --strict --verbose=2 "$DEST" >/dev/null 2>&1 || {
    echo "install-mstream.sh: post-install codesign verify failed for ${DEST}" >&2
    exit 1
  }
fi

echo ">> Verifying ${DEST} --version"
"$DEST" --version || {
  echo "install-mstream.sh: ${DEST} --version failed" >&2
  exit 1
}

echo ""
echo "Installed: ${DEST} (${BIN_VERSION} from ${RELEASE_TAG})"
