#!/usr/bin/env sh
# install-mmux.sh — mmux 0.1.0 (release 2026-05-apex-anchor)
#
# Downloads the platform-matched archive from the GitHub Release, verifies its
# SHA256, extracts bin/mmux, installs to /usr/local/bin/mmux, and re-signs on
# macOS so the installed-path execution invariant holds on Apple Silicon.
#
# Usage:
#   curl -fsSL https://github.com/chungers/motlie/releases/download/2026-05-apex-anchor/install-mmux.sh | sh
#
# Options:
#   --prefix <dir>   Install prefix (default: /usr/local). Binary lands at <prefix>/bin/mmux.
#   --source archive|npm
#                    Source mode (default: archive). npm mode is reserved; see release notes.
#   --version <tag>  Override the release tag (default: 2026-05-apex-anchor).
#
# This is the v0 installer. Future releases should generalize this into a
# canonical template at bins/mmux/install-template.sh on main.

set -eu

RELEASE_TAG="${RELEASE_TAG:-2026-05-apex-anchor}"
BIN_VERSION="${BIN_VERSION:-0.1.0}"
PREFIX="${PREFIX:-/usr/local}"
SOURCE="${SOURCE:-archive}"
REPO="chungers/motlie"
ASSET_PREFIX="motlie-mmux-v${BIN_VERSION}"
BIN_NAME="mmux"

while [ $# -gt 0 ]; do
  case "$1" in
    --prefix)  PREFIX="$2"; shift 2 ;;
    --source)  SOURCE="$2"; shift 2 ;;
    --version) RELEASE_TAG="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "install-mmux.sh: unknown option: $1" >&2; exit 2 ;;
  esac
done

if [ "$SOURCE" != "archive" ]; then
  echo "install-mmux.sh: source=$SOURCE not yet supported in this release. Use --source archive." >&2
  exit 2
fi

# ---- detect OS / arch / libc ----
detect_os() {
  case "$(uname -s)" in
    Linux)  echo linux ;;
    Darwin) echo darwin ;;
    *) echo "install-mmux.sh: unsupported OS: $(uname -s)" >&2; exit 1 ;;
  esac
}

detect_arch() {
  case "$(uname -m)" in
    x86_64|amd64)        echo x64 ;;
    aarch64|arm64)       echo arm64 ;;
    *) echo "install-mmux.sh: unsupported arch: $(uname -m)" >&2; exit 1 ;;
  esac
}

detect_libc_suffix() {
  # Only meaningful on linux; darwin returns empty.
  if [ "$OS" = "linux" ]; then
    echo "-musl"
  else
    echo ""
  fi
}

OS="$(detect_os)"
ARCH="$(detect_arch)"
LIBC_SUFFIX="$(detect_libc_suffix)"
TARGET="${OS}-${ARCH}${LIBC_SUFFIX}"
ASSET="${ASSET_PREFIX}-${TARGET}.tar.gz"
URL="https://github.com/${REPO}/releases/download/${RELEASE_TAG}/${ASSET}"

# ---- prereq check ----
need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "install-mmux.sh: missing required command: $1" >&2
    exit 1
  }
}
need curl
need tar
need install
case "$OS" in
  darwin) need codesign ;;
esac

# Choose sha256 tool
if command -v shasum >/dev/null 2>&1; then
  SHA256_CMD="shasum -a 256"
elif command -v sha256sum >/dev/null 2>&1; then
  SHA256_CMD="sha256sum"
else
  echo "install-mmux.sh: missing sha256 tool (shasum or sha256sum)" >&2
  exit 1
fi

# ---- download archive + SHA256SUMS, verify ----
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo ">> Downloading ${ASSET}"
curl -fsSL -o "${TMP}/${ASSET}" "${URL}"

echo ">> Downloading SHA256SUMS"
curl -fsSL -o "${TMP}/SHA256SUMS" "https://github.com/${REPO}/releases/download/${RELEASE_TAG}/SHA256SUMS"

echo ">> Verifying checksum"
# SHA256SUMS contains every archive in this release; grep for our asset and verify it alone.
( cd "$TMP" && grep "  ${ASSET}\$" SHA256SUMS > expected.sum && $SHA256_CMD -c expected.sum >/dev/null ) || {
  echo "install-mmux.sh: checksum mismatch for ${ASSET}" >&2
  exit 1
}

# ---- extract + install ----
echo ">> Extracting"
tar -xzf "${TMP}/${ASSET}" -C "$TMP"
[ -x "${TMP}/bin/${BIN_NAME}" ] || {
  echo "install-mmux.sh: archive missing bin/${BIN_NAME}" >&2
  exit 1
}

DEST_DIR="${PREFIX}/bin"
DEST="${DEST_DIR}/${BIN_NAME}"

echo ">> Installing to ${DEST}"
SUDO=""
if [ -w "$DEST_DIR" ] || { [ ! -e "$DEST_DIR" ] && [ -w "$(dirname "$DEST_DIR")" ]; }; then
  :
else
  if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
  else
    echo "install-mmux.sh: ${DEST_DIR} not writable and sudo unavailable" >&2
    exit 1
  fi
fi
$SUDO install -d "$DEST_DIR"
$SUDO install -m 755 "${TMP}/bin/${BIN_NAME}" "$DEST"

# ---- macOS re-sign + verify installed path ----
if [ "$OS" = "darwin" ]; then
  echo ">> Re-signing installed binary (ad-hoc)"
  $SUDO codesign --force --sign - "$DEST"
  codesign --verify --strict --verbose=2 "$DEST" >/dev/null 2>&1 || {
    echo "install-mmux.sh: post-install codesign verify failed for ${DEST}" >&2
    exit 1
  }
fi

# ---- final smoke check ----
echo ">> Verifying ${DEST} --version"
"$DEST" --version || {
  echo "install-mmux.sh: ${DEST} --version failed" >&2
  exit 1
}

echo ""
echo "Installed: ${DEST} (${BIN_VERSION} from ${RELEASE_TAG})"
