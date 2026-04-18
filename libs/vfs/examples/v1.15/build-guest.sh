#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
ARTIFACTS_DIR="$SCRIPT_DIR/artifacts"
BASE_VM_NAME="${MOTLIE_VZ_BASE_VM_NAME:-motlie-v1-15-base}"
SOURCE_IMAGE="${MOTLIE_VZ_SOURCE_IMAGE:-ghcr.io/cirruslabs/ubuntu@sha256:1e23e6fe5a6d3fb2089652229a09d71742617758b15aa311cecf1c05985d3021}"
RUN_LOG="$ARTIFACTS_DIR/build-run.log"
RESULT_JSON="$ARTIFACTS_DIR/build-result.json"
SOURCE_TARBALL="$ARTIFACTS_DIR/motlie-src.tar.gz"
TIMEOUT_SECONDS="${MOTLIE_VZ_TIMEOUT_SECONDS:-300}"
GUEST_SRC_DIR="/home/admin/motlie-src"
SERVICE_FILE="$REPO_ROOT/libs/vfs/examples/v1.15/motlie-vfs-guest.service"

zmodload zsh/datetime

mkdir -p "$ARTIFACTS_DIR"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

require_cmd tart
require_cmd git
require_cmd tar
require_cmd python3

local_vm_exists() {
  tart list --source local -q 2>/dev/null | grep -Fx "$1" >/dev/null 2>&1
}

cleanup() {
  if local_vm_exists "$BASE_VM_NAME"; then
    tart stop "$BASE_VM_NAME" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT

echo "=== Vz v1.15 base guest build ==="
echo "Base VM:      $BASE_VM_NAME"
echo "Source image: $SOURCE_IMAGE"

if local_vm_exists "$BASE_VM_NAME"; then
  echo "--- deleting stale base VM clone ---"
  tart delete "$BASE_VM_NAME" >/dev/null
fi

echo "--- cloning base image ---"
tart clone "$SOURCE_IMAGE" "$BASE_VM_NAME" >/dev/null

echo "--- packing Motlie source tree ---"
COPYFILE_DISABLE=1 COPY_EXTENDED_ATTRIBUTES_DISABLE=1 git -C "$REPO_ROOT" ls-files -z | COPYFILE_DISABLE=1 COPY_EXTENDED_ATTRIBUTES_DISABLE=1 tar --disable-copyfile --no-mac-metadata --no-xattrs --null -czf "$SOURCE_TARBALL" -C "$REPO_ROOT" --files-from -

echo "--- starting guest ---"
: > "$RUN_LOG"
START_EPOCH="$EPOCHREALTIME"
tart run --no-graphics "$BASE_VM_NAME" >"$RUN_LOG" 2>&1 &
RUN_PID="$!"

IP_ADDR=""
ATTEMPTS=0
MAX_ATTEMPTS=$(( TIMEOUT_SECONDS * 2 ))
while [[ $ATTEMPTS -lt $MAX_ATTEMPTS ]]; do
  if ! kill -0 "$RUN_PID" >/dev/null 2>&1; then
    echo "tart run exited early; log follows:" >&2
    cat "$RUN_LOG" >&2
    exit 1
  fi

  IP_ADDR="$(tart ip "$BASE_VM_NAME" 2>/dev/null || true)"
  if [[ -n "$IP_ADDR" ]]; then
    break
  fi

  sleep 0.5
  ATTEMPTS=$(( ATTEMPTS + 1 ))
done

if [[ -z "$IP_ADDR" ]]; then
  echo "timed out waiting for guest IP after ${TIMEOUT_SECONDS}s" >&2
  cat "$RUN_LOG" >&2 || true
  exit 1
fi

READY_EPOCH="$EPOCHREALTIME"
BOOT_SECONDS="$(awk -v start="$START_EPOCH" -v ready="$READY_EPOCH" 'BEGIN { printf "%.3f", ready - start }')"

guest_bash() {
  tart exec -i "$BASE_VM_NAME" bash -seuo pipefail
}

echo "--- installing guest prerequisites ---"
guest_bash <<'EOF'
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libfuse3-dev curl ca-certificates tar gzip iproute2
EOF

echo "--- installing Rust toolchain in guest if needed ---"
guest_bash <<'EOF'
if ! command -v cargo >/dev/null 2>&1; then
  curl https://sh.rustup.rs -sSf | sh -s -- -y
fi
EOF

echo "--- uploading Motlie source tree into guest ---"
cat "$SOURCE_TARBALL" | tart exec -i "$BASE_VM_NAME" bash -lc "cat > /tmp/motlie-src.tar.gz"
cat "$SERVICE_FILE" | tart exec -i "$BASE_VM_NAME" bash -lc "cat > /tmp/motlie-vfs-guest.service"
guest_bash <<EOF
rm -rf '$GUEST_SRC_DIR'
mkdir -p '$GUEST_SRC_DIR'
tar -xzf /tmp/motlie-src.tar.gz -C '$GUEST_SRC_DIR'
EOF

echo "--- building motlie-vfs-guest-v1_15 in guest ---"
guest_bash <<EOF
export PATH="\$HOME/.cargo/bin:\$PATH"
export CARGO_TARGET_DIR="\$HOME/motlie-target"
python3 - <<'PY'
from pathlib import Path

root = Path("$GUEST_SRC_DIR/Cargo.toml")
text = root.read_text(encoding="utf-8")
if '"libs/vfs",' not in text:
    text = text.replace('members = [\n', 'members = [\n    "libs/vfs",\n', 1)
    root.write_text(text, encoding="utf-8")
PY
cargo build --manifest-path '$GUEST_SRC_DIR/libs/vfs/Cargo.toml' --release --features client --bin motlie-vfs-guest-v1_15
EOF

echo "--- installing generic guest contract ---"
guest_bash <<EOF
sudo install -D -m 0755 "\$HOME/motlie-target/release/motlie-vfs-guest-v1_15" /usr/local/bin/motlie-vfs-guest-v1_15
sudo install -D -m 0644 /tmp/motlie-vfs-guest.service /etc/systemd/system/motlie-vfs-guest.service
sudo mkdir -p /etc/motlie-vfs
sudo systemctl unmask motlie-vfs-guest.service || true
sudo systemctl daemon-reload
EOF

GUEST_BINARY="$(guest_bash <<'EOF'
which motlie-vfs-guest-v1_15
EOF
)"

python3 - "$RESULT_JSON" "$BASE_VM_NAME" "$IP_ADDR" "$BOOT_SECONDS" "$GUEST_BINARY" <<'PY'
import json
import sys

path, vm_name, ip_addr, boot_seconds, guest_binary = sys.argv[1:]
with open(path, "w", encoding="utf-8") as fh:
    json.dump(
        {
            "backend": "vz-tart",
            "vm_name": vm_name,
            "ip_addr": ip_addr,
            "boot_to_ip_seconds": float(boot_seconds),
            "guest_contract": {
                "motlie_vfs_guest_v1_15_path": guest_binary,
            },
        },
        fh,
        indent=2,
        sort_keys=True,
    )
    fh.write("\n")
PY

echo "--- result written ---"
cat "$RESULT_JSON"

echo "--- stopping base guest ---"
tart stop "$BASE_VM_NAME" >/dev/null

echo "=== success ==="
