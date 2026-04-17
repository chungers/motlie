#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
ARTIFACTS_DIR="$SCRIPT_DIR/artifacts"
BASE_VM_NAME="${MOTLIE_VZ_BASE_VM_NAME:-motlie-v1-05-ubuntu}"
PROVISIONED_VM_NAME="${MOTLIE_VZ_PROVISIONED_VM_NAME:-motlie-v1-05-vfs}"
SOURCE_IMAGE="${MOTLIE_VZ_SOURCE_IMAGE:-ghcr.io/cirruslabs/ubuntu:latest}"
RUN_LOG="$ARTIFACTS_DIR/build-run.log"
RESULT_JSON="$ARTIFACTS_DIR/build-result.json"
SOURCE_TARBALL="$ARTIFACTS_DIR/motlie-src.tar.gz"
TIMEOUT_SECONDS="${MOTLIE_VZ_TIMEOUT_SECONDS:-300}"
GUEST_SRC_DIR="/home/admin/motlie-src"

mkdir -p "$ARTIFACTS_DIR"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

require_cmd tart
require_cmd python3
require_cmd git
require_cmd tar

cleanup() {
  if tart list 2>/dev/null | awk 'NR>1 {print $1}' | grep -Fx "$PROVISIONED_VM_NAME" >/dev/null 2>&1; then
    tart stop "$PROVISIONED_VM_NAME" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT

echo "=== Vz v1.05 Tart guest build ==="
echo "Base VM:        $BASE_VM_NAME"
echo "Provisioned VM: $PROVISIONED_VM_NAME"
echo "Source image:   $SOURCE_IMAGE"
echo "Repo root:      $REPO_ROOT"

if ! tart list 2>/dev/null | awk 'NR>1 {print $1}' | grep -Fx "$BASE_VM_NAME" >/dev/null 2>&1; then
  echo "--- cloning base image ---"
  tart clone "$SOURCE_IMAGE" "$BASE_VM_NAME"
fi

if tart list 2>/dev/null | awk 'NR>1 {print $1}' | grep -Fx "$PROVISIONED_VM_NAME" >/dev/null 2>&1; then
  echo "--- deleting stale provisioned VM clone ---"
  tart delete "$PROVISIONED_VM_NAME" >/dev/null
fi

echo "--- cloning provisioned work VM ---"
tart clone "$BASE_VM_NAME" "$PROVISIONED_VM_NAME" >/dev/null

echo "--- packing Motlie source tree ---"
git -C "$REPO_ROOT" ls-files -z | tar --null -czf "$SOURCE_TARBALL" -C "$REPO_ROOT" --files-from -

echo "--- starting guest ---"
: > "$RUN_LOG"
START_EPOCH="$(python3 - <<'PY'
import time
print(f"{time.time():.6f}")
PY
)"
/bin/zsh -lc "tart run --no-graphics '$PROVISIONED_VM_NAME'" >"$RUN_LOG" 2>&1 &

IP_ADDR=""
ATTEMPTS=0
MAX_ATTEMPTS=$(( TIMEOUT_SECONDS * 2 ))
while [[ $ATTEMPTS -lt $MAX_ATTEMPTS ]]; do
  IP_ADDR="$(tart ip "$PROVISIONED_VM_NAME" 2>/dev/null || true)"
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

READY_EPOCH="$(python3 - <<'PY'
import time
print(f"{time.time():.6f}")
PY
)"
BOOT_SECONDS="$(python3 - "$START_EPOCH" "$READY_EPOCH" <<'PY'
import sys
print(f"{float(sys.argv[2]) - float(sys.argv[1]):.3f}")
PY
)"

guest() {
  tart exec "$PROVISIONED_VM_NAME" bash -lc "$1"
}

echo "--- installing guest prerequisites ---"
guest "set -euo pipefail; sudo apt-get update; sudo apt-get install -y build-essential pkg-config libfuse3-dev curl ca-certificates tar gzip"

echo "--- installing Rust toolchain in guest if needed ---"
guest "set -euo pipefail; if ! command -v cargo >/dev/null 2>&1; then curl https://sh.rustup.rs -sSf | sh -s -- -y; fi"

echo "--- uploading Motlie source tree into guest ---"
cat "$SOURCE_TARBALL" | tart exec -i "$PROVISIONED_VM_NAME" bash -lc "cat > /tmp/motlie-src.tar.gz"
guest "set -euo pipefail; rm -rf '$GUEST_SRC_DIR'; mkdir -p '$GUEST_SRC_DIR'; tar -xzf /tmp/motlie-src.tar.gz -C '$GUEST_SRC_DIR'"

echo "--- building motlie-vfs-guest in guest ---"
guest "set -euo pipefail; export PATH=\"\$HOME/.cargo/bin:\$PATH\"; export CARGO_TARGET_DIR=\"\$HOME/motlie-target\"; cd '$GUEST_SRC_DIR'; cargo build --release --features vsock,client -p motlie-vfs --bin motlie-vfs-guest"

echo "--- installing Motlie guest contract ---"
guest "set -euo pipefail; sudo install -D -m 0755 \"\$HOME/motlie-target/release/motlie-vfs-guest\" /usr/local/bin/motlie-vfs-guest; sudo install -D -m 0644 '$GUEST_SRC_DIR/libs/vfs/examples/v1/mounts.yaml' /etc/motlie-vfs/mounts.yaml; cat <<'EOF' | sudo tee /etc/systemd/system/motlie-vfs-guest.service >/dev/null
[Unit]
Description=Motlie VFS Guest
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/local/bin/motlie-vfs-guest /etc/motlie-vfs/mounts.yaml
Restart=on-failure
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload; sudo systemctl enable motlie-vfs-guest"

echo "--- validating guest contract ---"
WHICH_OUTPUT="$(guest "which motlie-vfs-guest")"
MOUNTS_OUTPUT="$(guest "cat /etc/motlie-vfs/mounts.yaml")"
UNIT_OUTPUT="$(guest "systemctl cat motlie-vfs-guest.service")"

python3 - "$RESULT_JSON" "$PROVISIONED_VM_NAME" "$IP_ADDR" "$BOOT_SECONDS" "$WHICH_OUTPUT" "$MOUNTS_OUTPUT" "$UNIT_OUTPUT" <<'PY'
import json
import sys

path, vm_name, ip_addr, boot_seconds, which_output, mounts_output, unit_output = sys.argv[1:]
with open(path, "w", encoding="utf-8") as fh:
    json.dump(
        {
            "backend": "vz-tart",
            "vm_name": vm_name,
            "ip_addr": ip_addr,
            "boot_to_ip_seconds": float(boot_seconds),
            "guest_contract": {
                "motlie_vfs_guest_path": which_output,
                "mounts_yaml": mounts_output,
                "systemd_unit": unit_output,
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

echo "--- stopping guest ---"
tart stop "$PROVISIONED_VM_NAME" >/dev/null

echo "=== success ==="
