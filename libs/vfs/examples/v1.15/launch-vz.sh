#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ARTIFACTS_DIR="$SCRIPT_DIR/artifacts"
BASE_VM_NAME="${MOTLIE_VZ_BASE_VM_NAME:-motlie-v1-15-base}"
TIMEOUT_SECONDS="${MOTLIE_VZ_TIMEOUT_SECONDS:-180}"
HOST_BIND_ALL="${MOTLIE_VZ_HOST_BIND_ALL:-0.0.0.0}"
SERVICE_FILE="$SCRIPT_DIR/motlie-vfs-guest.service"
KEEP_RUNNING=0

zmodload zsh/datetime

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

require_cmd tart
require_cmd python3

GUEST_NAME="alice"
RUN_VM_NAME=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --guest) GUEST_NAME="$2"; shift 2 ;;
    --vm-name) RUN_VM_NAME="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

case "$GUEST_NAME" in
  alice)
    RUN_VM_NAME="${RUN_VM_NAME:-motlie-v1-15-alice-$(date +%s)}"
    TEMPLATE_FILE="$SCRIPT_DIR/mounts.alice.yaml"
    SERVICE_PORT=5501
    LOGIN_USER="alice"
    UID_NUM=2000
    GID_NUM=2000
    ;;
  bob)
    RUN_VM_NAME="${RUN_VM_NAME:-motlie-v1-15-bob-$(date +%s)}"
    TEMPLATE_FILE="$SCRIPT_DIR/mounts.bob.yaml"
    SERVICE_PORT=5502
    LOGIN_USER="bob"
    UID_NUM=2001
    GID_NUM=2001
    ;;
  *)
    echo "guest must be alice or bob" >&2
    exit 1
    ;;
esac

RUN_LOG="$ARTIFACTS_DIR/${GUEST_NAME}-run.log"
RESULT_JSON="$ARTIFACTS_DIR/${GUEST_NAME}-launch-result.json"

local_vm_exists() {
  tart list --source local -q 2>/dev/null | grep -Fx "$1" >/dev/null 2>&1
}

cleanup() {
  if [[ "$KEEP_RUNNING" -eq 0 ]] && local_vm_exists "$RUN_VM_NAME"; then
    tart stop "$RUN_VM_NAME" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT

if ! local_vm_exists "$BASE_VM_NAME"; then
  echo "base VM '$BASE_VM_NAME' not found; run ./build-guest.sh first" >&2
  exit 1
fi

if local_vm_exists "$RUN_VM_NAME"; then
  echo "guest VM '$RUN_VM_NAME' already exists; pass --vm-name with a unique value" >&2
  exit 1
fi

echo "--- cloning guest VM ---"
tart clone "$BASE_VM_NAME" "$RUN_VM_NAME" >/dev/null

echo "--- starting guest ---"
: > "$RUN_LOG"
START_EPOCH="$EPOCHREALTIME"
nohup tart run --no-graphics "$RUN_VM_NAME" >"$RUN_LOG" 2>&1 &
RUN_PID="$!"
disown "$RUN_PID" 2>/dev/null || true

IP_ADDR=""
ATTEMPTS=0
MAX_ATTEMPTS=$(( TIMEOUT_SECONDS * 2 ))
while [[ $ATTEMPTS -lt $MAX_ATTEMPTS ]]; do
  if ! kill -0 "$RUN_PID" >/dev/null 2>&1; then
    echo "tart run exited early; log follows:" >&2
    cat "$RUN_LOG" >&2
    exit 1
  fi
  IP_ADDR="$(tart ip "$RUN_VM_NAME" 2>/dev/null || true)"
  if [[ -n "$IP_ADDR" ]]; then
    break
  fi
  sleep 0.5
  ATTEMPTS=$(( ATTEMPTS + 1 ))
done

if [[ -z "$IP_ADDR" ]]; then
  echo "timed out waiting for guest IP" >&2
  cat "$RUN_LOG" >&2 || true
  exit 1
fi

HOST_GATEWAY="${MOTLIE_VZ_HOST_GATEWAY:-${IP_ADDR%.*}.1}"
READY_EPOCH="$EPOCHREALTIME"
BOOT_SECONDS="$(awk -v start="$START_EPOCH" -v ready="$READY_EPOCH" 'BEGIN { printf "%.3f", ready - start }')"

MOUNTS_YAML="$(python3 - "$TEMPLATE_FILE" "$HOST_GATEWAY" "$SERVICE_PORT" <<'PY'
from pathlib import Path
import sys

template = Path(sys.argv[1]).read_text(encoding="utf-8")
template = template.replace("__HOST_GATEWAY__", sys.argv[2])
template = template.replace("__HOST_PORT__", sys.argv[3])
print(template, end="")
PY
)"

guest_bash() {
  tart exec -i "$RUN_VM_NAME" bash -seuo pipefail
}

echo "--- provisioning guest identity and mounts ---"
printf "%s" "$MOUNTS_YAML" | tart exec -i "$RUN_VM_NAME" bash -lc "cat > /tmp/mounts.yaml"
cat "$SERVICE_FILE" | tart exec -i "$RUN_VM_NAME" bash -lc "cat > /tmp/motlie-vfs-guest.service"
guest_bash <<EOF
if ! getent group '$LOGIN_USER' >/dev/null 2>&1; then
  sudo groupadd -g $GID_NUM '$LOGIN_USER'
fi
if ! id -u '$LOGIN_USER' >/dev/null 2>&1; then
  sudo useradd -m -u $UID_NUM -g $GID_NUM -s /bin/bash '$LOGIN_USER'
fi
sudo install -d -m 0755 /workspace
sudo install -d -m 0755 /etc/motlie-vfs
sudo install -m 0644 /tmp/mounts.yaml /etc/motlie-vfs/mounts.yaml
sudo install -D -m 0644 /tmp/motlie-vfs-guest.service /etc/systemd/system/motlie-vfs-guest.service
sudo systemctl unmask motlie-vfs-guest.service || true
sudo systemctl daemon-reload
sudo systemctl restart motlie-vfs-guest.service
EOF

echo "--- validating guest mounts ---"
EXPECTED_WORKSPACE_README="$(
  [[ "$GUEST_NAME" == "alice" ]] &&
    printf "Alice workspace mounted from the host over the v1.15 Vz TCP bridge." ||
    printf "Bob workspace mounted from the host over the v1.15 Vz TCP bridge."
)"
EXPECTED_ENV_LINE="$(
  [[ "$GUEST_NAME" == "alice" ]] &&
    printf "ALICE_API_KEY=demo-alice" ||
    printf "BOB_API_KEY=demo-bob"
)"

VALIDATION_OK=0
for _ in $(seq 1 60); do
  if tart exec "$RUN_VM_NAME" bash -seuo pipefail <<EOF >/dev/null 2>&1
mount | grep -F ' /home/$LOGIN_USER ' >/dev/null
mount | grep -F ' /workspace ' >/dev/null
test "\$(cat /workspace/README.md)" = "$EXPECTED_WORKSPACE_README"
test "\$(cat /home/$LOGIN_USER/.env)" = "$EXPECTED_ENV_LINE"
test -s /home/$LOGIN_USER/.ssh/authorized_keys
systemctl is-active --quiet motlie-vfs-guest.service
EOF
  then
    VALIDATION_OK=1
    break
  fi
  sleep 1
done

if [[ "$VALIDATION_OK" -ne 1 ]]; then
  echo "guest mount validation failed" >&2
  tart exec "$RUN_VM_NAME" bash -lc "mount || true; printf '\n=== motlie-vfs-guest.service ===\n'; systemctl --no-pager --full status motlie-vfs-guest.service || true; printf '\n=== /workspace ===\n'; ls -la /workspace || true; printf '\n=== /home/$LOGIN_USER ===\n'; ls -la /home/$LOGIN_USER || true; printf '\n=== /home/$LOGIN_USER/.ssh ===\n'; ls -la /home/$LOGIN_USER/.ssh || true; printf '\n=== README ===\n'; cat /workspace/README.md || true; printf '\n=== ENV ===\n'; cat /home/$LOGIN_USER/.env || true" >&2 || true
  exit 1
fi

python3 - "$RESULT_JSON" "$RUN_VM_NAME" "$IP_ADDR" "$HOST_GATEWAY" "$SERVICE_PORT" "$BOOT_SECONDS" "$LOGIN_USER" <<'PY'
import json
import sys

path, vm_name, ip_addr, host_gateway, service_port, boot_seconds, login_user = sys.argv[1:]
with open(path, "w", encoding="utf-8") as fh:
    json.dump(
        {
            "backend": "vz-tart",
            "vm_name": vm_name,
            "ip_addr": ip_addr,
            "host_gateway": host_gateway,
            "service_port": int(service_port),
            "boot_to_ip_seconds": float(boot_seconds),
            "validation": {
                "guest_user": login_user,
                "mounts_ready": True,
                "service_active": True,
            },
        },
        fh,
        indent=2,
        sort_keys=True,
    )
    fh.write("\n")
PY

KEEP_RUNNING=1
echo "--- result written ---"
cat "$RESULT_JSON"
echo "=== guest running ==="
