#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ARTIFACTS_DIR="$SCRIPT_DIR/artifacts"
BASE_VM_NAME="${MOTLIE_VZ_BASE_VM_NAME:-motlie-v1-15-base}"
TIMEOUT_SECONDS="${MOTLIE_VZ_TIMEOUT_SECONDS:-180}"
SERVICE_FILE="$SCRIPT_DIR/motlie-vfs-guest.service"
VALIDATE_UNIT_FILE="$SCRIPT_DIR/motlie-vfs-validate.service"
RUNNER_BUILD_SCRIPT="$SCRIPT_DIR/build-vz-runner.sh"
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
require_cmd stat

GUEST_NAME="alice"
RUN_VM_NAME=""
ENABLE_NAT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --guest) GUEST_NAME="$2"; shift 2 ;;
    --vm-name) RUN_VM_NAME="$2"; shift 2 ;;
    --enable-nat) ENABLE_NAT=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

case "$GUEST_NAME" in
  alice)
    RUN_VM_NAME="${RUN_VM_NAME:-motlie-v1-15-alice-$(date +%s)}"
    MOUNTS_FILE="$SCRIPT_DIR/mounts.alice.yaml"
    LOGIN_USER="alice"
    UID_NUM=2000
    GID_NUM=2000
    SOCKET_PATH="/tmp/motlie-vfs-alice.vsock_5000"
    HOST_HOME_DIR="/tmp/motlie-vfs-demo/alice-home"
    HOST_WORKSPACE_DIR="/tmp/motlie-vfs-demo/alice-workspace"
    EXPECTED_WORKSPACE_README="Alice workspace mounted from the host over the v1.15 Vz virtio-socket bridge."
    EXPECTED_ENV_LINE="ALICE_API_KEY=demo-alice"
    ;;
  bob)
    RUN_VM_NAME="${RUN_VM_NAME:-motlie-v1-15-bob-$(date +%s)}"
    MOUNTS_FILE="$SCRIPT_DIR/mounts.bob.yaml"
    LOGIN_USER="bob"
    UID_NUM=2001
    GID_NUM=2001
    SOCKET_PATH="/tmp/motlie-vfs-bob.vsock_5000"
    HOST_HOME_DIR="/tmp/motlie-vfs-demo/bob-home"
    HOST_WORKSPACE_DIR="/tmp/motlie-vfs-demo/bob-workspace"
    EXPECTED_WORKSPACE_README="Bob workspace mounted from the host over the v1.15 Vz virtio-socket bridge."
    EXPECTED_ENV_LINE="BOB_API_KEY=demo-bob"
    ;;
  *)
    echo "guest must be alice or bob" >&2
    exit 1
    ;;
esac

RUN_LOG="$ARTIFACTS_DIR/${GUEST_NAME}-run.log"
PROVISION_LOG="$ARTIFACTS_DIR/${GUEST_NAME}-provision.log"
RESULT_JSON="$ARTIFACTS_DIR/${GUEST_NAME}-launch-result.json"
SERIAL_LOG="$ARTIFACTS_DIR/${GUEST_NAME}-serial.log"
VALIDATION_TOKEN="$RUN_VM_NAME"
WORKSPACE_SENTINEL="$HOST_WORKSPACE_DIR/.motlie-vfs-validation-${VALIDATION_TOKEN}.json"
HOME_SENTINEL="$HOST_HOME_DIR/.motlie-vfs-validation-${VALIDATION_TOKEN}.json"
RUNNER_PID_FILE="$ARTIFACTS_DIR/${GUEST_NAME}-runner.pid"

local_vm_exists() {
  tart list --source local -q 2>/dev/null | grep -Fx "$1" >/dev/null 2>&1
}

cleanup() {
  if [[ "$KEEP_RUNNING" -eq 0 ]]; then
    if [[ -f "$RUNNER_PID_FILE" ]]; then
      kill "$(cat "$RUNNER_PID_FILE")" >/dev/null 2>&1 || true
      rm -f "$RUNNER_PID_FILE"
    fi
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

mkdir -p "$ARTIFACTS_DIR"
rm -f "$WORKSPACE_SENTINEL" "$HOME_SENTINEL" "$RUN_LOG" "$PROVISION_LOG" "$SERIAL_LOG" "$RUNNER_PID_FILE"

echo "--- cloning guest VM for provisioning ---"
tart clone "$BASE_VM_NAME" "$RUN_VM_NAME" >/dev/null

echo "--- booting guest with Tart for provisioning ---"
: > "$PROVISION_LOG"
START_EPOCH="$EPOCHREALTIME"
tart run --no-graphics "$RUN_VM_NAME" >"$PROVISION_LOG" 2>&1 &
TART_PID="$!"

IP_ADDR=""
ATTEMPTS=0
MAX_ATTEMPTS=$(( TIMEOUT_SECONDS * 2 ))
while [[ $ATTEMPTS -lt $MAX_ATTEMPTS ]]; do
  if ! kill -0 "$TART_PID" >/dev/null 2>&1; then
    echo "tart run exited early during provisioning; log follows:" >&2
    cat "$PROVISION_LOG" >&2
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
  echo "timed out waiting for guest IP during provisioning" >&2
  cat "$PROVISION_LOG" >&2 || true
  exit 1
fi

READY_EPOCH="$EPOCHREALTIME"
PROVISION_BOOT_SECONDS="$(awk -v start="$START_EPOCH" -v ready="$READY_EPOCH" 'BEGIN { printf "%.3f", ready - start }')"

guest_bash() {
  tart exec -i "$RUN_VM_NAME" bash -seuo pipefail
}

VALIDATE_SCRIPT_CONTENT="$(python3 - "$LOGIN_USER" "$EXPECTED_WORKSPACE_README" "$EXPECTED_ENV_LINE" "$VALIDATION_TOKEN" <<'PY'
import json
import sys

login_user, expected_readme, expected_env, token = sys.argv[1:]
script = f"""#!/bin/bash
set -euo pipefail

LOGIN_USER={json.dumps(login_user)}
EXPECTED_README={json.dumps(expected_readme)}
EXPECTED_ENV={json.dumps(expected_env)}
TOKEN={json.dumps(token)}
WORKSPACE_SENTINEL=\"/workspace/.motlie-vfs-validation-${{TOKEN}}.json\"
HOME_SENTINEL=\"/home/${{LOGIN_USER}}/.motlie-vfs-validation-${{TOKEN}}.json\"
deadline=$((SECONDS + 120))

while [ \"$SECONDS\" -lt \"$deadline\" ]; do
  if mountpoint -q /workspace && mountpoint -q /home/${{LOGIN_USER}} && [ -f /workspace/README.md ] && [ -f /home/${{LOGIN_USER}}/.env ] && [ -s /home/${{LOGIN_USER}}/.ssh/authorized_keys ]; then
    python3 - <<'PY2'
import json
from pathlib import Path

login_user = {json.dumps(login_user)}
expected_readme = {json.dumps(expected_readme)}
expected_env = {json.dumps(expected_env)}
token = {json.dumps(token)}

actual_readme = Path('/workspace/README.md').read_text(encoding='utf-8').strip()
actual_env = Path(f'/home/{{login_user}}/.env').read_text(encoding='utf-8').strip()
authorized_keys = Path(f'/home/{{login_user}}/.ssh/authorized_keys').read_text(encoding='utf-8').strip()

payload = {{
    'token': token,
    'guest_user': login_user,
    'status': 'ok' if actual_readme == expected_readme and actual_env == expected_env and bool(authorized_keys) else 'mismatch',
    'workspace_readme': actual_readme,
    'expected_workspace_readme': expected_readme,
    'env_line': actual_env,
    'expected_env_line': expected_env,
    'authorized_keys_present': bool(authorized_keys),
}}

workspace_path = Path(f'/workspace/.motlie-vfs-validation-{{token}}.json')
home_path = Path(f'/home/{{login_user}}/.motlie-vfs-validation-{{token}}.json')
workspace_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\\n', encoding='utf-8')
home_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\\n', encoding='utf-8')
PY2
    exit 0
  fi
  sleep 1
done

python3 - <<'PY2'
import json
from pathlib import Path

token = {json.dumps(token)}
login_user = {json.dumps(login_user)}
payload = {{
    'token': token,
    'guest_user': login_user,
    'status': 'timeout',
}}
workspace_path = Path(f'/workspace/.motlie-vfs-validation-{{token}}.json')
home_path = Path(f'/home/{{login_user}}/.motlie-vfs-validation-{{token}}.json')
workspace_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\\n', encoding='utf-8')
home_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\\n', encoding='utf-8')
PY2

exit 1
"""
print(script, end="")
PY
)"

echo "--- provisioning guest identity and mount contract ---"
cat "$MOUNTS_FILE" | tart exec -i "$RUN_VM_NAME" bash -lc "cat > /tmp/mounts.yaml"
cat "$SERVICE_FILE" | tart exec -i "$RUN_VM_NAME" bash -lc "cat > /tmp/motlie-vfs-guest.service"
cat "$VALIDATE_UNIT_FILE" | tart exec -i "$RUN_VM_NAME" bash -lc "cat > /tmp/motlie-vfs-validate.service"
printf "%s" "$VALIDATE_SCRIPT_CONTENT" | tart exec -i "$RUN_VM_NAME" bash -lc "cat > /tmp/motlie-vfs-v1_15-validate.sh"

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
sudo install -D -m 0644 /tmp/motlie-vfs-validate.service /etc/systemd/system/motlie-vfs-validate.service
sudo install -D -m 0755 /tmp/motlie-vfs-v1_15-validate.sh /usr/local/bin/motlie-vfs-v1_15-validate.sh
sudo systemctl unmask motlie-vfs-guest.service || true
sudo systemctl daemon-reload
sudo systemctl enable motlie-vfs-validate.service >/dev/null 2>&1 || true
EOF

echo "--- stopping Tart provisioning guest ---"
tart stop "$RUN_VM_NAME" >/dev/null
wait "$TART_PID" || true

echo "--- building Vz virtio-socket helper ---"
RUNNER_BIN="$($RUNNER_BUILD_SCRIPT)"
chmod +x "$RUNNER_BIN"

VM_DIR="$HOME/.tart/vms/$RUN_VM_NAME"
DISK_PATH="$VM_DIR/disk.img"
NVRAM_PATH="$VM_DIR/nvram.bin"
MACHINE_ID_PATH="$VM_DIR/machine-id.bin"

echo "--- launching Apple Vz helper ---"
: > "$RUN_LOG"
RUNNER_ARGS=(
  --disk "$DISK_PATH"
  --nvram "$NVRAM_PATH"
  --machine-id "$MACHINE_ID_PATH"
  --serial-log "$SERIAL_LOG"
  --unix-socket "$SOCKET_PATH"
  --vsock-port 5000
  --memory-mib 4096
  --cpu-count 4
)
if [[ "$ENABLE_NAT" -eq 1 ]]; then
  RUNNER_ARGS+=(--enable-nat)
fi
"$RUNNER_BIN" \
  "${RUNNER_ARGS[@]}" \
  >"$RUN_LOG" 2>&1 &
RUNNER_PID="$!"
echo "$RUNNER_PID" > "$RUNNER_PID_FILE"

VALIDATION_OK=0
for _ in $(seq 1 "$TIMEOUT_SECONDS"); do
  if ! kill -0 "$RUNNER_PID" >/dev/null 2>&1; then
    echo "vz-vsock-runner exited early; log follows:" >&2
    cat "$RUN_LOG" >&2 || true
    exit 1
  fi
  if [[ -f "$WORKSPACE_SENTINEL" && -f "$HOME_SENTINEL" ]]; then
    STATUS="$(python3 - "$WORKSPACE_SENTINEL" "$HOME_SENTINEL" <<'PY'
import json
import sys

paths = sys.argv[1:]
statuses = []
for path in paths:
    with open(path, 'r', encoding='utf-8') as fh:
        statuses.append(json.load(fh).get('status'))
print('ok' if statuses and all(s == 'ok' for s in statuses) else 'mismatch')
PY
)"
    if [[ "$STATUS" == "ok" ]]; then
      VALIDATION_OK=1
      break
    fi
    echo "validation sentinel reported mismatch" >&2
    cat "$WORKSPACE_SENTINEL" >&2 || true
    cat "$HOME_SENTINEL" >&2 || true
    exit 1
  fi
  sleep 1
done

if [[ "$VALIDATION_OK" -ne 1 ]]; then
  echo "timed out waiting for host-backed validation sentinels" >&2
  cat "$RUN_LOG" >&2 || true
  [[ -f "$SERIAL_LOG" ]] && cat "$SERIAL_LOG" >&2 || true
  exit 1
fi

python3 - "$RESULT_JSON" "$RUN_VM_NAME" "$PROVISION_BOOT_SECONDS" "$SOCKET_PATH" "$RUNNER_PID" "$WORKSPACE_SENTINEL" "$HOME_SENTINEL" <<'PY'
import json
import sys

path, vm_name, boot_seconds, socket_path, runner_pid, workspace_sentinel, home_sentinel = sys.argv[1:]
with open(path, "w", encoding="utf-8") as fh:
    json.dump(
        {
            "backend": "vz-vsock-runner",
            "vm_name": vm_name,
            "provision_boot_to_ip_seconds": float(boot_seconds),
            "unix_socket_path": socket_path,
            "runner_pid": int(runner_pid),
            "validation": {
                "workspace_sentinel": workspace_sentinel,
                "home_sentinel": home_sentinel,
                "mounts_ready": True,
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
echo "=== guest running through Apple virtio-socket ==="
