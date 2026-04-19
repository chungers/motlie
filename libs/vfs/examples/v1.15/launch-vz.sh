#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
ARTIFACTS_DIR="$SCRIPT_DIR/artifacts"
BASE_VM_NAME="${MOTLIE_VZ_BASE_VM_NAME:-motlie-v1-15-base}"
TIMEOUT_SECONDS="${MOTLIE_VZ_TIMEOUT_SECONDS:-900}"
SERVICE_FILE="$SCRIPT_DIR/motlie-vfs-guest.service"
VALIDATE_UNIT_FILE="$SCRIPT_DIR/motlie-vfs-validate.service"
RUNNER_BUILD_SCRIPT="$SCRIPT_DIR/build-vz-runner.sh"
RUNNER_BIN_OVERRIDE="${MOTLIE_VZ_RUNNER_BIN:-}"
SKIP_RUNNER_BUILD="${MOTLIE_VZ_SKIP_RUNNER_BUILD:-0}"
SOURCE_TARBALL="$ARTIFACTS_DIR/motlie-src.tar.gz"
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
require_cmd git
require_cmd tar
require_cmd expect
require_cmd scp
require_cmd ssh
require_cmd nc
require_cmd arp

GUEST_NAME="alice"
RUN_VM_NAME=""
ENABLE_NAT=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --guest) GUEST_NAME="$2"; shift 2 ;;
    --vm-name) RUN_VM_NAME="$2"; shift 2 ;;
    --enable-nat) ENABLE_NAT=1; shift ;;
    --no-nat) ENABLE_NAT=0; shift ;;
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
    GUEST_HOSTNAME="motlie-v1-15-alice"
    NAT_MAC="02:4d:6f:74:61:11"
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
    GUEST_HOSTNAME="motlie-v1-15-bob"
    NAT_MAC="02:4d:6f:74:62:15"
    ;;
  *)
    echo "guest must be alice or bob" >&2
    exit 1
    ;;
esac

RUN_LOG="$ARTIFACTS_DIR/${GUEST_NAME}-run.log"
RESULT_JSON="$ARTIFACTS_DIR/${GUEST_NAME}-launch-result.json"
SERIAL_LOG="$ARTIFACTS_DIR/${GUEST_NAME}-serial.log"
VALIDATION_TOKEN="$RUN_VM_NAME"
RUNNER_PID_FILE="$ARTIFACTS_DIR/${GUEST_NAME}-runner.pid"
GUEST_IP_FILE="$ARTIFACTS_DIR/${GUEST_NAME}-ip.txt"

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
rm -f "$RUN_LOG" "$SERIAL_LOG" "$RUNNER_PID_FILE" "$GUEST_IP_FILE"

guest_copy() {
  local src="$1"
  local dst="$2"
  local ip_addr="$3"
  expect <<EOF
set timeout -1
spawn scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$src" admin@${ip_addr}:$dst
expect {
  "password:" {
    send "admin\r"
    exp_continue
  }
  eof
}
EOF
}

guest_bash() {
  local ip_addr="$1"
  local remote_script
  remote_script="$(mktemp)"
  cat >"$remote_script"
  expect <<EOF
set timeout -1
spawn scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$remote_script" admin@${ip_addr}:/tmp/motlie-vfs-remote.sh
expect {
  "password:" {
    send "admin\r"
    exp_continue
  }
  eof
}
EOF
  rm -f "$remote_script"
  expect <<EOF
set timeout -1
log_user 0
spawn ssh -n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null admin@${ip_addr} "bash -seuo pipefail /tmp/motlie-vfs-remote.sh </dev/null"
expect {
  "password:" {
    send "admin\r"
    exp_continue
  }
  eof {
    if {[info exists expect_out(buffer)]} {
      puts \$expect_out(buffer)
    }
  }
}
EOF
}

probe_guest_ip() {
  arp -an | python3 -c '
import re
import sys

target = sys.argv[1].strip().lower()

def normalize(mac: str) -> str:
    parts = [part for part in mac.split(":") if part]
    return ":".join(f"{int(part, 16):x}" for part in parts)

target = normalize(target)

for line in sys.stdin:
    match = re.search(r"\(([^)]+)\)\s+at\s+([0-9a-fA-F:]+)", line)
    if not match:
        continue
    ip_addr, mac = match.groups()
    if normalize(mac.lower()) == target:
        print(ip_addr)
        break
' "$NAT_MAC"
}

wait_for_guest_ip() {
  local attempts=0
  local max_attempts=$(( TIMEOUT_SECONDS * 2 ))
  local ip_addr=""
  while [[ $attempts -lt $max_attempts ]]; do
    if ! kill -0 "$RUNNER_PID" >/dev/null 2>&1; then
      echo "vz-vsock-runner exited early; log follows:" >&2
      cat "$RUN_LOG" >&2 || true
      exit 1
    fi
    ip_addr="$(probe_guest_ip || true)"
    if [[ -n "$ip_addr" ]]; then
      echo "$ip_addr"
      return 0
    fi
    if (( attempts % 10 == 0 )); then
      local candidate
      for candidate in 192.168.64.{2..40}; do
        nc -z -w 1 "$candidate" 22 >/dev/null 2>&1 || true
      done
    fi
    sleep 0.5
    attempts=$(( attempts + 1 ))
  done
  return 1
}

wait_for_guest_ssh() {
  local ip_addr="$1"
  local attempts=0
  local max_attempts=$(( TIMEOUT_SECONDS * 2 ))
  while [[ $attempts -lt $max_attempts ]]; do
    if nc -z "$ip_addr" 22 >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
    attempts=$(( attempts + 1 ))
  done
  return 1
}

guest_capture() {
  local ip_addr="$1"
  local remote_script
  remote_script="$(mktemp)"
  cat >"$remote_script"
  expect <<EOF
set timeout -1
spawn scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$remote_script" admin@${ip_addr}:/tmp/motlie-vfs-capture.sh
expect {
  "password:" {
    send "admin\r"
    exp_continue
  }
  eof
}
EOF
  rm -f "$remote_script"
  expect <<EOF
set timeout -1
log_user 0
spawn ssh -n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null admin@${ip_addr} "bash -seuo pipefail /tmp/motlie-vfs-capture.sh </dev/null"
expect {
  "password:" {
    send "admin\r"
    exp_continue
  }
  eof {
    if {[info exists expect_out(buffer)]} {
      puts \$expect_out(buffer)
    }
  }
}
EOF
}

echo "--- cloning guest VM disk ---"
tart clone "$BASE_VM_NAME" "$RUN_VM_NAME" >/dev/null

echo "--- packing Motlie source tree ---"
COPYFILE_DISABLE=1 COPY_EXTENDED_ATTRIBUTES_DISABLE=1 git -C "$REPO_ROOT" ls-files -z | COPYFILE_DISABLE=1 COPY_EXTENDED_ATTRIBUTES_DISABLE=1 tar --disable-copyfile --no-mac-metadata --no-xattrs --null -czf "$SOURCE_TARBALL" -C "$REPO_ROOT" --files-from -

if [[ -n "$RUNNER_BIN_OVERRIDE" ]]; then
  RUNNER_BIN="$RUNNER_BIN_OVERRIDE"
elif [[ "$SKIP_RUNNER_BUILD" == "1" ]]; then
  RUNNER_BIN="$SCRIPT_DIR/artifacts/build/vz-vsock-runner"
else
  echo "--- building Vz virtio-socket helper ---"
  RUNNER_BIN="$($RUNNER_BUILD_SCRIPT)"
fi
chmod +x "$RUNNER_BIN"

VM_DIR="$HOME/.tart/vms/$RUN_VM_NAME"
DISK_PATH="$VM_DIR/disk.img"
NVRAM_PATH="$VM_DIR/nvram.bin"
MACHINE_ID_PATH="$VM_DIR/machine-id.bin"

echo "--- launching Apple Vz helper ---"
: > "$RUN_LOG"
START_EPOCH="$EPOCHREALTIME"
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
  RUNNER_ARGS+=(--enable-nat --nat-mac "$NAT_MAC")
fi
"$RUNNER_BIN" \
  "${RUNNER_ARGS[@]}" \
  >"$RUN_LOG" 2>&1 &
RUNNER_PID="$!"
echo "$RUNNER_PID" > "$RUNNER_PID_FILE"

echo "--- resolving native guest IP ---"
IP_ADDR="$(wait_for_guest_ip)" || {
  echo "timed out waiting for native guest IP" >&2
  cat "$RUN_LOG" >&2 || true
  exit 1
}
echo "$IP_ADDR" > "$GUEST_IP_FILE"

echo "--- waiting for guest SSH ---"
wait_for_guest_ssh "$IP_ADDR" || {
  echo "timed out waiting for guest SSH at $IP_ADDR" >&2
  [[ -f "$SERIAL_LOG" ]] && cat "$SERIAL_LOG" >&2 || true
  exit 1
}

echo "--- provisioning guest over native Vz NAT ---"
guest_copy "$SOURCE_TARBALL" /tmp/motlie-src.tar.gz "$IP_ADDR"
guest_copy "$MOUNTS_FILE" "/tmp/mounts.${GUEST_NAME}.yaml" "$IP_ADDR"
guest_copy "$SERVICE_FILE" /tmp/motlie-vfs-guest.service "$IP_ADDR"

guest_bash "$IP_ADDR" <<EOF
printf 'admin\n' | sudo -S hostnamectl set-hostname '$GUEST_HOSTNAME'
printf 'admin\n' | sudo -S mkdir -p /var/lib/motlie /workspace /etc/motlie-vfs
if ! getent group '$LOGIN_USER' >/dev/null 2>&1; then
  printf 'admin\n' | sudo -S groupadd -g $GID_NUM '$LOGIN_USER'
fi
if ! id -u '$LOGIN_USER' >/dev/null 2>&1; then
  printf 'admin\n' | sudo -S useradd -m -u $UID_NUM -g $GID_NUM -s /bin/bash '$LOGIN_USER'
fi
printf 'admin\n' | sudo -S install -d -m 0700 -o $UID_NUM -g $GID_NUM /home/$LOGIN_USER/.ssh
printf 'admin\n' | sudo -S apt-get update
printf 'admin\n' | sudo -S DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential pkg-config libfuse3-dev curl ca-certificates tar gzip iproute2
export PATH="/home/admin/.cargo/bin:\$PATH"
if ! command -v cargo >/dev/null 2>&1; then
  curl https://sh.rustup.rs -sSf | sh -s -- -y
fi
printf 'admin\n' | sudo -S rm -rf /var/lib/motlie/src /var/lib/motlie/target
printf 'admin\n' | sudo -S mkdir -p /var/lib/motlie/src
printf 'admin\n' | sudo -S tar -xzf /tmp/motlie-src.tar.gz -C /var/lib/motlie/src
printf 'admin\n' | sudo -S chown -R admin:admin /var/lib/motlie
cargo build --manifest-path /var/lib/motlie/src/libs/vfs/Cargo.toml --release --features vsock,client --bin motlie-vfs-guest-v1_15 --target-dir /var/lib/motlie/src/target
printf 'admin\n' | sudo -S install -D -m 0755 /var/lib/motlie/src/target/release/motlie-vfs-guest-v1_15 /usr/local/bin/motlie-vfs-guest-v1_15
printf 'admin\n' | sudo -S install -D -m 0644 /tmp/mounts.${GUEST_NAME}.yaml /etc/motlie-vfs/mounts.yaml
printf 'admin\n' | sudo -S install -D -m 0644 /tmp/motlie-vfs-guest.service /etc/systemd/system/motlie-vfs-guest.service
printf 'admin\n' | sudo -S systemctl daemon-reload
printf 'admin\n' | sudo -S systemctl enable motlie-vfs-guest.service
printf 'admin\n' | sudo -S systemctl restart motlie-vfs-guest.service
EOF

POST_PROVISION_CHECK="$(guest_capture "$IP_ADDR" <<'EOF'
python3 - <<'PY2'
import json
from pathlib import Path

payload = {
    "mounts_yaml": Path("/etc/motlie-vfs/mounts.yaml").exists(),
    "guest_bin": Path("/usr/local/bin/motlie-vfs-guest-v1_15").exists(),
    "service_unit": Path("/etc/systemd/system/motlie-vfs-guest.service").exists(),
}
print(json.dumps(payload, sort_keys=True))
PY2
EOF
)"
POST_PROVISION_CHECK="$(printf '%s' "$POST_PROVISION_CHECK" | tr -d '\r' | sed -n '/^{.*}$/p' | tail -n 1)"
if [[ -z "$POST_PROVISION_CHECK" ]]; then
  echo "guest post-provision verification returned no JSON" >&2
  exit 1
fi
POST_PROVISION_OK="$(printf '%s' "$POST_PROVISION_CHECK" | python3 - <<'PY'
import json
import sys

payload = json.loads(sys.stdin.read())
print("ok" if all(payload.values()) else "bad")
PY
)"
if [[ "$POST_PROVISION_OK" != "ok" ]]; then
  echo "guest post-provision verification failed" >&2
  printf '%s\n' "$POST_PROVISION_CHECK" >&2
  exit 1
fi

VALIDATION_OK=0
for _ in $(seq 1 "$TIMEOUT_SECONDS"); do
  if ! kill -0 "$RUNNER_PID" >/dev/null 2>&1; then
    echo "vz-vsock-runner exited early; log follows:" >&2
    cat "$RUN_LOG" >&2 || true
    exit 1
  fi
  VALIDATION_JSON="$(guest_capture "$IP_ADDR" <<EOF
python3 - <<'PY2'
import json
from pathlib import Path

login_user = '$LOGIN_USER'
expected_readme = '$EXPECTED_WORKSPACE_README'
expected_env = '$EXPECTED_ENV_LINE'
token = '$VALIDATION_TOKEN'

workspace_mount = Path('/workspace')
home_mount = Path(f'/home/{login_user}')
workspace_readme_path = workspace_mount / 'README.md'
env_path = home_mount / '.env'
auth_path = home_mount / '.ssh' / 'authorized_keys'

result = {
    'token': token,
    'guest_user': login_user,
    'workspace_mountpoint': workspace_mount.is_mount(),
    'home_mountpoint': home_mount.is_mount(),
}

if not (result['workspace_mountpoint'] and result['home_mountpoint'] and workspace_readme_path.exists() and env_path.exists() and auth_path.exists()):
    print('')
    raise SystemExit(0)

workspace_readme = workspace_readme_path.read_text(encoding='utf-8').strip()
env_line = env_path.read_text(encoding='utf-8').strip()
authorized_keys = auth_path.read_text(encoding='utf-8').strip()
result.update({
    'workspace_readme': workspace_readme,
    'expected_workspace_readme': expected_readme,
    'env_line': env_line,
    'expected_env_line': expected_env,
    'authorized_keys_present': bool(authorized_keys),
})
result['status'] = 'ok' if (
    workspace_readme == expected_readme
    and env_line == expected_env
    and bool(authorized_keys)
) else 'mismatch'
print(json.dumps(result, sort_keys=True))
PY2
EOF
)" 
  VALIDATION_JSON="$(printf '%s' "$VALIDATION_JSON" | tr -d '\r' | sed -n '/^{.*}$/p' | tail -n 1)"
  if [[ -n "${VALIDATION_JSON//[[:space:]]/}" ]]; then
    STATUS="$(printf '%s' "$VALIDATION_JSON" | python3 - <<'PY'
import json
import sys

payload = json.loads(sys.stdin.read())
print(payload['status'])
PY
)"
    if [[ "$STATUS" == "ok" ]]; then
      printf '%s\n' "$VALIDATION_JSON" >"$ARTIFACTS_DIR/${GUEST_NAME}-validation.json"
      VALIDATION_OK=1
      break
    fi
    echo "guest validation reported mismatch" >&2
    printf '%s\n' "$VALIDATION_JSON" >&2
    exit 1
  fi
  sleep 1
done

if [[ "$VALIDATION_OK" -ne 1 ]]; then
  echo "timed out waiting for guest validation sentinels" >&2
  cat "$RUN_LOG" >&2 || true
  [[ -f "$SERIAL_LOG" ]] && cat "$SERIAL_LOG" >&2 || true
  exit 1
fi

READY_EPOCH="$EPOCHREALTIME"
BOOT_SECONDS="$(awk -v start="$START_EPOCH" -v ready="$READY_EPOCH" 'BEGIN { printf "%.3f", ready - start }')"

python3 - "$RESULT_JSON" "$RUN_VM_NAME" "$BOOT_SECONDS" "$SOCKET_PATH" "$RUNNER_PID" "$IP_ADDR" "$ARTIFACTS_DIR/${GUEST_NAME}-validation.json" <<'PY'
import json
import sys

path, vm_name, boot_seconds, socket_path, runner_pid, ip_addr, validation_json_path = sys.argv[1:]
with open(validation_json_path, "r", encoding="utf-8") as fh:
    validation_payload = json.load(fh)
with open(path, "w", encoding="utf-8") as fh:
    json.dump(
        {
            "backend": "vz-vsock-runner",
            "vm_name": vm_name,
            "boot_to_validation_seconds": float(boot_seconds),
            "unix_socket_path": socket_path,
            "runner_pid": int(runner_pid),
            "guest_ip": ip_addr,
            "validation": {
                "guest_validation_json": validation_json_path,
                "mounts_ready": True,
                "guest": validation_payload,
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
