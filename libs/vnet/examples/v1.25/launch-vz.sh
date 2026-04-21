#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
ARTIFACTS_DIR="$SCRIPT_DIR/artifacts"
BASE_VM_NAME="${MOTLIE_VZ_BASE_VM_NAME:-motlie-v1-25-base-iter}"
BASE_VM_DIR_OVERRIDE="${MOTLIE_VZ_BASE_VM_DIR:-}"
TIMEOUT_SECONDS="${MOTLIE_VZ_TIMEOUT_SECONDS:-900}"
SERVICE_FILE="$SCRIPT_DIR/motlie-vfs-guest.service"
VALIDATE_UNIT_FILE="$SCRIPT_DIR/motlie-vfs-validate.service"
AGENT_STATE_SETUP_FILE="$SCRIPT_DIR/motlie-agent-state-setup.sh"
AGENT_STATE_UNIT_FILE="$SCRIPT_DIR/motlie-agent-state.service"
RUNNER_BUILD_SCRIPT="$SCRIPT_DIR/build-vz-runner.sh"
RUNNER_BIN_OVERRIDE="${MOTLIE_VZ_RUNNER_BIN:-}"
SKIP_RUNNER_BUILD="${MOTLIE_VZ_SKIP_RUNNER_BUILD:-0}"
SOURCE_TARBALL="$ARTIFACTS_DIR/motlie-src.tar.gz"
KEEP_RUNNING="${MOTLIE_VZ_KEEP_RUNNING:-0}"
REUSE_VM="${MOTLIE_VZ_REUSE_VM:-0}"
NATIVE_SOURCE_VM_DIR="${MOTLIE_VZ_NATIVE_SOURCE_VM_DIR:-$ARTIFACTS_DIR/source-base.vm}"
BASE_SOURCE_DIR=""
RUN_VM_DIR=""

zmodload zsh/datetime

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

require_cmd python3
require_cmd git
require_cmd tar
require_cmd expect
require_cmd scp
require_cmd ssh
require_cmd nc
require_cmd arp

if [[ -n "$BASE_VM_DIR_OVERRIDE" ]]; then
  BASE_SOURCE_DIR="$BASE_VM_DIR_OVERRIDE"
elif [[ -f "$NATIVE_SOURCE_VM_DIR/disk.img" && -f "$NATIVE_SOURCE_VM_DIR/nvram.bin" ]]; then
  BASE_SOURCE_DIR="$NATIVE_SOURCE_VM_DIR"
else
  cat >&2 <<EOF
native base artifacts are required for v1.25 guest launches

Set:
  MOTLIE_VZ_BASE_VM_DIR

Or build/populate the default native cache first:
  $NATIVE_SOURCE_VM_DIR
EOF
  exit 1
fi

GUEST_NAME="alice"
RUN_VM_NAME=""
ENABLE_NAT=1
REUSED_VM=0
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
    RUN_VM_NAME="${RUN_VM_NAME:-motlie-v1-25-alice-iter}"
    MOUNTS_FILE="$SCRIPT_DIR/mounts.alice.yaml"
    LOGIN_USER="alice"
    UID_NUM=1000
    GID_NUM=1000
    SOCKET_PATH="/tmp/motlie-vnet-alice.vsock_5000"
    HOST_HOME_DIR="/tmp/motlie-vnet-demo/alice-home"
    HOST_AGENT_STATE_DIR="/tmp/motlie-vnet-demo/alice-agent-state"
    HOST_WORKSPACE_DIR="/tmp/motlie-vnet-demo/alice-workspace"
    EXPECTED_WORKSPACE_README="Alice workspace mounted from the host."
    EXPECTED_AGENT_STATE_README="Dedicated read-write agent-state layer for Codex and Claude lives here."
    EXPECTED_ENV_LINE="ALICE_API_KEY=demo-alice"
    GUEST_HOSTNAME="motlie-alice"
    NAT_MAC="02:4d:6f:74:61:11"
    ;;
  bob)
    RUN_VM_NAME="${RUN_VM_NAME:-motlie-v1-25-bob-iter}"
    MOUNTS_FILE="$SCRIPT_DIR/mounts.bob.yaml"
    LOGIN_USER="bob"
    UID_NUM=1001
    GID_NUM=1001
    SOCKET_PATH="/tmp/motlie-vnet-bob.vsock_5000"
    HOST_HOME_DIR="/tmp/motlie-vnet-demo/bob-home"
    HOST_AGENT_STATE_DIR="/tmp/motlie-vnet-demo/bob-agent-state"
    HOST_WORKSPACE_DIR="/tmp/motlie-vnet-demo/bob-workspace"
    EXPECTED_WORKSPACE_README="Bob workspace mounted from the host."
    EXPECTED_AGENT_STATE_README="Dedicated read-write agent-state layer for Codex and Claude lives here."
    EXPECTED_ENV_LINE="BOB_API_KEY=demo-bob"
    GUEST_HOSTNAME="motlie-bob"
    NAT_MAC="02:4d:6f:74:62:15"
    ;;
  *)
    echo "guest must be alice or bob" >&2
    exit 1
    ;;
esac

CONTROL_USER="$LOGIN_USER"
CONTROL_PASSWORD="testpass"

RUN_LOG="$ARTIFACTS_DIR/${GUEST_NAME}-run.log"
RESULT_JSON="$ARTIFACTS_DIR/${GUEST_NAME}-launch-result.json"
SERIAL_LOG="$ARTIFACTS_DIR/${GUEST_NAME}-serial.log"
VALIDATION_TOKEN="$RUN_VM_NAME"
RUNNER_PID_FILE="$ARTIFACTS_DIR/${GUEST_NAME}-runner.pid"
GUEST_IP_FILE="$ARTIFACTS_DIR/${GUEST_NAME}-ip.txt"

require_host_socket_ready() {
  python3 - "$SOCKET_PATH" <<'PY'
import socket
import sys

path = sys.argv[1]
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.settimeout(1.0)
try:
    sock.connect(path)
except Exception as exc:
    print(f"host unix socket not ready: {path}: {exc}", file=sys.stderr)
    raise SystemExit(1)
finally:
    sock.close()
PY
}

print_failure_context() {
  echo "--- runner log tail ---" >&2
  if [[ -f "$RUN_LOG" ]]; then
    tail -n 80 "$RUN_LOG" >&2 || true
  else
    echo "(missing $RUN_LOG)" >&2
  fi
  echo "--- serial log tail ---" >&2
  if [[ -f "$SERIAL_LOG" ]]; then
    tail -n 80 "$SERIAL_LOG" >&2 || true
  else
    echo "(missing $SERIAL_LOG)" >&2
  fi
  if [[ -f "$GUEST_IP_FILE" ]]; then
    echo "--- guest ip ---" >&2
    cat "$GUEST_IP_FILE" >&2 || true
  fi
}

cleanup() {
  if [[ "$KEEP_RUNNING" -eq 0 ]]; then
    if [[ -f "$RUNNER_PID_FILE" ]]; then
      kill "$(cat "$RUNNER_PID_FILE")" >/dev/null 2>&1 || true
      rm -f "$RUNNER_PID_FILE"
    fi
    if [[ "$REUSED_VM" -eq 0 ]] && [[ -n "$RUN_VM_DIR" && -d "$RUN_VM_DIR" ]]; then
      rm -rf "$RUN_VM_DIR"
    fi
  fi
}

trap cleanup EXIT

kill_stale_runners() {
  local stale_pids=()
  local stale_pid=""
  while read -r stale_pid; do
    [[ -n "$stale_pid" ]] || continue
    stale_pids+=("$stale_pid")
  done < <(ps -Ao pid=,command= -ww | awk -v mac="$NAT_MAC" -v sock="$SOCKET_PATH" '
    /vz-vsock-runner/ && index($0, mac) && index($0, sock) { print $1 }
  ')

  if [[ "${#stale_pids[@]}" -eq 0 ]]; then
    return 0
  fi

  printf '%s\n' "--- terminating stale Vz runners for $GUEST_NAME ---"
  kill "${stale_pids[@]}" >/dev/null 2>&1 || true

  local attempts=0
  while [[ $attempts -lt 20 ]]; do
    local survivors=()
    while read -r stale_pid; do
      [[ -n "$stale_pid" ]] || continue
      survivors+=("$stale_pid")
    done < <(ps -Ao pid=,command= -ww | awk -v mac="$NAT_MAC" -v sock="$SOCKET_PATH" '
      /vz-vsock-runner/ && index($0, mac) && index($0, sock) { print $1 }
    ')
    if [[ "${#survivors[@]}" -eq 0 ]]; then
      return 0
    fi
    sleep 0.5
    attempts=$(( attempts + 1 ))
    if [[ $attempts -eq 10 ]]; then
      kill -9 "${survivors[@]}" >/dev/null 2>&1 || true
    fi
  done

  echo "failed to terminate stale Vz runners for $GUEST_NAME" >&2
  ps -Ao pid,etime,command | awk -v mac="$NAT_MAC" -v sock="$SOCKET_PATH" '
    /vz-vsock-runner/ && index($0, mac) && index($0, sock) { print }
  ' >&2 || true
  exit 1
}

if [[ -f "$RUNNER_PID_FILE" ]]; then
  kill "$(cat "$RUNNER_PID_FILE")" >/dev/null 2>&1 || true
  rm -f "$RUNNER_PID_FILE"
fi
kill_stale_runners

if [[ ! -f "$BASE_SOURCE_DIR/disk.img" || ! -f "$BASE_SOURCE_DIR/nvram.bin" ]]; then
  echo "base VM artifacts not found at '$BASE_SOURCE_DIR'; run ./build-guest.sh first" >&2
  exit 1
fi
require_host_socket_ready

RUN_VM_DIR="$ARTIFACTS_DIR/${RUN_VM_NAME}.vm"
if [[ -d "$RUN_VM_DIR" ]]; then
  if [[ "$REUSE_VM" == "1" ]]; then
    echo "--- reusing existing guest VM '$RUN_VM_NAME' ---"
    REUSED_VM=1
  else
    echo "--- removing existing guest VM '$RUN_VM_NAME' for rerun ---"
    rm -rf "$RUN_VM_DIR"
  fi
fi

mkdir -p "$ARTIFACTS_DIR"
rm -f "$RUN_LOG" "$SERIAL_LOG" "$RUNNER_PID_FILE" "$GUEST_IP_FILE" "$RESULT_JSON" \
  "$ARTIFACTS_DIR/${GUEST_NAME}-validation.json"

guest_copy() {
  local src="$1"
  local dst="$2"
  local ip_addr="$3"
  expect <<EOF
set timeout -1
spawn scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$src" ${CONTROL_USER}@${ip_addr}:$dst
expect {
  "password:" {
    send "${CONTROL_PASSWORD}\r"
    exp_continue
  }
  eof {}
  timeout {
    puts stderr "scp timed out: $src -> $dst"
    exit 124
  }
}
catch wait result
set exit_code [lindex \$result 3]
if {\$exit_code != 0} {
  exit \$exit_code
}
EOF
}

guest_fetch() {
  local src="$1"
  local dst="$2"
  local ip_addr="$3"
  expect <<EOF
set timeout -1
spawn scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ${CONTROL_USER}@${ip_addr}:$src "$dst"
expect {
  "password:" {
    send "${CONTROL_PASSWORD}\r"
    exp_continue
  }
  eof {}
  timeout {
    puts stderr "scp timed out: ${src} -> ${dst}"
    exit 124
  }
}
catch wait result
set exit_code [lindex \$result 3]
if {\$exit_code != 0} {
  exit \$exit_code
}
EOF
}

require_host_fixture_dirs() {
  mkdir -p "$HOST_HOME_DIR/.ssh" "$HOST_HOME_DIR/.config" "$HOST_AGENT_STATE_DIR" "$HOST_WORKSPACE_DIR"
}

require_host_fixture_dirs

guest_bash() {
  local ip_addr="$1"
  local remote_script
  remote_script="$(mktemp)"
  cat >"$remote_script"
  expect <<EOF
set timeout -1
spawn scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$remote_script" ${CONTROL_USER}@${ip_addr}:/tmp/motlie-vfs-remote.sh
expect {
  "password:" {
    send "${CONTROL_PASSWORD}\r"
    exp_continue
  }
  eof {}
  timeout {
    puts stderr "scp timed out: remote guest bash script"
    exit 124
  }
}
catch wait result
set exit_code [lindex \$result 3]
if {\$exit_code != 0} {
  exit \$exit_code
}
EOF
  rm -f "$remote_script"
  expect <<EOF
set timeout -1
log_user 0
set output ""
spawn ssh -n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ${CONTROL_USER}@${ip_addr} "bash -euo pipefail /tmp/motlie-vfs-remote.sh </dev/null"
expect {
  "password:" {
    send "${CONTROL_PASSWORD}\r"
    exp_continue
  }
  -re ".+" {
    append output \$expect_out(0,string)
    exp_continue
  }
  eof {
    if {\$output ne ""} {
      puts \$output
    }
  }
  timeout {
    puts stderr "ssh timed out: remote guest bash script"
    exit 124
  }
}
catch wait result
set exit_code [lindex \$result 3]
if {\$exit_code != 0} {
  exit \$exit_code
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

probe_guest_ip_via_ssh() {
  python3 - "$NAT_MAC" <<'PY'
import sys

target = sys.argv[1].strip().lower()

def normalize(mac: str) -> str:
    parts = [part for part in mac.split(":") if part]
    return ":".join(f"{int(part, 16):x}" for part in parts)

print(normalize(target))
PY
}

guest_mac_for_ip() {
  local ip_addr="$1"
  expect <<EOF
set timeout 10
log_user 0
set output ""
spawn ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ${CONTROL_USER}@${ip_addr} "ip -o link show enp0s1 | grep link/ether"
expect {
  "password:" {
    send "${CONTROL_PASSWORD}\r"
    exp_continue
  }
  -re ".+" {
    append output \$expect_out(0,string)
    exp_continue
  }
  eof {
    if {\$output ne ""} {
      puts \$output
    }
  }
  timeout {
    exit 124
  }
}
catch wait result
set exit_code [lindex \$result 3]
if {\$exit_code != 0} {
  exit \$exit_code
}
EOF
}

probe_guest_ip_by_ssh() {
  local target_mac=""
  target_mac="$(probe_guest_ip_via_ssh)"
  local candidate=""
  local candidate_mac=""
  for candidate in 192.168.64.{2..40}; do
    ssh_ready_for_password_prompt "$candidate" || continue
    candidate_mac="$(guest_mac_for_ip "$candidate" 2>/dev/null | python3 -c '
import re
import sys

def normalize(mac: str) -> str:
    parts = [part for part in mac.split(":") if part]
    return ":".join(f"{int(part, 16):x}" for part in parts)

text = sys.stdin.read()
match = re.search(r"link/ether\s+([0-9a-fA-F:]+)", text)
if match:
    print(normalize(match.group(1).lower()))
' 2>/dev/null || true)"
    if [[ -n "$candidate_mac" && "$candidate_mac" == "$target_mac" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

ssh_ready_for_password_prompt() {
  local ip_addr="$1"
  expect <<EOF
set timeout 5
log_user 0
spawn ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=1 -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 ${CONTROL_USER}@${ip_addr} true
expect {
  "password:" {
    exit 0
  }
  "Permission denied" {
    exit 0
  }
  eof {
    catch wait result
    set exit_code [lindex \$result 3]
    exit \$exit_code
  }
  timeout {
    exit 124
  }
}
EOF
}

wait_for_guest_ip() {
  local attempts=0
  local max_attempts=$(( TIMEOUT_SECONDS * 2 ))
  local ip_addr=""
  while [[ $attempts -lt $max_attempts ]]; do
    if ! kill -0 "$RUNNER_PID" >/dev/null 2>&1; then
      echo "vz-vsock-runner exited early; log follows:" >&2
      print_failure_context
      exit 1
    fi
    ip_addr="$(probe_guest_ip || true)"
    if [[ -n "$ip_addr" ]]; then
      echo "$ip_addr"
      return 0
    fi
    if (( attempts % 10 == 0 )); then
      for candidate in 192.168.64.{2..40}; do
        ssh_ready_for_password_prompt "$candidate" >/dev/null 2>&1 || true
      done
      ip_addr="$(probe_guest_ip_by_ssh || true)"
      if [[ -n "$ip_addr" ]]; then
        echo "$ip_addr"
        return 0
      fi
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
    if ssh_ready_for_password_prompt "$ip_addr" >/dev/null 2>&1; then
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
spawn scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$remote_script" ${CONTROL_USER}@${ip_addr}:/tmp/motlie-vfs-capture.sh
expect {
  "password:" {
    send "${CONTROL_PASSWORD}\r"
    exp_continue
  }
  eof {}
  timeout {
    puts stderr "scp timed out: remote guest capture script"
    exit 124
  }
}
catch wait result
set exit_code [lindex \$result 3]
if {\$exit_code != 0} {
  exit \$exit_code
}
EOF
  rm -f "$remote_script"
  expect <<EOF
set timeout -1
log_user 0
set output ""
spawn ssh -n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ${CONTROL_USER}@${ip_addr} "bash -euo pipefail /tmp/motlie-vfs-capture.sh </dev/null"
expect {
  "password:" {
    send "${CONTROL_PASSWORD}\r"
    exp_continue
  }
  -re ".+" {
    append output \$expect_out(0,string)
    exp_continue
  }
  eof {
    if {\$output ne ""} {
      puts \$output
    }
  }
  timeout {
    puts stderr "ssh timed out: remote guest capture script"
    exit 124
  }
}
catch wait result
set exit_code [lindex \$result 3]
if {\$exit_code != 0} {
  exit \$exit_code
}
EOF
}

if [[ "$REUSED_VM" -eq 0 ]]; then
  echo "--- materializing guest VM from base artifacts ---"
  mkdir -p "$RUN_VM_DIR"
  BASE_DISK_PATH="$BASE_SOURCE_DIR/disk.img"
  BASE_NVRAM_PATH="$BASE_SOURCE_DIR/nvram.bin"
  BASE_MACHINE_ID_PATH="$BASE_SOURCE_DIR/machine-id.bin"
  python3 - "$BASE_DISK_PATH" "$BASE_NVRAM_PATH" "$BASE_MACHINE_ID_PATH" "$RUN_VM_DIR" <<'PY'
import os
import shutil
import sys

src_disk, src_nvram, src_machine_id, out_dir = sys.argv[1:]
os.makedirs(out_dir, exist_ok=True)
for src, name in (
    (src_disk, "disk.img"),
    (src_nvram, "nvram.bin"),
):
    dst = os.path.join(out_dir, name)
    tmp = dst + ".tmp-copy"
    if os.path.exists(tmp):
        os.remove(tmp)
    with open(src, "rb") as rf, open(tmp, "wb") as wf:
        shutil.copyfileobj(rf, wf, length=16 * 1024 * 1024)
    os.replace(tmp, dst)

machine_id_dst = os.path.join(out_dir, "machine-id.bin")
if src_machine_id and os.path.exists(src_machine_id):
    tmp = machine_id_dst + ".tmp-copy"
    if os.path.exists(tmp):
        os.remove(tmp)
    with open(src_machine_id, "rb") as rf, open(tmp, "wb") as wf:
        shutil.copyfileobj(rf, wf, length=1024 * 1024)
    os.replace(tmp, machine_id_dst)
else:
    with open(machine_id_dst, "wb"):
        pass
PY
fi

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

VM_DIR="$RUN_VM_DIR"
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
  print_failure_context
  exit 1
}
echo "$IP_ADDR" > "$GUEST_IP_FILE"

echo "--- waiting for guest SSH ---"
wait_for_guest_ssh "$IP_ADDR" || {
  echo "timed out waiting for guest SSH at $IP_ADDR" >&2
  print_failure_context
  exit 1
}

echo "--- provisioning guest over native Vz NAT ---"
guest_copy "$SOURCE_TARBALL" /tmp/motlie-src.tar.gz "$IP_ADDR"
guest_copy "$MOUNTS_FILE" "/tmp/mounts.${GUEST_NAME}.yaml" "$IP_ADDR"
guest_copy "$SERVICE_FILE" /tmp/motlie-vfs-guest.service "$IP_ADDR"
guest_copy "$AGENT_STATE_SETUP_FILE" /tmp/motlie-agent-state-setup "$IP_ADDR"
guest_copy "$AGENT_STATE_UNIT_FILE" /tmp/motlie-agent-state.service "$IP_ADDR"

guest_bash "$IP_ADDR" <<EOF
printf '${CONTROL_PASSWORD}\n' | sudo -S hostnamectl set-hostname '$GUEST_HOSTNAME' || true
printf '${CONTROL_PASSWORD}\n' | sudo -S umount -lf /workspace >/dev/null 2>&1 || true
printf '${CONTROL_PASSWORD}\n' | sudo -S umount -lf /agent-state >/dev/null 2>&1 || true
printf '${CONTROL_PASSWORD}\n' | sudo -S umount -lf /home/$LOGIN_USER >/dev/null 2>&1 || true
printf '${CONTROL_PASSWORD}\n' | sudo -S mkdir -p /var/lib/motlie /workspace /agent-state /etc/motlie-vfs
existing_gid="\$(getent group '$LOGIN_USER' | cut -d: -f3 || true)"
if [[ -z "\$existing_gid" ]]; then
  gid_owner="\$(getent group $GID_NUM | cut -d: -f1 || true)"
  if [[ -n "\$gid_owner" && "\$gid_owner" != '$LOGIN_USER' ]]; then
    echo "gid $GID_NUM already belongs to \$gid_owner; cannot provision $LOGIN_USER" >&2
    exit 1
  fi
  printf '${CONTROL_PASSWORD}\n' | sudo -S groupadd -g $GID_NUM '$LOGIN_USER'
elif [[ "\$existing_gid" != "$GID_NUM" ]]; then
  echo "guest group $LOGIN_USER has gid \$existing_gid but expected $GID_NUM" >&2
  exit 1
fi

existing_uid="\$(id -u '$LOGIN_USER' 2>/dev/null || true)"
if [[ -z "\$existing_uid" ]]; then
  uid_owner="\$(getent passwd $UID_NUM | cut -d: -f1 || true)"
  if [[ -n "\$uid_owner" && "\$uid_owner" != '$LOGIN_USER' ]]; then
    echo "uid $UID_NUM already belongs to \$uid_owner; cannot provision $LOGIN_USER" >&2
    exit 1
  fi
  printf '${CONTROL_PASSWORD}\n' | sudo -S useradd -m -u $UID_NUM -g $GID_NUM -s /bin/bash '$LOGIN_USER'
elif [[ "\$existing_uid" != "$UID_NUM" ]]; then
  echo "guest user $LOGIN_USER has uid \$existing_uid but expected $UID_NUM" >&2
  exit 1
fi
printf '${CONTROL_PASSWORD}\n' | sudo -S bash -c "printf '%s:%s\n' '$LOGIN_USER' 'testpass' | chpasswd"
printf '${CONTROL_PASSWORD}\n' | sudo -S install -d -m 0700 -o $UID_NUM -g $GID_NUM /home/$LOGIN_USER/.ssh
printf '${CONTROL_PASSWORD}\n' | sudo -S chown root:root /workspace
printf '${CONTROL_PASSWORD}\n' | sudo -S chmod 0755 /workspace
printf '${CONTROL_PASSWORD}\n' | sudo -S chown root:root /agent-state
printf '${CONTROL_PASSWORD}\n' | sudo -S chmod 0755 /agent-state
if ! dpkg -s build-essential pkg-config libfuse3-dev curl ca-certificates tar gzip iproute2 dnsutils >/dev/null 2>&1; then
  printf '${CONTROL_PASSWORD}\n' | sudo -S apt-get update
  printf '${CONTROL_PASSWORD}\n' | sudo -S DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential pkg-config libfuse3-dev curl ca-certificates tar gzip iproute2 dnsutils
fi
if ! dpkg -s npm >/dev/null 2>&1; then
  printf '${CONTROL_PASSWORD}\n' | sudo -S apt-get update
  printf '${CONTROL_PASSWORD}\n' | sudo -S DEBIAN_FRONTEND=noninteractive apt-get install -y npm
fi
if [[ ! -x /usr/local/bin/motlie-vfs-guest ]]; then
  export PATH="/home/$LOGIN_USER/.cargo/bin:\$PATH"
  if ! command -v cargo >/dev/null 2>&1 || ! rustc -vV >/dev/null 2>&1; then
    rm -rf /home/$LOGIN_USER/.cargo /home/$LOGIN_USER/.rustup
    curl https://sh.rustup.rs -sSf | sh -s -- -y
    export PATH="/home/$LOGIN_USER/.cargo/bin:\$PATH"
  fi
  printf '${CONTROL_PASSWORD}\n' | sudo -S rm -rf /var/lib/motlie/src /var/lib/motlie/target
  printf '${CONTROL_PASSWORD}\n' | sudo -S mkdir -p /var/lib/motlie/src
  printf '${CONTROL_PASSWORD}\n' | sudo -S tar -xzf /tmp/motlie-src.tar.gz -C /var/lib/motlie/src
  printf '${CONTROL_PASSWORD}\n' | sudo -S chown -R $LOGIN_USER:$LOGIN_USER /var/lib/motlie
  cargo build --manifest-path /var/lib/motlie/src/libs/vfs/Cargo.toml --release --features vsock,client --bin motlie-vfs-guest-v1_1 --target-dir /var/lib/motlie/src/target
  printf '${CONTROL_PASSWORD}\n' | sudo -S install -D -m 0755 /var/lib/motlie/src/target/release/motlie-vfs-guest-v1_1 /usr/local/bin/motlie-vfs-guest
fi
if ! command -v codex >/dev/null 2>&1; then
  printf '${CONTROL_PASSWORD}\n' | sudo -S npm install -g @openai/codex
fi
if ! command -v claude >/dev/null 2>&1; then
  printf '${CONTROL_PASSWORD}\n' | sudo -S npm install -g @anthropic-ai/claude-code
fi
NPM_GLOBAL_PREFIX="$(npm prefix -g 2>/dev/null || true)"
if [[ -n "\$NPM_GLOBAL_PREFIX" ]]; then
  if [[ -x "\$NPM_GLOBAL_PREFIX/bin/codex" && ! -x /usr/local/bin/codex ]]; then
    printf '${CONTROL_PASSWORD}\n' | sudo -S ln -s "\$NPM_GLOBAL_PREFIX/bin/codex" /usr/local/bin/codex
  fi
  if [[ -x "\$NPM_GLOBAL_PREFIX/bin/claude" && ! -x /usr/local/bin/claude ]]; then
    printf '${CONTROL_PASSWORD}\n' | sudo -S ln -s "\$NPM_GLOBAL_PREFIX/bin/claude" /usr/local/bin/claude
  fi
fi
if ! command -v codex >/dev/null 2>&1; then
  cat <<'CODEXWRAP' >/tmp/codex
#!/bin/sh
exec npm exec --yes @openai/codex -- "$@"
CODEXWRAP
  chmod 0755 /tmp/codex
  printf '${CONTROL_PASSWORD}\n' | sudo -S install -D -m 0755 /tmp/codex /usr/local/bin/codex
fi
if ! command -v claude >/dev/null 2>&1; then
  cat <<'CLAUDEWRAP' >/tmp/claude
#!/bin/sh
exec npm exec --yes @anthropic-ai/claude-code -- "$@"
CLAUDEWRAP
  chmod 0755 /tmp/claude
  printf '${CONTROL_PASSWORD}\n' | sudo -S install -D -m 0755 /tmp/claude /usr/local/bin/claude
fi
printf '${CONTROL_PASSWORD}\n' | sudo -S install -D -m 0644 /tmp/mounts.${GUEST_NAME}.yaml /etc/motlie-vfs/mounts.yaml
printf '${CONTROL_PASSWORD}\n' | sudo -S install -D -m 0644 /tmp/motlie-vfs-guest.service /etc/systemd/system/motlie-vfs-guest.service
printf '${CONTROL_PASSWORD}\n' | sudo -S install -D -m 0755 /tmp/motlie-agent-state-setup /usr/local/bin/motlie-agent-state-setup
printf '${CONTROL_PASSWORD}\n' | sudo -S install -D -m 0644 /tmp/motlie-agent-state.service /etc/systemd/system/motlie-agent-state.service
printf '${CONTROL_PASSWORD}\n' | sudo -S systemctl daemon-reload
printf '${CONTROL_PASSWORD}\n' | sudo -S systemctl unmask motlie-vfs-guest.service || true
printf '${CONTROL_PASSWORD}\n' | sudo -S systemctl unmask motlie-agent-state.service || true
printf '${CONTROL_PASSWORD}\n' | sudo -S systemctl enable motlie-vfs-guest.service
printf '${CONTROL_PASSWORD}\n' | sudo -S systemctl enable motlie-agent-state.service || true
printf '${CONTROL_PASSWORD}\n' | sudo -S systemctl restart motlie-vfs-guest.service
printf '${CONTROL_PASSWORD}\n' | sudo -S systemctl restart motlie-agent-state.service || true
printf '${CONTROL_PASSWORD}\n' | sudo -S /usr/local/bin/motlie-agent-state-setup || true
EOF

POST_PROVISION_REMOTE_JSON="/tmp/motlie-vfs-post-provision.json"
POST_PROVISION_OK=0
POST_PROVISION_CHECK=""
for _ in $(seq 1 10); do
  guest_bash "$IP_ADDR" <<EOF
python3 - <<'PY2'
import json
from pathlib import Path

payload = {
    "mounts_yaml": Path("/etc/motlie-vfs/mounts.yaml").exists(),
    "guest_bin": Path("/usr/local/bin/motlie-vfs-guest").exists(),
    "service_unit": Path("/etc/systemd/system/motlie-vfs-guest.service").exists(),
    "agent_state_service_unit": Path("/etc/systemd/system/motlie-agent-state.service").exists(),
}
Path("$POST_PROVISION_REMOTE_JSON").write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
PY2
EOF
  POST_PROVISION_LOCAL_JSON="$(mktemp)"
  guest_fetch "$POST_PROVISION_REMOTE_JSON" "$POST_PROVISION_LOCAL_JSON" "$IP_ADDR"
  POST_PROVISION_CHECK="$(cat "$POST_PROVISION_LOCAL_JSON")"
  rm -f "$POST_PROVISION_LOCAL_JSON"
  if [[ -n "$POST_PROVISION_CHECK" ]]; then
    POST_PROVISION_STATE="$(printf '%s' "$POST_PROVISION_CHECK" | python3 -c 'import json,sys; payload=json.loads(sys.stdin.read()); print("ok" if all(payload.values()) else "bad")')"
    if [[ "$POST_PROVISION_STATE" == "ok" ]]; then
      POST_PROVISION_OK=1
      break
    fi
  fi
  sleep 1
done
if [[ "$POST_PROVISION_OK" -ne 1 ]]; then
  echo "guest post-provision verification failed" >&2
  printf '%s\n' "$POST_PROVISION_CHECK" >&2
  exit 1
fi

VALIDATION_OK=0
VALIDATION_REMOTE_JSON="/tmp/motlie-vfs-validation.json"
for _ in $(seq 1 "$TIMEOUT_SECONDS"); do
  if ! kill -0 "$RUNNER_PID" >/dev/null 2>&1; then
    echo "vz-vsock-runner exited early; log follows:" >&2
    print_failure_context
    exit 1
  fi
  guest_bash "$IP_ADDR" <<EOF
python3 - <<'PY2'
import json
import os
import subprocess
from pathlib import Path

login_user = '$LOGIN_USER'
expected_readme = '$EXPECTED_WORKSPACE_README'
expected_agent_state_readme = '$EXPECTED_AGENT_STATE_README'
expected_env = '$EXPECTED_ENV_LINE'
token = '$VALIDATION_TOKEN'

workspace_mount = Path('/workspace')
home_mount = Path(f'/home/{login_user}')
agent_state_mount = Path('/agent-state')
workspace_readme_path = workspace_mount / 'README.md'
agent_state_readme_path = agent_state_mount / 'README.md'
env_path = home_mount / '.env'
auth_path = home_mount / '.ssh' / 'authorized_keys'
codex_link = home_mount / '.codex'
claude_link = home_mount / '.claude'
claude_code_link = home_mount / '.config' / 'claude-code'

result = {
    'token': token,
    'guest_user': login_user,
    'workspace_mountpoint': workspace_mount.is_mount(),
    'home_mountpoint': home_mount.is_mount(),
    'agent_state_mountpoint': agent_state_mount.is_mount(),
}

required_paths_exist = (
    workspace_readme_path.exists()
    and agent_state_readme_path.exists()
    and env_path.exists()
    and auth_path.exists()
    and codex_link.exists()
    and claude_link.exists()
    and claude_code_link.exists()
)

if not (result['workspace_mountpoint'] and result['home_mountpoint'] and result['agent_state_mountpoint'] and required_paths_exist):
    Path('$VALIDATION_REMOTE_JSON').write_text('', encoding='utf-8')
    raise SystemExit(0)

workspace_readme = workspace_readme_path.read_text(encoding='utf-8').strip()
agent_state_readme = agent_state_readme_path.read_text(encoding='utf-8').strip()
env_line = env_path.read_text(encoding='utf-8').strip()
authorized_keys = auth_path.read_text(encoding='utf-8').strip()

def link_target(path: Path) -> str:
    return os.readlink(path) if path.is_symlink() else ''

def command_ok(cmd):
    env = os.environ.copy()
    env["PATH"] = "/usr/local/bin:/usr/bin:/bin:" + env.get("PATH", "")
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30, env=env)
    return completed.returncode == 0, completed.stdout.strip(), completed.stderr.strip()

def first_executable(candidates):
    for candidate in candidates:
        if Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    return ''

dns_ok, dns_stdout, dns_stderr = command_ok(['getent', 'ahostsv4', 'example.com'])
curl_ok, curl_stdout, curl_stderr = command_ok(['curl', '-fsS', '--max-time', '20', 'https://example.com'])
codex_path = first_executable(['/usr/local/bin/codex', '/usr/bin/codex', '/bin/codex'])
claude_path = first_executable(['/usr/local/bin/claude', '/usr/bin/claude', '/bin/claude'])
codex_shell_ok, codex_shell_out, codex_shell_err = command_ok(['env', '-u', 'SSH_CONNECTION', 'TMUX=1', 'bash', '--noprofile', '--norc', '-lc', 'export PATH=/usr/local/bin:/usr/bin:/bin:$PATH; command -v codex'])
claude_shell_ok, claude_shell_out, claude_shell_err = command_ok(['env', '-u', 'SSH_CONNECTION', 'TMUX=1', 'bash', '--noprofile', '--norc', '-lc', 'export PATH=/usr/local/bin:/usr/bin:/bin:$PATH; command -v claude'])
codex_ok = bool(codex_path)
claude_ok = bool(claude_path)

result.update({
    'workspace_readme': workspace_readme,
    'expected_workspace_readme': expected_readme,
    'agent_state_readme': agent_state_readme,
    'expected_agent_state_readme': expected_agent_state_readme,
    'env_line': env_line,
    'expected_env_line': expected_env,
    'authorized_keys_present': bool(authorized_keys),
    'codex_link': link_target(codex_link),
    'claude_link': link_target(claude_link),
    'claude_code_link': link_target(claude_code_link),
    'codex_cli_path': codex_path,
    'codex_cli_present': codex_ok,
    'codex_shell_lookup_ok': codex_shell_ok,
    'codex_shell_lookup': codex_shell_out,
    'codex_shell_lookup_error': codex_shell_err,
    'claude_cli_path': claude_path,
    'claude_cli_present': claude_ok,
    'claude_shell_lookup_ok': claude_shell_ok,
    'claude_shell_lookup': claude_shell_out,
    'claude_shell_lookup_error': claude_shell_err,
    'dns_lookup_ok': dns_ok,
    'dns_lookup_sample': dns_stdout.splitlines()[0] if dns_stdout else '',
    'dns_lookup_error': dns_stderr,
    'internet_ok': curl_ok,
    'internet_error': curl_stderr,
    'internet_sample': curl_stdout[:120],
})
result['status'] = 'ok' if (
    workspace_readme == expected_readme
    and agent_state_readme == expected_agent_state_readme
    and env_line == expected_env
    and bool(authorized_keys)
    and link_target(codex_link) == '/agent-state/codex'
    and link_target(claude_link) == '/agent-state/claude'
    and link_target(claude_code_link) == '/agent-state/claude-code'
    and codex_ok
    and claude_ok
    and dns_ok
    and curl_ok
) else 'mismatch'
Path('$VALIDATION_REMOTE_JSON').write_text(json.dumps(result, sort_keys=True) + '\n', encoding='utf-8')
PY2
EOF
  VALIDATION_LOCAL_JSON="$(mktemp)"
  guest_fetch "$VALIDATION_REMOTE_JSON" "$VALIDATION_LOCAL_JSON" "$IP_ADDR"
  VALIDATION_JSON="$(cat "$VALIDATION_LOCAL_JSON")"
  rm -f "$VALIDATION_LOCAL_JSON"
  if [[ -n "${VALIDATION_JSON//[[:space:]]/}" ]]; then
    STATUS="$(printf '%s' "$VALIDATION_JSON" | python3 -c 'import json,sys; payload=json.loads(sys.stdin.read()); print(payload["status"])')"
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
  print_failure_context
  exit 1
fi

READY_EPOCH="$EPOCHREALTIME"
BOOT_SECONDS="$(awk -v start="$START_EPOCH" -v ready="$READY_EPOCH" 'BEGIN { printf "%.3f", ready - start }')"

python3 - "$RESULT_JSON" "$RUN_VM_NAME" "$BOOT_SECONDS" "$SOCKET_PATH" "$RUNNER_PID" "$IP_ADDR" "$ARTIFACTS_DIR/${GUEST_NAME}-validation.json" "$KEEP_RUNNING" <<'PY'
import json
import sys

path, vm_name, boot_seconds, socket_path, runner_pid, ip_addr, validation_json_path, keep_running = sys.argv[1:]
with open(validation_json_path, "r", encoding="utf-8") as fh:
    validation_payload = json.load(fh)
keep_running_bool = keep_running == "1"
with open(path, "w", encoding="utf-8") as fh:
    json.dump(
        {
            "backend": "vz-vsock-runner",
            "vm_name": vm_name,
            "boot_to_validation_seconds": float(boot_seconds),
            "unix_socket_path": socket_path,
            "runner_pid": int(runner_pid) if keep_running_bool else None,
            "kept_running": keep_running_bool,
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

echo "--- result written ---"
cat "$RESULT_JSON"
if [[ "$KEEP_RUNNING" -eq 1 ]]; then
  echo "=== guest left running through Apple virtio-socket ==="
else
  echo "=== guest validated and scheduled for cleanup ==="
fi
