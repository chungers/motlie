#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
ARTIFACTS_DIR="$SCRIPT_DIR/artifacts"
BASE_VM_NAME="${MOTLIE_VZ_BASE_VM_NAME:-motlie-v1-35-base-iter}"
BASE_VM_DIR_OVERRIDE="${MOTLIE_VZ_BASE_VM_DIR:-}"
TIMEOUT_SECONDS="${MOTLIE_VZ_TIMEOUT_SECONDS:-900}"
SERVICE_FILE="$SCRIPT_DIR/motlie-vfs-guest.service"
AGENT_STATE_SETUP_FILE="$SCRIPT_DIR/motlie-agent-state-setup.sh"
AGENT_STATE_UNIT_FILE="$SCRIPT_DIR/motlie-agent-state.service"
SSH_BRIDGE_LOOP_FILE="$SCRIPT_DIR/motlie-vmm-vsock-ssh-loop.sh"
SSH_BRIDGE_UNIT_FILE="$SCRIPT_DIR/motlie-vmm-vsock-ssh.service"
RUNNER_BUILD_SCRIPT="$SCRIPT_DIR/build-vz-runner.sh"
RUNNER_BIN_OVERRIDE="${MOTLIE_VZ_RUNNER_BIN:-}"
SKIP_RUNNER_BUILD="${MOTLIE_VZ_SKIP_RUNNER_BUILD:-0}"
EGRESS_HELPER_BIN_OVERRIDE="${MOTLIE_VZ_EGRESS_HELPER_BIN:-}"
SOURCE_TARBALL="$ARTIFACTS_DIR/motlie-src.tar.gz"
KEEP_RUNNING="${MOTLIE_VZ_KEEP_RUNNING:-0}"
REUSE_VM="${MOTLIE_VZ_REUSE_VM:-0}"
NATIVE_SOURCE_VM_DIR="${MOTLIE_VZ_NATIVE_SOURCE_VM_DIR:-$ARTIFACTS_DIR/source-base.vm}"
BASE_SOURCE_DIR=""
RUN_VM_DIR=""
EGRESS_HELPER_PID_FILE=""
EGRESS_SOCKET_PATH=""
EGRESS_LOG=""
NET_MAC=""
CONTROL_HOST="127.0.0.1"
CONTROL_PORT=""
GUEST_IPV4="10.0.2.15"
EGRESS_HELPER_LOG_FRAMES="${MOTLIE_VZ_LOG_FRAMES:-0}"

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
require_cmd cargo
require_cmd hdiutil

if [[ -n "$BASE_VM_DIR_OVERRIDE" ]]; then
  BASE_SOURCE_DIR="$BASE_VM_DIR_OVERRIDE"
elif [[ -f "$NATIVE_SOURCE_VM_DIR/disk.img" && -f "$NATIVE_SOURCE_VM_DIR/nvram.bin" ]]; then
  BASE_SOURCE_DIR="$NATIVE_SOURCE_VM_DIR"
else
  cat >&2 <<EOF
native base artifacts are required for v1.35 guest launches

Set:
  MOTLIE_VZ_BASE_VM_DIR

Or build/populate the default native cache first:
  $NATIVE_SOURCE_VM_DIR
EOF
  exit 1
fi

GUEST_NAME="alice"
RUN_VM_NAME=""
REUSED_VM=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --guest) GUEST_NAME="$2"; shift 2 ;;
    --vm-name) RUN_VM_NAME="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

case "$GUEST_NAME" in
  alice)
    RUN_VM_NAME="${RUN_VM_NAME:-motlie-v1-35-alice-iter}"
    MOUNTS_FILE="$SCRIPT_DIR/mounts.alice.yaml"
    LOGIN_USER="alice"
    UID_NUM=1000
    GID_NUM=1000
    SOCKET_PATH="/tmp/motlie-vmm-alice.vsock_5000"
    SSH_VSOCK_SOCKET="/tmp/motlie-vmm-alice.vsock_2222"
    HOST_HOME_DIR="/tmp/motlie-vmm-demo/alice-home"
    HOST_AGENT_STATE_DIR="/tmp/motlie-vmm-demo/alice-agent-state"
    HOST_WORKSPACE_DIR="/tmp/motlie-vmm-demo/alice-workspace"
    EXPECTED_WORKSPACE_README="Alice workspace mounted from the host."
    EXPECTED_AGENT_STATE_README="Dedicated read-write agent-state layer for Codex and Claude lives here."
    EXPECTED_ENV_LINE="ALICE_API_KEY=demo-alice"
    GUEST_HOSTNAME="motlie-alice"
    NET_MAC="02:4d:6f:74:61:11"
    CONTROL_PORT="${MOTLIE_VZ_ALICE_SSH_PORT:-2226}"
    ;;
  bob)
    RUN_VM_NAME="${RUN_VM_NAME:-motlie-v1-35-bob-iter}"
    MOUNTS_FILE="$SCRIPT_DIR/mounts.bob.yaml"
    LOGIN_USER="bob"
    UID_NUM=1001
    GID_NUM=1001
    SOCKET_PATH="/tmp/motlie-vmm-bob.vsock_5000"
    SSH_VSOCK_SOCKET="/tmp/motlie-vmm-bob.vsock_2222"
    HOST_HOME_DIR="/tmp/motlie-vmm-demo/bob-home"
    HOST_AGENT_STATE_DIR="/tmp/motlie-vmm-demo/bob-agent-state"
    HOST_WORKSPACE_DIR="/tmp/motlie-vmm-demo/bob-workspace"
    EXPECTED_WORKSPACE_README="Bob workspace mounted from the host."
    EXPECTED_AGENT_STATE_README="Dedicated read-write agent-state layer for Codex and Claude lives here."
    EXPECTED_ENV_LINE="BOB_API_KEY=demo-bob"
    GUEST_HOSTNAME="motlie-bob"
    NET_MAC="02:4d:6f:74:62:15"
    CONTROL_PORT="${MOTLIE_VZ_BOB_SSH_PORT:-2227}"
    ;;
  *)
    echo "guest must be alice or bob" >&2
    exit 1
    ;;
esac

BOOTSTRAP_USER="${MOTLIE_VZ_BOOTSTRAP_USER:-admin}"
BOOTSTRAP_PASSWORD="${MOTLIE_VZ_BOOTSTRAP_PASS:-admin}"
CONTROL_USER="$BOOTSTRAP_USER"
CONTROL_PASSWORD="$BOOTSTRAP_PASSWORD"
SSH_PRINCIPAL="${MOTLIE_VZ_SSH_PRINCIPAL:-$LOGIN_USER}"
SSH_CA_PUBKEY="${MOTLIE_VZ_SSH_CA_PUBKEY:-}"

if [[ -z "$SSH_CA_PUBKEY" ]]; then
  echo "MOTLIE_VZ_SSH_CA_PUBKEY must be set for v1.35 launches" >&2
  exit 1
fi

RUN_LOG="$ARTIFACTS_DIR/${GUEST_NAME}-run.log"
RESULT_JSON="$ARTIFACTS_DIR/${GUEST_NAME}-launch-result.json"
SERIAL_LOG="$ARTIFACTS_DIR/${GUEST_NAME}-serial.log"
SEED_DIR="$ARTIFACTS_DIR/${GUEST_NAME}-seed"
SEED_IMAGE="$ARTIFACTS_DIR/${GUEST_NAME}-seed.dmg"
VALIDATION_TOKEN="$RUN_VM_NAME"
RUNNER_PID_FILE="$ARTIFACTS_DIR/${GUEST_NAME}-runner.pid"
GUEST_IP_FILE="$ARTIFACTS_DIR/${GUEST_NAME}-ip.txt"
EGRESS_HELPER_PID_FILE="$ARTIFACTS_DIR/${GUEST_NAME}-egress-helper.pid"
EGRESS_SOCKET_PATH="/tmp/motlie-vmm-${GUEST_NAME}.egress.sock"
EGRESS_LOG="$ARTIFACTS_DIR/${GUEST_NAME}-egress.log"

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
    if [[ -f "$EGRESS_HELPER_PID_FILE" ]]; then
      kill "$(cat "$EGRESS_HELPER_PID_FILE")" >/dev/null 2>&1 || true
      rm -f "$EGRESS_HELPER_PID_FILE"
    fi
    rm -f "$EGRESS_SOCKET_PATH"
    if [[ "$REUSED_VM" -eq 0 ]] && [[ -n "$RUN_VM_DIR" && -d "$RUN_VM_DIR" ]]; then
      rm -rf "$RUN_VM_DIR"
    fi
    rm -f "$SEED_IMAGE"
    rm -rf "$SEED_DIR"
  fi
}

trap cleanup EXIT

kill_stale_runners() {
  local stale_pids=()
  local stale_pid=""
  while read -r stale_pid; do
    [[ -n "$stale_pid" ]] || continue
    stale_pids+=("$stale_pid")
  done < <(ps -Ao pid=,command= -ww | awk -v mac="$NET_MAC" -v sock="$SOCKET_PATH" '
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
    done < <(ps -Ao pid=,command= -ww | awk -v mac="$NET_MAC" -v sock="$SOCKET_PATH" '
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
  ps -Ao pid,etime,command | awk -v mac="$NET_MAC" -v sock="$SOCKET_PATH" '
    /vz-vsock-runner/ && index($0, mac) && index($0, sock) { print }
  ' >&2 || true
  exit 1
}

kill_stale_egress_helpers() {
  local stale_pids=()
  local stale_pid=""

  if [[ -f "$EGRESS_HELPER_PID_FILE" ]]; then
    stale_pid="$(cat "$EGRESS_HELPER_PID_FILE" 2>/dev/null || true)"
    if [[ -n "$stale_pid" ]]; then
      stale_pids+=("$stale_pid")
    fi
  fi

  while read -r stale_pid; do
    [[ -n "$stale_pid" ]] || continue
    stale_pids+=("$stale_pid")
  done < <(lsof -tiTCP:"$CONTROL_PORT" -sTCP:LISTEN 2>/dev/null || true)

  if [[ "${#stale_pids[@]}" -eq 0 ]]; then
    rm -f "$EGRESS_SOCKET_PATH" "$EGRESS_HELPER_PID_FILE"
    return 0
  fi

  local unique_pids=()
  local seen=""
  for stale_pid in "${stale_pids[@]}"; do
    [[ -n "$stale_pid" ]] || continue
    if [[ " $seen " == *" $stale_pid "* ]]; then
      continue
    fi
    seen+=" $stale_pid"
    unique_pids+=("$stale_pid")
  done

  printf '%s\n' "--- terminating stale egress helpers for $GUEST_NAME on tcp:${CONTROL_PORT} ---"
  kill "${unique_pids[@]}" >/dev/null 2>&1 || true

  local attempts=0
  while [[ $attempts -lt 20 ]]; do
    local survivors=()
    while read -r stale_pid; do
      [[ -n "$stale_pid" ]] || continue
      survivors+=("$stale_pid")
    done < <(lsof -tiTCP:"$CONTROL_PORT" -sTCP:LISTEN 2>/dev/null || true)
    if [[ "${#survivors[@]}" -eq 0 ]]; then
      rm -f "$EGRESS_SOCKET_PATH" "$EGRESS_HELPER_PID_FILE"
      return 0
    fi
    sleep 0.5
    attempts=$(( attempts + 1 ))
    if [[ $attempts -eq 10 ]]; then
      kill -9 "${survivors[@]}" >/dev/null 2>&1 || true
    fi
  done

  echo "failed to terminate stale egress helper on tcp:${CONTROL_PORT}" >&2
  lsof -iTCP:"$CONTROL_PORT" -n -P >&2 || true
  exit 1
}

start_egress_helper() {
  local helper_bin=""
  if [[ -n "$EGRESS_HELPER_BIN_OVERRIDE" ]]; then
    helper_bin="$EGRESS_HELPER_BIN_OVERRIDE"
  else
    cargo build -p motlie-vnet --example vz_egress_helper_v1_25 >/dev/null
    helper_bin="$REPO_ROOT/target/debug/examples/vz_egress_helper_v1_25"
  fi
  kill_stale_egress_helpers
  rm -f "$EGRESS_SOCKET_PATH" "$EGRESS_HELPER_PID_FILE"
  : > "$EGRESS_LOG"
  "$helper_bin" \
    --socket-path "$EGRESS_SOCKET_PATH" \
    --host-forward-tcp "127.0.0.1:${CONTROL_PORT}:22" \
    $( [[ "$EGRESS_HELPER_LOG_FRAMES" == "1" ]] && print -- "--log-frames" ) \
    >"$EGRESS_LOG" 2>&1 &
  EGRESS_HELPER_PID="$!"
  echo "$EGRESS_HELPER_PID" > "$EGRESS_HELPER_PID_FILE"
  local attempts=0
  while [[ $attempts -lt 40 ]]; do
    if [[ -S "$EGRESS_SOCKET_PATH" ]]; then
      return 0
    fi
    if ! kill -0 "$EGRESS_HELPER_PID" >/dev/null 2>&1; then
      echo "egress helper exited early" >&2
      cat "$EGRESS_LOG" >&2 || true
      exit 1
    fi
    sleep 0.25
    attempts=$(( attempts + 1 ))
  done
  echo "timed out waiting for egress helper socket" >&2
  cat "$EGRESS_LOG" >&2 || true
  exit 1
}

if [[ -f "$RUNNER_PID_FILE" ]]; then
  kill "$(cat "$RUNNER_PID_FILE")" >/dev/null 2>&1 || true
  rm -f "$RUNNER_PID_FILE"
fi
kill_stale_runners
kill_stale_egress_helpers

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
  "$ARTIFACTS_DIR/${GUEST_NAME}-validation.json" "$EGRESS_SOCKET_PATH" "$EGRESS_LOG" "$EGRESS_HELPER_PID_FILE" "$SEED_IMAGE"
rm -rf "$SEED_DIR"

guest_copy() {
  local src="$1"
  local dst="$2"
  local size_bytes
  size_bytes="$(wc -c < "$src" | tr -d '[:space:]')"
  if [[ "${size_bytes:-0}" -gt 262144 ]]; then
    local chunk_dir
    local dst_dir
    chunk_dir="$(mktemp -d)"
    dst_dir="${dst:h}"
    split -b 256k -a 4 "$src" "$chunk_dir/chunk."
    expect <<EOF
set timeout -1
spawn ssh -p ${CONTROL_PORT} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ${CONTROL_USER}@${CONTROL_HOST} "mkdir -p \"$dst_dir\" && : > \"$dst\""
expect {
  "password:" {
    send "${CONTROL_PASSWORD}\r"
    exp_continue
  }
  eof {}
  timeout {
    puts stderr "ssh timed out: prepare remote file $dst"
    exit 124
  }
}
catch wait result
set exit_code [lindex \$result 3]
if {\$exit_code != 0} {
  exit \$exit_code
}
EOF
    local chunk=""
    for chunk in "$chunk_dir"/chunk.*; do
      local pump_script
      pump_script="$(mktemp)"
      cat >"$pump_script" <<EOF2
#!/bin/sh
cat "$chunk" | ssh -p ${CONTROL_PORT} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ${CONTROL_USER}@${CONTROL_HOST} "cat >> \"$dst\""
EOF2
      chmod +x "$pump_script"
      expect <<EOF
set timeout -1
spawn "$pump_script"
expect {
  "password:" {
    send "${CONTROL_PASSWORD}\r"
    exp_continue
  }
  eof {}
  timeout {
    puts stderr "ssh stream timed out: $chunk -> $dst"
    exit 124
  }
}
catch wait result
set exit_code [lindex \$result 3]
if {\$exit_code != 0} {
  exit \$exit_code
}
EOF
      rm -f "$pump_script"
    done
    rm -rf "$chunk_dir"
    return 0
  fi
  expect <<EOF
set timeout -1
spawn scp -P ${CONTROL_PORT} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$src" ${CONTROL_USER}@${CONTROL_HOST}:$dst
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
  expect <<EOF
set timeout -1
spawn scp -P ${CONTROL_PORT} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ${CONTROL_USER}@${CONTROL_HOST}:$src "$dst"
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
  local remote_script
  remote_script="$(mktemp)"
  local remote_path="/home/${CONTROL_USER}/.motlie-vfs-remote.sh"
  cat >"$remote_script"
  expect <<EOF
set timeout -1
spawn scp -P ${CONTROL_PORT} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$remote_script" ${CONTROL_USER}@${CONTROL_HOST}:${remote_path}
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
set output ""
spawn ssh -tt -p ${CONTROL_PORT} -n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ${CONTROL_USER}@${CONTROL_HOST} "chmod 0700 ${remote_path} && bash -euo pipefail ${remote_path} </dev/null"
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

ssh_ready_for_password_prompt() {
  expect <<EOF
set timeout 5
log_user 0
spawn ssh -p ${CONTROL_PORT} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=1 -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 ${CONTROL_USER}@${CONTROL_HOST} true
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

wait_for_guest_ssh() {
  local attempts=0
  local max_attempts=$(( TIMEOUT_SECONDS * 2 ))
  while [[ $attempts -lt $max_attempts ]]; do
    if ssh_ready_for_password_prompt >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$RUNNER_PID" >/dev/null 2>&1; then
      echo "vz-vsock-runner exited early; log follows:" >&2
      print_failure_context
      exit 1
    fi
    sleep 0.5
    attempts=$(( attempts + 1 ))
  done
  return 1
}

guest_capture() {
  local remote_script
  remote_script="$(mktemp)"
  local remote_path="/home/${CONTROL_USER}/.motlie-vfs-capture.sh"
  cat >"$remote_script"
  expect <<EOF
set timeout -1
spawn scp -P ${CONTROL_PORT} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$remote_script" ${CONTROL_USER}@${CONTROL_HOST}:${remote_path}
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
spawn ssh -tt -p ${CONTROL_PORT} -n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ${CONTROL_USER}@${CONTROL_HOST} "chmod 0700 ${remote_path} && bash -euo pipefail ${remote_path} </dev/null"
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

echo "--- rendering guest seed disk ---"
mkdir -p "$SEED_DIR"
cp "$SOURCE_TARBALL" "$SEED_DIR/motlie-src.tar.gz"
cp "$MOUNTS_FILE" "$SEED_DIR/mounts.${GUEST_NAME}.yaml"
cp "$SERVICE_FILE" "$SEED_DIR/motlie-vfs-guest.service"
cp "$AGENT_STATE_SETUP_FILE" "$SEED_DIR/motlie-agent-state-setup"
cp "$AGENT_STATE_UNIT_FILE" "$SEED_DIR/motlie-agent-state.service"
cp "$SSH_BRIDGE_LOOP_FILE" "$SEED_DIR/motlie-vmm-vsock-ssh-loop"
cp "$SSH_BRIDGE_UNIT_FILE" "$SEED_DIR/motlie-vmm-vsock-ssh.service"
hdiutil create -quiet -fs FAT32 -volname MOTLIESEED -srcfolder "$SEED_DIR" -ov -format UDRW "$SEED_IMAGE"

if [[ -n "$RUNNER_BIN_OVERRIDE" ]]; then
  RUNNER_BIN="$RUNNER_BIN_OVERRIDE"
elif [[ "$SKIP_RUNNER_BUILD" == "1" ]]; then
  RUNNER_BIN="$SCRIPT_DIR/artifacts/build/vz-vsock-runner"
else
  echo "--- building Vz virtio-socket helper ---"
  RUNNER_BIN="$($RUNNER_BUILD_SCRIPT)"
fi
chmod +x "$RUNNER_BIN"
start_egress_helper

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
  --seed-disk "$SEED_IMAGE"
  --vsock-forward "5000:$SOCKET_PATH"
  --vsock-forward "2222:$SSH_VSOCK_SOCKET"
  --net-backend-socket "$EGRESS_SOCKET_PATH"
  --net-mac "$NET_MAC"
  --memory-mib 4096
  --cpu-count 4
)
"$RUNNER_BIN" \
  "${RUNNER_ARGS[@]}" \
  >"$RUN_LOG" 2>&1 &
RUNNER_PID="$!"
echo "$RUNNER_PID" > "$RUNNER_PID_FILE"

IP_ADDR="$GUEST_IPV4"
echo "$IP_ADDR" > "$GUEST_IP_FILE"

echo "--- waiting for guest SSH ---"
wait_for_guest_ssh || {
  echo "timed out waiting for guest SSH at ${CONTROL_HOST}:${CONTROL_PORT}" >&2
  print_failure_context
  exit 1
}

echo "--- provisioning guest over native Vz userspace egress ---"
guest_bash <<'EOF'
set -x
if ! id -u motlie-build >/dev/null 2>&1; then
  printf 'admin\n' | sudo -S groupadd -f -g 2002 motlie-build
  printf 'admin\n' | sudo -S useradd -m -u 2002 -g 2002 -s /bin/bash motlie-build
fi
printf 'admin\n' | sudo -S bash -c "printf '%s:%s\n' 'motlie-build' 'admin' | chpasswd"
printf 'admin\n' | sudo -S usermod -U motlie-build || true
printf 'admin\n' | sudo -S usermod -aG sudo motlie-build || true
printf 'admin\n' | sudo -S sh -c "printf '%s\n' 'motlie-build ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/90-motlie-build"
printf 'admin\n' | sudo -S chown root:root /etc/sudoers.d/90-motlie-build
printf 'admin\n' | sudo -S chmod 0440 /etc/sudoers.d/90-motlie-build
EOF
CONTROL_USER="motlie-build"
CONTROL_PASSWORD="admin"

NEED_SOURCE_UPLOAD=1
if guest_bash <<'EOF'
set -x
if [[ -s /usr/local/bin/motlie-vfs-guest && -x /usr/local/bin/motlie-vfs-guest ]]; then
  exit 0
fi
exit 1
EOF
then
  NEED_SOURCE_UPLOAD=0
fi

if [[ "$NEED_SOURCE_UPLOAD" -eq 1 ]]; then
  :
fi

guest_bash <<EOF
set -x
remap_conflicting_identity() {
  user_name="\$1"
  target_uid="\$2"
  target_gid="\$3"
  remap_uid="\$4"
  remap_gid="\$5"

  if ! id -u "\$user_name" >/dev/null 2>&1; then
    return 0
  fi

  current_uid="\$(id -u "\$user_name" 2>/dev/null || true)"
  current_gid="\$(getent group "\$user_name" | cut -d: -f3 || true)"

  if [[ "\$current_gid" == "\$target_gid" ]]; then
    sudo groupmod -g "\$remap_gid" "\$user_name"
    current_gid="\$remap_gid"
  fi
  if [[ "\$current_uid" == "\$target_uid" ]]; then
    while read -r victim_pid; do
      [[ -n "\$victim_pid" ]] || continue
      sudo kill -KILL "\$victim_pid" >/dev/null 2>&1 || true
    done < <(pgrep -u "\$user_name" || true)
    sleep 1
    sudo usermod -u "\$remap_uid" -g "\$current_gid" "\$user_name"
  fi
}

remap_conflicting_identity admin $UID_NUM $GID_NUM 2000 2000
remap_conflicting_identity ubuntu $UID_NUM $GID_NUM 2001 2001

SEED_MOUNT="/mnt/motlie-seed"
seed_dev="\$(blkid -L MOTLIESEED 2>/dev/null || true)"
if [[ -z "\$seed_dev" && -e /dev/disk/by-label/MOTLIESEED ]]; then
  seed_dev="/dev/disk/by-label/MOTLIESEED"
fi
if [[ -z "\$seed_dev" ]]; then
  echo "failed to locate MOTLIESEED device" >&2
  exit 1
fi
printf '${CONTROL_PASSWORD}\n' | sudo -S mkdir -p "\$SEED_MOUNT"
printf '${CONTROL_PASSWORD}\n' | sudo -S umount -lf "\$SEED_MOUNT" >/dev/null 2>&1 || true
printf '${CONTROL_PASSWORD}\n' | sudo -S mount -o ro "\$seed_dev" "\$SEED_MOUNT"
if [[ "$NEED_SOURCE_UPLOAD" -eq 1 ]]; then
  printf '${CONTROL_PASSWORD}\n' | sudo -S cp "\$SEED_MOUNT/motlie-src.tar.gz" /tmp/motlie-src.tar.gz
fi
printf '${CONTROL_PASSWORD}\n' | sudo -S cp "\$SEED_MOUNT/mounts.${GUEST_NAME}.yaml" /tmp/mounts.${GUEST_NAME}.yaml
printf '${CONTROL_PASSWORD}\n' | sudo -S cp "\$SEED_MOUNT/motlie-vfs-guest.service" /tmp/motlie-vfs-guest.service
printf '${CONTROL_PASSWORD}\n' | sudo -S cp "\$SEED_MOUNT/motlie-agent-state-setup" /tmp/motlie-agent-state-setup
printf '${CONTROL_PASSWORD}\n' | sudo -S cp "\$SEED_MOUNT/motlie-agent-state.service" /tmp/motlie-agent-state.service
printf '${CONTROL_PASSWORD}\n' | sudo -S cp "\$SEED_MOUNT/motlie-vmm-vsock-ssh-loop" /tmp/motlie-vmm-vsock-ssh-loop
printf '${CONTROL_PASSWORD}\n' | sudo -S cp "\$SEED_MOUNT/motlie-vmm-vsock-ssh.service" /tmp/motlie-vmm-vsock-ssh.service
printf '${CONTROL_PASSWORD}\n' | sudo -S umount -lf "\$SEED_MOUNT" >/dev/null 2>&1 || true
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
typeset -a missing_pkgs=()
for cmd_pkg in \
  "cargo:cargo" \
  "rustc:rustc" \
  "cc:build-essential" \
  "pkg-config:pkg-config" \
  "curl:curl" \
  "tar:tar" \
  "gzip:gzip" \
  "npm:npm"
do
  required_cmd="\${cmd_pkg%%:*}"
  required_pkg="\${cmd_pkg#*:}"
  if ! command -v "\$required_cmd" >/dev/null 2>&1; then
    missing_pkgs+=("\$required_pkg")
  fi
done
if ! pkg-config --exists fuse3 >/dev/null 2>&1; then
  missing_pkgs+=("libfuse3-dev")
fi
if (( \${#missing_pkgs[@]} > 0 )); then
  unique_pkgs=()
  seen_pkgs=""
  for required_pkg in "\${missing_pkgs[@]}"; do
    if [[ " \$seen_pkgs " == *" \$required_pkg "* ]]; then
      continue
    fi
    seen_pkgs+=" \$required_pkg"
    unique_pkgs+=("\$required_pkg")
  done
  printf '${CONTROL_PASSWORD}\n' | sudo -S systemctl stop apt-daily.service apt-daily-upgrade.service unattended-upgrades.service >/dev/null 2>&1 || true
  printf '${CONTROL_PASSWORD}\n' | sudo -S systemctl kill apt-daily.service apt-daily-upgrade.service unattended-upgrades.service >/dev/null 2>&1 || true
  printf '${CONTROL_PASSWORD}\n' | sudo -S apt-get update
  printf '${CONTROL_PASSWORD}\n' | sudo -S DEBIAN_FRONTEND=noninteractive apt-get install -y "\${unique_pkgs[@]}"
fi
if [[ ! -s /usr/local/bin/motlie-vfs-guest || ! -x /usr/local/bin/motlie-vfs-guest ]]; then
  export PATH="/home/$LOGIN_USER/.cargo/bin:\$PATH"
  printf '${CONTROL_PASSWORD}\n' | sudo -S rm -rf /var/lib/motlie/src /var/lib/motlie/target
  printf '${CONTROL_PASSWORD}\n' | sudo -S mkdir -p /var/lib/motlie/src
  printf '${CONTROL_PASSWORD}\n' | sudo -S tar -xzf /tmp/motlie-src.tar.gz -C /var/lib/motlie/src
  printf '${CONTROL_PASSWORD}\n' | sudo -S chown -R $LOGIN_USER:$LOGIN_USER /var/lib/motlie
  if ! cargo metadata --manifest-path /var/lib/motlie/src/libs/vfs/Cargo.toml --format-version 1 >/dev/null 2>&1; then
    if command -v rustup >/dev/null 2>&1; then
      rustup toolchain install stable --profile minimal
      rustup default stable
    else
      curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable
    fi
    export PATH="/home/$LOGIN_USER/.cargo/bin:\$PATH"
  fi
  cargo build --manifest-path /var/lib/motlie/src/libs/vfs/Cargo.toml --release --features vsock,client --bin motlie-vfs-guest-v1_1 --target-dir /var/lib/motlie/src/target
  printf '${CONTROL_PASSWORD}\n' | sudo -S install -D -m 0755 /var/lib/motlie/src/target/release/motlie-vfs-guest-v1_1 /usr/local/bin/motlie-vfs-guest
fi
if ! command -v codex >/dev/null 2>&1; then
  printf '${CONTROL_PASSWORD}\n' | sudo -S npm install -g @openai/codex
fi
if ! command -v claude >/dev/null 2>&1; then
  printf '${CONTROL_PASSWORD}\n' | sudo -S npm install -g @anthropic-ai/claude-code
fi
NPM_GLOBAL_PREFIX="\$(npm prefix -g 2>/dev/null || true)"
if [[ -n "\$NPM_GLOBAL_PREFIX" ]]; then
  if [[ -x "\$NPM_GLOBAL_PREFIX/bin/codex" && ! -x /usr/local/bin/codex ]]; then
    printf '${CONTROL_PASSWORD}\n' | sudo -S ln -sf "\$NPM_GLOBAL_PREFIX/bin/codex" /usr/local/bin/codex
  fi
  if [[ -x "\$NPM_GLOBAL_PREFIX/bin/claude" && ! -x /usr/local/bin/claude ]]; then
    printf '${CONTROL_PASSWORD}\n' | sudo -S ln -sf "\$NPM_GLOBAL_PREFIX/bin/claude" /usr/local/bin/claude
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
printf '${CONTROL_PASSWORD}\n' | sudo -S install -D -m 0755 /tmp/motlie-vmm-vsock-ssh-loop /usr/local/bin/motlie-vmm-vsock-ssh-loop
printf '${CONTROL_PASSWORD}\n' | sudo -S install -D -m 0644 /tmp/motlie-vmm-vsock-ssh.service /etc/systemd/system/motlie-vmm-vsock-ssh.service
cat <<'EOFCA' >/tmp/motlie-vmm-user-ca.pub
${SSH_CA_PUBKEY}
EOFCA
cat <<'EOFPRINCIPAL' >/tmp/motlie-vmm-principal
${SSH_PRINCIPAL}
EOFPRINCIPAL
printf '${CONTROL_PASSWORD}\n' | sudo -S install -D -m 0644 /tmp/motlie-vmm-user-ca.pub /etc/ssh/ca/user_ca.pub
printf '${CONTROL_PASSWORD}\n' | sudo -S install -D -m 0644 /tmp/motlie-vmm-principal /etc/ssh/auth_principals/${LOGIN_USER}
if ! printf '${CONTROL_PASSWORD}\n' | sudo -S grep -q '^TrustedUserCAKeys /etc/ssh/ca/user_ca.pub$' /etc/ssh/sshd_config; then
  cat <<'EOFSSHCFG' >/tmp/motlie-vmm-sshd-ca.conf
TrustedUserCAKeys /etc/ssh/ca/user_ca.pub
AuthorizedPrincipalsFile /etc/ssh/auth_principals/%u
EOFSSHCFG
  printf '${CONTROL_PASSWORD}\n' | sudo -S sh -c 'cat /tmp/motlie-vmm-sshd-ca.conf >> /etc/ssh/sshd_config'
fi
printf '${CONTROL_PASSWORD}\n' | sudo -S chown root:root /etc/ssh/ca/user_ca.pub /etc/ssh/auth_principals/${LOGIN_USER}
printf '${CONTROL_PASSWORD}\n' | sudo -S chmod 0644 /etc/ssh/ca/user_ca.pub /etc/ssh/auth_principals/${LOGIN_USER}
printf '${CONTROL_PASSWORD}\n' | sudo -S systemctl daemon-reload
printf '${CONTROL_PASSWORD}\n' | sudo -S systemctl unmask motlie-vfs-guest.service || true
printf '${CONTROL_PASSWORD}\n' | sudo -S systemctl unmask motlie-agent-state.service || true
printf '${CONTROL_PASSWORD}\n' | sudo -S systemctl unmask motlie-vmm-vsock-ssh.service || true
printf '${CONTROL_PASSWORD}\n' | sudo -S systemctl enable motlie-vfs-guest.service
printf '${CONTROL_PASSWORD}\n' | sudo -S systemctl enable motlie-agent-state.service || true
printf '${CONTROL_PASSWORD}\n' | sudo -S systemctl enable motlie-vmm-vsock-ssh.service || true
printf '${CONTROL_PASSWORD}\n' | sudo -S systemctl restart ssh.service || printf '${CONTROL_PASSWORD}\n' | sudo -S systemctl restart ssh || true
printf '${CONTROL_PASSWORD}\n' | sudo -S systemctl restart motlie-vfs-guest.service
printf '${CONTROL_PASSWORD}\n' | sudo -S systemctl restart motlie-agent-state.service || true
printf '${CONTROL_PASSWORD}\n' | sudo -S systemctl restart motlie-vmm-vsock-ssh.service || true
printf '${CONTROL_PASSWORD}\n' | sudo -S /usr/local/bin/motlie-agent-state-setup || true
EOF

CONTROL_USER="$LOGIN_USER"
CONTROL_PASSWORD="testpass"

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
  guest_fetch "$POST_PROVISION_REMOTE_JSON" "$POST_PROVISION_LOCAL_JSON"
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
  guest_bash <<EOF
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
  guest_fetch "$VALIDATION_REMOTE_JSON" "$VALIDATION_LOCAL_JSON"
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
