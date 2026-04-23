#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
ARTIFACTS_DIR="$SCRIPT_DIR/artifacts"
BASE_VM_NAME="${MOTLIE_VZ_BASE_VM_NAME:-motlie-v1-25-base-iter}"
SOURCE_DISK_PATH="${MOTLIE_VZ_SOURCE_DISK:-}"
SOURCE_NVRAM_PATH="${MOTLIE_VZ_SOURCE_NVRAM:-}"
SOURCE_MACHINE_ID_PATH="${MOTLIE_VZ_SOURCE_MACHINE_ID:-}"
RUN_LOG="$ARTIFACTS_DIR/build-run.log"
RESULT_JSON="$ARTIFACTS_DIR/build-result.json"
IDENTITY_PROBE_JSON="$ARTIFACTS_DIR/identity-probe.json"
SOURCE_TARBALL="$ARTIFACTS_DIR/motlie-src.tar.gz"
TIMEOUT_SECONDS="${MOTLIE_VZ_TIMEOUT_SECONDS:-300}"
GUEST_SRC_DIR="/home/admin/motlie-src"
BOOTSTRAP_USER="${MOTLIE_VZ_BOOTSTRAP_USER:-admin}"
BOOTSTRAP_PASS="${MOTLIE_VZ_BOOTSTRAP_PASS:-admin}"
SERVICE_FILE="$SCRIPT_DIR/motlie-vfs-guest.service"
DATASOURCE_CFG_FILE="$SCRIPT_DIR/99_motlie_vz.cfg"
AGENT_STATE_SETUP_FILE="$SCRIPT_DIR/motlie-agent-state-setup.sh"
AGENT_STATE_UNIT_FILE="$SCRIPT_DIR/motlie-agent-state.service"
RUNNER_BUILD_SCRIPT="$SCRIPT_DIR/build-vz-runner.sh"
RUNNER_BIN_OVERRIDE="${MOTLIE_VZ_RUNNER_BIN:-}"
SKIP_RUNNER_BUILD="${MOTLIE_VZ_SKIP_RUNNER_BUILD:-0}"
EGRESS_HELPER_BIN_OVERRIDE="${MOTLIE_VZ_EGRESS_HELPER_BIN:-}"
RUNNER_PID_FILE="$ARTIFACTS_DIR/build-runner.pid"
EGRESS_HELPER_PID_FILE="$ARTIFACTS_DIR/build-egress-helper.pid"
SERIAL_LOG="$ARTIFACTS_DIR/build-serial.log"
SOCKET_PATH="/tmp/motlie-vnet-build.vsock_5000"
EGRESS_SOCKET_PATH="/tmp/motlie-vnet-build.egress.sock"
EGRESS_LOG="$ARTIFACTS_DIR/build-egress.log"
EGRESS_HELPER_LOG_FRAMES="${MOTLIE_VZ_LOG_FRAMES:-0}"
NET_MAC="${MOTLIE_VZ_BUILD_NET_MAC:-02:4d:6f:74:62:25}"
CONTROL_HOST="127.0.0.1"
CONTROL_PORT="${MOTLIE_VZ_BUILD_SSH_PORT:-2225}"
GUEST_IPV4="10.0.2.15"
SEED_DIR="$ARTIFACTS_DIR/build-seed"
SEED_IMAGE="$ARTIFACTS_DIR/build-seed.dmg"
WORK_VM_DIR="$ARTIFACTS_DIR/${BASE_VM_NAME}.vm"
NATIVE_SOURCE_VM_DIR="${MOTLIE_VZ_NATIVE_SOURCE_VM_DIR:-$ARTIFACTS_DIR/source-base.vm}"

zmodload zsh/datetime

mkdir -p "$ARTIFACTS_DIR"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

require_cmd git
require_cmd tar
require_cmd python3
require_cmd expect
require_cmd scp
require_cmd ssh
require_cmd hdiutil
require_cmd cargo

if [[ -n "$SOURCE_DISK_PATH" || -n "$SOURCE_NVRAM_PATH" || -n "$SOURCE_MACHINE_ID_PATH" ]]; then
  if [[ -z "$SOURCE_DISK_PATH" || -z "$SOURCE_NVRAM_PATH" ]]; then
    echo "MOTLIE_VZ_SOURCE_DISK and MOTLIE_VZ_SOURCE_NVRAM must be set together" >&2
    exit 1
  fi
else
  if [[ -f "$NATIVE_SOURCE_VM_DIR/disk.img" && -f "$NATIVE_SOURCE_VM_DIR/nvram.bin" ]]; then
    SOURCE_DISK_PATH="$NATIVE_SOURCE_VM_DIR/disk.img"
    SOURCE_NVRAM_PATH="$NATIVE_SOURCE_VM_DIR/nvram.bin"
    if [[ -f "$NATIVE_SOURCE_VM_DIR/machine-id.bin" ]]; then
      SOURCE_MACHINE_ID_PATH="$NATIVE_SOURCE_VM_DIR/machine-id.bin"
    else
      SOURCE_MACHINE_ID_PATH=""
    fi
  else
    cat >&2 <<EOF
native source artifacts are required for v1.25 guest builds

Set:
  MOTLIE_VZ_SOURCE_DISK
  MOTLIE_VZ_SOURCE_NVRAM
optionally:
  MOTLIE_VZ_SOURCE_MACHINE_ID

Or populate the native source cache first:
  $NATIVE_SOURCE_VM_DIR
EOF
    exit 1
  fi
fi

cleanup() {
  if [[ -f "$RUNNER_PID_FILE" ]]; then
    kill "$(cat "$RUNNER_PID_FILE")" >/dev/null 2>&1 || true
    rm -f "$RUNNER_PID_FILE"
  fi
  if [[ -f "$EGRESS_HELPER_PID_FILE" ]]; then
    kill "$(cat "$EGRESS_HELPER_PID_FILE")" >/dev/null 2>&1 || true
    rm -f "$EGRESS_HELPER_PID_FILE"
  fi
  rm -f "$EGRESS_SOCKET_PATH"
  rm -f "$SEED_IMAGE"
  rm -rf "$SEED_DIR"
}

trap cleanup EXIT

echo "=== Vz v1.25 base guest build ==="
echo "Base VM:      $BASE_VM_NAME"
echo "Source disk:  $SOURCE_DISK_PATH"
echo "Native cache: $NATIVE_SOURCE_VM_DIR"

rm -rf "$WORK_VM_DIR"
mkdir -p "$WORK_VM_DIR"

materialize_work_vm() {
  local src_disk="$1"
  local src_nvram="$2"
  local src_machine_id="${3:-}"
  python3 - "$src_disk" "$src_nvram" "$src_machine_id" "$WORK_VM_DIR" <<'PY'
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
if src_machine_id:
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
}

echo "--- materializing native build VM from source artifacts ---"
materialize_work_vm "$SOURCE_DISK_PATH" "$SOURCE_NVRAM_PATH" "$SOURCE_MACHINE_ID_PATH"

echo "--- packing Motlie source tree ---"
COPYFILE_DISABLE=1 COPY_EXTENDED_ATTRIBUTES_DISABLE=1 git -C "$REPO_ROOT" ls-files -z | COPYFILE_DISABLE=1 COPY_EXTENDED_ATTRIBUTES_DISABLE=1 tar --disable-copyfile --no-mac-metadata --no-xattrs --null -czf "$SOURCE_TARBALL" -C "$REPO_ROOT" --files-from -

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

  echo "failed to terminate stale build Vz runners" >&2
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

ssh_ready_for_password_prompt() {
  expect <<EOF
set timeout 5
log_user 0
spawn ssh -p ${CONTROL_PORT} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=1 -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 ${BOOTSTRAP_USER}@${CONTROL_HOST} true
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
      cat "$RUN_LOG" >&2 || true
      [[ -f "$SERIAL_LOG" ]] && tail -n 80 "$SERIAL_LOG" >&2 || true
      [[ -f "$EGRESS_LOG" ]] && tail -n 80 "$EGRESS_LOG" >&2 || true
      exit 1
    fi
    sleep 0.5
    attempts=$(( attempts + 1 ))
  done
  return 1
}

kill_stale_runners
rm -f "$RUNNER_PID_FILE" "$SERIAL_LOG" "$SEED_IMAGE" "$EGRESS_HELPER_PID_FILE" "$EGRESS_SOCKET_PATH" "$EGRESS_LOG"
rm -rf "$SEED_DIR"

echo "--- rendering native NoCloud seed disk ---"
mkdir -p "$SEED_DIR"
cat >"$SEED_DIR/meta-data" <<EOF
instance-id: ${BASE_VM_NAME}
local-hostname: motlie-v1-25-build
EOF
cat >"$SEED_DIR/user-data" <<'EOF'
#cloud-config
users:
  - default
  - name: admin
    gecos: Motlie Build Bootstrap
    plain_text_passwd: admin
    lock_passwd: false
    shell: /bin/bash
    sudo: ALL=(ALL) NOPASSWD:ALL
ssh_pwauth: true
disable_root: true
package_update: false
chpasswd:
  expire: false
runcmd:
  - [ systemctl, enable, --now, ssh ]
  - [ sh, -lc, "echo '=== motlie-v1.25 bootdiag begin ===' >/dev/hvc0" ]
  - [ sh, -lc, "ip -o link show >/dev/hvc0 2>&1 || true" ]
  - [ sh, -lc, "ip -o addr show >/dev/hvc0 2>&1 || true" ]
  - [ sh, -lc, "ip route show >/dev/hvc0 2>&1 || true" ]
  - [ sh, -lc, "ss -ltnp >/dev/hvc0 2>&1 || true" ]
  - [ sh, -lc, "systemctl status ssh --no-pager >/dev/hvc0 2>&1 || true" ]
  - [ sh, -lc, "echo '=== motlie-v1.25 bootdiag end ===' >/dev/hvc0" ]
EOF
hdiutil create -quiet -fs FAT32 -volname CIDATA -srcfolder "$SEED_DIR" -ov -format UDRW "$SEED_IMAGE"

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

VM_DIR="$WORK_VM_DIR"
DISK_PATH="$VM_DIR/disk.img"
NVRAM_PATH="$VM_DIR/nvram.bin"
MACHINE_ID_PATH="$VM_DIR/machine-id.bin"

echo "--- starting guest via native Apple Vz runner ---"
: > "$RUN_LOG"
START_EPOCH="$EPOCHREALTIME"
"$RUNNER_BIN" \
  --disk "$DISK_PATH" \
  --nvram "$NVRAM_PATH" \
  --machine-id "$MACHINE_ID_PATH" \
  --serial-log "$SERIAL_LOG" \
  --seed-disk "$SEED_IMAGE" \
  --unix-socket "$SOCKET_PATH" \
  --net-backend-socket "$EGRESS_SOCKET_PATH" \
  --net-mac "$NET_MAC" \
  --vsock-port 5000 \
  --memory-mib 4096 \
  --cpu-count 4 \
  >"$RUN_LOG" 2>&1 &
RUNNER_PID="$!"
echo "$RUNNER_PID" > "$RUNNER_PID_FILE"

IP_ADDR="$GUEST_IPV4"

echo "--- waiting for guest SSH ---"
wait_for_guest_ssh || {
  echo "timed out waiting for guest SSH after ${TIMEOUT_SECONDS}s" >&2
  cat "$RUN_LOG" >&2 || true
  [[ -f "$SERIAL_LOG" ]] && tail -n 80 "$SERIAL_LOG" >&2 || true
  [[ -f "$EGRESS_LOG" ]] && tail -n 80 "$EGRESS_LOG" >&2 || true
  exit 1
}

READY_EPOCH="$EPOCHREALTIME"
BOOT_SECONDS="$(awk -v start="$START_EPOCH" -v ready="$READY_EPOCH" 'BEGIN { printf "%.3f", ready - start }')"

guest_bash() {
  guest_bash_as "$BOOTSTRAP_USER" "$BOOTSTRAP_PASS"
}

guest_bash_as() {
  local run_user="$1"
  local run_pass="$2"
  local sudo_pass="${3:-}"
  local remote_path="/home/${run_user}/.motlie-vnet-remote.sh"
  local remote_exec="chmod 0700 ${remote_path} && bash -euo pipefail ${remote_path} </dev/null"
  if [[ -n "$sudo_pass" ]]; then
    remote_exec="chmod 0700 ${remote_path} && printf '%s\n' '${sudo_pass}' | sudo -S -p '' bash -euo pipefail ${remote_path} </dev/null"
  fi
  local remote_script
  remote_script="$(mktemp)"
  cat >"$remote_script"
  expect <<EOF
set timeout -1
set password_tries 0
spawn scp -P ${CONTROL_PORT} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$remote_script" ${run_user}@${CONTROL_HOST}:${remote_path}
expect {
  "password:" {
    incr password_tries
    if {\$password_tries > 3} {
      puts stderr "scp auth failed for ${run_user}@${CONTROL_HOST}:${CONTROL_PORT}"
      exit 97
    }
    send "${run_pass}\r"
    exp_continue
  }
  "Permission denied" {
    puts stderr "scp permission denied for ${run_user}@${CONTROL_HOST}:${CONTROL_PORT}"
    exit 98
  }
  eof {
    catch wait result
    set exit_code [lindex \$result 3]
    if {\$exit_code != 0} {
      exit \$exit_code
    }
  }
}
EOF
  rm -f "$remote_script"
  expect <<EOF
set timeout -1
set password_tries 0
spawn ssh -p ${CONTROL_PORT} -n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ${run_user}@${CONTROL_HOST} "${remote_exec}"
expect {
  "password:" {
    incr password_tries
    if {\$password_tries > 3} {
      puts stderr "ssh auth failed for ${run_user}@${CONTROL_HOST}:${CONTROL_PORT}"
      exit 97
    }
    send "${run_pass}\r"
    exp_continue
  }
  "Permission denied" {
    puts stderr "ssh permission denied for ${run_user}@${CONTROL_HOST}:${CONTROL_PORT}"
    exit 98
  }
  eof {
    catch wait result
    set exit_code [lindex \$result 3]
    if {\$exit_code != 0} {
      exit \$exit_code
    }
  }
}
EOF
}

guest_bash_as_capture() {
  local run_user="$1"
  local run_pass="$2"
  local sudo_pass="${3:-}"
  local remote_path="/home/${run_user}/.motlie-vnet-remote.sh"
  local remote_exec="chmod 0700 ${remote_path} && bash -euo pipefail ${remote_path} </dev/null"
  if [[ -n "$sudo_pass" ]]; then
    remote_exec="chmod 0700 ${remote_path} && printf '%s\n' '${sudo_pass}' | sudo -S -p '' bash -euo pipefail ${remote_path} </dev/null"
  fi
  local remote_script
  remote_script="$(mktemp)"
  cat >"$remote_script"
  expect <<EOF
set timeout -1
set password_tries 0
spawn scp -P ${CONTROL_PORT} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$remote_script" ${run_user}@${CONTROL_HOST}:${remote_path}
expect {
  "password:" {
    incr password_tries
    if {\$password_tries > 3} {
      puts stderr "scp auth failed for ${run_user}@${CONTROL_HOST}:${CONTROL_PORT}"
      exit 97
    }
    send "${run_pass}\r"
    exp_continue
  }
  "Permission denied" {
    puts stderr "scp permission denied for ${run_user}@${CONTROL_HOST}:${CONTROL_PORT}"
    exit 98
  }
  eof {
    catch wait result
    set exit_code [lindex \$result 3]
    if {\$exit_code != 0} {
      exit \$exit_code
    }
  }
}
EOF
  rm -f "$remote_script"
  expect <<EOF
set timeout -1
spawn ssh -p ${CONTROL_PORT} -n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ${run_user}@${CONTROL_HOST} "${remote_exec}"
expect {
  "password:" {
    send_user "password-prompt\\n"
    send "${run_pass}\r"
    exp_continue
  }
  eof {
    catch wait result
    set exit_code [lindex \$result 3]
    exit \$exit_code
  }
}
EOF
}

guest_copy() {
  local src="$1"
  local dst="$2"
  expect <<EOF
set timeout -1
set password_tries 0
spawn scp -P ${CONTROL_PORT} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$src" ${BOOTSTRAP_USER}@${CONTROL_HOST}:$dst
expect {
  "password:" {
    incr password_tries
    if {\$password_tries > 3} {
      puts stderr "scp auth failed for ${BOOTSTRAP_USER}@${CONTROL_HOST}:${CONTROL_PORT}"
      exit 97
    }
    send "${BOOTSTRAP_PASS}\r"
    exp_continue
  }
  "Permission denied" {
    puts stderr "scp permission denied for ${BOOTSTRAP_USER}@${CONTROL_HOST}:${CONTROL_PORT}"
    exit 98
  }
  eof {
    catch wait result
    set exit_code [lindex \$result 3]
    if {\$exit_code != 0} {
      exit \$exit_code
    }
  }
}
EOF
}

guest_fetch() {
  local src="$1"
  local dst="$2"
  expect <<EOF
set timeout -1
set password_tries 0
spawn scp -P ${CONTROL_PORT} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ${BOOTSTRAP_USER}@${CONTROL_HOST}:$src "$dst"
expect {
  "password:" {
    incr password_tries
    if {\$password_tries > 3} {
      puts stderr "scp auth failed for ${BOOTSTRAP_USER}@${CONTROL_HOST}:${CONTROL_PORT}"
      exit 97
    }
    send "${BOOTSTRAP_PASS}\r"
    exp_continue
  }
  "Permission denied" {
    puts stderr "scp permission denied for ${BOOTSTRAP_USER}@${CONTROL_HOST}:${CONTROL_PORT}"
    exit 98
  }
  eof {
    catch wait result
    set exit_code [lindex \$result 3]
    if {\$exit_code != 0} {
      exit \$exit_code
    }
  }
}
EOF
}

guest_capture() {
  guest_capture_as "$BOOTSTRAP_USER" "$BOOTSTRAP_PASS"
}

guest_capture_as() {
  local run_user="$1"
  local run_pass="$2"
  local sudo_pass="${3:-}"
  local remote_path="/home/${run_user}/.motlie-vnet-capture.sh"
  local remote_exec="chmod 0700 ${remote_path} && bash -euo pipefail ${remote_path} </dev/null"
  if [[ -n "$sudo_pass" ]]; then
    remote_exec="chmod 0700 ${remote_path} && printf '%s\n' '${sudo_pass}' | sudo -S -p '' bash -euo pipefail ${remote_path} </dev/null"
  fi
  local remote_script
  remote_script="$(mktemp)"
  cat >"$remote_script"
  expect <<EOF
set timeout -1
set password_tries 0
spawn scp -P ${CONTROL_PORT} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$remote_script" ${run_user}@${CONTROL_HOST}:${remote_path}
expect {
  "password:" {
    incr password_tries
    if {\$password_tries > 3} {
      puts stderr "scp auth failed for ${run_user}@${CONTROL_HOST}:${CONTROL_PORT}"
      exit 97
    }
    send "${run_pass}\r"
    exp_continue
  }
  "Permission denied" {
    puts stderr "scp permission denied for ${run_user}@${CONTROL_HOST}:${CONTROL_PORT}"
    exit 98
  }
  eof {
    catch wait result
    set exit_code [lindex \$result 3]
    if {\$exit_code != 0} {
      exit \$exit_code
    }
  }
}
EOF
  rm -f "$remote_script"
expect <<EOF
set timeout -1
log_user 0
set output ""
set password_tries 0
spawn ssh -p ${CONTROL_PORT} -n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ${run_user}@${CONTROL_HOST} "${remote_exec}"
expect {
  "password:" {
    incr password_tries
    if {\$password_tries > 3} {
      puts stderr "ssh auth failed for ${run_user}@${CONTROL_HOST}:${CONTROL_PORT}"
      exit 97
    }
    send "${run_pass}\r"
    exp_continue
  }
  "Permission denied" {
    puts stderr "ssh permission denied for ${run_user}@${CONTROL_HOST}:${CONTROL_PORT}"
    exit 98
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
}
catch wait result
set exit_code [lindex \$result 3]
if {\$exit_code != 0} {
  exit \$exit_code
}
EOF
}

echo "--- installing guest prerequisites ---"
guest_bash <<'EOF'
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential \
  pkg-config \
  libfuse3-dev \
  libfuse3-3 \
  fuse3 \
  curl \
  ca-certificates \
  tar \
  gzip \
  iproute2 \
  tmux \
  locales \
  bubblewrap \
  dnsutils \
  python3 \
  npm
sudo sed -i '/^en_US.UTF-8 UTF-8$/d' /etc/locale.gen
printf 'en_US.UTF-8 UTF-8\n' | sudo tee -a /etc/locale.gen >/dev/null
sudo locale-gen en_US.UTF-8
sudo update-locale LANG=en_US.UTF-8
EOF

echo "--- installing Rust toolchain in guest if needed ---"
guest_bash <<'EOF'
if ! command -v cargo >/dev/null 2>&1; then
  curl https://sh.rustup.rs -sSf | sh -s -- -y
fi
EOF

echo "--- uploading Motlie source tree into guest ---"
guest_copy "$SOURCE_TARBALL" /tmp/motlie-src.tar.gz
guest_copy "$SERVICE_FILE" /tmp/motlie-vfs-guest.service
guest_copy "$DATASOURCE_CFG_FILE" /tmp/99_motlie_vz.cfg
guest_copy "$AGENT_STATE_SETUP_FILE" /tmp/motlie-agent-state-setup
guest_copy "$AGENT_STATE_UNIT_FILE" /tmp/motlie-agent-state.service
guest_bash <<EOF
rm -rf '$GUEST_SRC_DIR'
mkdir -p '$GUEST_SRC_DIR'
tar -xzf /tmp/motlie-src.tar.gz -C '$GUEST_SRC_DIR'
EOF

echo "--- building guest binaries and CLIs in guest ---"
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
cargo build --manifest-path '$GUEST_SRC_DIR/libs/vfs/Cargo.toml' --release --features vsock,client --bin motlie-vfs-guest-v1_1
sudo npm install -g @openai/codex
sudo npm install -g @anthropic-ai/claude-code
EOF

echo "--- installing converged v1.25 guest contract ---"
guest_bash <<'EOF'
sudo install -D -m 0755 "$HOME/motlie-target/release/motlie-vfs-guest-v1_1" /usr/local/bin/motlie-vfs-guest
sudo install -D -m 0644 /tmp/motlie-vfs-guest.service /etc/systemd/system/motlie-vfs-guest.service
sudo install -D -m 0644 /tmp/99_motlie_vz.cfg /etc/cloud/cloud.cfg.d/99_motlie_vz.cfg
sudo mkdir -p /etc/motlie-vfs
sudo mkdir -p /etc/profile.d
EOF

echo "--- creating bootstrap user for identity remap ---"
guest_bash_as admin admin <<'EOF'
if ! id -u motlie-build >/dev/null 2>&1; then
    sudo groupadd -f -g 2002 motlie-build
    sudo useradd -m -u 2002 -g 2002 -s /bin/bash motlie-build
fi
echo "motlie-build:admin" | sudo chpasswd
sudo usermod -U motlie-build || true
sudo usermod -aG sudo motlie-build || true
printf '%s\n' 'motlie-build ALL=(ALL) NOPASSWD:ALL' | sudo tee /etc/sudoers.d/90-motlie-build >/dev/null
sudo chown root:root /etc/sudoers.d/90-motlie-build
sudo chmod 0440 /etc/sudoers.d/90-motlie-build
EOF

if ! guest_bash_as motlie-build admin admin <<'EOF'
true
EOF
then
  echo "--- motlie-build bootstrap diagnostics ---" >&2
  guest_bash <<'EOF' >&2 || true
set -x
getent passwd motlie-build || true
sudo passwd -S motlie-build || true
sudo -l -U motlie-build || true
grep -R "AllowUsers\\|PasswordAuthentication\\|KbdInteractiveAuthentication\\|UsePAM" /etc/ssh/sshd_config /etc/ssh/sshd_config.d 2>/dev/null || true
EOF
  guest_bash_as_capture motlie-build admin admin <<'EOF' >&2 || true
id
true
EOF
  echo "motlie-build bootstrap user is not ready for sudo-backed remap execution" >&2
  exit 1
fi

guest_bash_as motlie-build admin admin <<'EOF'

remap_conflicting_identity() {
    user_name="$1"
    target_uid="$2"
    target_gid="$3"
    remap_uid="$4"
    remap_gid="$5"

    current_uid="$(id -u "$user_name" 2>/dev/null || true)"
    current_gid="$(getent group "$user_name" | cut -d: -f3 || true)"

    if [ "$current_uid" = "$target_uid" ]; then
        loginctl terminate-user "$user_name" >/dev/null 2>&1 || true
        systemctl stop "user@${target_uid}.service" >/dev/null 2>&1 || true
        pkill -KILL -u "$target_uid" >/dev/null 2>&1 || true
        for _ in 1 2 3 4 5; do
            if ! pgrep -u "$target_uid" >/dev/null 2>&1; then
                break
            fi
            sleep 1
        done
        if pgrep -u "$target_uid" >/dev/null 2>&1; then
            echo "uid $target_uid is still active; cannot remap $user_name" >&2
            ps -u "$target_uid" -o pid,ppid,user,tty,comm,args >&2 || true
            exit 1
        fi
        usermod -u "$remap_uid" "$user_name"
        find / -xdev -uid "$target_uid" -exec chown -h "$remap_uid" {} + 2>/dev/null || true
    fi
    if [ "$current_gid" = "$target_gid" ]; then
        groupmod -g "$remap_gid" "$user_name"
        find / -xdev -gid "$target_gid" -exec chgrp -h "$remap_gid" {} + 2>/dev/null || true
    fi
}

ensure_guest_identity() {
    user_name="$1"
    target_uid="$2"
    target_gid="$3"
    password="$4"

    existing_gid="$(getent group "$user_name" | cut -d: -f3 || true)"
    if [ -z "$existing_gid" ]; then
        gid_owner="$(getent group "$target_gid" | cut -d: -f1 || true)"
        if [ -n "$gid_owner" ] && [ "$gid_owner" != "$user_name" ]; then
            echo "gid $target_gid already belongs to $gid_owner" >&2
            exit 1
        fi
        groupadd -g "$target_gid" "$user_name"
    elif [ "$existing_gid" != "$target_gid" ]; then
        echo "group $user_name has gid $existing_gid but expected $target_gid" >&2
        exit 1
    fi

    existing_uid="$(id -u "$user_name" 2>/dev/null || true)"
    if [ -z "$existing_uid" ]; then
        uid_owner="$(getent passwd "$target_uid" | cut -d: -f1 || true)"
        if [ -n "$uid_owner" ] && [ "$uid_owner" != "$user_name" ]; then
            echo "uid $target_uid already belongs to $uid_owner" >&2
            exit 1
        fi
        useradd -m -u "$target_uid" -g "$target_gid" -s /bin/bash "$user_name"
    elif [ "$existing_uid" != "$target_uid" ]; then
        echo "user $user_name has uid $existing_uid but expected $target_uid" >&2
        exit 1
    fi

    usermod -aG sudo "$user_name" || true
    echo "$user_name:$password" | chpasswd
}

remap_conflicting_identity admin 1000 1000 2000 2000
remap_conflicting_identity ubuntu 1001 1001 2001 2001
ensure_guest_identity alice 1000 1000 testpass
ensure_guest_identity bob 1001 1001 testpass

cat <<'TMUXEOF' > /etc/profile.d/tmux-auto.sh
if [ -n "$SSH_CONNECTION" ] && [ -z "$TMUX" ] && [ -t 0 ] && [ -t 1 ] && command -v tmux >/dev/null 2>&1; then
    if tmux has-session -t "$USER" 2>/dev/null; then
        echo "Attaching to existing tmux session..."
        sleep 1
        exec tmux attach-session -t "$USER"
    else
        printf "Start tmux session? [Y/n] (auto-yes in 3s) "
        if read -r -n 1 -t 3 answer; then
            echo
        else
            answer=Y
            echo
        fi
        case "$answer" in
            n|N) ;;
            *) exec tmux new-session -s "$USER" ;;
        esac
    fi
fi
TMUXEOF
cat <<'DOTENVEOF' > /etc/profile.d/dotenv.sh
if [ -f "$HOME/.env" ]; then
    set -a
    . "$HOME/.env"
    set +a
fi
DOTENVEOF
cat <<'AGENTEOF' > /etc/profile.d/agent-state.sh
agent_state_root=/agent-state
codex_root="$agent_state_root/codex"
codex_sqlite_root="$codex_root/sqlite"
claude_root="$agent_state_root/claude"
claude_code_root="$agent_state_root/claude-code"
if [ -d "$agent_state_root" ] && [ -n "${HOME:-}" ] && [ -d "$HOME" ] && [ "${USER:-}" != "root" ] && [ "${HOME#"/home/"}" != "$HOME" ]; then
    mkdir -p "$codex_root" "$codex_sqlite_root" "$claude_root" "$claude_code_root" "$HOME/.config" >/dev/null 2>&1 || true
    export CODEX_HOME="$codex_root"
    export CODEX_SQLITE_HOME="$codex_sqlite_root"
fi
AGENTEOF
cat <<'MOTDEOF' > /etc/motd
                    _   _ _
  _ __ ___   ___ | |_| (_) ___
 | '_ ` _ \ / _ \| __| | |/ _ \
 | | | | | | (_) | |_| | |  __/
 |_| |_| |_|\___/ \__|_|_|\___|

v1.25 Apple Vz vnet / agent-state demo
MOTDEOF
if ! grep -qx 'user_allow_other' /etc/fuse.conf 2>/dev/null; then
  printf 'user_allow_other\n' >> /etc/fuse.conf
fi
chmod 0644 /etc/profile.d/tmux-auto.sh /etc/profile.d/dotenv.sh /etc/profile.d/agent-state.sh
install -D -m 0755 /tmp/motlie-agent-state-setup /usr/local/bin/motlie-agent-state-setup
install -D -m 0644 /tmp/motlie-agent-state.service /etc/systemd/system/motlie-agent-state.service
systemctl unmask motlie-vfs-guest.service || true
systemctl unmask motlie-agent-state.service || true
systemctl daemon-reload
systemctl enable motlie-vfs-guest.service >/dev/null 2>&1 || true
systemctl enable motlie-agent-state.service >/dev/null 2>&1 || true
EOF

echo "--- cleaning cloud-init state for reusable base image ---"
guest_bash_as admin admin <<'EOF'
sudo cloud-init clean --logs --machine-id --seed
sudo rm -rf /var/lib/cloud/*
sudo truncate -s 0 /etc/machine-id
sudo mkdir -p /var/lib/cloud
EOF

GUEST_BINARY="/usr/local/bin/motlie-vfs-guest"

guest_bash_as admin admin <<'EOF'
python3 - <<'PY'
import json
import pwd
import grp

def passwd_entry(name: str):
    try:
        entry = pwd.getpwnam(name)
        return {"name": entry.pw_name, "uid": entry.pw_uid, "gid": entry.pw_gid, "home": entry.pw_dir}
    except KeyError:
        return None

def group_entry(name: str):
    try:
        entry = grp.getgrnam(name)
        return {"name": entry.gr_name, "gid": entry.gr_gid}
    except KeyError:
        return None

payload = {
    "passwd": {
        "admin": passwd_entry("admin"),
        "alice": passwd_entry("alice"),
        "bob": passwd_entry("bob"),
    },
    "group": {
        "admin": group_entry("admin"),
        "alice": group_entry("alice"),
        "bob": group_entry("bob"),
    },
}
with open("/tmp/motlie-identity-probe.json", "w", encoding="utf-8") as fh:
    json.dump(payload, fh, sort_keys=True)
PY
EOF
guest_fetch /tmp/motlie-identity-probe.json "$IDENTITY_PROBE_JSON"

python3 - "$RESULT_JSON" "$BASE_VM_NAME" "$IP_ADDR" "$BOOT_SECONDS" "$GUEST_BINARY" "$IDENTITY_PROBE_JSON" <<'PY'
import json
import sys

path, vm_name, ip_addr, boot_seconds, guest_binary, identity_probe_json = sys.argv[1:]
with open(identity_probe_json, "r", encoding="utf-8") as fh:
    identity_payload = json.load(fh)
with open(path, "w", encoding="utf-8") as fh:
    json.dump(
        {
            "backend": "vz-native",
            "vm_name": vm_name,
            "ip_addr": ip_addr,
            "boot_to_ip_seconds": float(boot_seconds),
            "guest_contract": {
                "motlie_vfs_guest_path": guest_binary,
                "users": {
                    "alice": {"uid": 1000, "gid": 1000, "password": "testpass"},
                    "bob": {"uid": 1001, "gid": 1001, "password": "testpass"},
                },
                "agent_state": "/agent-state",
            },
            "identity_probe": identity_payload,
        },
        fh,
        indent=2,
        sort_keys=True,
    )
    fh.write("\n")
PY

echo "--- result written ---"
cat "$RESULT_JSON"

echo "--- shutting down base guest gracefully ---"
guest_bash_as admin admin <<'EOF'
sudo shutdown -h now || true
EOF
for _ in {1..40}; do
  if [[ ! -f "$RUNNER_PID_FILE" ]]; then
    break
  fi
  if ! kill -0 "$(cat "$RUNNER_PID_FILE")" >/dev/null 2>&1; then
    rm -f "$RUNNER_PID_FILE"
    break
  fi
  sleep 0.5
done
if [[ -f "$RUNNER_PID_FILE" ]]; then
  kill "$(cat "$RUNNER_PID_FILE")" >/dev/null 2>&1 || true
  rm -f "$RUNNER_PID_FILE"
fi

echo "--- caching native source artifacts ---"
rm -rf "$NATIVE_SOURCE_VM_DIR"
mkdir -p "$NATIVE_SOURCE_VM_DIR"
python3 - "$WORK_VM_DIR" "$NATIVE_SOURCE_VM_DIR" <<'PY'
import os
import shutil
import sys

src_dir, dst_dir = sys.argv[1:]
for name in ("disk.img", "nvram.bin", "machine-id.bin"):
    src = os.path.join(src_dir, name)
    if not os.path.exists(src):
        continue
    dst = os.path.join(dst_dir, name)
    tmp = dst + ".tmp-copy"
    if os.path.exists(tmp):
        os.remove(tmp)
    with open(src, "rb") as rf, open(tmp, "wb") as wf:
        shutil.copyfileobj(rf, wf, length=16 * 1024 * 1024)
    os.replace(tmp, dst)
PY

echo "=== success ==="
