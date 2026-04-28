#!/bin/zsh
set -euo pipefail

# v1.45 Vz convergence contract:
# - base-image/seed content owns long-pole immutable setup (packages, toolchains,
#   guest binaries, baseline agent tooling)
# - first-contact SSH auto-provisioning may only wait for interactive readiness:
#   CA login plus required VFS/agent-state mounts
# - full VFS/VNET/egress/package validation is a separate harness step
# Keep this sequence aligned with libs/vmm/docs/CONVERGENCE.md.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
ARTIFACTS_DIR="${MOTLIE_VZ_ARTIFACTS_DIR:-$SCRIPT_DIR/artifacts}"
BASE_VM_NAME="${MOTLIE_VZ_BASE_VM_NAME:-motlie-v1-45-base-iter}"
BASE_VM_DIR_OVERRIDE="${MOTLIE_VZ_BASE_VM_DIR:-}"
TIMEOUT_SECONDS="${MOTLIE_VZ_TIMEOUT_SECONDS:-900}"
RUNNER_BUILD_SCRIPT="$SCRIPT_DIR/build-vz-runner.sh"
RUNNER_BIN_OVERRIDE="${MOTLIE_VZ_RUNNER_BIN:-}"
SKIP_RUNNER_BUILD="${MOTLIE_VZ_SKIP_RUNNER_BUILD:-1}"
ENABLE_GUEST_SSH_VSOCK="${MOTLIE_VZ_ENABLE_GUEST_SSH_VSOCK:-0}"
EGRESS_HELPER_BIN_OVERRIDE="${MOTLIE_VZ_EGRESS_HELPER_BIN:-}"
SKIP_EGRESS_HELPER_BUILD="${MOTLIE_VZ_SKIP_EGRESS_HELPER_BUILD:-1}"
KEEP_RUNNING="${MOTLIE_VZ_KEEP_RUNNING:-0}"
REUSE_VM="${MOTLIE_VZ_REUSE_VM:-0}"
NATIVE_SOURCE_VM_DIR="${MOTLIE_VZ_NATIVE_SOURCE_VM_DIR:-$SCRIPT_DIR/../v1.35/artifacts/source-base.vm}"
BASE_SOURCE_DIR=""
RUN_VM_DIR=""
EGRESS_HELPER_PID_FILE=""
EGRESS_SOCKET_PATH=""
EGRESS_LOG=""
NET_MAC=""
CONTROL_HOST="127.0.0.1"
CONTROL_PORT=""
CONTROL_PORT_FILE="${MOTLIE_VZ_CONTROL_PORT_FILE:-}"
EGRESS_GUEST_IP="${MOTLIE_VZ_EGRESS_GUEST_IP:-10.0.2.15}"
EGRESS_HOST_IP="${MOTLIE_VZ_EGRESS_HOST_IP:-10.0.2.2}"
EGRESS_DNS_IP="${MOTLIE_VZ_EGRESS_DNS_IP:-10.0.2.3}"
EGRESS_NETMASK="${MOTLIE_VZ_EGRESS_NETMASK:-255.255.255.0}"
GUEST_IPV4="$EGRESS_GUEST_IP"
EGRESS_HELPER_LOG_FRAMES="${MOTLIE_VZ_LOG_FRAMES:-0}"
BENCHMARK_APT="${MOTLIE_VZ_BENCHMARK_APT:-0}"
BENCHMARK_HTTP="${MOTLIE_VZ_BENCHMARK_HTTP:-0}"
BENCHMARK_HTTP_URL="${MOTLIE_VZ_BENCHMARK_HTTP_URL:-http://ports.ubuntu.com/ubuntu-ports/dists/noble-updates/main/binary-arm64/Packages.gz}"
BENCHMARK_TIMEOUT_SECONDS="${MOTLIE_VZ_BENCHMARK_TIMEOUT_SECONDS:-90}"
BENCHMARK_MTU="${MOTLIE_VZ_BENCHMARK_MTU:-0}"
CONTROL_READY_FILE="${MOTLIE_VZ_CONTROL_READY_FILE:-}"
INTERACTIVE_READY_FILE="${MOTLIE_VZ_INTERACTIVE_READY_FILE:-}"
VALIDATION_COMPLETE_FILE="${MOTLIE_VZ_VALIDATION_COMPLETE_FILE:-}"
PHASES_LOG="${MOTLIE_VZ_PHASES_LOG:-}"
INLINE_VALIDATION="${MOTLIE_VZ_INLINE_VALIDATION:-}"
MOUNTS_FILE_OVERRIDE="${MOTLIE_VZ_MOUNTS_FILE:-}"
LOGIN_USER_OVERRIDE="${MOTLIE_VZ_LOGIN_USER:-}"
UID_NUM_OVERRIDE="${MOTLIE_VZ_UID_NUM:-}"
GID_NUM_OVERRIDE="${MOTLIE_VZ_GID_NUM:-}"
HOST_HOME_DIR_OVERRIDE="${MOTLIE_VZ_HOST_HOME_DIR:-}"
HOST_AGENT_STATE_DIR_OVERRIDE="${MOTLIE_VZ_HOST_AGENT_STATE_DIR:-}"
HOST_WORKSPACE_DIR_OVERRIDE="${MOTLIE_VZ_HOST_WORKSPACE_DIR:-}"
GUEST_HOSTNAME_OVERRIDE="${MOTLIE_VZ_GUEST_HOSTNAME:-}"
NET_MAC_OVERRIDE="${MOTLIE_VZ_NET_MAC:-}"
RUN_LOG_OVERRIDE="${MOTLIE_VZ_RUN_LOG:-}"
RESULT_JSON_OVERRIDE="${MOTLIE_VZ_RESULT_JSON:-}"
SERIAL_LOG_OVERRIDE="${MOTLIE_VZ_SERIAL_LOG:-}"
SEED_DIR_OVERRIDE="${MOTLIE_VZ_SEED_DIR:-}"
SEED_IMAGE_OVERRIDE="${MOTLIE_VZ_SEED_IMAGE:-}"
USE_SEED_DISK="${MOTLIE_VZ_USE_SEED_DISK:-0}"
RUNNER_PID_FILE_OVERRIDE="${MOTLIE_VZ_RUNNER_PID_FILE:-}"
GUEST_IP_FILE_OVERRIDE="${MOTLIE_VZ_GUEST_IP_FILE:-}"
EGRESS_HELPER_PID_FILE_OVERRIDE="${MOTLIE_VZ_EGRESS_HELPER_PID_FILE:-}"
EGRESS_SOCKET_PATH_OVERRIDE="${MOTLIE_VZ_EGRESS_SOCKET_PATH:-}"
EGRESS_LOG_OVERRIDE="${MOTLIE_VZ_EGRESS_LOG:-}"
INTERACTIVE_SECONDS=""
VALIDATION_SECONDS=""

zmodload zsh/datetime
SCRIPT_START_EPOCH="$EPOCHREALTIME"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

ensure_parent_dir() {
  local target_path="$1"
  mkdir -p "${target_path:h}"
}

require_cmd python3
require_cmd base64
require_cmd expect
require_cmd scp
require_cmd ssh
if [[ "$SKIP_RUNNER_BUILD" != "1" || "$SKIP_EGRESS_HELPER_BUILD" != "1" ]]; then
  require_cmd cargo
fi
if [[ "$USE_SEED_DISK" == "1" ]]; then
  require_cmd hdiutil
fi

if [[ -n "$BASE_VM_DIR_OVERRIDE" ]]; then
  BASE_SOURCE_DIR="$BASE_VM_DIR_OVERRIDE"
elif [[ -f "$NATIVE_SOURCE_VM_DIR/disk.img" && -f "$NATIVE_SOURCE_VM_DIR/nvram.bin" ]]; then
  BASE_SOURCE_DIR="$NATIVE_SOURCE_VM_DIR"
else
  cat >&2 <<EOF
native base artifacts are required for v1.45 guest launches

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

SEED_DIR="${SEED_DIR_OVERRIDE:-$ARTIFACTS_DIR/${GUEST_NAME}-seed}"

case "$GUEST_NAME" in
  alice)
    RUN_VM_NAME="${RUN_VM_NAME:-motlie-v1-45-alice-iter}"
    MOUNTS_FILE="$SCRIPT_DIR/mounts.alice.yaml"
    LOGIN_USER="alice"
    UID_NUM=1000
    GID_NUM=1000
    SOCKET_PATH="${MOTLIE_VZ_VFS_VSOCK_SOCKET:-/tmp/motlie-vmm-alice.vsock_5000}"
    SSH_VSOCK_SOCKET="${MOTLIE_VZ_SSH_VSOCK_SOCKET:-/tmp/motlie-vmm-alice.vsock_2222}"
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
    RUN_VM_NAME="${RUN_VM_NAME:-motlie-v1-45-bob-iter}"
    MOUNTS_FILE="$SCRIPT_DIR/mounts.bob.yaml"
    LOGIN_USER="bob"
    UID_NUM=1001
    GID_NUM=1001
    SOCKET_PATH="${MOTLIE_VZ_VFS_VSOCK_SOCKET:-/tmp/motlie-vmm-bob.vsock_5000}"
    SSH_VSOCK_SOCKET="${MOTLIE_VZ_SSH_VSOCK_SOCKET:-/tmp/motlie-vmm-bob.vsock_2222}"
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
    LOGIN_USER="${LOGIN_USER_OVERRIDE:-$GUEST_NAME}"
    UID_NUM="${UID_NUM_OVERRIDE:-}"
    GID_NUM="${GID_NUM_OVERRIDE:-}"
    if [[ -z "$UID_NUM" || -z "$GID_NUM" ]]; then
      echo "MOTLIE_VZ_UID_NUM and MOTLIE_VZ_GID_NUM must be set for guest '$GUEST_NAME'" >&2
      exit 1
    fi
    MOUNTS_FILE="${MOUNTS_FILE_OVERRIDE:-$SEED_DIR/mounts.yaml}"
    SOCKET_PATH="${MOTLIE_VZ_VFS_VSOCK_SOCKET:-/tmp/motlie-vmm-${GUEST_NAME}.vsock_5000}"
    SSH_VSOCK_SOCKET="${MOTLIE_VZ_SSH_VSOCK_SOCKET:-/tmp/motlie-vmm-${GUEST_NAME}.vsock_2222}"
    HOST_HOME_DIR="${HOST_HOME_DIR_OVERRIDE:-/tmp/motlie-vmm-demo/${GUEST_NAME}-home}"
    HOST_AGENT_STATE_DIR="${HOST_AGENT_STATE_DIR_OVERRIDE:-/tmp/motlie-vmm-demo/${GUEST_NAME}-agent-state}"
    HOST_WORKSPACE_DIR="${HOST_WORKSPACE_DIR_OVERRIDE:-/tmp/motlie-vmm-demo/${GUEST_NAME}-workspace}"
    EXPECTED_WORKSPACE_README="${(C)GUEST_NAME} workspace mounted from the host."
    EXPECTED_AGENT_STATE_README="Dedicated read-write agent-state layer for Codex and Claude lives here."
    EXPECTED_ENV_LINE="${(U)GUEST_NAME}_API_KEY=demo-${GUEST_NAME}"
    GUEST_HOSTNAME="${GUEST_HOSTNAME_OVERRIDE:-motlie-${GUEST_NAME}}"
    NET_MAC="${NET_MAC_OVERRIDE:-02:4d:6f:74:00:01}"
    CONTROL_PORT="${MOTLIE_VZ_SSH_PORT:-2226}"
    ;;
esac

MOUNTS_FILE="${MOUNTS_FILE_OVERRIDE:-$MOUNTS_FILE}"
LOGIN_USER="${LOGIN_USER_OVERRIDE:-$LOGIN_USER}"
UID_NUM="${UID_NUM_OVERRIDE:-$UID_NUM}"
GID_NUM="${GID_NUM_OVERRIDE:-$GID_NUM}"
SOCKET_PATH="${MOTLIE_VZ_VFS_VSOCK_SOCKET:-$SOCKET_PATH}"
SSH_VSOCK_SOCKET="${MOTLIE_VZ_SSH_VSOCK_SOCKET:-$SSH_VSOCK_SOCKET}"
HOST_HOME_DIR="${HOST_HOME_DIR_OVERRIDE:-$HOST_HOME_DIR}"
HOST_AGENT_STATE_DIR="${HOST_AGENT_STATE_DIR_OVERRIDE:-$HOST_AGENT_STATE_DIR}"
HOST_WORKSPACE_DIR="${HOST_WORKSPACE_DIR_OVERRIDE:-$HOST_WORKSPACE_DIR}"
GUEST_HOSTNAME="${GUEST_HOSTNAME_OVERRIDE:-$GUEST_HOSTNAME}"
NET_MAC="${NET_MAC_OVERRIDE:-$NET_MAC}"
CONTROL_PORT="${MOTLIE_VZ_SSH_PORT:-$CONTROL_PORT}"

if [[ -z "$INLINE_VALIDATION" ]]; then
  if [[ "$KEEP_RUNNING" == "1" ]]; then
    INLINE_VALIDATION=0
  else
    INLINE_VALIDATION=1
  fi
fi

BOOTSTRAP_USER="${MOTLIE_VZ_BOOTSTRAP_USER:-admin}"
BOOTSTRAP_PASSWORD="${MOTLIE_VZ_BOOTSTRAP_PASS:-admin}"
CONTROL_USER="$BOOTSTRAP_USER"
CONTROL_PASSWORD="$BOOTSTRAP_PASSWORD"
SSH_PRINCIPAL="${MOTLIE_VZ_SSH_PRINCIPAL:-$LOGIN_USER}"
SSH_CA_PUBKEY="${MOTLIE_VZ_SSH_CA_PUBKEY:-}"

if [[ -z "$SSH_CA_PUBKEY" ]]; then
  echo "MOTLIE_VZ_SSH_CA_PUBKEY must be set for v1.45 launches" >&2
  exit 1
fi

RUN_LOG="${RUN_LOG_OVERRIDE:-$ARTIFACTS_DIR/${GUEST_NAME}-run.log}"
RESULT_JSON="${RESULT_JSON_OVERRIDE:-$ARTIFACTS_DIR/${GUEST_NAME}-launch-result.json}"
SERIAL_LOG="${SERIAL_LOG_OVERRIDE:-$ARTIFACTS_DIR/${GUEST_NAME}-serial.log}"
SEED_IMAGE="${SEED_IMAGE_OVERRIDE:-$ARTIFACTS_DIR/${GUEST_NAME}-seed.dmg}"
VALIDATION_TOKEN="$RUN_VM_NAME"
RUNNER_PID_FILE="${RUNNER_PID_FILE_OVERRIDE:-$ARTIFACTS_DIR/${GUEST_NAME}-runner.pid}"
GUEST_IP_FILE="${GUEST_IP_FILE_OVERRIDE:-$ARTIFACTS_DIR/${GUEST_NAME}-ip.txt}"
EGRESS_HELPER_PID_FILE="${EGRESS_HELPER_PID_FILE_OVERRIDE:-$ARTIFACTS_DIR/${GUEST_NAME}-egress-helper.pid}"
EGRESS_SOCKET_PATH="${EGRESS_SOCKET_PATH_OVERRIDE:-/tmp/motlie-vmm-${GUEST_NAME}.egress.sock}"
EGRESS_LOG="${EGRESS_LOG_OVERRIDE:-$ARTIFACTS_DIR/${GUEST_NAME}-egress.log}"

mkdir -p "$ARTIFACTS_DIR"
for artifact_path in \
  "$RUN_LOG" \
  "$RESULT_JSON" \
  "$SERIAL_LOG" \
  "$SEED_IMAGE" \
  "$RUNNER_PID_FILE" \
  "$GUEST_IP_FILE" \
  "$EGRESS_HELPER_PID_FILE" \
  "$EGRESS_SOCKET_PATH" \
  "$EGRESS_LOG"
do
  ensure_parent_dir "$artifact_path"
done
for optional_artifact_path in \
  "$INTERACTIVE_READY_FILE" \
  "$VALIDATION_COMPLETE_FILE" \
  "$PHASES_LOG"
do
  if [[ -n "$optional_artifact_path" ]]; then
    ensure_parent_dir "$optional_artifact_path"
    rm -f "$optional_artifact_path"
  fi
done

if [[ -n "$CONTROL_READY_FILE" ]]; then
  mkdir -p "$(dirname "$CONTROL_READY_FILE")"
  rm -f "$CONTROL_READY_FILE"
fi
if [[ -n "$CONTROL_PORT_FILE" ]]; then
  mkdir -p "$(dirname "$CONTROL_PORT_FILE")"
  rm -f "$CONTROL_PORT_FILE"
fi

if [[ -n "$CONTROL_PORT_FILE" ]]; then
  CONTROL_PORT="$(python3 - <<'PY'
import socket
sock = socket.socket()
sock.bind(("127.0.0.1", 0))
port = sock.getsockname()[1]
sock.close()
print(port)
PY
)"
  printf '%s\n' "$CONTROL_PORT" >"$CONTROL_PORT_FILE"
fi

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

elapsed_since_start() {
  awk -v start="$SCRIPT_START_EPOCH" -v now="$EPOCHREALTIME" 'BEGIN { printf "%.3f", now - start }'
}

mark_phase() {
  local phase_name="$1"
  if [[ -n "$PHASES_LOG" ]]; then
    printf '%s\t%s\n' "$phase_name" "$(elapsed_since_start)" >>"$PHASES_LOG"
  fi
}

mark_interactive_ready() {
  INTERACTIVE_SECONDS="$(elapsed_since_start)"
  mark_phase "interactive-ready"
  if [[ -n "$CONTROL_READY_FILE" ]]; then
    : >"$CONTROL_READY_FILE"
  fi
  if [[ -n "$INTERACTIVE_READY_FILE" ]]; then
    : >"$INTERACTIVE_READY_FILE"
  fi
}

mark_validation_complete() {
  VALIDATION_SECONDS="$(elapsed_since_start)"
  mark_phase "validation-complete"
  if [[ -n "$VALIDATION_COMPLETE_FILE" ]]; then
    : >"$VALIDATION_COMPLETE_FILE"
  fi
}

write_skipped_validation_json() {
  local validation_json_path="$1"
  python3 - "$validation_json_path" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "w", encoding="utf-8") as fh:
    json.dump(
        {
            "status": "skipped",
            "reason": "inline validation disabled; run harness validate <guest> or a saved scenario for full certification",
        },
        fh,
        sort_keys=True,
    )
    fh.write("\n")
PY
}

write_result_json() {
  local validation_json_path="$1"
  local benchmark_json_path="${2:-}"
  local result_phase="${3:-interactive-ready}"
  python3 - "$RESULT_JSON" "$RUN_VM_NAME" "${INTERACTIVE_SECONDS:-}" "${VALIDATION_SECONDS:-}" "$SOCKET_PATH" "$RUNNER_PID" "$IP_ADDR" "$validation_json_path" "$KEEP_RUNNING" "$benchmark_json_path" "$INLINE_VALIDATION" "$result_phase" "$PHASES_LOG" <<'PY'
import json
import os
import sys

(
    path,
    vm_name,
    interactive_seconds,
    validation_seconds,
    socket_path,
    runner_pid,
    ip_addr,
    validation_json_path,
    keep_running,
    benchmark_json_path,
    inline_validation,
    result_phase,
    phase_log_path,
) = sys.argv[1:]


def maybe_float(value):
    return float(value) if value else None


validation_payload = None
if validation_json_path and os.path.exists(validation_json_path):
    with open(validation_json_path, "r", encoding="utf-8") as fh:
        validation_payload = json.load(fh)

benchmark_payload = None
if benchmark_json_path:
    with open(benchmark_json_path, "r", encoding="utf-8") as fh:
        benchmark_payload = json.load(fh)

phases = []
if phase_log_path and os.path.exists(phase_log_path):
    with open(phase_log_path, "r", encoding="utf-8") as fh:
        for line in fh:
            name, _, seconds = line.rstrip("\n").partition("\t")
            if name:
                phases.append({"phase": name, "seconds": maybe_float(seconds)})

validation_skipped = bool(
    isinstance(validation_payload, dict) and validation_payload.get("status") == "skipped"
)
keep_running_bool = keep_running == "1"
validation_complete_seconds = maybe_float(validation_seconds)
with open(path, "w", encoding="utf-8") as fh:
    json.dump(
        {
            "backend": "vz-vsock-runner",
            "vm_name": vm_name,
            "ready_phase": result_phase,
            "interactive_ready_seconds": maybe_float(interactive_seconds),
            "validation_complete_seconds": validation_complete_seconds,
            "boot_to_validation_seconds": validation_complete_seconds,
            "inline_validation": inline_validation == "1",
            "unix_socket_path": socket_path,
            "runner_pid": int(runner_pid) if keep_running_bool else None,
            "kept_running": keep_running_bool,
            "guest_ip": ip_addr,
            "phase_log": phase_log_path or None,
            "phases": phases,
            "validation": {
                "guest_validation_json": validation_json_path or None,
                "mounts_ready": True,
                "skipped": validation_skipped,
                "guest": validation_payload,
            },
            "benchmark": benchmark_payload,
        },
        fh,
        indent=2,
        sort_keys=True,
    )
    fh.write("\n")
PY
}

cleanup() {
  if [[ "$KEEP_RUNNING" -eq 0 ]]; then
    # @opus47-mac 2026-04-28 -- The previous body sent kill + rm without
    # waiting for the runner's stopWithCompletionHandler flush to drain,
    # then immediately rm -rf'd RUN_VM_DIR — racing the
    # VZDiskImageSynchronizationModeFsync window against disk deletion.
    # It also removed the PID file unconditionally, so a subsequent
    # launch-vz.sh invocation could not find a prior orphan via the
    # file path. Delegate to the same kill_stale_* helpers used at
    # script start so the signal/wait/SIGKILL-escalate/remove flow is
    # the single source of truth for the runner and egress lifecycle.
    kill_stale_runners
    kill_stale_egress_helpers
    rm -f "$EGRESS_SOCKET_PATH"
    if [[ "$REUSED_VM" -eq 0 ]] && [[ -n "$RUN_VM_DIR" && -d "$RUN_VM_DIR" ]]; then
      rm -rf "$RUN_VM_DIR"
    fi
    rm -f "$SEED_IMAGE"
    rm -rf "$SEED_DIR"
  fi
}

trap cleanup EXIT

# @opus47-mac 2026-04-26 -- kill_stale_runners treats $RUNNER_PID_FILE as the
# authoritative source for the previous run's PID, and uses the ps/awk grep
# only as a fallback for orphans whose PID file was lost. The previous
# implementation relied solely on the ps grep matching `vz-vsock-runner` plus
# the MAC + socket path as substrings, which was fragile under multi-guest
# (issue #215 / PR #212 finding #9): adjacent socket paths could collide as
# substrings, ps output truncation could drop entries, and a debugger-wrapped
# runner would not match the literal binary name. Same shape as
# kill_stale_egress_helpers above.
kill_stale_runners() {
  local stale_pids=()
  local stale_pid=""

  if [[ -f "$RUNNER_PID_FILE" ]]; then
    stale_pid="$(cat "$RUNNER_PID_FILE" 2>/dev/null || true)"
    if [[ -n "$stale_pid" ]]; then
      stale_pids+=("$stale_pid")
    fi
  fi

  while read -r stale_pid; do
    [[ -n "$stale_pid" ]] || continue
    stale_pids+=("$stale_pid")
  done < <(ps -Ao pid=,command= -ww | awk -v mac="$NET_MAC" -v sock="$SOCKET_PATH" '
    /vz-vsock-runner/ && index($0, mac) && index($0, sock) { print $1 }
  ')

  if [[ "${#stale_pids[@]}" -eq 0 ]]; then
    rm -f "$RUNNER_PID_FILE"
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

  printf '%s\n' "--- terminating stale Vz runners for $GUEST_NAME ---"
  kill "${unique_pids[@]}" >/dev/null 2>&1 || true

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
      rm -f "$RUNNER_PID_FILE"
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
    helper_bin="$REPO_ROOT/target/debug/examples/vz_egress_helper_v1_25"
    if [[ ! -x "$helper_bin" && "$SKIP_EGRESS_HELPER_BUILD" != "1" ]]; then
      cargo build -p motlie-vnet --example vz_egress_helper_v1_25 >/dev/null
    fi
  fi
  if [[ ! -x "$helper_bin" ]]; then
    cat >&2 <<EOF
missing prebuilt v1.45 Vz egress helper: $helper_bin

Build host artifacts before launch:
  cargo build -p motlie-vnet --example vz_egress_helper_v1_25

Set MOTLIE_VZ_SKIP_EGRESS_HELPER_BUILD=0 only for explicit developer rebuilds;
first-contact guest startup must not hide host tool builds.
EOF
    exit 1
  fi
  kill_stale_egress_helpers
  rm -f "$EGRESS_SOCKET_PATH" "$EGRESS_HELPER_PID_FILE"
  : > "$EGRESS_LOG"
  nohup "$helper_bin" \
    --socket-path "$EGRESS_SOCKET_PATH" \
    --guest-ip "$EGRESS_GUEST_IP" \
    --host-ip "$EGRESS_HOST_IP" \
    --dns-ip "$EGRESS_DNS_IP" \
    --netmask "$EGRESS_NETMASK" \
    --host-forward-tcp "127.0.0.1:${CONTROL_PORT}:22" \
    $( [[ "$EGRESS_HELPER_LOG_FRAMES" == "1" ]] && print -- "--log-frames" ) \
    >"$EGRESS_LOG" 2>&1 </dev/null &
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

# @opus47-mac 2026-04-28 -- The previous pre-block here read the PID file,
# sent SIGTERM, and removed the file before calling kill_stale_runners. That
# preempted the new PID-file authority path in kill_stale_runners (the file
# was gone by the time it ran) and skipped the wait/escalate loop, so a
# runner mid-flush via stopWithCompletionHandler could be left racing the
# script's cleanup/relaunch. kill_stale_runners now owns the full lifecycle:
# read/validate PID file, union with ps-grep candidates, signal, poll,
# SIGKILL escalate, and remove the file on success.
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

mark_phase "host-fixtures-ready"

guest_bash() {
  local remote_timeout="${MOTLIE_VZ_GUEST_BASH_TIMEOUT:--1}"
  local remote_script
  local pump_script
  remote_script="$(mktemp)"
  cat >"$remote_script"
  pump_script="$(mktemp)"
  cat >"$pump_script" <<EOF
#!/bin/sh
exec ssh -T -p ${CONTROL_PORT} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ${CONTROL_USER}@${CONTROL_HOST} 'bash -euo pipefail -s' < "$remote_script"
EOF
  chmod +x "$pump_script"
  expect <<EOF
set timeout ${remote_timeout}
set output ""
spawn "$pump_script"
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
  rm -f "$remote_script" "$pump_script"
}

guest_exec() {
  local remote_timeout="${MOTLIE_VZ_GUEST_EXEC_TIMEOUT:--1}"
  local remote_script
  remote_script="$(mktemp)"
  local remote_path="/home/${CONTROL_USER}/.motlie-vfs-exec.sh"
  cat >"$remote_script"
  expect <<EOF
set timeout ${remote_timeout}
spawn scp -P ${CONTROL_PORT} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$remote_script" ${CONTROL_USER}@${CONTROL_HOST}:${remote_path}
expect {
  "password:" {
    send "${CONTROL_PASSWORD}\r"
    exp_continue
  }
  eof {}
  timeout {
    puts stderr "scp timed out: remote guest exec script"
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
set timeout ${remote_timeout}
log_user 0
set output ""
spawn ssh -T -p ${CONTROL_PORT} -n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ${CONTROL_USER}@${CONTROL_HOST} "chmod 0700 ${remote_path} && bash -euo pipefail ${remote_path} </dev/null"
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
    puts stderr "ssh timed out: remote guest exec script"
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
set timeout ${remote_timeout}
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
import ctypes
import errno
import os
import shutil
import sys

src_disk, src_nvram, src_machine_id, out_dir = sys.argv[1:]
os.makedirs(out_dir, exist_ok=True)


def clonefile(src, dst):
    if sys.platform != "darwin":
        return False
    try:
        libc = ctypes.CDLL("libc.dylib", use_errno=True)
        call = libc.clonefile
    except AttributeError:
        return False
    call.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32]
    call.restype = ctypes.c_int
    if call(os.fsencode(src), os.fsencode(dst), 0) == 0:
        return True
    err = ctypes.get_errno()
    if err in (
        errno.ENOTSUP,
        errno.EXDEV,
        errno.EINVAL,
        errno.ENOSYS,
        getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
    ):
        return False
    raise OSError(err, os.strerror(err), dst)


def materialize_file(src, dst, chunk_size):
    tmp = dst + ".tmp-copy"
    if os.path.exists(tmp):
        os.remove(tmp)
    try:
        if not clonefile(src, tmp):
            with open(src, "rb") as rf, open(tmp, "wb") as wf:
                shutil.copyfileobj(rf, wf, length=chunk_size)
        os.replace(tmp, dst)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


for src, name in (
    (src_disk, "disk.img"),
    (src_nvram, "nvram.bin"),
):
    materialize_file(src, os.path.join(out_dir, name), 16 * 1024 * 1024)

machine_id_dst = os.path.join(out_dir, "machine-id.bin")
if src_machine_id and os.path.exists(src_machine_id):
    materialize_file(src_machine_id, machine_id_dst, 1024 * 1024)
else:
    with open(machine_id_dst, "wb"):
        pass
PY
  mark_phase "disk-materialized"
else
  mark_phase "disk-reused"
fi

# CONVERGENCE CONTRACT:
# Vz cannot yet inject the same CH overlay before boot, so the transitional
# path sends only the dynamic mount declaration over first-contact SSH. Service
# units, agent-state setup scripts, package state, and SSH CA config belong to
# the base image and must fail fast below if missing.
echo "--- rendering guest runtime config ---"
MOUNTS_B64="$(base64 < "$MOUNTS_FILE" | tr -d '\n')"
mark_phase "runtime-config-rendered"
if [[ "$USE_SEED_DISK" == "1" ]]; then
  echo "--- rendering diagnostic guest seed disk ---"
  mkdir -p "$SEED_DIR"
  cp "$MOUNTS_FILE" "$SEED_DIR/mounts.${GUEST_NAME}.yaml"
  hdiutil create -quiet -fs FAT32 -volname MOTLIESEED -srcfolder "$SEED_DIR" -ov -format UDRW "$SEED_IMAGE"
  mark_phase "seed-rendered"
fi

if [[ -n "$RUNNER_BIN_OVERRIDE" ]]; then
  RUNNER_BIN="$RUNNER_BIN_OVERRIDE"
elif [[ "$SKIP_RUNNER_BUILD" == "1" ]]; then
  RUNNER_BIN="$SCRIPT_DIR/artifacts/build/vz-vsock-runner"
else
  echo "--- building Vz virtio-socket helper ---"
  RUNNER_BIN="$($RUNNER_BUILD_SCRIPT)"
fi
if [[ ! -x "$RUNNER_BIN" ]]; then
  cat >&2 <<EOF
missing prebuilt v1.45 Vz runner: $RUNNER_BIN

Build and sign host artifacts before launch:
  $RUNNER_BUILD_SCRIPT
  $SCRIPT_DIR/sign-vz-runner.sh

Set MOTLIE_VZ_SKIP_RUNNER_BUILD=0 only for explicit developer rebuilds;
first-contact guest startup must not hide host tool builds.
EOF
  exit 1
fi
chmod +x "$RUNNER_BIN"
start_egress_helper
mark_phase "egress-helper-ready"

VM_DIR="$RUN_VM_DIR"
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
  --vsock-forward "5000:$SOCKET_PATH"
  --net-backend-socket "$EGRESS_SOCKET_PATH"
  --net-mac "$NET_MAC"
  --memory-mib 4096
  --cpu-count 4
)
if [[ "$USE_SEED_DISK" == "1" ]]; then
  RUNNER_ARGS+=(--seed-disk "$SEED_IMAGE")
fi
if [[ "$ENABLE_GUEST_SSH_VSOCK" == "1" ]]; then
  RUNNER_ARGS+=(--vsock-forward "2222:$SSH_VSOCK_SOCKET")
fi
nohup "$RUNNER_BIN" \
  "${RUNNER_ARGS[@]}" \
  >"$RUN_LOG" 2>&1 </dev/null &
RUNNER_PID="$!"
echo "$RUNNER_PID" > "$RUNNER_PID_FILE"
mark_phase "runner-started"

IP_ADDR="$EGRESS_GUEST_IP"
echo "$IP_ADDR" > "$GUEST_IP_FILE"

echo "--- waiting for guest SSH ---"
wait_for_guest_ssh || {
  echo "timed out waiting for guest SSH at ${CONTROL_HOST}:${CONTROL_PORT}" >&2
  print_failure_context
  exit 1
}
mark_phase "guest-ssh-password-ready"

echo "--- provisioning guest over native Vz userspace egress ---"
mark_phase "provision-start"
guest_bash <<EOF
set -x
motlie_guest_phase() {
  printf '[motlie-guest-phase] %s %s\n' "\$(date +%s.%N)" "\$1" >&2 || true
}
motlie_guest_phase provision-remote-start
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
motlie_guest_phase identity-remapped

set -x
SEED_MOUNT="/mnt/motlie-seed"
if [[ "$USE_SEED_DISK" == "1" ]]; then
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
  printf '${CONTROL_PASSWORD}\n' | sudo -S cp "\$SEED_MOUNT/mounts.${GUEST_NAME}.yaml" /tmp/mounts.${GUEST_NAME}.yaml
  printf '${CONTROL_PASSWORD}\n' | sudo -S umount -lf "\$SEED_MOUNT" >/dev/null 2>&1 || true
else
  cat <<'MOTLIE_MOUNTS_B64' >/tmp/mounts.${GUEST_NAME}.yaml.b64
${MOUNTS_B64}
MOTLIE_MOUNTS_B64
  base64 -d /tmp/mounts.${GUEST_NAME}.yaml.b64 >/tmp/mounts.${GUEST_NAME}.yaml
  rm -f /tmp/mounts.${GUEST_NAME}.yaml.b64
fi
motlie_guest_phase mounts-config-staged
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
printf '${CONTROL_PASSWORD}\n' | sudo -S usermod -aG sudo '$LOGIN_USER' || true
printf '%s\n' '${CONTROL_PASSWORD}' | sudo -S tee /etc/sudoers.d/90-motlie-demo >/dev/null <<SUDOERSEOF
alice ALL=(ALL) NOPASSWD:ALL
bob ALL=(ALL) NOPASSWD:ALL
$LOGIN_USER ALL=(ALL) NOPASSWD:ALL
SUDOERSEOF
printf '${CONTROL_PASSWORD}\n' | sudo -S chown root:root /etc/sudoers.d/90-motlie-demo
printf '${CONTROL_PASSWORD}\n' | sudo -S chmod 0440 /etc/sudoers.d/90-motlie-demo
printf '${CONTROL_PASSWORD}\n' | sudo -S install -d -m 0700 -o $UID_NUM -g $GID_NUM /home/$LOGIN_USER/.ssh
printf '${CONTROL_PASSWORD}\n' | sudo -S chown root:root /workspace
printf '${CONTROL_PASSWORD}\n' | sudo -S chmod 0755 /workspace
printf '${CONTROL_PASSWORD}\n' | sudo -S chown root:root /agent-state
printf '${CONTROL_PASSWORD}\n' | sudo -S chmod 0755 /agent-state
motlie_guest_phase user-and-base-dirs-ready
# CONVERGENCE CONTRACT:
# Do not install packages, fetch Rust toolchains, or compile guest binaries in
# the first SSH/proxy path. Those are image-ready responsibilities. If this
# fails, rebuild the v1.45 base image instead of hiding the long pole here.
typeset -a contract_missing=()
for cmd_pkg in \
  "cargo:cargo" \
  "rustc:rustc" \
  "cc:build-essential" \
  "pkg-config:pkg-config" \
  "curl:curl" \
  "tar:tar" \
  "gzip:gzip" \
  "npm:npm" \
  "codex:@openai/codex" \
  "claude:@anthropic-ai/claude-code"
do
  required_cmd="\${cmd_pkg%%:*}"
  required_pkg="\${cmd_pkg#*:}"
  if ! command -v "\$required_cmd" >/dev/null 2>&1; then
    contract_missing+=("command '\$required_cmd' from base package '\$required_pkg'")
  fi
done
if ! pkg-config --exists fuse3 >/dev/null 2>&1; then
  contract_missing+=("pkg-config fuse3 metadata from base package 'libfuse3-dev'")
fi
if [[ ! -s /usr/local/bin/motlie-vfs-guest || ! -x /usr/local/bin/motlie-vfs-guest ]]; then
  contract_missing+=("/usr/local/bin/motlie-vfs-guest executable from the base image")
fi
if [[ ! -s /usr/local/bin/motlie-agent-state-setup || ! -x /usr/local/bin/motlie-agent-state-setup ]]; then
  contract_missing+=("/usr/local/bin/motlie-agent-state-setup executable from the base image")
elif ! grep -q 'MOTLIE_CONVERGENCE_AGENT_STATE_SETUP_V3' /usr/local/bin/motlie-agent-state-setup; then
  contract_missing+=("/usr/local/bin/motlie-agent-state-setup is stale; rebuild the base image with the converged VFS-backed agent-state setup")
fi
if [[ ! -s /usr/local/bin/motlie-vmm-vsock-ssh-loop || ! -x /usr/local/bin/motlie-vmm-vsock-ssh-loop ]]; then
  contract_missing+=("/usr/local/bin/motlie-vmm-vsock-ssh-loop executable from the base image")
fi
if [[ ! -s /etc/systemd/system/motlie-vfs-guest.service ]]; then
  contract_missing+=("/etc/systemd/system/motlie-vfs-guest.service from the base image")
fi
if [[ ! -s /etc/systemd/system/motlie-agent-state.service ]]; then
  contract_missing+=("/etc/systemd/system/motlie-agent-state.service from the base image")
fi
if [[ ! -s /etc/systemd/system/motlie-vmm-vsock-ssh.service ]]; then
  contract_missing+=("/etc/systemd/system/motlie-vmm-vsock-ssh.service from the base image")
fi
if [[ ! -s /etc/profile.d/tmux-auto.sh ]]; then
  contract_missing+=("/etc/profile.d/tmux-auto.sh from the base image")
fi
if [[ ! -s /etc/profile.d/agent-state.sh ]]; then
  contract_missing+=("/etc/profile.d/agent-state.sh from the base image")
fi
if ! grep -q '^TrustedUserCAKeys /etc/ssh/ca/user_ca.pub$' /etc/ssh/sshd_config; then
  contract_missing+=("TrustedUserCAKeys /etc/ssh/ca/user_ca.pub baked into /etc/ssh/sshd_config")
fi
if ! grep -q '^AuthorizedPrincipalsFile /etc/ssh/auth_principals/%u$' /etc/ssh/sshd_config; then
  contract_missing+=("AuthorizedPrincipalsFile /etc/ssh/auth_principals/%u baked into /etc/ssh/sshd_config")
fi
if [[ ! -s /usr/local/bin/codex || ! -x /usr/local/bin/codex ]]; then
  contract_missing+=("/usr/local/bin/codex executable from the base image")
fi
if [[ ! -s /usr/local/bin/claude || ! -x /usr/local/bin/claude ]]; then
  contract_missing+=("/usr/local/bin/claude executable from the base image")
fi
if (( \${#contract_missing[@]} > 0 )); then
  echo "v1.45 Vz convergence contract violation: base image is missing required immutable content" >&2
  printf '  - %s\n' "\${contract_missing[@]}" >&2
  echo "Rebuild the v1.45 base image/seed path; first-contact SSH must not apt-get, npm, rustup, or cargo build." >&2
  exit 1
fi
motlie_guest_phase base-image-contract-ok
echo "[v1.45 launch] provisioning block complete"
# Dynamic first-contact writes are limited to CH-equivalent per-guest overlay
# inputs. Do not install service units or patch sshd_config here; those are
# image-ready responsibilities in the convergence contract.
printf '${CONTROL_PASSWORD}\n' | sudo -S install -D -m 0644 /tmp/mounts.${GUEST_NAME}.yaml /etc/motlie-vfs/mounts.yaml
cat <<'EOFCA' >/tmp/motlie-vmm-user-ca.pub
${SSH_CA_PUBKEY}
EOFCA
cat <<'EOFPRINCIPAL' >/tmp/motlie-vmm-principal
${SSH_PRINCIPAL}
EOFPRINCIPAL
printf '${CONTROL_PASSWORD}\n' | sudo -S install -D -m 0644 /tmp/motlie-vmm-user-ca.pub /etc/ssh/ca/user_ca.pub
printf '${CONTROL_PASSWORD}\n' | sudo -S install -D -m 0644 /tmp/motlie-vmm-principal /etc/ssh/auth_principals/${LOGIN_USER}
printf '${CONTROL_PASSWORD}\n' | sudo -S chown root:root /etc/ssh/ca/user_ca.pub /etc/ssh/auth_principals/${LOGIN_USER}
printf '${CONTROL_PASSWORD}\n' | sudo -S chmod 0644 /etc/ssh/ca/user_ca.pub /etc/ssh/auth_principals/${LOGIN_USER}
motlie_guest_phase guest-runtime-files-installed
printf '${CONTROL_PASSWORD}\n' | sudo -S systemctl restart motlie-vfs-guest.service
motlie_guest_phase vfs-service-restarted
mounts_ready=0
for _ in \$(seq 1 60); do
  if test -d /agent-state \
    && ls /agent-state >/dev/null 2>&1 \
    && test -d /home/${LOGIN_USER} \
    && ls /home/${LOGIN_USER} >/dev/null 2>&1; then
    mounts_ready=1
    break
  fi
  sleep 1
done
if [[ "\$mounts_ready" -ne 1 ]]; then
  echo "guest mounts did not become readable after motlie-vfs restart" >&2
  systemctl status motlie-vfs-guest.service --no-pager || true
  exit 1
fi
motlie_guest_phase mounts-readable
printf '${CONTROL_PASSWORD}\n' | sudo -S /usr/local/bin/motlie-agent-state-setup || true
motlie_guest_phase agent-state-setup-complete
EOF
mark_phase "provision-complete"

CONTROL_USER="$LOGIN_USER"
CONTROL_PASSWORD="testpass"

# This is the first-contact SSH gate used by auto-provisioning. It is
# intentionally earlier than full validation: the guest can now accept the
# requested principal and required VFS/agent-state mounts have become readable.
echo "--- v1.45 Vz interactive-ready ---"
mark_interactive_ready
echo "interactive_ready_seconds=${INTERACTIVE_SECONDS}"

if [[ "$INLINE_VALIDATION" != "1" ]]; then
  VALIDATION_JSON_PATH="$ARTIFACTS_DIR/${GUEST_NAME}-validation.json"
  write_skipped_validation_json "$VALIDATION_JSON_PATH"
  write_result_json "$VALIDATION_JSON_PATH" "" "interactive-ready"
  echo "--- result written ---"
  cat "$RESULT_JSON"
  exit 0
fi

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
LAST_VALIDATION_JSON=""
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
codex_dir = home_mount / '.codex'
claude_dir = home_mount / '.claude'
claude_code_dir = home_mount / '.config' / 'claude-code'

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
    and codex_dir.exists()
    and claude_dir.exists()
    and claude_code_dir.exists()
)

if not (result['workspace_mountpoint'] and result['home_mountpoint'] and result['agent_state_mountpoint'] and required_paths_exist):
    Path('$VALIDATION_REMOTE_JSON').write_text('', encoding='utf-8')
    raise SystemExit(0)

workspace_readme = workspace_readme_path.read_text(encoding='utf-8').strip()
agent_state_readme = agent_state_readme_path.read_text(encoding='utf-8').strip()
env_line = env_path.read_text(encoding='utf-8').strip()
authorized_keys = auth_path.read_text(encoding='utf-8').strip()

def command_ok(cmd):
    env = os.environ.copy()
    env["PATH"] = "/usr/local/bin:/usr/bin:/bin:" + env.get("PATH", "")
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30, env=env)
    return completed.returncode == 0, completed.stdout.strip(), completed.stderr.strip()

def first_executable(candidates):
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file() and path.stat().st_size > 0 and os.access(candidate, os.X_OK):
            return candidate
    return ''

dns_ok, dns_stdout, dns_stderr = command_ok(['getent', 'ahostsv4', 'example.com'])
curl_ok, curl_stdout, curl_stderr = command_ok(['curl', '-fsS', '--max-time', '20', 'https://example.com'])
codex_path = first_executable(['/usr/local/bin/codex', '/usr/bin/codex', '/bin/codex'])
claude_path = first_executable(['/usr/local/bin/claude', '/usr/bin/claude', '/bin/claude'])
codex_shell_ok, codex_shell_out, codex_shell_err = command_ok(['env', '-u', 'SSH_CONNECTION', 'TMUX=1', 'bash', '--noprofile', '--norc', '-lc', 'export PATH=/usr/local/bin:/usr/bin:/bin:$PATH; command -v codex'])
claude_shell_ok, claude_shell_out, claude_shell_err = command_ok(['env', '-u', 'SSH_CONNECTION', 'TMUX=1', 'bash', '--noprofile', '--norc', '-lc', 'export PATH=/usr/local/bin:/usr/bin:/bin:$PATH; command -v claude'])
codex_dir_ok = codex_dir.is_dir() and os.access(codex_dir, os.W_OK)
claude_dir_ok = claude_dir.is_dir() and os.access(claude_dir, os.W_OK)
claude_code_dir_ok = claude_code_dir.is_dir() and os.access(claude_code_dir, os.W_OK)
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
    'codex_dir_writable': codex_dir_ok,
    'claude_dir_writable': claude_dir_ok,
    'claude_code_dir_writable': claude_code_dir_ok,
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
    and codex_dir_ok
    and claude_dir_ok
    and claude_code_dir_ok
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
    LAST_VALIDATION_JSON="$VALIDATION_JSON"
    STATUS="$(printf '%s' "$VALIDATION_JSON" | python3 -c 'import json,sys; payload=json.loads(sys.stdin.read()); print(payload["status"])')"
    if [[ "$STATUS" == "ok" ]]; then
      printf '%s\n' "$VALIDATION_JSON" >"$ARTIFACTS_DIR/${GUEST_NAME}-validation.json"
      VALIDATION_OK=1
      break
    fi
  fi
  sleep 1
done

if [[ "$VALIDATION_OK" -ne 1 ]]; then
  echo "timed out waiting for guest validation sentinels" >&2
  if [[ -n "$LAST_VALIDATION_JSON" ]]; then
    echo "last guest validation mismatch:" >&2
    printf '%s\n' "$LAST_VALIDATION_JSON" >&2
  fi
  print_failure_context
  exit 1
fi

echo "--- waiting for package manager quiescence ---"
MOTLIE_VZ_GUEST_EXEC_TIMEOUT=150 guest_exec <<'EOF'
/bin/sh -lc '
cloud-init status --wait >/dev/null 2>&1 || true
idle=0
for _ in $(seq 1 120); do
  if ! pgrep -x apt >/dev/null 2>&1 \
    && ! pgrep -x apt-get >/dev/null 2>&1 \
    && ! pgrep -x dpkg >/dev/null 2>&1 \
    && ! pgrep -x unattended-upgr >/dev/null 2>&1 \
    && ! pgrep -x unattended-upgrade >/dev/null 2>&1; then
    idle=$((idle+1))
    if [ "$idle" -ge 5 ]; then
      echo PKG_IDLE_OK
      exit 0
    fi
  else
    idle=0
  fi
  sleep 1
done
exit 1
'
EOF

mark_validation_complete
echo "validation_complete_seconds=${VALIDATION_SECONDS}"

BENCHMARK_REMOTE_JSON="/tmp/motlie-vmm-benchmark.json"
BENCHMARK_LOCAL_JSON=""
if [[ "$BENCHMARK_APT" == "1" || "$BENCHMARK_HTTP" == "1" ]]; then
  echo "--- running in-guest benchmark phase ---"
  MOTLIE_VZ_GUEST_EXEC_TIMEOUT="$(( BENCHMARK_TIMEOUT_SECONDS + 30 ))" guest_exec <<EOF
export MOTLIE_BENCHMARK_TIMEOUT_SECONDS='${BENCHMARK_TIMEOUT_SECONDS}'
export MOTLIE_BENCHMARK_MTU='${BENCHMARK_MTU}'
export MOTLIE_BENCHMARK_APT='${BENCHMARK_APT}'
export MOTLIE_BENCHMARK_HTTP='${BENCHMARK_HTTP}'
export MOTLIE_BENCHMARK_HTTP_URL='${BENCHMARK_HTTP_URL}'
export MOTLIE_BENCHMARK_REMOTE_JSON='${BENCHMARK_REMOTE_JSON}'
python3 - <<'PY2'
import json
import os
import subprocess
import time
from pathlib import Path

result = {}
benchmark_timeout = int(os.environ["MOTLIE_BENCHMARK_TIMEOUT_SECONDS"])
benchmark_mtu = int(os.environ["MOTLIE_BENCHMARK_MTU"])
benchmark_apt = os.environ["MOTLIE_BENCHMARK_APT"] == "1"
benchmark_http = os.environ["MOTLIE_BENCHMARK_HTTP"] == "1"
benchmark_http_url = os.environ["MOTLIE_BENCHMARK_HTTP_URL"]
benchmark_remote_json = os.environ["MOTLIE_BENCHMARK_REMOTE_JSON"]

if benchmark_mtu > 0:
    iface_proc = subprocess.run(
        ["bash", "-lc", "ip route show default | awk '{print \$5; exit}'"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    iface = iface_proc.stdout.strip()
    result["mtu_clamp"] = {
        "requested_mtu": benchmark_mtu,
        "iface": iface,
        "detect_exit_code": iface_proc.returncode,
    }
    if iface:
        mtu_proc = subprocess.run(
            [
                "bash",
                "-lc",
                f"printf 'testpass\\n' | sudo -S ip link set dev {iface} mtu {benchmark_mtu}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        result["mtu_clamp"]["set_exit_code"] = mtu_proc.returncode
        result["mtu_clamp"]["set_output"] = "\n".join(mtu_proc.stdout.splitlines()[-20:])

if benchmark_apt:
    start = time.time()
    proc = subprocess.run(
        [
            "bash",
            "-lc",
            f"timeout --signal=TERM --kill-after=10 {benchmark_timeout}s bash -lc \"printf 'testpass\\\\n' | sudo -S apt-get update -o Acquire::Retries=0\"",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    apt_payload = {
        "elapsed_seconds": round(time.time() - start, 3),
        "exit_code": proc.returncode,
        "timed_out": proc.returncode == 124,
        "tail": "\n".join(proc.stdout.splitlines()[-40:]),
    }
    result["apt_update"] = apt_payload

if benchmark_http:
    proc = subprocess.run(
        [
            "bash",
            "-lc",
            f"timeout --signal=TERM --kill-after=10 {benchmark_timeout}s curl -4 -o /dev/null -sS -w 'time_total=%{{time_total}}\\nspeed_download=%{{speed_download}}\\nsize_download=%{{size_download}}\\nhttp_code=%{{http_code}}\\n' {benchmark_http_url!r}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = proc.stdout
    payload = {
        "exit_code": proc.returncode,
        "url": benchmark_http_url,
        "timed_out": proc.returncode == 124,
    }
    for line in output.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            payload[key] = value
    result["http_fetch"] = payload

Path(benchmark_remote_json).write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
PY2
EOF
  BENCHMARK_LOCAL_JSON="$(mktemp)"
  guest_fetch "$BENCHMARK_REMOTE_JSON" "$BENCHMARK_LOCAL_JSON"
  cp "$BENCHMARK_LOCAL_JSON" "$ARTIFACTS_DIR/${GUEST_NAME}-benchmark.json"
fi

write_result_json "$ARTIFACTS_DIR/${GUEST_NAME}-validation.json" "${BENCHMARK_LOCAL_JSON:-}" "validation-complete"

if [[ -n "$BENCHMARK_LOCAL_JSON" ]]; then
  rm -f "$BENCHMARK_LOCAL_JSON"
fi

echo "--- result written ---"
cat "$RESULT_JSON"
if [[ "$KEEP_RUNNING" -eq 1 ]]; then
  echo "=== guest left running through Apple virtio-socket ==="
else
  echo "=== guest validated and scheduled for cleanup ==="
fi
