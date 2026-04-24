#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ARTIFACTS_DIR="$SCRIPT_DIR/artifacts"

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
  alice) RUN_VM_NAME="${RUN_VM_NAME:-motlie-v1-45-alice-iter}" ;;
  bob) RUN_VM_NAME="${RUN_VM_NAME:-motlie-v1-45-bob-iter}" ;;
  *) echo "guest must be alice or bob" >&2; exit 1 ;;
esac

RUNNER_PID_FILE="$ARTIFACTS_DIR/${GUEST_NAME}-runner.pid"
EGRESS_HELPER_PID_FILE="$ARTIFACTS_DIR/${GUEST_NAME}-egress-helper.pid"
RUN_VM_DIR="$ARTIFACTS_DIR/${RUN_VM_NAME}.vm"
EGRESS_SOCKET_PATH="/tmp/motlie-vmm-${GUEST_NAME}.egress.sock"

terminate_pid_file() {
  local pid_file="$1"
  if [[ ! -f "$pid_file" ]]; then
    return 0
  fi
  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
    kill "$pid" >/dev/null 2>&1 || true
    for _ in $(seq 1 20); do
      if ! kill -0 "$pid" >/dev/null 2>&1; then
        break
      fi
      sleep 0.25
    done
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill -9 "$pid" >/dev/null 2>&1 || true
    fi
  fi
  rm -f "$pid_file"
}

terminate_pid_file "$RUNNER_PID_FILE"
terminate_pid_file "$EGRESS_HELPER_PID_FILE"
rm -f "$EGRESS_SOCKET_PATH"
rm -rf "$RUN_VM_DIR"
