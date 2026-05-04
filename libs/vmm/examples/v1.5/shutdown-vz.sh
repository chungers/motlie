#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ARTIFACTS_DIR="${MOTLIE_VZ_ARTIFACTS_DIR:-$SCRIPT_DIR/artifacts}"

GUEST_NAME="alice"
RUN_VM_NAME=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --guest) GUEST_NAME="$2"; shift 2 ;;
    --vm-name) RUN_VM_NAME="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

RUN_VM_NAME="${RUN_VM_NAME:-motlie-v1-5-${GUEST_NAME}-iter}"

RUNNER_PID_FILE="${MOTLIE_VZ_RUNNER_PID_FILE:-$ARTIFACTS_DIR/${GUEST_NAME}-runner.pid}"
RUN_VM_DIR="$ARTIFACTS_DIR/${RUN_VM_NAME}.vm"

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
# v1.5 convergence contract: VZ egress is owned by the VMM/harness runtime,
# not by launch/shutdown shell helpers. Only the Apple VZ runner is external.
rm -rf "$RUN_VM_DIR"
