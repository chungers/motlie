#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ARTIFACTS_DIR="$SCRIPT_DIR/artifacts"
BASE_VM_NAME="${MOTLIE_VZ_BASE_VM_NAME:-motlie-v1-05-ubuntu}"
RUN_VM_NAME="${MOTLIE_VZ_RUN_VM_NAME:-motlie-v1-05-ubuntu-run}"
SOURCE_IMAGE="${MOTLIE_VZ_SOURCE_IMAGE:-ghcr.io/cirruslabs/ubuntu@sha256:1e23e6fe5a6d3fb2089652229a09d71742617758b15aa311cecf1c05985d3021}"
RUN_LOG="$ARTIFACTS_DIR/tart-run.log"
RESULT_JSON="$ARTIFACTS_DIR/result.json"
TIMEOUT_SECONDS="${MOTLIE_VZ_TIMEOUT_SECONDS:-180}"

zmodload zsh/datetime

mkdir -p "$ARTIFACTS_DIR"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

require_cmd tart
require_cmd python3

local_vm_exists() {
  tart list 2>/dev/null | awk 'NR>1 {print $2}' | grep -Fx "$1" >/dev/null 2>&1
}

cleanup() {
  if local_vm_exists "$RUN_VM_NAME"; then
    tart stop "$RUN_VM_NAME" >/dev/null 2>&1 || true
    tart delete "$RUN_VM_NAME" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT

echo "=== Vz v1.05 Tart smoke ==="
echo "Base VM name: $BASE_VM_NAME"
echo "Run VM name:  $RUN_VM_NAME"
echo "Source image: $SOURCE_IMAGE"
echo "Artifacts:    $ARTIFACTS_DIR"

if ! local_vm_exists "$BASE_VM_NAME"; then
  echo "--- cloning remote image ---"
  tart clone "$SOURCE_IMAGE" "$BASE_VM_NAME"
else
  echo "--- reusing existing local base VM ---"
fi

if local_vm_exists "$RUN_VM_NAME"; then
  echo "--- deleting stale run VM clone ---"
  tart delete "$RUN_VM_NAME" >/dev/null
fi

echo "--- cloning throwaway run VM ---"
tart clone "$BASE_VM_NAME" "$RUN_VM_NAME" >/dev/null

echo "--- starting headless guest ---"
: > "$RUN_LOG"
START_EPOCH="$EPOCHREALTIME"
tart run --no-graphics "$RUN_VM_NAME" >"$RUN_LOG" 2>&1 &
RUN_PID="$!"

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
  echo "timed out waiting for tart ip after ${TIMEOUT_SECONDS}s" >&2
  cat "$RUN_LOG" >&2 || true
  exit 1
fi

READY_EPOCH="$EPOCHREALTIME"
BOOT_SECONDS="$(awk -v start="$START_EPOCH" -v ready="$READY_EPOCH" 'BEGIN { printf "%.3f", ready - start }')"

echo "--- guest is reachable ---"
echo "IP: $IP_ADDR"
echo "Boot-to-IP: ${BOOT_SECONDS}s"

UNAME_OUTPUT="$(tart exec "$RUN_VM_NAME" uname -a)"
ID_OUTPUT="$(tart exec "$RUN_VM_NAME" id)"

python3 - "$RESULT_JSON" "$BASE_VM_NAME" "$RUN_VM_NAME" "$SOURCE_IMAGE" "$IP_ADDR" "$BOOT_SECONDS" "$UNAME_OUTPUT" "$ID_OUTPUT" <<'PY'
import json
import sys

path, base_vm_name, run_vm_name, source_image, ip_addr, boot_seconds, uname_output, id_output = sys.argv[1:]
with open(path, "w", encoding="utf-8") as fh:
    json.dump(
        {
            "backend": "vz-tart",
            "base_vm_name": base_vm_name,
            "run_vm_name": run_vm_name,
            "source_image": source_image,
            "ip_addr": ip_addr,
            "boot_to_ip_seconds": float(boot_seconds),
            "guest_validation": {
                "uname_a": uname_output,
                "id": id_output,
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
tart stop "$RUN_VM_NAME" >/dev/null
tart delete "$RUN_VM_NAME" >/dev/null

echo "=== success ==="
