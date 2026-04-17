#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ARTIFACTS_DIR="$SCRIPT_DIR/artifacts"
VM_NAME="${MOTLIE_VZ_VM_NAME:-motlie-v1-05-ubuntu}"
SOURCE_IMAGE="${MOTLIE_VZ_SOURCE_IMAGE:-ghcr.io/cirruslabs/ubuntu:latest}"
RUN_LOG="$ARTIFACTS_DIR/tart-run.log"
RESULT_JSON="$ARTIFACTS_DIR/result.json"
TIMEOUT_SECONDS="${MOTLIE_VZ_TIMEOUT_SECONDS:-180}"

mkdir -p "$ARTIFACTS_DIR"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

require_cmd tart
require_cmd python3

cleanup() {
  if tart list 2>/dev/null | awk 'NR>1 {print $1}' | grep -Fx "$VM_NAME" >/dev/null 2>&1; then
    tart stop "$VM_NAME" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT

echo "=== Vz v1.05 Tart smoke ==="
echo "VM name:      $VM_NAME"
echo "Source image: $SOURCE_IMAGE"
echo "Artifacts:    $ARTIFACTS_DIR"

if ! tart list 2>/dev/null | awk 'NR>1 {print $1}' | grep -Fx "$VM_NAME" >/dev/null 2>&1; then
  echo "--- cloning remote image ---"
  tart clone "$SOURCE_IMAGE" "$VM_NAME"
else
  echo "--- reusing existing local VM ---"
fi

echo "--- starting headless guest ---"
: > "$RUN_LOG"
START_EPOCH="$(python3 - <<'PY'
import time
print(f"{time.time():.6f}")
PY
)"
/bin/zsh -lc "tart run --no-graphics '$VM_NAME'" >"$RUN_LOG" 2>&1 &
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

  IP_ADDR="$(tart ip "$VM_NAME" 2>/dev/null || true)"
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

READY_EPOCH="$(python3 - <<'PY'
import time
print(f"{time.time():.6f}")
PY
)"
BOOT_SECONDS="$(python3 - "$START_EPOCH" "$READY_EPOCH" <<'PY'
import sys
start = float(sys.argv[1])
ready = float(sys.argv[2])
print(f"{ready - start:.3f}")
PY
)"

echo "--- guest is reachable ---"
echo "IP: $IP_ADDR"
echo "Boot-to-IP: ${BOOT_SECONDS}s"

UNAME_OUTPUT="$(tart exec "$VM_NAME" uname -a)"
ID_OUTPUT="$(tart exec "$VM_NAME" id)"

python3 - "$RESULT_JSON" "$VM_NAME" "$SOURCE_IMAGE" "$IP_ADDR" "$BOOT_SECONDS" "$UNAME_OUTPUT" "$ID_OUTPUT" <<'PY'
import json
import sys

path, vm_name, source_image, ip_addr, boot_seconds, uname_output, id_output = sys.argv[1:]
with open(path, "w", encoding="utf-8") as fh:
    json.dump(
        {
            "backend": "vz-tart",
            "vm_name": vm_name,
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
tart stop "$VM_NAME" >/dev/null

echo "=== success ==="
