#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V15_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$V15_DIR/../../../.." && pwd)"

HARNESS_BIN="$REPO_ROOT/target/debug/examples/harness_v1_5"
CONTROL_BASE="${MOTLIE_HARNESS_TMPDIR:-/tmp}"
CONTROL_ROOT="$(mktemp -d "$CONTROL_BASE/motlie-vmm-v15-harness-auto-provision.XXXXXX")"
RUN_ROOT="$CONTROL_ROOT/runroot"
HARNESS_FIFO="$CONTROL_ROOT/harness.fifo"
HARNESS_LOG="$CONTROL_ROOT/harness.log"
HARNESS_PID=""
PROXY_PORT=""
AUTO_GUEST="joe"
MANUAL_GUEST_OFF="alice"
MANUAL_GUEST_ON="bob"
DISABLED_GUEST="zoe"
SSH_TIMEOUT_SECS="${SSH_TIMEOUT_SECS:-900}"
BASE_VM_DIR="${MOTLIE_VZ_BASE_VM_DIR:-$V15_DIR/artifacts/source-base.vm}"

cleanup() {
    if [ -n "$HARNESS_PID" ]; then
        kill "$HARNESS_PID" >/dev/null 2>&1 || true
        wait "$HARNESS_PID" >/dev/null 2>&1 || true
    fi
    exec 3>&- || true
    rm -rf "$CONTROL_ROOT"
}

capture_log() {
    cat "$HARNESS_LOG"
}

wait_for_pattern() {
    local pattern="$1"
    local timeout_secs="${2:-60}"
    local deadline=$((SECONDS + timeout_secs))
    while [ "$SECONDS" -lt "$deadline" ]; do
        if capture_log | grep -F "$pattern" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done

    echo "timed out waiting for pattern: $pattern" >&2
    echo "--- harness log ---" >&2
    capture_log >&2 || true
    return 1
}

extract_proxy_port() {
    PROXY_PORT="$(
        grep -oE 'proxy=ssh://localhost:[0-9]+' "$HARNESS_LOG" |
            tail -n1 |
            sed 's/.*://'
    )"
    if [ -z "$PROXY_PORT" ]; then
        echo "failed to parse proxy port from harness log" >&2
        capture_log >&2 || true
        exit 1
    fi
}

send_harness() {
    printf '%s\n' "$1" >&3
}

guest_pid_from_status() {
    local guest="$1"
    grep -E "${guest} pid=Some\\([0-9]+\\)" "$HARNESS_LOG" | tail -n1 | sed -E 's/.*pid=Some\(([0-9]+)\).*/\1/'
}

wait_for_guest_status() {
    local guest="$1"
    local timeout_secs="${2:-60}"
    local deadline=$((SECONDS + timeout_secs))
    while [ "$SECONDS" -lt "$deadline" ]; do
        if guest_pid_from_status "$guest" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done

    echo "timed out waiting for guest status for ${guest}" >&2
    echo "--- harness log ---" >&2
    capture_log >&2 || true
    return 1
}

record_status() {
    local guest="$1"
    send_harness "status"
    wait_for_guest_status "$guest" 30
    guest_pid_from_status "$guest"
}

run_uname_probe() {
    local guest="$1"
    local output_file="$2"
    local error_file="$3"

    run_with_timeout \
        ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=30 -o ServerAliveInterval=5 -o ServerAliveCountMax=3 -p "$PROXY_PORT" "$guest@localhost" uname -s \
        >"$output_file" 2>"$error_file"
}

run_with_timeout() {
    if command -v timeout >/dev/null 2>&1; then
        timeout "$SSH_TIMEOUT_SECS" "$@"
    elif command -v gtimeout >/dev/null 2>&1; then
        gtimeout "$SSH_TIMEOUT_SECS" "$@"
    else
        "$@"
    fi
}

trap cleanup EXIT

cd "$REPO_ROOT"

test -f "$BASE_VM_DIR/disk.img" || {
    echo "missing $BASE_VM_DIR/disk.img; run $V15_DIR/build-guest.sh first or set MOTLIE_VZ_BASE_VM_DIR" >&2
    exit 1
}
test -f "$BASE_VM_DIR/nvram.bin" || {
    echo "missing $BASE_VM_DIR/nvram.bin; run $V15_DIR/build-guest.sh first or set MOTLIE_VZ_BASE_VM_DIR" >&2
    exit 1
}

cargo build -p motlie-vmm --example harness_v1_5 >/dev/null

mkfifo "$HARNESS_FIFO"
mkdir -p "$RUN_ROOT"
"$HARNESS_BIN" shell --backend vz --root "$RUN_ROOT" <"$HARNESS_FIFO" >"$HARNESS_LOG" 2>&1 &
HARNESS_PID="$!"
exec 3>"$HARNESS_FIFO"

wait_for_pattern "proxy=ssh://localhost:" 20
wait_for_pattern "v15-harness>" 20
extract_proxy_port

send_harness "auto-provision status"
wait_for_pattern "auto-provision=off" 10

send_harness "boot ${MANUAL_GUEST_OFF}"
wait_for_pattern "ok: booted ${MANUAL_GUEST_OFF}" 900

MANUAL_OFF_OUTPUT="$CONTROL_ROOT/${MANUAL_GUEST_OFF}.out"
MANUAL_OFF_ERROR="$CONTROL_ROOT/${MANUAL_GUEST_OFF}.err"
run_uname_probe "$MANUAL_GUEST_OFF" "$MANUAL_OFF_OUTPUT" "$MANUAL_OFF_ERROR"
grep -Fx "Linux" "$MANUAL_OFF_OUTPUT" >/dev/null

send_harness "auto-provision on"
wait_for_pattern "ok: auto-provision enabled" 10
send_harness "auto-provision status"
wait_for_pattern "auto-provision=on" 10

send_harness "boot ${MANUAL_GUEST_ON}"
wait_for_pattern "ok: booted ${MANUAL_GUEST_ON}" 900

MANUAL_ON_OUTPUT="$CONTROL_ROOT/${MANUAL_GUEST_ON}.out"
MANUAL_ON_ERROR="$CONTROL_ROOT/${MANUAL_GUEST_ON}.err"
run_uname_probe "$MANUAL_GUEST_ON" "$MANUAL_ON_OUTPUT" "$MANUAL_ON_ERROR"
grep -Fx "Linux" "$MANUAL_ON_OUTPUT" >/dev/null

FIRST_OUTPUT="$CONTROL_ROOT/${AUTO_GUEST}-first.out"
FIRST_ERROR="$CONTROL_ROOT/${AUTO_GUEST}-first.err"
SECOND_OUTPUT="$CONTROL_ROOT/${AUTO_GUEST}-second.out"
SECOND_ERROR="$CONTROL_ROOT/${AUTO_GUEST}-second.err"

run_uname_probe "$AUTO_GUEST" "$FIRST_OUTPUT" "$FIRST_ERROR"
grep -Fx "Linux" "$FIRST_OUTPUT" >/dev/null

FIRST_PID="$(record_status "$AUTO_GUEST")"
if [ -z "$FIRST_PID" ]; then
    echo "failed to capture first guest pid from harness status" >&2
    capture_log >&2 || true
    exit 1
fi

run_uname_probe "$AUTO_GUEST" "$SECOND_OUTPUT" "$SECOND_ERROR"
grep -Fx "Linux" "$SECOND_OUTPUT" >/dev/null

SECOND_PID="$(record_status "$AUTO_GUEST")"
if [ -z "$SECOND_PID" ]; then
    echo "failed to capture second guest pid from harness status" >&2
    capture_log >&2 || true
    exit 1
fi

if [ "$FIRST_PID" != "$SECOND_PID" ]; then
    echo "expected guest reuse for ${AUTO_GUEST}, but pid changed: ${FIRST_PID} -> ${SECOND_PID}" >&2
    echo "--- harness log ---" >&2
    capture_log >&2 || true
    exit 1
fi

send_harness "auto-provision off"
wait_for_pattern "ok: auto-provision disabled" 10
send_harness "auto-provision status"
wait_for_pattern "auto-provision=off" 10

DISABLED_OUTPUT="$CONTROL_ROOT/${DISABLED_GUEST}.out"
DISABLED_ERROR="$CONTROL_ROOT/${DISABLED_GUEST}.err"
if run_uname_probe "$DISABLED_GUEST" "$DISABLED_OUTPUT" "$DISABLED_ERROR"; then
    echo "expected SSH as ${DISABLED_GUEST} to fail with auto-provision disabled" >&2
    echo "--- ssh stdout ---" >&2
    cat "$DISABLED_OUTPUT" >&2 || true
    echo "--- ssh stderr ---" >&2
    cat "$DISABLED_ERROR" >&2 || true
    exit 1
fi

if grep -E "${DISABLED_GUEST} pid=Some\\([0-9]+\\)" "$HARNESS_LOG" >/dev/null 2>&1; then
    echo "unexpected guest provisioning for ${DISABLED_GUEST} while auto-provision was disabled" >&2
    echo "--- harness log ---" >&2
    capture_log >&2 || true
    exit 1
fi

send_harness "quit"
sleep 1

echo "v1.5 harness auto-provision smoke passed (manual_off=${MANUAL_GUEST_OFF} manual_on=${MANUAL_GUEST_ON} auto_guest=${AUTO_GUEST} pid=${FIRST_PID} proxy_port=${PROXY_PORT})"
