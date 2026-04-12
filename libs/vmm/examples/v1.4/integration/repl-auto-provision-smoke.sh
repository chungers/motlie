#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V14_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$V14_DIR/../../../.." && pwd)"

REPL_BIN="$REPO_ROOT/target/debug/examples/repl_host_v1_4"
CONTROL_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/motlie-vmm-v14-repl-auto-provision.XXXXXX")"
RUN_ROOT="$CONTROL_ROOT/runroot"
REPL_FIFO="$CONTROL_ROOT/repl.fifo"
REPL_LOG="$CONTROL_ROOT/repl.log"
REPL_PID=""
PROXY_PORT=""
AUTO_GUEST="joe"
MANUAL_GUEST_OFF="alice"
MANUAL_GUEST_ON="bob"
DISABLED_GUEST="zoe"
SSH_TIMEOUT_SECS="${SSH_TIMEOUT_SECS:-180}"

cleanup() {
    if [ -n "$REPL_PID" ]; then
        kill "$REPL_PID" >/dev/null 2>&1 || true
        wait "$REPL_PID" >/dev/null 2>&1 || true
    fi
    exec 3>&- || true
    rm -rf "$CONTROL_ROOT"
}

capture_log() {
    cat "$REPL_LOG"
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
    echo "--- repl log ---" >&2
    capture_log >&2 || true
    return 1
}

extract_proxy_port() {
    PROXY_PORT="$(
        grep -oE 'SSH proxy: listening on 127\.0\.0\.1:[0-9]+' "$REPL_LOG" |
            tail -n1 |
            sed 's/.*://'
    )"
    if [ -z "$PROXY_PORT" ]; then
        echo "failed to parse proxy port from repl log" >&2
        capture_log >&2 || true
        exit 1
    fi
}

send_repl() {
    printf '%s\n' "$1" >&3
}

guest_pid_from_status() {
    local guest="$1"
    grep -E "${guest} pid=Some\\([0-9]+\\)" "$REPL_LOG" | tail -n1 | sed -E 's/.*pid=Some\(([0-9]+)\).*/\1/'
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
    echo "--- repl log ---" >&2
    capture_log >&2 || true
    return 1
}

record_status() {
    local guest="$1"
    send_repl "status"
    wait_for_guest_status "$guest" 30
    guest_pid_from_status "$guest"
}

run_uname_probe() {
    local guest="$1"
    local output_file="$2"
    local error_file="$3"

    timeout "$SSH_TIMEOUT_SECS" \
        ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$PROXY_PORT" "$guest@localhost" uname -s \
        >"$output_file" 2>"$error_file"
}

trap cleanup EXIT

cd "$REPO_ROOT"

test -f "$V14_DIR/artifacts/base/rootfs.squashfs" || {
    echo "missing $V14_DIR/artifacts/base/rootfs.squashfs; run $V14_DIR/build-guest.sh first" >&2
    exit 1
}
test -f "$V14_DIR/artifacts/base/Image" || {
    echo "missing $V14_DIR/artifacts/base/Image; run $V14_DIR/build-guest.sh first" >&2
    exit 1
}

cargo build -p motlie-vmm --example repl_host_v1_4 >/dev/null

mkfifo "$REPL_FIFO"
mkdir -p "$RUN_ROOT"
"$REPL_BIN" --root "$RUN_ROOT" <"$REPL_FIFO" >"$REPL_LOG" 2>&1 &
REPL_PID="$!"
exec 3>"$REPL_FIFO"

wait_for_pattern "SSH proxy: listening on 127.0.0.1:" 20
wait_for_pattern "v14>" 20
extract_proxy_port

send_repl "auto-provision status"
wait_for_pattern "auto-provision=off" 10

send_repl "boot ${MANUAL_GUEST_OFF}"
wait_for_pattern "ok: booted ${MANUAL_GUEST_OFF}" 60

MANUAL_OFF_OUTPUT="$CONTROL_ROOT/${MANUAL_GUEST_OFF}.out"
MANUAL_OFF_ERROR="$CONTROL_ROOT/${MANUAL_GUEST_OFF}.err"
run_uname_probe "$MANUAL_GUEST_OFF" "$MANUAL_OFF_OUTPUT" "$MANUAL_OFF_ERROR"
grep -Fx "Linux" "$MANUAL_OFF_OUTPUT" >/dev/null

send_repl "auto-provision on"
wait_for_pattern "ok: auto-provision enabled" 10
send_repl "auto-provision status"
wait_for_pattern "auto-provision=on" 10

send_repl "boot ${MANUAL_GUEST_ON}"
wait_for_pattern "ok: booted ${MANUAL_GUEST_ON}" 60

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
    echo "failed to capture first guest pid from repl status" >&2
    capture_log >&2 || true
    exit 1
fi

run_uname_probe "$AUTO_GUEST" "$SECOND_OUTPUT" "$SECOND_ERROR"
grep -Fx "Linux" "$SECOND_OUTPUT" >/dev/null

SECOND_PID="$(record_status "$AUTO_GUEST")"
if [ -z "$SECOND_PID" ]; then
    echo "failed to capture second guest pid from repl status" >&2
    capture_log >&2 || true
    exit 1
fi

if [ "$FIRST_PID" != "$SECOND_PID" ]; then
    echo "expected guest reuse for ${AUTO_GUEST}, but pid changed: ${FIRST_PID} -> ${SECOND_PID}" >&2
    echo "--- repl log ---" >&2
    capture_log >&2 || true
    exit 1
fi

send_repl "auto-provision off"
wait_for_pattern "ok: auto-provision disabled" 10
send_repl "auto-provision status"
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

if grep -E "${DISABLED_GUEST} pid=Some\\([0-9]+\\)" "$REPL_LOG" >/dev/null 2>&1; then
    echo "unexpected guest provisioning for ${DISABLED_GUEST} while auto-provision was disabled" >&2
    echo "--- repl log ---" >&2
    capture_log >&2 || true
    exit 1
fi

send_repl "quit"
sleep 1

echo "v1.4 repl auto-provision smoke passed (manual_off=${MANUAL_GUEST_OFF} manual_on=${MANUAL_GUEST_ON} auto_guest=${AUTO_GUEST} pid=${FIRST_PID} proxy_port=${PROXY_PORT})"
