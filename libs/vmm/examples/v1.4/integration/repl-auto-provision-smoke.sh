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
GUEST="joe"
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

guest_pid_from_status() {
    grep -E "${GUEST} pid=Some\\([0-9]+\\)" "$REPL_LOG" | tail -n1 | sed -E 's/.*pid=Some\(([0-9]+)\).*/\1/'
}

wait_for_guest_status() {
    local timeout_secs="${1:-60}"
    local deadline=$((SECONDS + timeout_secs))
    while [ "$SECONDS" -lt "$deadline" ]; do
        if guest_pid_from_status >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done

    echo "timed out waiting for guest status for ${GUEST}" >&2
    echo "--- repl log ---" >&2
    capture_log >&2 || true
    return 1
}

record_status() {
    printf 'status\n' >&3
    wait_for_guest_status 30
    guest_pid_from_status
}

run_uname_probe() {
    local output_file="$1"

    timeout "$SSH_TIMEOUT_SECS" \
        ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$PROXY_PORT" "$GUEST@localhost" uname -s \
        >"$output_file"
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

FIRST_OUTPUT="$CONTROL_ROOT/${GUEST}-first.out"
SECOND_OUTPUT="$CONTROL_ROOT/${GUEST}-second.out"

run_uname_probe "$FIRST_OUTPUT"
grep -Fx "Linux" "$FIRST_OUTPUT" >/dev/null

FIRST_PID="$(record_status)"
if [ -z "$FIRST_PID" ]; then
    echo "failed to capture first guest pid from repl status" >&2
    capture_log >&2 || true
    exit 1
fi

run_uname_probe "$SECOND_OUTPUT"
grep -Fx "Linux" "$SECOND_OUTPUT" >/dev/null

SECOND_PID="$(record_status)"
if [ -z "$SECOND_PID" ]; then
    echo "failed to capture second guest pid from repl status" >&2
    capture_log >&2 || true
    exit 1
fi

if [ "$FIRST_PID" != "$SECOND_PID" ]; then
    echo "expected guest reuse for ${GUEST}, but pid changed: ${FIRST_PID} -> ${SECOND_PID}" >&2
    echo "--- repl log ---" >&2
    capture_log >&2 || true
    exit 1
fi

printf 'shutdown %s\n' "$GUEST" >&3
wait_for_pattern "ok: shutdown ${GUEST}" 30

printf 'quit\n' >&3
sleep 1

echo "v1.4 repl auto-provision smoke passed (guest=${GUEST} pid=${FIRST_PID} proxy_port=${PROXY_PORT})"
