#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V14_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$V14_DIR/../../../.." && pwd)"

REPL_BIN="$REPO_ROOT/target/debug/examples/repl_host_v1_4"
CONTROL_ROOT="${1:-$(mktemp -d "${TMPDIR:-/tmp}/motlie-v14-headless.XXXXXX")}"
RUN_ROOT="$CONTROL_ROOT/live"
REPL_FIFO="$CONTROL_ROOT/repl.fifo"
REPL_LOG="$CONTROL_ROOT/repl.log"
REPL_PID=""
PROXY_PORT=""
SSH_TIMEOUT_SECS="${SSH_TIMEOUT_SECS:-180}"
SSH_CONNECT_TIMEOUT_SECS="${SSH_CONNECT_TIMEOUT_SECS:-15}"
users=(alice bob jane joe zoe)

declare -A BOOT_IDS

cleanup() {
    if [ -n "${REPL_PID:-}" ]; then
        kill "$REPL_PID" >/dev/null 2>&1 || true
        wait "$REPL_PID" >/dev/null 2>&1 || true
    fi
    exec 3>&- || true
}

capture_log() {
    cat "$REPL_LOG"
}

wait_for_pattern() {
    local pattern="$1"
    local timeout_secs="${2:-20}"
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
    PROXY_PORT="$({
        grep -oE 'SSH proxy: listening on 127\.0\.0\.1:[0-9]+' "$REPL_LOG" || true
    } | tail -n1 | sed 's/.*://')"
    if [ -z "$PROXY_PORT" ]; then
        echo "failed to parse proxy port from repl log" >&2
        capture_log >&2 || true
        exit 1
    fi
}

send_repl() {
    printf '%s\n' "$1" >&3
}

run_remote_probe() {
    local user="$1"
    timeout "$SSH_TIMEOUT_SECS" \
        ssh \
            -o StrictHostKeyChecking=no \
            -o UserKnownHostsFile=/dev/null \
            -o ConnectTimeout="$SSH_CONNECT_TIMEOUT_SECS" \
            -p "$PROXY_PORT" \
            "$user@localhost" \
            'whoami; cat /proc/sys/kernel/random/boot_id; uname -s'
}

probe_user() {
    local user="$1"
    local stdout_file="$CONTROL_ROOT/ssh-${user}.out"
    local stderr_file="$CONTROL_ROOT/ssh-${user}.err"
    local status

    run_remote_probe "$user" >"$stdout_file" 2>"$stderr_file"
    status=$?
    if [ "$status" -eq 0 ]; then
        cat "$stdout_file"
        rm -f "$stdout_file" "$stderr_file"
        return 0
    fi

    echo "FAILED: ssh probe for $user exited with status $status" >&2
    cat "$stderr_file" >&2 || true
    echo "--- repl log ---" >&2
    capture_log >&2 || true
    rm -f "$stdout_file" "$stderr_file"
    return $status
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

test -x "$REPL_BIN" || {
    echo "missing repl binary at $REPL_BIN" >&2
    exit 1
}

mkdir -p "$CONTROL_ROOT" "$RUN_ROOT"
rm -f "$REPL_FIFO"
mkfifo "$REPL_FIFO"
"$REPL_BIN" --root "$RUN_ROOT" <"$REPL_FIFO" >"$REPL_LOG" 2>&1 &
REPL_PID="$!"
exec 3>"$REPL_FIFO"

wait_for_pattern "SSH proxy: listening on 127.0.0.1:" 20
wait_for_pattern "v14>" 20
extract_proxy_port

send_repl "auto-provision on"
wait_for_pattern "ok: auto-provision enabled" 20

exec 3>&-
wait_for_pattern "notice: stdin closed; continuing headless." 20

if ! kill -0 "$REPL_PID" 2>/dev/null; then
    echo "FAILED: repl_host_v1_4 exited after stdin closed" >&2
    capture_log >&2 || true
    exit 1
fi

for user in "${users[@]}"; do
    if ! result="$(probe_user "$user")"; then
        exit 1
    fi
    who="$(printf '%s\n' "$result" | sed -n '1p')"
    boot_id="$(printf '%s\n' "$result" | sed -n '2p')"
    os_name="$(printf '%s\n' "$result" | sed -n '3p')"
    if [ "$who" != "$user" ] || [ -z "$boot_id" ] || [ "$os_name" != "Linux" ]; then
        echo "FAILED: first login for $user" >&2
        printf '%s\n' "$result" >&2
        capture_log >&2 || true
        exit 1
    fi
    BOOT_IDS["$user"]="$boot_id"
done

for round in 1 2 3; do
    for user in "${users[@]}"; do
        if ! result="$(probe_user "$user")"; then
            exit 1
        fi
        who="$(printf '%s\n' "$result" | sed -n '1p')"
        boot_id="$(printf '%s\n' "$result" | sed -n '2p')"
        os_name="$(printf '%s\n' "$result" | sed -n '3p')"
        if [ "$who" != "$user" ] || [ "$boot_id" != "${BOOT_IDS[$user]}" ] || [ "$os_name" != "Linux" ]; then
            echo "FAILED: reconnect round $round for $user" >&2
            printf 'expected boot_id=%s\n' "${BOOT_IDS[$user]}" >&2
            printf '%s\n' "$result" >&2
            capture_log >&2 || true
            exit 1
        fi
    done
done

kill -TERM "$REPL_PID"
wait "$REPL_PID"
REPL_PID=""

echo "PASS proxy_port=${PROXY_PORT} root=${CONTROL_ROOT} users=${users[*]}"
