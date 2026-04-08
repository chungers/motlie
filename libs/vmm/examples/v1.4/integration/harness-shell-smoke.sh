#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V14_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$V14_DIR/../../../.." && pwd)"

HARNESS_BIN="$REPO_ROOT/target/debug/examples/harness_v1_4"
HARNESS_SCRIPT="$V14_DIR/setup-multiguest.harness"
CONTROL_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/motlie-vmm-v14-harness-shell-smoke.XXXXXX")"
RUN_ROOT="$CONTROL_ROOT/runroot"
HARNESS_FIFO="$CONTROL_ROOT/harness.fifo"
HARNESS_LOG="$CONTROL_ROOT/harness.log"
HARNESS_PID=""
PROXY_PORT=""
NAMESPACE=""

cleanup() {
    if [ -n "$HARNESS_PID" ]; then
        kill "$HARNESS_PID" >/dev/null 2>&1 || true
        wait "$HARNESS_PID" >/dev/null 2>&1 || true
    fi
    exec 3>&- || true
    rm -f "$CONTROL_ROOT/alice.out" "$CONTROL_ROOT/bob.out"
    if [ -n "$NAMESPACE" ]; then
        rm -rf \
            "$RUN_ROOT/${NAMESPACE}-demo" \
            "$RUN_ROOT/${NAMESPACE}-sockets" \
            "$RUN_ROOT/${NAMESPACE}-runtime" \
            "$RUN_ROOT/${NAMESPACE}-launch" \
            "$RUN_ROOT"/${NAMESPACE}-cloud-init-* \
            "$RUN_ROOT/${NAMESPACE}-alice.sock" \
            "$RUN_ROOT/${NAMESPACE}-bob.sock" \
            "$RUN_ROOT/${NAMESPACE}-alice.vsock" \
            "$RUN_ROOT/${NAMESPACE}-bob.vsock" \
            "$RUN_ROOT/${NAMESPACE}-alice-api.sock" \
            "$RUN_ROOT/${NAMESPACE}-bob-api.sock" \
            "$RUN_ROOT/${NAMESPACE}-alice.vsock_2222" \
            "$RUN_ROOT/${NAMESPACE}-bob.vsock_2222" \
            "$RUN_ROOT/${NAMESPACE}-alice.vsock_5000" \
            "$RUN_ROOT/${NAMESPACE}-bob.vsock_5000"
    fi
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

extract_runtime_values() {
    PROXY_PORT="$(grep -oE 'proxy=ssh://localhost:[0-9]+' "$HARNESS_LOG" | tail -n1 | sed 's/.*://')"
    NAMESPACE="$(grep -oE 'namespace=[^[:space:]]+' "$HARNESS_LOG" | tail -n1 | cut -d= -f2)"
    if [ -z "$PROXY_PORT" ] || [ -z "$NAMESPACE" ]; then
        echo "failed to parse proxy port or namespace from harness log" >&2
        capture_log >&2 || true
        exit 1
    fi
}

run_shell_check() {
    local guest="$1"
    local expected_env="$2"
    local output_file="$CONTROL_ROOT/${guest}.out"

    : >"$output_file"
    printf 'pwd\ncat ~/.env\ncurl -fsSL https://example.com -o ~/example.html && echo FETCH_OK && stat ~/example.html\nstat -c "%%U:%%G %%n" ~ ~/.env /workspace /agent-state\nexit\n' |
        ssh -tt -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$PROXY_PORT" "$guest@localhost" \
        >"$output_file"

    grep -F "v1.4 extraction / agent-state demo" "$output_file" >/dev/null
    grep -F "/home/$guest" "$output_file" >/dev/null
    grep -F "$expected_env" "$output_file" >/dev/null
    grep -F "FETCH_OK" "$output_file" >/dev/null
    grep -F "$guest:$guest /home/$guest" "$output_file" >/dev/null
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

cargo build -p motlie-vmm --example harness_v1_4 >/dev/null

mkdir -p "$CONTROL_ROOT"
mkfifo "$HARNESS_FIFO"
mkdir -p "$RUN_ROOT"
"$HARNESS_BIN" shell --root "$RUN_ROOT" <"$HARNESS_FIFO" >"$HARNESS_LOG" 2>&1 &
HARNESS_PID="$!"
exec 3>"$HARNESS_FIFO"

wait_for_pattern "v14-harness>" 20

cat "$HARNESS_SCRIPT" >&3
wait_for_pattern "ok: booted alice" 90
wait_for_pattern "ok: booted bob" 90
wait_for_pattern "validation: 5 passed, 0 failed" 90
extract_runtime_values

run_shell_check "alice" "ALICE_API_KEY=demo-alice"
run_shell_check "bob" "BOB_API_KEY=demo-bob"

printf 'shutdown bob\n' >&3
wait_for_pattern "ok: shutdown bob" 30

printf 'shutdown alice\n' >&3
wait_for_pattern "ok: shutdown alice" 30

printf 'quit\n' >&3
sleep 1

echo "v1.4 harness shell smoke passed"
