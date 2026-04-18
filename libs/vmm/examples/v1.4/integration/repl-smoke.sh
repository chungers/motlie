#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V14_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$V14_DIR/../../../.." && pwd)"

TMUX_SOCKET="v14-smoke"
REPL_BIN="$REPO_ROOT/target/debug/examples/repl_host_v1_4"
REPL_FIFO="/tmp/motlie-vmm-v14-repl-smoke.fifo"
REPL_LOG="/tmp/motlie-vmm-v14-repl-smoke.log"
REPL_PID=""

cleanup() {
    pkill -f repl_host_v1_4 >/dev/null 2>&1 || true
    if [ -n "$REPL_PID" ]; then
        kill "$REPL_PID" >/dev/null 2>&1 || true
    fi
    exec 3>&- || true
    rm -rf \
        /tmp/motlie-vmm-v14-demo \
        /tmp/motlie-vmm-v14-runtime \
        /tmp/motlie-vmm-v14-launch \
        /tmp/motlie-vmm-v14-cloud-init-alice \
        /tmp/motlie-vmm-v14-cloud-init-bob \
        /tmp/motlie-vmm-v14-sockets \
        /tmp/motlie-vmm-v14-alice.sock \
        /tmp/motlie-vmm-v14-bob.sock \
        /tmp/motlie-vmm-v14-alice.vsock \
        /tmp/motlie-vmm-v14-bob.vsock \
        /tmp/motlie-vmm-v14-alice-api.sock \
        /tmp/motlie-vmm-v14-bob-api.sock \
        /tmp/motlie-vmm-v14-alice.vsock_2222 \
        /tmp/motlie-vmm-v14-bob.vsock_2222 \
        /tmp/motlie-vmm-v14-alice.vsock_5000 \
        /tmp/motlie-vmm-v14-bob.vsock_5000 \
        "$REPL_FIFO" \
        "$REPL_LOG" \
        /tmp/alice-repl-smoke.out \
        /tmp/bob-repl-smoke.out
}

capture_pane() {
    cat "$REPL_LOG"
}

wait_for_pattern() {
    local pattern="$1"
    local timeout_secs="${2:-60}"
    local deadline=$((SECONDS + timeout_secs))
    while [ "$SECONDS" -lt "$deadline" ]; do
        if capture_pane | grep -F "$pattern" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done

    echo "timed out waiting for pattern: $pattern" >&2
    echo "--- repl pane ---" >&2
    capture_pane >&2 || true
    return 1
}

run_shell_check() {
    local guest="$1"
    local expected_env="$2"
    local output_file="/tmp/${guest}-repl-smoke.out"

    : >"$output_file"
    printf 'pwd\ncat ~/.env\ncurl -fsSL https://example.com -o ~/example.html && echo FETCH_OK && stat ~/example.html\nstat -c "%%U:%%G %%n" ~ ~/.env /workspace /agent-state\nexit\n' |
        ssh -tt -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 2224 "$guest@localhost" \
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

cargo build -p motlie-vmm --example repl_host_v1_4 >/dev/null

cleanup

mkfifo "$REPL_FIFO"
"$REPL_BIN" <"$REPL_FIFO" >"$REPL_LOG" 2>&1 &
REPL_PID="$!"
exec 3>"$REPL_FIFO"

wait_for_pattern "v14>" 10

printf 'boot alice\n' >&3
wait_for_pattern "ok: booted alice" 60

printf 'boot bob\n' >&3
wait_for_pattern "ok: booted bob" 60

printf 'validate alice\n' >&3
wait_for_pattern "=== 5 passed, 0 failed ===" 60

printf 'validate bob\n' >&3
wait_for_pattern "=== 5 passed, 0 failed ===" 60

run_shell_check "alice" "ALICE_API_KEY=demo-alice"
run_shell_check "bob" "BOB_API_KEY=demo-bob"

printf 'shutdown bob\n' >&3
wait_for_pattern "ok: shutdown bob" 20

printf 'shutdown alice\n' >&3
wait_for_pattern "ok: shutdown alice" 20

printf 'quit\n' >&3
sleep 1

echo "v1.4 repl smoke passed"
