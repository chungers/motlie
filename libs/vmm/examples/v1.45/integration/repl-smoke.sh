#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V145_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$V145_DIR/../../../.." && pwd)"

REPL_BIN="$REPO_ROOT/target/debug/examples/repl_host_v1_45"
CONTROL_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/motlie-vmm-v145-repl-smoke.XXXXXX")"
RUN_ROOT="$CONTROL_ROOT/runroot"
REPL_FIFO="$CONTROL_ROOT/repl.fifo"
REPL_LOG="$CONTROL_ROOT/repl.log"
REPL_PID=""
PROXY_PORT=""
BASE_VM_DIR="${MOTLIE_VZ_BASE_VM_DIR:-$V145_DIR/../v1.35/artifacts/source-base.vm}"

cleanup() {
    if [ -n "$REPL_PID" ]; then
        kill "$REPL_PID" >/dev/null 2>&1 || true
    fi
    exec 3>&- || true
    rm -rf "$CONTROL_ROOT"
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

extract_proxy_port() {
    PROXY_PORT="$(
        grep -oE 'SSH proxy: listening on 127\.0\.0\.1:[0-9]+' "$REPL_LOG" |
            tail -n1 |
            sed 's/.*://'
    )"
    if [ -z "$PROXY_PORT" ]; then
        echo "failed to parse proxy port from repl log" >&2
        capture_pane >&2 || true
        exit 1
    fi
}

run_shell_check() {
    local guest="$1"
    local expected_env="$2"
    local output_file="$CONTROL_ROOT/${guest}-repl-smoke.out"

    : >"$output_file"
    printf 'pwd\ncat ~/.env\ncurl -fsSL https://example.com -o ~/example.html && echo FETCH_OK && stat ~/example.html\nstat -c "%%U:%%G %%n" ~ ~/.env /workspace /agent-state\nexit\n' |
        ssh -tt -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$PROXY_PORT" "$guest@localhost" \
        >"$output_file"

    grep -F "/home/$guest" "$output_file" >/dev/null
    grep -F "$expected_env" "$output_file" >/dev/null
    grep -F "FETCH_OK" "$output_file" >/dev/null
    grep -F "$guest:$guest /home/$guest" "$output_file" >/dev/null
}

trap cleanup EXIT

cd "$REPO_ROOT"

test -f "$BASE_VM_DIR/disk.img" || {
    echo "missing $BASE_VM_DIR/disk.img; run $V145_DIR/build-guest.sh first or set MOTLIE_VZ_BASE_VM_DIR" >&2
    exit 1
}
test -f "$BASE_VM_DIR/nvram.bin" || {
    echo "missing $BASE_VM_DIR/nvram.bin; run $V145_DIR/build-guest.sh first or set MOTLIE_VZ_BASE_VM_DIR" >&2
    exit 1
}

cargo build -p motlie-vmm --example repl_host_v1_45 >/dev/null
cargo build -p motlie-vnet --example vz_egress_helper_v1_25 >/dev/null

cleanup

mkdir -p "$RUN_ROOT"
mkfifo "$REPL_FIFO"
"$REPL_BIN" --root "$RUN_ROOT" <"$REPL_FIFO" >"$REPL_LOG" 2>&1 &
REPL_PID="$!"
exec 3>"$REPL_FIFO"

wait_for_pattern "SSH proxy: listening on 127.0.0.1:" 20
wait_for_pattern "v145>" 20
extract_proxy_port

printf 'boot alice\n' >&3
wait_for_pattern "ok: booted alice" 900

printf 'boot bob\n' >&3
wait_for_pattern "ok: booted bob" 900

printf 'validate alice\n' >&3
wait_for_pattern "=== 5 passed, 0 failed ===" 900

printf 'validate bob\n' >&3
wait_for_pattern "=== 5 passed, 0 failed ===" 900

run_shell_check "alice" "ALICE_API_KEY=demo-alice"
run_shell_check "bob" "BOB_API_KEY=demo-bob"

printf 'shutdown bob\n' >&3
wait_for_pattern "ok: shutdown bob" 20

printf 'shutdown alice\n' >&3
wait_for_pattern "ok: shutdown alice" 20

printf 'quit\n' >&3
sleep 1

echo "v1.45 repl smoke passed"
