#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V14_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$V14_DIR/../../../.." && pwd)"

HARNESS_BIN="$REPO_ROOT/target/debug/examples/harness_v1_4"
CONTROL_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/motlie-vmm-v14-harness-isolation.XXXXXX")"

cleanup_instance() {
    local fifo="$1"
    local pid="$2"
    if [ -n "$pid" ]; then
        kill "$pid" >/dev/null 2>&1 || true
        wait "$pid" >/dev/null 2>&1 || true
    fi
    rm -f "$fifo"
}

cleanup() {
    cleanup_instance "$FIFO1" "${PID1:-}"
    cleanup_instance "$FIFO2" "${PID2:-}"
    rm -rf "$CONTROL_ROOT"
}

wait_for_pattern() {
    local log="$1"
    local pattern="$2"
    local timeout_secs="${3:-60}"
    local deadline=$((SECONDS + timeout_secs))
    while [ "$SECONDS" -lt "$deadline" ]; do
        if grep -F "$pattern" "$log" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    echo "timed out waiting for pattern '$pattern' in $log" >&2
    tail -n 200 "$log" >&2 || true
    return 1
}

extract_proxy_port() {
    local log="$1"
    grep -oE 'proxy=ssh://localhost:[0-9]+' "$log" | tail -n1 | sed 's/.*://'
}

extract_namespace() {
    local log="$1"
    grep -oE '^namespace=.*' "$log" | tail -n1 | cut -d= -f2
}

run_shell_check() {
    local guest="$1"
    local port="$2"
    local output_file="$3"

    : >"$output_file"
    printf 'pwd\ncat ~/.env\ncurl -fsSL https://example.com -o ~/example.html && echo FETCH_OK && stat ~/example.html\nexit\n' |
        ssh -tt -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$port" "$guest@localhost" \
        >"$output_file"

    grep -F "/home/$guest" "$output_file" >/dev/null
    grep -F "FETCH_OK" "$output_file" >/dev/null
}

trap cleanup EXIT

cd "$REPO_ROOT"
cargo build -p motlie-vmm --example harness_v1_4 >/dev/null

RUN_ROOT1="$CONTROL_ROOT/runroot-1"
RUN_ROOT2="$CONTROL_ROOT/runroot-2"
LOG1="$CONTROL_ROOT/harness-1.log"
LOG2="$CONTROL_ROOT/harness-2.log"
FIFO1="$CONTROL_ROOT/harness-1.fifo"
FIFO2="$CONTROL_ROOT/harness-2.fifo"
OUT1="$CONTROL_ROOT/harness-1-alice.out"
OUT2="$CONTROL_ROOT/harness-2-alice.out"

mkdir -p "$RUN_ROOT1" "$RUN_ROOT2"
mkfifo "$FIFO1" "$FIFO2"

"$HARNESS_BIN" shell --root "$RUN_ROOT1" <"$FIFO1" >"$LOG1" 2>&1 &
PID1="$!"
"$HARNESS_BIN" shell --root "$RUN_ROOT2" <"$FIFO2" >"$LOG2" 2>&1 &
PID2="$!"

exec 3>"$FIFO1"
exec 4>"$FIFO2"

wait_for_pattern "$LOG1" "v14-harness>" 20
wait_for_pattern "$LOG2" "v14-harness>" 20

printf 'where\nboot alice\nwhere alice\n' >&3
printf 'where\nboot alice\nwhere alice\n' >&4

wait_for_pattern "$LOG1" "ok: booted alice" 90
wait_for_pattern "$LOG2" "ok: booted alice" 90

PORT1="$(extract_proxy_port "$LOG1")"
PORT2="$(extract_proxy_port "$LOG2")"
NS1="$(extract_namespace "$LOG1")"
NS2="$(extract_namespace "$LOG2")"

test -n "$PORT1" && test -n "$PORT2"
test -n "$NS1" && test -n "$NS2"

if [ "$PORT1" = "$PORT2" ] || [ "$NS1" = "$NS2" ]; then
    echo "instance collision detected" >&2
    echo "PORT1=$PORT1 PORT2=$PORT2 NS1=$NS1 NS2=$NS2" >&2
    exit 1
fi

run_shell_check alice "$PORT1" "$OUT1"
run_shell_check alice "$PORT2" "$OUT2"

printf 'shutdown alice\nquit\n' >&3
printf 'shutdown alice\nquit\n' >&4

wait_for_pattern "$LOG1" "ok: shutdown alice" 30
wait_for_pattern "$LOG2" "ok: shutdown alice" 30

echo "v1.4 harness isolation smoke passed"
