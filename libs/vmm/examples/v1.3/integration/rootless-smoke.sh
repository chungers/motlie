#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$EXAMPLE_DIR/../../../.." && pwd)"

TMUX_SOCKET="v13-integration"
TMUX_SESSION="rootless-smoke"
REPL_BIN="$REPO_ROOT/target/debug/examples/repl_host_v1_3"
PTY_PROBE_BIN="$REPO_ROOT/target/debug/examples/russh_pty_probe"

dump_debug() {
    echo ""
    echo "=== tmux pane ==="
    tmux -L "$TMUX_SOCKET" capture-pane -pt "$TMUX_SESSION" -S -250 2>/dev/null || true
    echo ""
    echo "=== alice launch.log ==="
    tail -200 /tmp/motlie-vmm-launch/alice/launch.log 2>/dev/null || true
    echo ""
    echo "=== alice serial.log ==="
    tail -200 /tmp/motlie-vmm-launch/alice/serial.log 2>/dev/null || true
    echo ""
    echo "=== bob launch.log ==="
    tail -200 /tmp/motlie-vmm-launch/bob/launch.log 2>/dev/null || true
    echo ""
    echo "=== bob serial.log ==="
    tail -200 /tmp/motlie-vmm-launch/bob/serial.log 2>/dev/null || true
}

cleanup() {
    tmux -L "$TMUX_SOCKET" kill-server >/dev/null 2>&1 || true
    pkill -f repl_host_v1_3 >/dev/null 2>&1 || true
    pkill -f cloud-hypervisor >/dev/null 2>&1 || true
}

trap cleanup EXIT
trap 'dump_debug' ERR

wait_for_pane() {
    local needle="$1"
    local attempts="${2:-60}"
    local delay="${3:-1}"
    local pane
    local i
    for ((i = 0; i < attempts; i++)); do
        pane="$(tmux -L "$TMUX_SOCKET" capture-pane -pt "$TMUX_SESSION" -S -250 2>/dev/null || true)"
        if grep -Fq "$needle" <<<"$pane"; then
            return 0
        fi
        sleep "$delay"
    done
    echo "Timed out waiting for pane text: $needle" >&2
    return 1
}

guest_ssh() {
    local guest="$1"
    shift
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p 2222 "${guest}@localhost" "$@"
}

assert_pane_count() {
    local needle="$1"
    local expected="${2:-1}"
    local actual
    actual="$(tmux -L "$TMUX_SOCKET" capture-pane -pt "$TMUX_SESSION" -S -300 | grep -Fc "$needle" || true)"
    if [ "$actual" -lt "$expected" ]; then
        echo "Expected at least $expected occurrences of '$needle' in tmux pane, saw $actual" >&2
        return 1
    fi
}

echo "=== v1.3 rootless smoke ==="
echo "repo:       $REPO_ROOT"
echo "example:    $EXAMPLE_DIR"
echo "tmux:       $TMUX_SOCKET/$TMUX_SESSION"

cleanup

rm -rf /tmp/motlie-vmm-runtime
rm -rf /tmp/motlie-vmm-launch
rm -rf /tmp/motlie-vmm-cloud-init-alice /tmp/motlie-vmm-cloud-init-bob
rm -f /tmp/motlie-vmm-alice.sock /tmp/motlie-vmm-bob.sock
rm -f /tmp/motlie-vmm-alice.vsock /tmp/motlie-vmm-bob.vsock
rm -f /tmp/motlie-vmm-alice.vsock_2222 /tmp/motlie-vmm-bob.vsock_2222
rm -f /tmp/motlie-vmm-alice-api.sock /tmp/motlie-vmm-bob-api.sock

echo ""
echo "=== build examples ==="
cargo build -p motlie-vmm --example repl_host_v1_3 --example russh_pty_probe --manifest-path "$REPO_ROOT/Cargo.toml"

echo ""
echo "=== build guest image ==="
(cd "$EXAMPLE_DIR" && ./build-guest.sh)

echo ""
echo "=== start repl_host_v1_3 (rootless) ==="
tmux -L "$TMUX_SOCKET" start-server
tmux -L "$TMUX_SOCKET" new-session -d -s "$TMUX_SESSION" \
    "cd '$REPO_ROOT' && '$REPL_BIN' --empty --script libs/vmm/examples/v1.3/setup-multiguest.sh.vfs --admin-net=none --egress-net=vhost-user"

wait_for_pane "vfs[bob]>"
tmux -L "$TMUX_SOCKET" send-keys -t "$TMUX_SESSION" "launch alice" Enter
wait_for_pane "SSH bridge ready for guest 'alice'" 90 1

echo ""
echo "=== alice ssh exec ==="
guest_ssh alice /bin/echo hello

echo ""
echo "=== alice vfs-backed guest view ==="
guest_ssh alice /bin/sh -lc \
    'pwd &&
     ls -ald /home/alice /workspace /agent-state &&
     test -d ~/.codex && test -w ~/.codex &&
     test -d ~/.claude && test -w ~/.claude &&
     test -d ~/.config/claude-code && test -w ~/.config/claude-code &&
     grep -q "Alice workspace mounted from the host." /workspace/README.md &&
     echo VFS_OK'

echo ""
echo "=== alice rootless route ==="
guest_ssh alice ip route

echo ""
echo "=== alice outbound https ==="
guest_ssh alice curl -sSf https://example.com -o /dev/null
echo "HTTPS_OK"

echo ""
echo "=== alice pty probe ==="
"$PTY_PROBE_BIN" 127.0.0.1:2222 "/bin/cat -v" true

echo ""
echo "=== repl validate alice ==="
tmux -L "$TMUX_SOCKET" send-keys -t "$TMUX_SESSION" "validate alice" Enter
wait_for_pane "=== 9 passed, 0 failed ===" 90 1
assert_pane_count "=== validate alice ===" 1
assert_pane_count "=== 9 passed, 0 failed ===" 1

echo ""
echo "=== launch bob ==="
tmux -L "$TMUX_SOCKET" send-keys -t "$TMUX_SESSION" "launch bob" Enter
wait_for_pane "SSH bridge ready for guest 'bob'" 90 1

echo ""
echo "=== resource uniqueness ==="
grep -F "CID 3" /tmp/motlie-vmm-launch/alice/launch.log
grep -F "CID 4" /tmp/motlie-vmm-launch/bob/launch.log
grep -F "/tmp/motlie-vmm-alice.sock" /tmp/motlie-vmm-launch/alice/launch.log
grep -F "/tmp/motlie-vmm-bob.sock" /tmp/motlie-vmm-launch/bob/launch.log

echo ""
echo "=== alice still reachable after bob boot ==="
guest_ssh alice /bin/echo alice-still-ok

echo ""
echo "=== bob ssh exec ==="
guest_ssh bob /bin/echo hello

echo ""
echo "=== bob vfs-backed guest view ==="
guest_ssh bob /bin/sh -lc \
    'pwd &&
     ls -ald /home/bob /workspace /agent-state &&
     test -d ~/.codex && test -w ~/.codex &&
     test -d ~/.claude && test -w ~/.claude &&
     test -d ~/.config/claude-code && test -w ~/.config/claude-code &&
     grep -q "Bob workspace mounted from the host." /workspace/README.md &&
     echo VFS_OK'

echo ""
echo "=== bob outbound https ==="
guest_ssh bob curl -sSf https://example.com -o /dev/null
echo "HTTPS_OK"

echo ""
echo "=== bob isolation ==="
guest_ssh alice /bin/sh -lc "echo alice-only > ~/.codex/owner.txt && echo alice-only > /workspace/owner.txt"
if guest_ssh bob test -e /home/bob/.codex/owner.txt; then
    echo "bob unexpectedly sees alice's ~/.codex/owner.txt" >&2
    exit 1
fi
if guest_ssh bob test -e /workspace/owner.txt; then
    echo "bob unexpectedly sees alice's /workspace/owner.txt" >&2
    exit 1
fi
echo "ISOLATION_OK"

echo ""
echo "=== repl validate bob ==="
tmux -L "$TMUX_SOCKET" send-keys -t "$TMUX_SESSION" "validate bob" Enter
wait_for_pane "=== validate bob ===" 90 1
sleep 2
assert_pane_count "=== 9 passed, 0 failed ===" 2
tmux -L "$TMUX_SOCKET" capture-pane -pt "$TMUX_SESSION" -S -180

echo ""
echo "=== shutdown ==="
tmux -L "$TMUX_SOCKET" send-keys -t "$TMUX_SESSION" "shutdown alice" Enter
wait_for_pane "ok: shutdown alice" 60 1
tmux -L "$TMUX_SOCKET" send-keys -t "$TMUX_SESSION" "shutdown bob" Enter
wait_for_pane "ok: shutdown bob" 60 1
tmux -L "$TMUX_SOCKET" send-keys -t "$TMUX_SESSION" "quit" Enter

echo ""
echo "v1.3 multi-guest rootless smoke passed"
