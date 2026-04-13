#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-$(mktemp -d "${TMPDIR:-/tmp}/motlie-v14-headless.XXXXXX")}"
FIFO="$ROOT_DIR/repl.fifo"
LOG="$ROOT_DIR/repl.log"
LIVE_ROOT="$ROOT_DIR/live"

cleanup() {
  if [[ -n "${writer_fd_open:-}" ]]; then
    exec 3>&- || true
  fi
  if [[ -n "${repl_pid:-}" ]] && kill -0 "$repl_pid" 2>/dev/null; then
    kill -TERM "$repl_pid" || true
    wait "$repl_pid" || true
  fi
}
trap cleanup EXIT

mkdir -p "$ROOT_DIR"
rm -f "$FIFO"
mkfifo "$FIFO"

./target/debug/examples/repl_host_v1_4 --root "$LIVE_ROOT" <"$FIFO" >"$LOG" 2>&1 &
repl_pid=$!
exec 3>"$FIFO"
writer_fd_open=1

for _ in $(seq 1 120); do
  if grep -q "SSH proxy: listening on 127.0.0.1:" "$LOG"; then
    break
  fi
  sleep 1
done

proxy_port="$(sed -n 's/.*SSH proxy: listening on 127\.0\.0\.1:\([0-9][0-9]*\).*/\1/p' "$LOG" | tail -n1)"
if [[ -z "$proxy_port" ]]; then
  echo "FAILED: proxy port not found"
  cat "$LOG"
  exit 1
fi

printf 'auto-provision on\n' >&3
for _ in $(seq 1 30); do
  if grep -q "ok: auto-provision enabled" "$LOG"; then
    break
  fi
  sleep 1
done

exec 3>&-
unset writer_fd_open
sleep 2

if ! kill -0 "$repl_pid" 2>/dev/null; then
  echo "FAILED: repl_host_v1_4 exited after stdin closed"
  cat "$LOG"
  exit 1
fi

declare -A boot_ids
users=(alice bob jane joe zoe)

run_remote_probe() {
  local user="$1"
  ssh \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -p "$proxy_port" \
    "$user@localhost" \
    'whoami; cat /proc/sys/kernel/random/boot_id; uname -s'
}

for user in "${users[@]}"; do
  result="$(run_remote_probe "$user" 2>/dev/null)"
  who="$(printf '%s\n' "$result" | sed -n '1p')"
  boot_id="$(printf '%s\n' "$result" | sed -n '2p')"
  os_name="$(printf '%s\n' "$result" | sed -n '3p')"
  if [[ "$who" != "$user" || -z "$boot_id" || "$os_name" != "Linux" ]]; then
    echo "FAILED: first login for $user"
    printf '%s\n' "$result"
    cat "$LOG"
    exit 1
  fi
  boot_ids["$user"]="$boot_id"
done

for round in 1 2 3; do
  for user in "${users[@]}"; do
    result="$(run_remote_probe "$user" 2>/dev/null)"
    who="$(printf '%s\n' "$result" | sed -n '1p')"
    boot_id="$(printf '%s\n' "$result" | sed -n '2p')"
    os_name="$(printf '%s\n' "$result" | sed -n '3p')"
    if [[ "$who" != "$user" || "$boot_id" != "${boot_ids[$user]}" || "$os_name" != "Linux" ]]; then
      echo "FAILED: reconnect round $round for $user"
      printf 'expected boot_id=%s\n' "${boot_ids[$user]}"
      printf '%s\n' "$result"
      cat "$LOG"
      exit 1
    fi
  done
done

kill -TERM "$repl_pid"
wait "$repl_pid"
repl_pid=""

printf 'PASS proxy_port=%s root=%s users=%s\n' "$proxy_port" "$ROOT_DIR" "${users[*]}"
