#!/usr/bin/env bash
# tests/vmm/driver.sh — run the golden vmm test suite against running guests.
#
# This driver is harness-agnostic. It takes an EXEC_CMD template that knows how
# to execute a single bash command in a named guest, and dispatches all suite
# scripts through it. The test scripts themselves never assume how they are
# invoked; they only require a normal Linux shell with the standard tools.
#
# Required env (or flags):
#   --guests alice,bob          comma-separated guest names (default: alice,bob)
#   --suite vnet,vfs            comma-separated suites to run (default: both)
#   --exec-cmd "ssh -p 2226 -o StrictHostKeyChecking=no GUEST@127.0.0.1 -- bash -s"
#                                template; literal token GUEST is replaced with
#                                each guest name. The script body is piped on
#                                stdin via "bash -s" or equivalent.
#   --out-dir tests/vmm/results last-run JSON results land here
#
# Example:
#   tests/vmm/driver.sh \
#     --guests alice,bob \
#     --suite vnet,vfs \
#     --exec-cmd 'ssh -p 2226 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null GUEST@127.0.0.1 -- bash -s'
#
# Output: prints one line per test in the form
#   TEST=<name> GUEST=<g> RESULT=pass|fail|skip [DETAIL=...]
# and writes a JSON summary to $OUT_DIR/results.json.
# Exit code: 0 if all tests pass (skips OK); 1 if any test fails; 2 on driver error.

set -euo pipefail

SUITE_DIR="$(cd "$(dirname "$0")" && pwd)"
GUESTS="alice,bob"
SUITES="vnet,vfs"
EXEC_CMD=""
OUT_DIR="${SUITE_DIR}/results"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --guests)   GUESTS="$2"; shift 2 ;;
    --suite)    SUITES="$2"; shift 2 ;;
    --exec-cmd) EXEC_CMD="$2"; shift 2 ;;
    --out-dir)  OUT_DIR="$2"; shift 2 ;;
    *) echo "driver: unknown arg $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$EXEC_CMD" ]]; then
  echo "driver: --exec-cmd required (template must contain literal GUEST token)" >&2
  exit 2
fi
if [[ "$EXEC_CMD" != *GUEST* ]]; then
  echo "driver: --exec-cmd must contain literal GUEST token to interpolate guest name" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
results_json="$OUT_DIR/results.json"
: > "$results_json.tmp"
echo "[" >> "$results_json.tmp"
first=1
fail_count=0
pass_count=0
skip_count=0

run_one() {
  local guest="$1" script="$2"
  local cmd="${EXEC_CMD//GUEST/$guest}"
  # Bundle the shared helper in front of the test body and pipe the bundle
  # through stdin to the exec-cmd's bash -s. This guarantees pass/fail/skip
  # are defined before the test body runs, regardless of $0 (which is "bash"
  # under stdin execution and would break a $(dirname "$0")/.. source path).
  local out rc
  if out=$(cat "$SUITE_DIR/shared/result.sh" "$script" | eval "$cmd" 2>&1); then rc=0; else rc=$?; fi
  # Each script is expected to print exactly one TEST=... line on stdout
  local line
  line=$(echo "$out" | grep -E '^TEST=' | tail -n 1 || true)
  if [[ -z "$line" ]]; then
    line="TEST=$(basename "$script" .sh) RESULT=fail DETAIL=no-output(rc=$rc)"
  fi
  echo "GUEST=$guest $line"
  # Tally
  case "$line" in
    *RESULT=pass*) pass_count=$((pass_count+1)) ;;
    *RESULT=skip*) skip_count=$((skip_count+1)) ;;
    *)             fail_count=$((fail_count+1)) ;;
  esac
  # Append JSON record
  local name result detail
  name=$(echo "$line"   | sed -nE 's/.*TEST=([^ ]+).*/\1/p')
  result=$(echo "$line" | sed -nE 's/.*RESULT=([a-z]+).*/\1/p')
  detail=$(echo "$line" | sed -nE 's/.*DETAIL=(.+)$/\1/p')
  if [[ $first -eq 1 ]]; then first=0; else echo "," >> "$results_json.tmp"; fi
  printf '  {"guest":"%s","test":"%s","result":"%s","detail":%s}' \
    "$guest" "$name" "$result" \
    "$(printf '%s' "${detail:-}" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))' 2>/dev/null || echo '""')" \
    >> "$results_json.tmp"
}

IFS=',' read -ra GUEST_ARR <<<"$GUESTS"
IFS=',' read -ra SUITE_ARR <<<"$SUITES"

for guest in "${GUEST_ARR[@]}"; do
  for suite in "${SUITE_ARR[@]}"; do
    suite_dir="$SUITE_DIR/$suite"
    if [[ ! -d "$suite_dir" ]]; then
      echo "driver: no such suite directory: $suite_dir" >&2
      exit 2
    fi
    for script in "$suite_dir"/*.sh; do
      [[ -f "$script" ]] || continue
      run_one "$guest" "$script"
    done
  done
done

echo "" >> "$results_json.tmp"
echo "]" >> "$results_json.tmp"
mv "$results_json.tmp" "$results_json"

echo ""
echo "SUMMARY: pass=$pass_count fail=$fail_count skip=$skip_count results=$results_json"

if [[ $fail_count -gt 0 ]]; then
  exit 1
fi
exit 0
