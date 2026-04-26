#!/usr/bin/env bash

set -euo pipefail

SSH_TARGET=""
FIFO_PATH=""
BACKEND="whisper"
QUIET_FLAG="--quiet"

render_shell_command() {
  local rendered=""
  local arg
  for arg in "$@"; do
    local quoted
    printf -v quoted '%q' "${arg}"
    if [[ -n "${rendered}" ]]; then
      rendered+=" "
    fi
    rendered+="${quoted}"
  done
  printf '%s\n' "${rendered}"
}

cleanup_stale_listeners() {
  local pids
  mapfile -t pids < <(
    ps -eo pid=,args= | awk -v fifo="${FIFO_PATH}" '
      index($0, fifo) && index($0, " listen ") { print $1 }
    '
  )

  if [[ ${#pids[@]} -eq 0 ]]; then
    return 0
  fi

  kill "${pids[@]}" 2>/dev/null || true
  sleep 1

  mapfile -t pids < <(
    ps -eo pid=,args= | awk -v fifo="${FIFO_PATH}" '
      index($0, fifo) && index($0, " listen ") { print $1 }
    '
  )
  if [[ ${#pids[@]} -gt 0 ]]; then
    kill -9 "${pids[@]}" 2>/dev/null || true
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ssh-target)
      SSH_TARGET="$2"
      shift 2
      ;;
    --seconds)
      shift 2
      ;;
    --fifo)
      FIFO_PATH="$2"
      shift 2
      ;;
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    --no-quiet)
      QUIET_FLAG=""
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 64
      ;;
  esac
done

if [[ -z "${SSH_TARGET}" ]]; then
  HOSTNAME_FALLBACK="$(hostname -s 2>/dev/null || hostname)"
  SSH_TARGET="${USER}@${HOSTNAME_FALLBACK}"
fi

if [[ -z "${FIFO_PATH}" ]]; then
  FIFO_PATH="/tmp/motlie-voice-listen.wav.pipe"
fi

cleanup_stale_listeners
rm -f "${FIFO_PATH}"
mkfifo "${FIFO_PATH}"

agent_listen_args=(./.agents/skills/voice/listen/scripts/run.sh --backend "${BACKEND}" --wav "${FIFO_PATH}")
if [[ -n "${QUIET_FLAG}" ]]; then
  agent_listen_args+=("${QUIET_FLAG}")
fi

AGENT_LISTEN_CMD="$(render_shell_command "${agent_listen_args[@]}")"
HUMAN_MAC_CMD="tmp=/tmp/motlie-voice-listen.wav; /opt/homebrew/bin/rec -q \"\$tmp\" 2>/dev/null; [ -s \"\$tmp\" ] && ssh ${SSH_TARGET} 'cat >${FIFO_PATH}' < \"\$tmp\"; rm -f \"\$tmp\""
HUMAN_MAC_CMD_PRETTY=$(cat <<CMD
 tmp=/tmp/motlie-voice-listen.wav
 /opt/homebrew/bin/rec -q "\$tmp" 2>/dev/null
 [ -s "\$tmp" ] && ssh ${SSH_TARGET} 'cat >${FIFO_PATH}' < "\$tmp"
 rm -f "\$tmp"
CMD
)
HUMAN_MAC_FALLBACK_CMD="tmp=/tmp/motlie-voice-listen.wav; if [ -x /opt/homebrew/bin/rec ]; then recbin=/opt/homebrew/bin/rec; elif [ -x /usr/local/bin/rec ]; then recbin=/usr/local/bin/rec; elif [ -x /opt/local/bin/rec ]; then recbin=/opt/local/bin/rec; elif command -v rec >/dev/null 2>&1; then recbin=rec; else echo 'rec not found; install sox with brew install sox' >&2; exit 127; fi; \"\$recbin\" -q \"\$tmp\" 2>/dev/null; [ -s \"\$tmp\" ] && ssh ${SSH_TARGET} 'cat >${FIFO_PATH}' < \"\$tmp\"; rm -f \"\$tmp\""
HUMAN_MAC_FALLBACK_CMD_PRETTY=$(cat <<CMD
 tmp=/tmp/motlie-voice-listen.wav
 if [ -x /opt/homebrew/bin/rec ]; then
   recbin=/opt/homebrew/bin/rec
 elif [ -x /usr/local/bin/rec ]; then
   recbin=/usr/local/bin/rec
 elif [ -x /opt/local/bin/rec ]; then
   recbin=/opt/local/bin/rec
 elif command -v rec >/dev/null 2>&1; then
   recbin=rec
 else
   echo 'rec not found; install sox with brew install sox' >&2
   exit 127
 fi
 "\$recbin" -q "\$tmp" 2>/dev/null
 [ -s "\$tmp" ] && ssh ${SSH_TARGET} 'cat >${FIFO_PATH}' < "\$tmp"
 rm -f "\$tmp"
CMD
)

cat <<OUT
SSH_TARGET=${SSH_TARGET}
FIFO_PATH=${FIFO_PATH}
AGENT_LISTEN_CMD=${AGENT_LISTEN_CMD}
HUMAN_MAC_CMD=${HUMAN_MAC_CMD}
HUMAN_MAC_CMD_PRETTY<<'CMD'
${HUMAN_MAC_CMD_PRETTY}
CMD
HUMAN_MAC_FALLBACK_CMD=${HUMAN_MAC_FALLBACK_CMD}
HUMAN_MAC_FALLBACK_CMD_PRETTY<<'CMD'
${HUMAN_MAC_FALLBACK_CMD_PRETTY}
CMD
OUT
