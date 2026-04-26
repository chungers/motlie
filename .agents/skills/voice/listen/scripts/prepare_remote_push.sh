#!/usr/bin/env bash

set -euo pipefail

SSH_TARGET=""
CAPTURE_SECONDS=""
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

mac_rec_source_command() {
  local format="$1"
  local trim_suffix=""
  if [[ -n "${CAPTURE_SECONDS}" ]]; then
    trim_suffix=" trim 0 ${CAPTURE_SECONDS}"
  fi

  cat <<EOF
if [ -x /opt/homebrew/bin/rec ]; then exec /opt/homebrew/bin/rec -q -c 1 -r 16000 -b 16 -e signed-integer -t ${format} -${trim_suffix}; elif [ -x /usr/local/bin/rec ]; then exec /usr/local/bin/rec -q -c 1 -r 16000 -b 16 -e signed-integer -t ${format} -${trim_suffix}; elif [ -x /opt/local/bin/rec ]; then exec /opt/local/bin/rec -q -c 1 -r 16000 -b 16 -e signed-integer -t ${format} -${trim_suffix}; elif command -v rec >/dev/null 2>&1; then exec rec -q -c 1 -r 16000 -b 16 -e signed-integer -t ${format} -${trim_suffix}; else echo 'rec not found; install sox with brew install sox' >&2; exit 127; fi
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ssh-target)
      SSH_TARGET="$2"
      shift 2
      ;;
    --seconds)
      CAPTURE_SECONDS="$2"
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

rm -f "${FIFO_PATH}"
mkfifo "${FIFO_PATH}"

agent_listen_args=(./.agents/skills/voice/listen/scripts/run.sh --backend "${BACKEND}" --wav "${FIFO_PATH}")
if [[ -n "${QUIET_FLAG}" ]]; then
  agent_listen_args+=("${QUIET_FLAG}")
fi

if [[ "${BACKEND}" == "moonshine" || "${BACKEND}" == "sherpa" ]]; then
  agent_listen_args+=(--input-format raw-s16le)
  HUMAN_MAC_SOURCE_CMD="$(mac_rec_source_command raw) 2>/dev/null"
else
  HUMAN_MAC_SOURCE_CMD="$(mac_rec_source_command wav) 2>/dev/null"
fi

REMOTE_FIFO_CMD="cat > $(printf '%q' "${FIFO_PATH}")"
AGENT_LISTEN_CMD="$(render_shell_command "${agent_listen_args[@]}")"
HUMAN_MAC_CMD="${HUMAN_MAC_SOURCE_CMD} | ssh ${SSH_TARGET} \"${REMOTE_FIFO_CMD}\""
HUMAN_MAC_CMD_PRETTY="${HUMAN_MAC_SOURCE_CMD} \
| ssh ${SSH_TARGET} \"${REMOTE_FIFO_CMD}\""

cat <<OUT
SSH_TARGET=${SSH_TARGET}
FIFO_PATH=${FIFO_PATH}
AGENT_LISTEN_CMD=${AGENT_LISTEN_CMD}
HUMAN_MAC_CMD=${HUMAN_MAC_CMD}
HUMAN_MAC_CMD_PRETTY<<'CMD'
${HUMAN_MAC_CMD_PRETTY}
CMD
OUT
