#!/usr/bin/env bash

set -euo pipefail

SSH_TARGET=""
CAPTURE_SECONDS=""
FIFO_PATH=""
BACKEND="whisper"
QUIET_FLAG="--quiet"

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

AGENT_LISTEN_CMD="./.agents/skills/voice/listen/scripts/run.sh --backend ${BACKEND} --wav ${FIFO_PATH}"
if [[ -n "${QUIET_FLAG}" ]]; then
  AGENT_LISTEN_CMD+=" ${QUIET_FLAG}"
fi

TRIM_SUFFIX=""
if [[ -n "${CAPTURE_SECONDS}" ]]; then
  TRIM_SUFFIX=" trim 0 ${CAPTURE_SECONDS}"
fi

if [[ "${BACKEND}" == "moonshine" || "${BACKEND}" == "sherpa" ]]; then
  AGENT_LISTEN_CMD+=" --input-format raw-s16le"
  HUMAN_MAC_SOURCE_CMD="/opt/homebrew/bin/rec -q -c 1 -r 16000 -b 16 -e signed-integer -t raw -${TRIM_SUFFIX} 2>/dev/null"
else
  HUMAN_MAC_SOURCE_CMD="/opt/homebrew/bin/rec -q -c 1 -r 16000 -b 16 -e signed-integer -t wav -${TRIM_SUFFIX} 2>/dev/null"
fi

HUMAN_MAC_CMD="${HUMAN_MAC_SOURCE_CMD} | ssh ${SSH_TARGET} 'cat > ${FIFO_PATH}'"
HUMAN_MAC_CMD_PRETTY="${HUMAN_MAC_SOURCE_CMD} \\
| ssh ${SSH_TARGET} 'cat > ${FIFO_PATH}'"

cat <<OUT
SSH_TARGET=${SSH_TARGET}
FIFO_PATH=${FIFO_PATH}
AGENT_LISTEN_CMD=${AGENT_LISTEN_CMD}
HUMAN_MAC_CMD=${HUMAN_MAC_CMD}
HUMAN_MAC_CMD_PRETTY<<'CMD'
${HUMAN_MAC_CMD_PRETTY}
CMD
OUT
