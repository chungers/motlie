#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ARGS=("$@")
WAV_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wav)
      WAV_PATH="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

cleanup_fifo() {
  if [[ -n "${WAV_PATH}" && -p "${WAV_PATH}" ]]; then
    rm -f "${WAV_PATH}"
  fi
}

trap cleanup_fifo EXIT HUP INT TERM

bash "${SCRIPT_DIR}/../../common/run_voice_agent.sh" listen "${RUN_ARGS[@]}"
