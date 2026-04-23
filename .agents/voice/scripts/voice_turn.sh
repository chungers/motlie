#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

voice_load_env

tts_backend='piper'
asr_backend='whisper'
playback_endpoint="${MOTLIE_VOICE_PLAYBACK_ENDPOINT}"
capture_endpoint="${MOTLIE_VOICE_CAPTURE_ENDPOINT}"
prompt_text=''
capture_seconds='8'
quiet=1

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --tts-backend)
      tts_backend="$2"
      shift 2
      ;;
    --asr-backend)
      asr_backend="$2"
      shift 2
      ;;
    --playback-endpoint)
      playback_endpoint="$2"
      shift 2
      ;;
    --capture-endpoint)
      capture_endpoint="$2"
      shift 2
      ;;
    --prompt)
      prompt_text="$2"
      shift 2
      ;;
    --seconds)
      capture_seconds="$2"
      shift 2
      ;;
    --verbose)
      quiet=0
      shift
      ;;
    --quiet)
      quiet=1
      shift
      ;;
    *)
      voice_die "unknown argument '$1'"
      ;;
  esac
done

[[ -n "${prompt_text}" ]] || voice_die "--prompt is required"

speak_cmd=("${SCRIPT_DIR}/voice_speak.sh" --backend "${tts_backend}" --endpoint "${playback_endpoint}" --text "${prompt_text}")
listen_cmd=("${SCRIPT_DIR}/voice_listen.sh" --backend "${asr_backend}" --endpoint "${capture_endpoint}" --seconds "${capture_seconds}")

if [[ "${quiet}" -eq 1 ]]; then
  speak_cmd+=(--quiet)
  listen_cmd+=(--quiet)
else
  speak_cmd+=(--verbose)
  listen_cmd+=(--verbose)
fi

"${speak_cmd[@]}"
"${listen_cmd[@]}"
