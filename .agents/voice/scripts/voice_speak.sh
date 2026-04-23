#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

voice_load_env

tts_backend='piper'
endpoint_name="${MOTLIE_VOICE_PLAYBACK_ENDPOINT}"
text_value=''
wav_path=''
quiet=1

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --backend)
      tts_backend="$2"
      shift 2
      ;;
    --endpoint)
      endpoint_name="$2"
      shift 2
      ;;
    --text)
      text_value="$2"
      shift 2
      ;;
    --wav)
      wav_path="$2"
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

"${SCRIPT_DIR}/ensure_examples.sh" "${tts_backend}"

IFS='|' read -r example_name _model_feature _cuda_feature artifact_root <<<"$(voice_backend_metadata "${tts_backend}")"
example_binary="$(voice_example_binary "${example_name}")"

[[ -x "${example_binary}" ]] || voice_die "missing example binary '${example_binary}'"
[[ -d "${artifact_root}" || -f "${artifact_root}" ]] || voice_die "artifact root '${artifact_root}' does not exist"

example_cmd=("${example_binary}" --artifact-root "${artifact_root}")
if [[ "${quiet}" -eq 1 ]]; then
  example_cmd+=(--quiet)
fi
if [[ -n "${wav_path}" ]]; then
  example_cmd+=(--wav "${wav_path}")
fi

run_example_input() {
  if [[ -n "${text_value}" ]]; then
    printf '%s\n' "${text_value}" | "${example_cmd[@]}"
  else
    "${example_cmd[@]}"
  fi
}

if [[ -n "${wav_path}" ]]; then
  run_example_input
  exit 0
fi

endpoint_kind="$(voice_require_endpoint_field "${endpoint_name}" KIND)"
play_cmd="$(voice_require_endpoint_field "${endpoint_name}" PLAY_CMD)"

case "${endpoint_kind}" in
  local)
    run_example_input | bash -lc "${play_cmd}"
    ;;
  ssh)
    ssh_target="$(voice_require_endpoint_field "${endpoint_name}" SSH_TARGET)"
    run_example_input | ssh "${ssh_target}" "${play_cmd}"
    ;;
  *)
    voice_die "unsupported playback endpoint kind '${endpoint_kind}' for endpoint '${endpoint_name}'"
    ;;
esac

