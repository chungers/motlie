#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

voice_load_env

asr_backend='whisper'
endpoint_name="${MOTLIE_VOICE_CAPTURE_ENDPOINT}"
wav_path=''
capture_seconds=''
partials=0
quiet=1

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --backend)
      asr_backend="$2"
      shift 2
      ;;
    --endpoint)
      endpoint_name="$2"
      shift 2
      ;;
    --seconds)
      capture_seconds="$2"
      shift 2
      ;;
    --wav)
      wav_path="$2"
      shift 2
      ;;
    --partials)
      partials=1
      shift
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

"${SCRIPT_DIR}/ensure_examples.sh" "${asr_backend}"

IFS='|' read -r example_name _model_feature _cuda_feature artifact_root <<<"$(voice_backend_metadata "${asr_backend}")"
example_binary="$(voice_example_binary "${example_name}")"

[[ -x "${example_binary}" ]] || voice_die "missing example binary '${example_binary}'"
[[ -d "${artifact_root}" || -f "${artifact_root}" ]] || voice_die "artifact root '${artifact_root}' does not exist"

example_cmd=("${example_binary}" --artifact-root "${artifact_root}")
if [[ "${quiet}" -eq 1 ]]; then
  example_cmd+=(--quiet)
fi
if [[ "${partials}" -eq 1 ]] && [[ "${asr_backend}" != 'whisper' ]]; then
  example_cmd+=(--partials)
fi

if [[ -n "${wav_path}" ]]; then
  exec <"${wav_path}"
  "${example_cmd[@]}"
  exit 0
fi

endpoint_kind="$(voice_require_endpoint_field "${endpoint_name}" KIND)"
record_cmd="$(voice_require_endpoint_field "${endpoint_name}" RECORD_CMD)"
if [[ -n "${capture_seconds}" ]]; then
  record_cmd="${record_cmd} trim 0 ${capture_seconds}"
fi

case "${endpoint_kind}" in
  local)
    bash -lc "${record_cmd}" | "${example_cmd[@]}"
    ;;
  ssh)
    ssh_target="$(voice_require_endpoint_field "${endpoint_name}" SSH_TARGET)"
    ssh "${ssh_target}" "${record_cmd}" | "${example_cmd[@]}"
    ;;
  *)
    voice_die "unsupported capture endpoint kind '${endpoint_kind}' for endpoint '${endpoint_name}'"
    ;;
esac

