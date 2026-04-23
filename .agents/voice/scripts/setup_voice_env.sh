#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

voice_load_env
voice_has_tty_prompt || voice_die "setup_voice_env.sh requires an interactive terminal"

playback_endpoint="$(voice_prompt_with_default "Default playback endpoint name" "${MOTLIE_VOICE_PLAYBACK_ENDPOINT}")"
capture_endpoint="$(voice_prompt_with_default "Default capture endpoint name" "${MOTLIE_VOICE_CAPTURE_ENDPOINT}")"

voice_store_env_var "MOTLIE_VOICE_PLAYBACK_ENDPOINT" "${playback_endpoint}"
voice_store_env_var "MOTLIE_VOICE_CAPTURE_ENDPOINT" "${capture_endpoint}"

setup_endpoint() {
  local endpoint_name="$1"
  local endpoint_key
  local kind
  local ssh_target=''
  local play_cmd=''
  local record_cmd=''
  local default_kind=''
  local default_play_cmd=''
  local default_record_cmd=''

  endpoint_key="$(voice_endpoint_key "${endpoint_name}")"
  default_kind="$(voice_endpoint_value "${endpoint_name}" KIND)"
  if [[ -z "${default_kind}" ]]; then
    if [[ "${endpoint_name}" == 'local-linux' ]]; then
      default_kind='local'
    else
      default_kind='ssh'
    fi
  fi

  kind="$(voice_prompt_with_default "Endpoint kind for '${endpoint_name}' (ssh/local)" "${default_kind}")"

  if [[ "${kind}" == 'local' ]]; then
    default_play_cmd="$(voice_endpoint_value "${endpoint_name}" PLAY_CMD)"
    default_record_cmd="$(voice_endpoint_value "${endpoint_name}" RECORD_CMD)"
    default_play_cmd="${default_play_cmd:-play -t wav -}"
    default_record_cmd="${default_record_cmd:-rec -q -c 1 -r 16000 -b 16 -e signed-integer -t wav -}"
  else
    default_play_cmd="$(voice_endpoint_value "${endpoint_name}" PLAY_CMD)"
    default_record_cmd="$(voice_endpoint_value "${endpoint_name}" RECORD_CMD)"
    default_play_cmd="${default_play_cmd:-/opt/homebrew/bin/play -t wav -}"
    default_record_cmd="${default_record_cmd:-/opt/homebrew/bin/rec -q -c 1 -r 16000 -b 16 -e signed-integer -t wav -}"
  fi

  play_cmd="$(voice_prompt_with_default "Playback command for '${endpoint_name}'" "${default_play_cmd}")"
  record_cmd="$(voice_prompt_with_default "Record command for '${endpoint_name}'" "${default_record_cmd}")"

  voice_store_env_var "MOTLIE_ENDPOINT_${endpoint_key}_KIND" "${kind}"
  if [[ "${kind}" == 'ssh' ]]; then
    ssh_target="$(voice_prompt_with_default "SSH target for '${endpoint_name}'" "$(voice_endpoint_value "${endpoint_name}" SSH_TARGET)")"
    ssh_target="${ssh_target:-${endpoint_name}}"
    voice_store_env_var "MOTLIE_ENDPOINT_${endpoint_key}_SSH_TARGET" "${ssh_target}"
  fi
  voice_store_env_var "MOTLIE_ENDPOINT_${endpoint_key}_PLAY_CMD" "${play_cmd}"
  voice_store_env_var "MOTLIE_ENDPOINT_${endpoint_key}_RECORD_CMD" "${record_cmd}"
}

setup_endpoint "${playback_endpoint}"
if [[ "${capture_endpoint}" != "${playback_endpoint}" ]]; then
  setup_endpoint "${capture_endpoint}"
fi

voice_store_env_var "MOTLIE_VOICE_REFERENCE_ROOT" "${MOTLIE_VOICE_REFERENCE_ROOT}"
voice_note "wrote configuration to ${VOICE_ENV_FILE}"
