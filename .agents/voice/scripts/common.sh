#!/usr/bin/env bash

set -euo pipefail

VOICE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOICE_ROOT="$(cd "${VOICE_SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${VOICE_ROOT}/../.." && pwd)"

voice_die() {
  echo "ERROR: $*" >&2
  exit 1
}

voice_note() {
  echo "[voice] $*" >&2
}

voice_has_cuda() {
  case "${MOTLIE_VOICE_ACCELERATION}" in
    auto)
      command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
      ;;
    cpu)
      return 1
      ;;
    cuda)
      command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 \
        || voice_die "MOTLIE_VOICE_ACCELERATION=cuda but no usable NVIDIA device is visible"
      ;;
    *)
      voice_die "unsupported MOTLIE_VOICE_ACCELERATION='${MOTLIE_VOICE_ACCELERATION}'"
      ;;
  esac
}

voice_prepend_path_var() {
  local var_name="$1"
  local value="$2"
  local current="${!var_name-}"

  if [[ -z "${current}" ]]; then
    printf -v "${var_name}" '%s' "${value}"
  else
    case ":${current}:" in
      *":${value}:"*) ;;
      *)
        printf -v "${var_name}" '%s:%s' "${value}" "${current}"
        ;;
    esac
  fi

  export "${var_name}"
}

voice_load_env() {
  export MOTLIE_VOICE_BUILD_PROFILE="${MOTLIE_VOICE_BUILD_PROFILE:-release}"
  export MOTLIE_VOICE_ACCELERATION="${MOTLIE_VOICE_ACCELERATION:-auto}"

  export PIPER_ARTIFACT_ROOT="${PIPER_ARTIFACT_ROOT:-$HOME/.cache/huggingface/hub}"
  export QWEN3_TTS_CPP_ARTIFACT_ROOT="${QWEN3_TTS_CPP_ARTIFACT_ROOT:-/tmp/qwen3-tts-models}"
  export WHISPER_ARTIFACT_ROOT="${WHISPER_ARTIFACT_ROOT:-$HOME/.cache/huggingface/hub}"
  export SHERPA_ARTIFACT_ROOT="${SHERPA_ARTIFACT_ROOT:-$HOME/.cache/huggingface/hub}"
  export MOONSHINE_ARTIFACT_ROOT="${MOONSHINE_ARTIFACT_ROOT:-$HOME/.cache/huggingface/hub}"

  export MOTLIE_VOICE_PLAYBACK_ENDPOINT="${MOTLIE_VOICE_PLAYBACK_ENDPOINT:-motliehost}"
  export MOTLIE_VOICE_CAPTURE_ENDPOINT="${MOTLIE_VOICE_CAPTURE_ENDPOINT:-motliehost}"

  export MOTLIE_ENDPOINT_MOTLIEHOST_KIND="${MOTLIE_ENDPOINT_MOTLIEHOST_KIND:-ssh}"
  export MOTLIE_ENDPOINT_MOTLIEHOST_SSH_TARGET="${MOTLIE_ENDPOINT_MOTLIEHOST_SSH_TARGET:-motliehost}"
  export MOTLIE_ENDPOINT_MOTLIEHOST_PLAY_CMD="${MOTLIE_ENDPOINT_MOTLIEHOST_PLAY_CMD:-/opt/homebrew/bin/play -t wav -}"
  export MOTLIE_ENDPOINT_MOTLIEHOST_RECORD_CMD="${MOTLIE_ENDPOINT_MOTLIEHOST_RECORD_CMD:-/opt/homebrew/bin/rec -q -c 1 -r 16000 -b 16 -e signed-integer -t wav -}"

  export MOTLIE_ENDPOINT_LOCAL_LINUX_KIND="${MOTLIE_ENDPOINT_LOCAL_LINUX_KIND:-local}"
  export MOTLIE_ENDPOINT_LOCAL_LINUX_PLAY_CMD="${MOTLIE_ENDPOINT_LOCAL_LINUX_PLAY_CMD:-play -t wav -}"
  export MOTLIE_ENDPOINT_LOCAL_LINUX_RECORD_CMD="${MOTLIE_ENDPOINT_LOCAL_LINUX_RECORD_CMD:-rec -q -c 1 -r 16000 -b 16 -e signed-integer -t wav -}"

  if [[ -f "${VOICE_ROOT}/voice.env" ]]; then
    # shellcheck disable=SC1091
    source "${VOICE_ROOT}/voice.env"
  fi

  if [[ -z "${ORT_LIB_PATH:-}" && -d "/tmp/onnxruntime-cuda/build/Linux-sm121/Release" ]]; then
    export ORT_LIB_PATH="/tmp/onnxruntime-cuda/build/Linux-sm121/Release"
  fi
  if [[ -n "${ORT_LIB_PATH:-}" ]]; then
    voice_prepend_path_var LD_LIBRARY_PATH "${ORT_LIB_PATH}"
  fi

  if [[ -z "${PIPER_ESPEAKNG_DATA_DIRECTORY:-}" && -d "/usr/lib/aarch64-linux-gnu/espeak-ng-data" ]]; then
    export PIPER_ESPEAKNG_DATA_DIRECTORY="/usr/lib/aarch64-linux-gnu"
  fi
}

voice_endpoint_key() {
  echo "$1" | tr '[:lower:]-' '[:upper:]_'
}

voice_endpoint_value() {
  local endpoint_name="$1"
  local field_name="$2"
  local endpoint_key
  local variable_name

  endpoint_key="$(voice_endpoint_key "${endpoint_name}")"
  variable_name="MOTLIE_ENDPOINT_${endpoint_key}_${field_name}"
  printf '%s' "${!variable_name-}"
}

voice_require_endpoint_field() {
  local endpoint_name="$1"
  local field_name="$2"
  local value

  value="$(voice_endpoint_value "${endpoint_name}" "${field_name}")"
  [[ -n "${value}" ]] || voice_die "missing endpoint field ${field_name} for endpoint '${endpoint_name}'"
  printf '%s' "${value}"
}

voice_profile_flag() {
  case "${MOTLIE_VOICE_BUILD_PROFILE}" in
    release)
      printf '%s' '--release'
      ;;
    debug)
      printf '%s' ''
      ;;
    *)
      voice_die "unsupported MOTLIE_VOICE_BUILD_PROFILE='${MOTLIE_VOICE_BUILD_PROFILE}'"
      ;;
  esac
}

voice_example_binary() {
  local example_name="$1"
  local profile_dir

  case "${MOTLIE_VOICE_BUILD_PROFILE}" in
    release) profile_dir='release' ;;
    debug) profile_dir='debug' ;;
    *) voice_die "unsupported MOTLIE_VOICE_BUILD_PROFILE='${MOTLIE_VOICE_BUILD_PROFILE}'" ;;
  esac

  printf '%s' "${REPO_ROOT}/target/${profile_dir}/examples/${example_name}"
}

voice_backend_metadata() {
  local backend_name="$1"

  case "${backend_name}" in
    piper)
      printf '%s|%s|%s|%s\n' \
        'tts_piper' \
        'model-piper-en-us-ljspeech-medium' \
        'piper-cuda' \
        "${PIPER_ARTIFACT_ROOT}"
      ;;
    qwen3cpp)
      printf '%s|%s|%s|%s\n' \
        'tts_qwen3_tts_cpp' \
        'model-qwen3-tts-cpp' \
        'qwen3-tts-cpp-cuda' \
        "${QWEN3_TTS_CPP_ARTIFACT_ROOT}"
      ;;
    whisper)
      printf '%s|%s|%s|%s\n' \
        'asr_whisper' \
        'model-whisper-base-en' \
        'whisper-cpp-cuda' \
        "${WHISPER_ARTIFACT_ROOT}"
      ;;
    sherpa)
      printf '%s|%s|%s|%s\n' \
        'asr_sherpa_onnx' \
        'model-sherpa-onnx-streaming' \
        'sherpa-onnx-cuda' \
        "${SHERPA_ARTIFACT_ROOT}"
      ;;
    moonshine)
      printf '%s|%s|%s|%s\n' \
        'asr_moonshine' \
        'model-moonshine-streaming' \
        '' \
        "${MOONSHINE_ARTIFACT_ROOT}"
      ;;
    *)
      voice_die "unsupported backend '${backend_name}'"
      ;;
  esac
}

voice_initialize_qwen_submodule_if_needed() {
  local qwen_root="${REPO_ROOT}/libs/model/backends/qwen3_tts_cpp/vendor/qwen3-tts.cpp"

  if [[ ! -f "${qwen_root}/CMakeLists.txt" ]]; then
    voice_note "initializing qwen3-tts.cpp submodule"
    (
      cd "${REPO_ROOT}"
      git submodule update --init --recursive libs/model/backends/qwen3_tts_cpp/vendor/qwen3-tts.cpp
    )
  fi
}

