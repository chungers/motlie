#!/usr/bin/env bash
set -euo pipefail

require_espeak=0
require_ort=0
require_qwen_submodule=0

while (($#)); do
  case "$1" in
    --require-espeak)
      require_espeak=1
      ;;
    --require-ort)
      require_ort=1
      ;;
    --require-qwen-submodule)
      require_qwen_submodule=1
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

note() {
  echo "[models-build] $*"
}

find_espeak_dir() {
  if [[ -n "${ESPEAK_NG_LIB_DIR:-}" ]]; then
    [[ -e "${ESPEAK_NG_LIB_DIR}/libespeak-ng.so" || -e "${ESPEAK_NG_LIB_DIR}/libespeak-ng.so.1" || -e "${ESPEAK_NG_LIB_DIR}/libespeak-ng.dylib" ]] && {
      printf '%s\n' "${ESPEAK_NG_LIB_DIR}"
      return 0
    }
  fi

  if command -v ldconfig >/dev/null 2>&1; then
    local match
    match="$(ldconfig -p 2>/dev/null | grep 'libespeak-ng\.so' | head -n1 | awk '{print $NF}')"
    if [[ -n "${match}" ]]; then
      dirname "${match}"
      return 0
    fi
  fi

  local dir
  for dir in /usr/lib /usr/local/lib /lib /usr/lib/aarch64-linux-gnu /lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu /opt/homebrew/lib; do
    [[ -e "${dir}/libespeak-ng.so" || -e "${dir}/libespeak-ng.so.1" || -e "${dir}/libespeak-ng.dylib" ]] && {
      printf '%s\n' "${dir}"
      return 0
    }
  done

  return 1
}

find_ort_dir() {
  if [[ -n "${ORT_LIB_PATH:-}" ]]; then
    local dir
    for dir in "${ORT_LIB_PATH}" "${ORT_LIB_PATH}/Release" "${ORT_LIB_PATH}/RelWithDebInfo" "${ORT_LIB_PATH}/MinSizeRel" "${ORT_LIB_PATH}/Debug"; do
      [[ -e "${dir}/libonnxruntime.a" || -e "${dir}/libonnxruntime_common.a" ]] && {
        printf '%s\n' "${dir}"
        return 0
      }
    done
  fi

  return 1
}

check_espeak() {
  local dir
  dir="$(find_espeak_dir)" || fail "libespeak-ng not found. Install libespeak-ng-dev or set ESPEAK_NG_LIB_DIR."
  note "found libespeak-ng in ${dir}"
}

check_ort() {
  case "${ORT_PREFER_DYNAMIC_LINK:-}" in
    1|true|TRUE|True)
      fail "ORT_PREFER_DYNAMIC_LINK must not be set for Motlie model checks; build static ONNX Runtime and set ORT_LIB_PATH to its build output."
      ;;
  esac

  local dir
  dir="$(find_ort_dir)" || fail "Static ONNX Runtime not found. Build ONNX Runtime from source and set ORT_LIB_PATH to a directory containing libonnxruntime.a or libonnxruntime_common.a."
  note "found static ONNX Runtime in ${dir}"
}

check_qwen_submodule() {
  local submodule_dir="${repo_root}/libs/model/backends/qwen3_tts_cpp/vendor/qwen3-tts.cpp"
  [[ -f "${submodule_dir}/CMakeLists.txt" ]] || fail "qwen3-tts.cpp submodule is missing. Run: git submodule update --init --recursive ${submodule_dir#${repo_root}/}"
  note "qwen3-tts.cpp submodule is initialized"
}

(( require_espeak )) && check_espeak
(( require_ort )) && check_ort
(( require_qwen_submodule )) && check_qwen_submodule

note "build prerequisites satisfied"
