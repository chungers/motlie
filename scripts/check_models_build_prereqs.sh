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

check_espeak() {
  local dir
  dir="$(find_espeak_dir)" || fail "libespeak-ng not found. Install libespeak-ng-dev or set ESPEAK_NG_LIB_DIR."
  note "found libespeak-ng in ${dir}"
}

check_ort() {
  [[ -z "${ORT_LIB_PATH:-}" ]] || fail "ORT_LIB_PATH must not be set for Motlie model checks; ort/download-binaries fetches and statically links libonnxruntime.a."
  [[ -z "${ORT_LIB_LOCATION:-}" ]] || fail "ORT_LIB_LOCATION must not be set for Motlie model checks; ort/download-binaries fetches and statically links libonnxruntime.a."

  case "${ORT_PREFER_DYNAMIC_LINK:-}" in
    1|true|TRUE|True)
      fail "ORT_PREFER_DYNAMIC_LINK must not be set for Motlie model checks; ort/download-binaries statically links libonnxruntime.a."
      ;;
  esac
  case "${ORT_SKIP_DOWNLOAD:-}" in
    1|true|TRUE|True)
      fail "ORT_SKIP_DOWNLOAD must not be set for Motlie model checks; ort/download-binaries must fetch the prebuilt static archive."
      ;;
  esac
  case "${ORT_OFFLINE:-}" in
    1|true|TRUE|True)
      fail "ORT_OFFLINE must not be set for Motlie model checks; ort/download-binaries must fetch the prebuilt static archive."
      ;;
  esac
  case "${CARGO_NET_OFFLINE:-}" in
    1|true|TRUE|True)
      fail "CARGO_NET_OFFLINE must not be set for Motlie ORT model checks; ort/download-binaries must fetch the prebuilt static archive."
      ;;
  esac

  note "ORT prebuilt static archive download is enabled through the workspace ort dependency"
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
