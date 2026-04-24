#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOICE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"

if [[ $# -lt 1 ]]; then
  echo "usage: run_voice_agent.sh <subcommand> [args...]" >&2
  exit 64
fi

SUBCOMMAND="$1"
shift

PROFILE="release"
BUILD_CMD=(cargo build -p voice-agent --release)
TARGET_DIR="release"

PLATFORM_OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
PLATFORM_ARCH="$(uname -m | tr '[:upper:]' '[:lower:]')"
VOICE_LIB_DIR=""

has_cuda() {
  nvidia-smi -L >/dev/null 2>&1
}

ensure_qwen_submodule() {
  local qwen_root="${REPO_ROOT}/libs/model/backends/qwen3_tts_cpp/vendor/qwen3-tts.cpp"
  [[ -f "${qwen_root}/CMakeLists.txt" ]] && return 0
  if ! command -v git >/dev/null 2>&1; then
    echo "voice-agent source is present, but git is unavailable to initialize the qwen3-tts.cpp submodule" >&2
    return 1
  fi
  echo "[voice-agent] initializing qwen3-tts.cpp submodule; please wait..." >&2
  git -C "${REPO_ROOT}" submodule update --init --recursive libs/model/backends/qwen3_tts_cpp/vendor/qwen3-tts.cpp
}

source_available() {
  [[ -f "${REPO_ROOT}/Cargo.toml" ]] && [[ -f "${REPO_ROOT}/bins/voice-agent/Cargo.toml" ]]
}

ort_sidecar_dir() {
  local candidate

  if [[ -n "${VOICE_LIB_DIR}" ]] && [[ -e "${VOICE_LIB_DIR}/libonnxruntime.so" || -e "${VOICE_LIB_DIR}/libonnxruntime.dylib" ]]; then
    printf '%s\n' "${VOICE_LIB_DIR}"
    return 0
  fi

  if [[ -n "${ORT_LIB_PATH:-}" ]] && [[ -e "${ORT_LIB_PATH}/libonnxruntime.so" || -e "${ORT_LIB_PATH}/libonnxruntime.dylib" ]]; then
    printf '%s\n' "${ORT_LIB_PATH}"
    return 0
  fi

  if [[ "${PLATFORM_OS}" == "linux" ]]; then
    for candidate in \
      /tmp/onnxruntime-cuda/build/Linux-sm121/Release \
      /tmp/onnxruntime/build/Linux/Release \
      /usr/local/lib \
      /usr/lib; do
      if [[ -e "${candidate}/libonnxruntime.so" || -e "${candidate}/libonnxruntime.so.1" ]]; then
        printf '%s\n' "${candidate}"
        return 0
      fi
    done
  elif [[ "${PLATFORM_OS}" == "darwin" ]]; then
    for candidate in \
      /opt/homebrew/lib \
      /usr/local/lib; do
      if [[ -e "${candidate}/libonnxruntime.dylib" ]]; then
        printf '%s\n' "${candidate}"
        return 0
      fi
    done
  fi

  return 1
}

print_ort_install_guidance() {
  if [[ "${PLATFORM_OS}" == "darwin" ]]; then
    cat >&2 <<'EOT'
voice-agent needs ONNX Runtime installed on this host before it can build or run the ONNX-backed voice backends.

Install it with:
  brew install onnxruntime

That should provide:
  /opt/homebrew/lib/libonnxruntime.dylib
EOT
    return 0
  fi

  cat >&2 <<'EOT'
voice-agent needs ONNX Runtime shared libraries on this host before it can build or run the ONNX-backed voice backends.

Required files:
  libonnxruntime.so
  libonnxruntime.so.1

For CUDA-backed ONNX paths, also:
  libonnxruntime_providers_shared.so
  libonnxruntime_providers_cuda.so

If this host does not provide distro packages, build ONNX Runtime from source as a shared library:
  git clone --recursive https://github.com/microsoft/onnxruntime.git
  cd onnxruntime
  python3 -m pip install cmake

CPU build:
  ./build.sh --config Release --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync

CUDA build:
  ./build.sh --config Release --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync \
    --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr/local/cuda
EOT
}

install_runtime_sidecars() {
  local qwen_lib
  qwen_lib="$(find "${REPO_ROOT}/target/${TARGET_DIR}/build" -path '*/out/build/vendor-build/libqwen3tts.so.0' 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -n "${qwen_lib}" ]]; then
    local qwen_lib_dir
    qwen_lib_dir="$(dirname "${qwen_lib}")"
    mkdir -p "${SKILL_BIN_DIR}"
    cp -a "${qwen_lib_dir}/libqwen3tts.so" "${SKILL_BIN_DIR}/"
    cp -a "${qwen_lib_dir}/libqwen3tts.so.0" "${SKILL_BIN_DIR}/"
    cp -a "${qwen_lib_dir}/libqwen3tts.so.0.1.0" "${SKILL_BIN_DIR}/"
  fi

  local ort_dir
  ort_dir="$(ort_sidecar_dir || true)"
  if [[ -n "${ort_dir}" ]]; then
    if [[ "${ort_dir}" == "${VOICE_LIB_DIR}" ]]; then
      return 0
    fi
    mkdir -p "${VOICE_LIB_DIR}"
    if [[ "${PLATFORM_OS}" == "linux" ]]; then
      cp -a "${ort_dir}/libonnxruntime.so"* "${VOICE_LIB_DIR}/"
      [[ -e "${ort_dir}/libonnxruntime_providers_shared.so" ]] && cp -a "${ort_dir}/libonnxruntime_providers_shared.so" "${VOICE_LIB_DIR}/"
      [[ -e "${ort_dir}/libonnxruntime_providers_cuda.so" ]] && cp -a "${ort_dir}/libonnxruntime_providers_cuda.so" "${VOICE_LIB_DIR}/"
    elif [[ "${PLATFORM_OS}" == "darwin" ]]; then
      cp -a "${ort_dir}/libonnxruntime.dylib" "${VOICE_LIB_DIR}/"
      for candidate in "${ort_dir}"/libonnxruntime.*.dylib; do
        [[ -e "${candidate}" ]] || continue
        cp -a "${candidate}" "${VOICE_LIB_DIR}/"
      done
    fi
  fi
}

exec_with_skill_env() {
  local binary="$1"
  shift

  local ort_dir
  ort_dir="$(ort_sidecar_dir || true)"
  if [[ -z "${ort_dir}" ]]; then
    print_ort_install_guidance
    exit 127
  fi

  local loader_var="LD_LIBRARY_PATH"
  if [[ "${PLATFORM_OS}" == "darwin" ]]; then
    loader_var="DYLD_FALLBACK_LIBRARY_PATH"
  fi

  local loader_path="${SKILL_BIN_DIR}:${ort_dir}"
  if [[ -n "${!loader_var:-}" ]]; then
    loader_path="${loader_path}:${!loader_var}"
  fi

  exec env \
    MOTLIE_VOICE_SKILL_ROOT="${VOICE_DIR}" \
    ORT_LIB_PATH="${ort_dir}" \
    ORT_PREFER_DYNAMIC_LINK=1 \
    "${loader_var}=${loader_path}" \
    "${binary}" "$@"
}

binary_is_current() {
  local binary="$1"

  [[ -x "${binary}" ]] || return 1

  if ! command -v cargo >/dev/null 2>&1; then
    return 0
  fi

  shopt -s globstar nullglob
  [[ -e "${SKILL_BIN_DIR}/libqwen3tts.so.0" ]] || return 1

  local path
  for path in \
    "${REPO_ROOT}/Cargo.toml" \
    "${REPO_ROOT}/Cargo.lock" \
    "${REPO_ROOT}/bins/voice-agent"/** \
    "${REPO_ROOT}/libs/model"/** \
    "${REPO_ROOT}/libs/models"/** \
    "${REPO_ROOT}/libs/voice"/**; do
    if [[ -f "${path}" && "${path}" -nt "${binary}" ]]; then
      return 1
    fi
  done
  return 0
}

resolve_flavor() {
  if has_cuda; then
    printf '%s\n' "cuda"
  else
    printf '%s\n' "cpu"
  fi
}

PREFERRED_FLAVOR="$(resolve_flavor)"
SKILL_BIN_DIR="${VOICE_DIR}/${SUBCOMMAND}/bin"
VOICE_LIB_DIR="${VOICE_DIR}/lib/${PLATFORM_OS}-${PLATFORM_ARCH}"
LEGACY_BINARY="${SKILL_BIN_DIR}/voice-agent-${PLATFORM_OS}-${PLATFORM_ARCH}-${PROFILE}"

if [[ "${PREFERRED_FLAVOR}" == "cuda" ]]; then
  BUILD_CMD+=(--features cuda)
fi

flavor_candidate_binaries() {
  if [[ "${PREFERRED_FLAVOR}" == "cuda" ]]; then
    printf '%s\n' \
      "${SKILL_BIN_DIR}/voice-agent-${PLATFORM_OS}-${PLATFORM_ARCH}-${PROFILE}-cuda" \
      "${SKILL_BIN_DIR}/voice-agent-${PLATFORM_OS}-${PLATFORM_ARCH}-${PROFILE}-cpu"
  else
    printf '%s\n' \
      "${SKILL_BIN_DIR}/voice-agent-${PLATFORM_OS}-${PLATFORM_ARCH}-${PROFILE}-cpu" \
      "${SKILL_BIN_DIR}/voice-agent-${PLATFORM_OS}-${PLATFORM_ARCH}-${PROFILE}-cuda"
  fi
}

while IFS= read -r INSTALLED_BINARY; do
  if binary_is_current "${INSTALLED_BINARY}"; then
    exec_with_skill_env "${INSTALLED_BINARY}" "${SUBCOMMAND}" "$@"
  fi
done < <(flavor_candidate_binaries)

INSTALL_BINARY="${SKILL_BIN_DIR}/voice-agent-${PLATFORM_OS}-${PLATFORM_ARCH}-${PROFILE}-${PREFERRED_FLAVOR}"

if ! command -v cargo >/dev/null 2>&1; then
  if [[ -x "${LEGACY_BINARY}" ]]; then
    exec_with_skill_env "${LEGACY_BINARY}" "${SUBCOMMAND}" "$@"
  fi
  echo "voice-agent binary '${INSTALL_BINARY}' is missing and cargo is not available to build it" >&2
  print_ort_install_guidance
  exit 127
fi

if ! source_available; then
  if [[ -x "${LEGACY_BINARY}" ]]; then
    exec_with_skill_env "${LEGACY_BINARY}" "${SUBCOMMAND}" "$@"
  fi
  echo "voice-agent binary '${INSTALL_BINARY}' is missing and the source tree is not available on this host" >&2
  echo "ship a prebuilt platform binary in .agents/skills/voice/${SUBCOMMAND}/bin/ or deploy the repo source to build it locally" >&2
  echo "on first use, the binary will also bootstrap model weights into .agents/skills/voice/artifacts/hf-cache/" >&2
  print_ort_install_guidance
  exit 127
fi

cd "${REPO_ROOT}"
ensure_qwen_submodule
mkdir -p "${SKILL_BIN_DIR}" "${VOICE_LIB_DIR}"
install_runtime_sidecars
ACTIVE_ORT_DIR="$(ort_sidecar_dir || true)"
if [[ -z "${ACTIVE_ORT_DIR}" ]]; then
  print_ort_install_guidance
  exit 127
fi

BUILD_ENV=(env "ORT_LIB_PATH=${ACTIVE_ORT_DIR}" "ORT_PREFER_DYNAMIC_LINK=1")
echo "[voice-agent] building optimized ${PREFERRED_FLAVOR} binary for ${PLATFORM_OS}-${PLATFORM_ARCH}; please wait..." >&2
"${BUILD_ENV[@]}" "${BUILD_CMD[@]}"
mkdir -p "${SKILL_BIN_DIR}"
cp "${REPO_ROOT}/target/${TARGET_DIR}/voice-agent" "${INSTALL_BINARY}"
chmod +x "${INSTALL_BINARY}"
install_runtime_sidecars
exec_with_skill_env "${INSTALL_BINARY}" "${SUBCOMMAND}" "$@"
