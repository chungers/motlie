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
VOICE_SUBSKILLS=(speak listen turn)

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

linux_ort_candidate_dirs() {
  local candidate
  local extra
  shopt -s nullglob
  if [[ -n "${MOTLIE_ORT_LIB_CANDIDATES:-}" ]]; then
    IFS=':' read -r -a extra <<<"${MOTLIE_ORT_LIB_CANDIDATES}"
    for candidate in "${extra[@]}"; do
      [[ -d "${candidate}" ]] || continue
      printf '%s\n' "${candidate}"
    done
  fi
  for candidate in \
    /usr/local/lib \
    /usr/lib \
    /usr/lib/aarch64-linux-gnu \
    /lib/aarch64-linux-gnu \
    /usr/lib/x86_64-linux-gnu \
    /lib/x86_64-linux-gnu \
    /opt/onnxruntime/lib \
    /opt/onnxruntime/lib64 \
    /tmp/onnxruntime*/build/*/Release \
    /tmp/onnxruntime*/build/*/*/Release; do
    [[ -d "${candidate}" ]] || continue
    printf '%s\n' "${candidate}"
  done
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
    while IFS= read -r candidate; do
      if [[ -e "${candidate}/libonnxruntime.so" || -e "${candidate}/libonnxruntime.so.1" ]]; then
        printf '%s\n' "${candidate}"
        return 0
      fi
    done < <(linux_ort_candidate_dirs)
  elif [[ "${PLATFORM_OS}" == "darwin" ]]; then
    for candidate in \
      /opt/homebrew/lib \
      /usr/local/lib \
      /opt/local/lib; do
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

subskill_bin_dir() {
  printf '%s\n' "${VOICE_DIR}/$1/bin"
}

platform_binary_name() {
  printf 'voice-agent-%s-%s-%s-%s\n' "${PLATFORM_OS}" "${PLATFORM_ARCH}" "${PROFILE}" "$1"
}

legacy_binary_name() {
  printf 'voice-agent-%s-%s-%s\n' "${PLATFORM_OS}" "${PLATFORM_ARCH}" "${PROFILE}"
}

qwen_sidecar_present() {
  local dir="$1"
  [[ -e "${dir}/libqwen3tts.so.0" || -e "${dir}/libqwen3tts.dylib" ]]
}

resolve_qwen_sidecar_dir() {
  local search_root="${REPO_ROOT}/target/${TARGET_DIR}"
  local candidate
  [[ -d "${search_root}" ]] || return 1
  while IFS= read -r candidate; do
    printf '%s\n' "$(dirname "${candidate}")"
    return 0
  done < <(find "${search_root}" \( -type f -o -type l \) \( -name 'libqwen3tts.so' -o -name 'libqwen3tts.so.*' -o -name 'libqwen3tts.dylib' -o -name 'libqwen3tts.*.dylib' \) 2>/dev/null | sort)
  return 1
}

copy_qwen_sidecars_to_dir() {
  local source_dir="$1"
  local target_dir="$2"
  mkdir -p "${target_dir}"
  if [[ "${PLATFORM_OS}" == "linux" ]]; then
    cp -a "${source_dir}/libqwen3tts.so"* "${target_dir}/"
  elif [[ "${PLATFORM_OS}" == "darwin" ]]; then
    cp -a "${source_dir}/libqwen3tts"*.dylib* "${target_dir}/"
  fi
}

install_qwen_sidecars() {
  local mode="$1"
  local qwen_dir
  qwen_dir="$(resolve_qwen_sidecar_dir || true)"
  if [[ -z "${qwen_dir}" ]]; then
    if [[ "${mode}" == "require" ]]; then
      echo "voice-agent build completed, but libqwen3tts sidecars were not found under target/${TARGET_DIR}" >&2
      echo "the qwen3-tts.cpp backend layout likely changed; update install_qwen_sidecars before shipping this skill" >&2
      exit 127
    fi
    return 0
  fi

  local subskill
  for subskill in "${VOICE_SUBSKILLS[@]}"; do
    copy_qwen_sidecars_to_dir "${qwen_dir}" "$(subskill_bin_dir "${subskill}")"
  done
}

install_ort_sidecars() {
  local ort_dir
  ort_dir="$(ort_sidecar_dir || true)"
  if [[ -z "${ort_dir}" || "${ort_dir}" == "${VOICE_LIB_DIR}" ]]; then
    return 0
  fi

  mkdir -p "${VOICE_LIB_DIR}"
  if [[ "${PLATFORM_OS}" == "linux" ]]; then
    cp -a "${ort_dir}/libonnxruntime.so"* "${VOICE_LIB_DIR}/"
    [[ -e "${ort_dir}/libonnxruntime_providers_shared.so" ]] && cp -a "${ort_dir}/libonnxruntime_providers_shared.so" "${VOICE_LIB_DIR}/"
    [[ -e "${ort_dir}/libonnxruntime_providers_cuda.so" ]] && cp -a "${ort_dir}/libonnxruntime_providers_cuda.so" "${VOICE_LIB_DIR}/"
  elif [[ "${PLATFORM_OS}" == "darwin" ]]; then
    cp -a "${ort_dir}/libonnxruntime.dylib" "${VOICE_LIB_DIR}/"
    local candidate
    for candidate in "${ort_dir}"/libonnxruntime.*.dylib; do
      [[ -e "${candidate}" ]] || continue
      cp -a "${candidate}" "${VOICE_LIB_DIR}/"
    done
  fi
}

link_or_copy_file() {
  local source="$1"
  local target="$2"
  rm -f "${target}"
  if ! ln "${source}" "${target}" 2>/dev/null; then
    cp -a "${source}" "${target}"
  fi
}

install_voice_agent_binary() {
  local source_binary="$1"
  local flavor="$2"
  local binary_name primary_dir primary_binary subskill dir target
  binary_name="$(platform_binary_name "${flavor}")"
  primary_dir="$(subskill_bin_dir "${SUBCOMMAND}")"
  mkdir -p "${primary_dir}"
  primary_binary="${primary_dir}/${binary_name}"
  rm -f "${primary_binary}"
  cp -a "${source_binary}" "${primary_binary}"
  chmod +x "${primary_binary}"

  for subskill in "${VOICE_SUBSKILLS[@]}"; do
    dir="$(subskill_bin_dir "${subskill}")"
    mkdir -p "${dir}"
    target="${dir}/${binary_name}"
    [[ "${target}" == "${primary_binary}" ]] && continue
    link_or_copy_file "${primary_binary}" "${target}"
    chmod +x "${target}"
  done
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

  local binary_dir
  binary_dir="$(dirname "${binary}")"
  if ! qwen_sidecar_present "${binary_dir}"; then
    echo "voice-agent binary '${binary}' is missing libqwen3tts sidecars in '${binary_dir}'" >&2
    echo "rebuild from source or repopulate the per-skill bin directories before running the voice skill" >&2
    exit 127
  fi

  local loader_var="LD_LIBRARY_PATH"
  if [[ "${PLATFORM_OS}" == "darwin" ]]; then
    loader_var="DYLD_FALLBACK_LIBRARY_PATH"
  fi

  local loader_path="${binary_dir}:${ort_dir}"
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

source_tree_newer_than_binary() {
  local binary="$1"
  find     "${REPO_ROOT}/Cargo.toml"     "${REPO_ROOT}/Cargo.lock"     "${REPO_ROOT}/bins/voice-agent"     "${REPO_ROOT}/libs/model"     "${REPO_ROOT}/libs/models"     "${REPO_ROOT}/libs/voice"     \( -path '*/target' -o -path '*/target/*' -o -path '*/vendor/qwen3-tts.cpp/ggml/build' -o -path '*/vendor/qwen3-tts.cpp/ggml/build/*' \) -prune -o     -type f -newer "${binary}" -print -quit 2>/dev/null | grep -q .
}

binary_is_current() {
  local binary="$1"

  [[ -x "${binary}" ]] || return 1
  if ! qwen_sidecar_present "$(dirname "${binary}")"; then
    return 1
  fi

  if ! command -v cargo >/dev/null 2>&1; then
    return 0
  fi

  ! source_tree_newer_than_binary "${binary}"
}

resolve_flavor() {
  if has_cuda; then
    printf '%s\n' "cuda"
  else
    printf '%s\n' "cpu"
  fi
}

PREFERRED_FLAVOR="$(resolve_flavor)"
SKILL_BIN_DIR="$(subskill_bin_dir "${SUBCOMMAND}")"
VOICE_LIB_DIR="${VOICE_DIR}/lib/${PLATFORM_OS}-${PLATFORM_ARCH}"
LEGACY_BINARY="${SKILL_BIN_DIR}/$(legacy_binary_name)"

if [[ "${PREFERRED_FLAVOR}" == "cuda" ]]; then
  BUILD_CMD+=(--features cuda)
fi

flavor_candidates() {
  if [[ "${PREFERRED_FLAVOR}" == "cuda" ]]; then
    printf '%s\n' cuda cpu
  else
    printf '%s\n' cpu cuda
  fi
}

installed_binary_candidates() {
  local flavor subskill
  while IFS= read -r flavor; do
    for subskill in "${VOICE_SUBSKILLS[@]}"; do
      printf '%s\n' "$(subskill_bin_dir "${subskill}")/$(platform_binary_name "${flavor}")"
    done
  done < <(flavor_candidates)
}

while IFS= read -r installed_binary; do
  if binary_is_current "${installed_binary}"; then
    exec_with_skill_env "${installed_binary}" "${SUBCOMMAND}" "$@"
  fi
done < <(installed_binary_candidates)

INSTALL_BINARY="${SKILL_BIN_DIR}/$(platform_binary_name "${PREFERRED_FLAVOR}")"

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
mkdir -p "${VOICE_LIB_DIR}"
install_ort_sidecars
ACTIVE_ORT_DIR="$(ort_sidecar_dir || true)"
if [[ -z "${ACTIVE_ORT_DIR}" ]]; then
  print_ort_install_guidance
  exit 127
fi

BUILD_ENV=(env "ORT_LIB_PATH=${ACTIVE_ORT_DIR}" "ORT_PREFER_DYNAMIC_LINK=1")
echo "[voice-agent] building optimized ${PREFERRED_FLAVOR} binary for ${PLATFORM_OS}-${PLATFORM_ARCH}; please wait..." >&2
"${BUILD_ENV[@]}" "${BUILD_CMD[@]}"
install_voice_agent_binary "${REPO_ROOT}/target/${TARGET_DIR}/voice-agent" "${PREFERRED_FLAVOR}"
install_qwen_sidecars require
install_ort_sidecars
exec_with_skill_env "${INSTALL_BINARY}" "${SUBCOMMAND}" "$@"
