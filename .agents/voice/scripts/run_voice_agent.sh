#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../" && pwd)"
SKILLS_BIN_DIR="${REPO_ROOT}/.agents/skills/bin"

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

has_cuda() {
  nvidia-smi -L >/dev/null 2>&1
}

source_available() {
  [[ -f "${REPO_ROOT}/Cargo.toml" ]] && [[ -f "${REPO_ROOT}/bins/voice-agent/Cargo.toml" ]]
}

binary_is_current() {
  local binary="$1"

  [[ -x "${binary}" ]] || return 1

  if ! command -v cargo >/dev/null 2>&1; then
    return 0
  fi

  shopt -s globstar nullglob
  local path
  for path in \
    "${REPO_ROOT}/Cargo.toml" \
    "${REPO_ROOT}/Cargo.lock" \
    "${REPO_ROOT}/bins/voice-agent"/** \
    "${REPO_ROOT}/.agents/voice"/** \
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
LEGACY_BINARY="${SKILLS_BIN_DIR}/voice-agent-${PLATFORM_OS}-${PLATFORM_ARCH}-${PROFILE}"

flavor_candidate_binaries() {
  if [[ "${PREFERRED_FLAVOR}" == "cuda" ]]; then
    printf '%s\n' \
      "${SKILLS_BIN_DIR}/voice-agent-${PLATFORM_OS}-${PLATFORM_ARCH}-${PROFILE}-cuda" \
      "${SKILLS_BIN_DIR}/voice-agent-${PLATFORM_OS}-${PLATFORM_ARCH}-${PROFILE}-cpu"
  else
    printf '%s\n' \
      "${SKILLS_BIN_DIR}/voice-agent-${PLATFORM_OS}-${PLATFORM_ARCH}-${PROFILE}-cpu" \
      "${SKILLS_BIN_DIR}/voice-agent-${PLATFORM_OS}-${PLATFORM_ARCH}-${PROFILE}-cuda"
  fi
}

while IFS= read -r INSTALLED_BINARY; do
  if binary_is_current "${INSTALLED_BINARY}"; then
    exec "${INSTALLED_BINARY}" "${SUBCOMMAND}" "$@"
  fi
done < <(flavor_candidate_binaries)

INSTALL_BINARY="${SKILLS_BIN_DIR}/voice-agent-${PLATFORM_OS}-${PLATFORM_ARCH}-${PROFILE}-${PREFERRED_FLAVOR}"

if ! command -v cargo >/dev/null 2>&1; then
  if [[ -x "${LEGACY_BINARY}" ]]; then
    exec "${LEGACY_BINARY}" "${SUBCOMMAND}" "$@"
  fi
  echo "voice-agent binary '${INSTALL_BINARY}' is missing and cargo is not available to build it" >&2
  exit 127
fi

if ! source_available; then
  if [[ -x "${LEGACY_BINARY}" ]]; then
    exec "${LEGACY_BINARY}" "${SUBCOMMAND}" "$@"
  fi
  echo "voice-agent binary '${INSTALL_BINARY}' is missing and the source tree is not available on this host" >&2
  echo "ship a prebuilt platform binary in .agents/skills/bin/ or deploy the repo source to build it locally" >&2
  exit 127
fi

cd "${REPO_ROOT}"
echo "[voice-agent] building optimized ${PREFERRED_FLAVOR} binary for ${PLATFORM_OS}-${PLATFORM_ARCH}; please wait..." >&2
"${BUILD_CMD[@]}"
mkdir -p "${SKILLS_BIN_DIR}"
cp "${REPO_ROOT}/target/${TARGET_DIR}/voice-agent" "${INSTALL_BINARY}"
chmod +x "${INSTALL_BINARY}"
exec "${INSTALL_BINARY}" "${SUBCOMMAND}" "$@"
