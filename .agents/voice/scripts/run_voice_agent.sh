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

PROFILE="${MOTLIE_VOICE_BUILD_PROFILE:-release}"
BUILD_CMD=(cargo build -p voice-agent)
TARGET_DIR="debug"
if [[ "${PROFILE}" == "release" ]]; then
  BUILD_CMD+=(--release)
  TARGET_DIR="release"
elif [[ "${PROFILE}" != "debug" ]]; then
  echo "unsupported MOTLIE_VOICE_BUILD_PROFILE='${PROFILE}'" >&2
  exit 64
fi

PLATFORM_OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
PLATFORM_ARCH="$(uname -m | tr '[:upper:]' '[:lower:]')"
INSTALLED_BINARY="${SKILLS_BIN_DIR}/voice-agent-${PLATFORM_OS}-${PLATFORM_ARCH}-${PROFILE}"

if [[ -x "${INSTALLED_BINARY}" ]]; then
  exec "${INSTALLED_BINARY}" "${SUBCOMMAND}" "$@"
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "voice-agent binary '${INSTALLED_BINARY}' is missing and cargo is not available to build it" >&2
  exit 127
fi

cd "${REPO_ROOT}"
"${BUILD_CMD[@]}"
mkdir -p "${SKILLS_BIN_DIR}"
cp "${REPO_ROOT}/target/${TARGET_DIR}/voice-agent" "${INSTALLED_BINARY}"
chmod +x "${INSTALLED_BINARY}"
exec "${INSTALLED_BINARY}" "${SUBCOMMAND}" "$@"
