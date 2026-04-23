#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

voice_load_env

if [[ "$#" -lt 1 ]]; then
  voice_die "usage: ensure_examples.sh <backend> [<backend> ...]"
fi

declare -a example_flags=()
declare -a feature_list=()
declare -a cuda_feature_list=()
declare -A seen_features=()
declare -A seen_examples=()
needs_qwen_submodule=0

for backend_name in "$@"; do
  IFS='|' read -r example_name model_feature cuda_feature _artifact_root <<<"$(voice_backend_metadata "${backend_name}")"

  if [[ -z "${seen_examples[${example_name}]-}" ]]; then
    example_flags+=(--example "${example_name}")
    seen_examples["${example_name}"]=1
  fi

  if [[ -z "${seen_features[${model_feature}]-}" ]]; then
    feature_list+=("${model_feature}")
    seen_features["${model_feature}"]=1
  fi

  if [[ -n "${cuda_feature}" ]] && voice_has_cuda && [[ -z "${seen_features[${cuda_feature}]-}" ]]; then
    cuda_feature_list+=("${cuda_feature}")
    seen_features["${cuda_feature}"]=1
  fi

  if [[ "${backend_name}" == 'qwen3cpp' ]]; then
    needs_qwen_submodule=1
  fi
done

if [[ "${needs_qwen_submodule}" -eq 1 ]]; then
  voice_initialize_qwen_submodule_if_needed
fi

feature_csv="$(IFS=,; echo "${feature_list[*]}${feature_list:+${cuda_feature_list:+,}}${cuda_feature_list[*]}")"
[[ -n "${feature_csv}" ]] || voice_die "no features selected for build"

build_cmd=(cargo build -p motlie-models)
profile_flag="$(voice_profile_flag)"
if [[ -n "${profile_flag}" ]]; then
  build_cmd+=("${profile_flag}")
fi
build_cmd+=("${example_flags[@]}" --no-default-features --features "${feature_csv}")

voice_note "building examples with features: ${feature_csv}"
(
  cd "${REPO_ROOT}"
  "${build_cmd[@]}"
)

