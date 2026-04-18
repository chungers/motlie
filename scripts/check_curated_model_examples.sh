#!/usr/bin/env bash
set -euo pipefail

mode="check"

while (($#)); do
  case "$1" in
    --mode)
      shift
      mode="${1:-}"
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

features="model-whisper-base-en,model-sherpa-onnx-streaming,model-moonshine-streaming,model-piper-en-us-ljspeech-medium,model-qwen3-tts-0_6b,model-qwen3-tts-cpp"

run_with_optional_ort() {
  if [[ -n "${ORT_LIB_PATH:-}" ]]; then
    ORT_LIB_PATH="${ORT_LIB_PATH}" "$@"
  else
    "$@"
  fi
}

case "${mode}" in
  check)
    ./scripts/check_models_build_prereqs.sh --require-espeak --require-qwen-submodule
    cargo check -p motlie-model-espeak-ng -p motlie-model-piper --lib
    run_with_optional_ort cargo check -p motlie-models --lib --examples --no-default-features --features "${features}"
    ;;
  build)
    ./scripts/check_models_build_prereqs.sh --require-espeak --require-qwen-submodule --require-ort
    ORT_LIB_PATH="${ORT_LIB_PATH}" cargo build -p motlie-models \
      --example tts_piper \
      --example tts_qwen3_onnx \
      --example tts_qwen3_tts_cpp \
      --example asr_whisper \
      --example asr_sherpa_onnx \
      --example asr_moonshine \
      --no-default-features \
      --features "${features}"
    ;;
  *)
    echo "unsupported mode: ${mode}" >&2
    exit 2
    ;;
esac
