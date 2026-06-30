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

speech_features="model-whisper-base-en,model-sherpa-onnx-streaming,model-moonshine-streaming,model-piper-en-us-ljspeech-medium,model-qwen3-tts-cpp"
chat_tool_features="model-qwen3-4b,model-gemma4-e2b"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

note() {
  echo "[curated-examples] $*"
}

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

    note "checking motlie-models lib with curated speech features"
    run_with_optional_ort cargo check -p motlie-models \
      --lib \
      --no-default-features \
      --features "${speech_features}"

    note "checking chat/tool examples without ORT-backed Piper/Moonshine/Sherpa features"
    cargo check -p motlie-models \
      --example chat_tool_binding \
      --example chat_multimodal_gemma4 \
      --example chat_gguf_gwen3_gemma4 \
      --example bench_chat \
      --no-default-features \
      --features "${chat_tool_features}"

    note "checking ASR/TTS examples with ORT-backed features isolated from chat/tool examples"
    run_with_optional_ort cargo check -p motlie-models \
      --example tts_piper \
      --example tts_qwen3_tts_cpp \
      --example asr_whisper \
      --example asr_sherpa_onnx \
      --example asr_moonshine \
      --no-default-features \
      --features "${speech_features}"
    ;;
  build)
    ./scripts/check_models_build_prereqs.sh --require-espeak --require-qwen-submodule --require-ort
    ORT_LIB_PATH="${ORT_LIB_PATH}" cargo build -p motlie-models \
      --example tts_piper \
      --example tts_qwen3_tts_cpp \
      --example asr_whisper \
      --example asr_sherpa_onnx \
      --example asr_moonshine \
      --no-default-features \
      --features "${speech_features}"
    ;;
  smoke-qwen3-whisper)
    ./scripts/check_models_build_prereqs.sh --require-qwen-submodule

    [[ -n "${QWEN3_TTS_CPP_ARTIFACT_ROOT:-}" ]] || fail "QWEN3_TTS_CPP_ARTIFACT_ROOT must point at the qwen3-tts.cpp GGUF artifact root"
    [[ -n "${WHISPER_ARTIFACT_ROOT:-}" ]] || fail "WHISPER_ARTIFACT_ROOT must point at the whisper.cpp artifact root"

    note "building qwen3-tts.cpp and whisper examples for the co-link smoke"
    cargo build -p motlie-models \
      --example tts_qwen3_tts_cpp \
      --example asr_whisper \
      --no-default-features \
      --features model-qwen3-tts-cpp,model-whisper-base-en

    prompt="hello from qwen three tts cpp to whisper"
    note "running qwen3-tts.cpp | whisper co-link smoke"
    transcript="$(
      printf '%s\n' "${prompt}" \
      | ./target/debug/examples/tts_qwen3_tts_cpp \
          --quiet \
          --artifact-root "${QWEN3_TTS_CPP_ARTIFACT_ROOT}" \
      | ./target/debug/examples/asr_whisper \
          --quiet \
          --artifact-root "${WHISPER_ARTIFACT_ROOT}"
    )"

    transcript_trimmed="$(printf '%s' "${transcript}" | tr -d '[:space:]')"
    [[ -n "${transcript_trimmed}" ]] || fail "qwen3-tts.cpp -> whisper smoke produced an empty transcript"

    transcript_lc="$(printf '%s' "${transcript}" | tr '[:upper:]' '[:lower:]')"
    [[ "${transcript_lc}" == *hello* ]] || fail "qwen3-tts.cpp -> whisper smoke transcript did not contain 'hello': ${transcript}"

    note "smoke transcript: ${transcript}"
    ;;
  *)
    echo "unsupported mode: ${mode}" >&2
    exit 2
    ;;
esac
