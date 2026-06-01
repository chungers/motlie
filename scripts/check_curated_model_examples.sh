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

features="model-whisper-base-en,model-sherpa-onnx-streaming,model-moonshine-streaming,model-piper-en-us-ljspeech-medium,model-qwen3-tts-cpp"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

note() {
  echo "[curated-examples] $*"
}

run_with_ort_policy() {
  ./scripts/check_models_build_prereqs.sh --require-ort
  env -u ORT_PREFER_DYNAMIC_LINK -u ORT_LIB_PATH -u ORT_LIB_LOCATION "$@"
}

case "${mode}" in
  check)
    ./scripts/check_models_build_prereqs.sh --require-espeak --require-qwen-submodule
    cargo check -p motlie-model-espeak-ng -p motlie-model-piper --lib
    run_with_ort_policy cargo check -p motlie-models --lib --examples --no-default-features --features "${features}"
    ;;
  build)
    ./scripts/check_models_build_prereqs.sh --require-espeak --require-qwen-submodule --require-ort
    env -u ORT_PREFER_DYNAMIC_LINK -u ORT_LIB_PATH -u ORT_LIB_LOCATION cargo build -p motlie-models \
      --example tts_piper \
      --example tts_qwen3_tts_cpp \
      --example asr_whisper \
      --example asr_sherpa_onnx \
      --example asr_moonshine \
      --no-default-features \
      --features "${features}"
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
