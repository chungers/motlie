# `tts_piper` — Piper TTS Example

This example demonstrates the Phase 1 TTS vertical slice using the curated
Piper `en_US-ljspeech-medium` ONNX voice.

## Run

```bash
cargo run -p motlie-models --example tts_piper \
  --no-default-features --features model-piper-en-us-ljspeech-medium \
  -- --text "Hello from Motlie." --wav /tmp/motlie-tts.wav
```

## Preconditions

- The curated Piper artifacts must already be downloaded under the default
  HF cache root, or `--artifact-root` may point directly at the resolved
  artifact directory that contains the voice ONNX + JSON sidecar.
- The env-gated test path uses `MOTLIE_TEST_PIPER_ARTIFACT_ROOT` to point at
  the same Hugging Face cache root.
- A compatible ONNX Runtime installation must be provided explicitly for the
  `ort` crate. Set `ORT_LIB_PATH` unless ONNX Runtime is already discoverable
  via `pkg-config` or another explicit system installation path.
- A system `libespeak-ng` installation must be available at build and runtime.
  Motlie no longer vendors or bootstraps `espeak-ng` during `cargo build`.
  If the library is installed in a non-standard location, set
  `ESPEAK_NG_LIB_DIR` to the directory containing `libespeak-ng.so` or
  `libespeak-ng.so.1`.
- eSpeak-ng data must be discoverable for phonemization. If it is not installed
  system-wide, set `PIPER_ESPEAKNG_DATA_DIRECTORY` to the directory that
  contains `espeak-ng-data/`.

## Expected Behavior

- The example opens the curated `Piper en_US ljspeech medium` bundle.
- Text is synthesized through the typed `SpeechSynthesizer` / `SpeechStream`
  contract and collected into a `.wav` file.
- In Phase 1, Piper performs whole-utterance synthesis in `synthesize()` and
  then emits buffered typed audio chunks through `next_chunk()`.
- The resulting file uses Piper's fixed `22050 Hz` mono `i16` output format.
