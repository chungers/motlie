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
  artifact root or the path passed with `--artifact-root`.
- The env-gated test path uses `MOTLIE_TEST_PIPER_ARTIFACT_ROOT` to point at
  the same Hugging Face cache root.
- A compatible ONNX Runtime installation must be provided explicitly for the
  `ort` crate. Set `ORT_LIB_PATH` unless ONNX Runtime is already discoverable
  via `pkg-config` or another explicit system installation path.
- eSpeak-ng data must be discoverable for phonemization. If it is not installed
  system-wide, set `PIPER_ESPEAKNG_DATA_DIRECTORY` to the directory that
  contains `espeak-ng-data/`.

## Expected Behavior

- The example opens the curated `Piper en_US ljspeech medium` bundle.
- Text is synthesized through the shared `SpeechModel` / `SpeechStream`
  contract and collected into a `.wav` file.
- In Phase 1, Piper performs whole-utterance synthesis in `open_stream()` and
  then emits buffered PCM chunks through `next_chunk()`.
- The resulting file uses the backend-reported sample rate and PCM encoding.
