# `asr_sherpa_onnx` — Sherpa ONNX Streaming ASR

This example demonstrates the Phase 2 ASR vertical slice using the
`sherpa-onnx` Zipformer backend.

Canonical build prerequisites live in
[`libs/models/docs/BUILD_MODELS.md`](../../docs/BUILD_MODELS.md).

## Run

```bash
cargo run -p motlie-models --example asr_sherpa_onnx \
  --no-default-features --features model-sherpa-onnx-streaming \
  -- --wav /path/to/audio.wav
```

If `--wav` is omitted, the example reads a WAV stream from stdin:

```bash
cat /path/to/audio.wav | cargo run -p motlie-models --example asr_sherpa_onnx \
  --no-default-features --features model-sherpa-onnx-streaming --
```

To preserve event-style streaming output on stdout, add `--partials`:

```bash
cat /path/to/audio.wav | cargo run -p motlie-models --example asr_sherpa_onnx \
  --no-default-features --features model-sherpa-onnx-streaming \
  -- --partials
```

## Preconditions

- The curated sherpa-onnx artifacts must already be downloaded under the
  default HF cache root, or `--artifact-root` may point directly at the
  resolved artifact directory that contains the ONNX files.
- A compatible ONNX Runtime installation must be provided explicitly for the
  `ort` crate. Set `ORT_LIB_PATH` to the ONNX Runtime library directory unless
  the runtime is already discoverable via `pkg-config` or another explicit
  system installation path.
- The `.wav` file must be PCM audio (`i16` or `f32`).
- For transcript-quality checks, prefer recorded speech. Synthetic TTS WAVs are
  useful for plumbing validation, but they are not representative of sherpa's
  real-speech accuracy and can produce visibly degraded tails.

## Expected Behavior

- The example opens the curated `Sherpa ONNX Streaming Zipformer EN` bundle.
- Audio is streamed through the shared PCM transcription contract.
- By default stdout prints one final plain-text transcript line.
- `--partials` switches stdout to `[partial]` / `[final]` event lines.
- If `--wav` is omitted, the example reads binary WAV input from stdin.
- Transcript text stays on stdout; diagnostics move to stderr in pipeline mode.
- `--quiet` suppresses example-layer and backend-native stderr diagnostics.
