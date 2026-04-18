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
- Incremental transcript updates are printed as `[partial]` and `[final]`
  segments while the file is processed.
