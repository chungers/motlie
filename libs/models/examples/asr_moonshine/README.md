# `asr_moonshine` — Moonshine ASR Example

This example demonstrates the curated Moonshine streaming bundle through the
typed streaming ASR contract.

Canonical build prerequisites live in
[`libs/models/docs/BUILD_MODELS.md`](../../docs/BUILD_MODELS.md).

## Run

```bash
cargo run -p motlie-models --example asr_moonshine \
  --no-default-features --features model-moonshine-streaming \
  -- --wav /path/to/audio.wav
```

## Preconditions

- The curated Moonshine artifacts must already be downloaded under the default
  HF cache root, or `--artifact-root` may point directly at the resolved
  artifact directory that contains the Moonshine `.ort` files.
- A compatible ONNX Runtime installation must be provided explicitly for the
  `ort` crate. Set `ORT_LIB_PATH` unless ONNX Runtime is already discoverable
  via `pkg-config` or another explicit system installation path.
- The `.wav` file must be PCM audio (`i16` or `f32`).

## Expected Behavior

- The example opens the curated `Moonshine Streaming EN` bundle.
- Audio is streamed through the typed `StreamingTranscriber` session.
- Partial output is disabled in this example, so only the final transcript is
  printed after `finish()`.
