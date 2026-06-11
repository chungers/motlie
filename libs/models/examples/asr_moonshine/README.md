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

If `--wav` is omitted, the example reads a WAV stream from stdin:

```bash
cat /path/to/audio.wav | cargo run -p motlie-models --example asr_moonshine \
  --no-default-features --features model-moonshine-streaming --
```

To preserve event-style streaming output on stdout, add `--partials`:

```bash
cat /path/to/audio.wav | cargo run -p motlie-models --example asr_moonshine \
  --no-default-features --features model-moonshine-streaming \
  -- --partials
```

## Preconditions

- The curated Moonshine artifacts must already be downloaded under the default
  HF cache root, or `--artifact-root` may point directly at the resolved
  artifact directory that contains the Moonshine `.ort` files.
- ONNX Runtime is provided by Cargo through the workspace `ort/download-binaries`
  static archive path. Do not set `ORT_LIB_PATH`, `ORT_PREFER_DYNAMIC_LINK`, or
  `LD_LIBRARY_PATH`.
- The `.wav` file must be PCM audio (`i16` or `f32`).

## Expected Behavior

- The example opens the curated `Moonshine Streaming EN` bundle.
- Audio is streamed through the typed `StreamingTranscriber` session.
- By default stdout prints one final plain-text transcript line after
  `finish()`.
- `--partials` switches stdout to `[partial]` / `[final]` event lines.
- If `--wav` is omitted, the example reads binary WAV input from stdin.
- Transcript text stays on stdout; diagnostics move to stderr in pipeline mode.
- `--quiet` suppresses example-layer and backend-native stderr diagnostics.
