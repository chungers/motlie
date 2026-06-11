# `asr_whisper` — Whisper ASR Example

Whisper `.wav` transcription through the typed batch contract.

Canonical build prerequisites live in
[`libs/models/docs/BUILD_MODELS.md`](../../docs/BUILD_MODELS.md).

## Preconditions

1. Download the curated `ggml-base.en.bin` model:
   ```sh
   cargo run -p motlie-models --bin motlie-models-download \
     --no-default-features --features model-whisper-base-en \
     -- whisper_base_en
   ```

2. Prepare a `.wav` audio file. The backend accepts any sample rate and channel
   count (mono or stereo) and normalizes to mono 16 kHz f32 internally via
   channel averaging and linear-interpolation resampling. Supported PCM
   encodings: 16-bit signed integer or 32-bit float.

## Run

```sh
cargo run -p motlie-models --example asr_whisper \
  --no-default-features --features model-whisper-base-en \
  -- --wav path/to/audio.wav
```

If `--wav` is omitted, the example reads a WAV stream from stdin:

```sh
cat path/to/audio.wav | cargo run -p motlie-models --example asr_whisper \
  --no-default-features --features model-whisper-base-en --
```

### Options

| Flag | Description |
|------|-------------|
| `--wav <path>` | Path to the `.wav` file to transcribe (optional; stdin is used when omitted) |
| `--artifact-root <path>` | Override the default HF cache root or pass a pre-resolved artifact directory |
| `--language <code>` | Language hint (e.g., `en`); defaults to auto-detect |
| `--quiet` | Suppress example-layer and backend-native stderr diagnostics |

## Expected Output

```text
Hello world, this is a test.
```

## Architecture

The example demonstrates the source-adaptation boundary from the strict typed API:
- `.wav` decoding is caller-side (using the `hound` crate)
- caller-side adaptation produces `AudioBuf<f32, 16000, Mono>`
- the backend runs as a batch transcriber rather than a chunked streaming session
- transcript text is written to stdout as one final plain-text line
- diagnostics move to stderr in stdin pipeline mode, and `--quiet` suppresses
  example-layer and backend-native stderr logging
