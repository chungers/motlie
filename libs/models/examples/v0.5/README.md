# v0.5 — ASR Vertical Slice

First ASR example: `.wav` file transcription via the streaming PCM contract.

## Preconditions

1. Download the curated `ggml-base.en.bin` model:
   ```sh
   cargo run -p motlie-models --bin motlie-models-download \
     --no-default-features --features model-whisper-base-en \
     -- whisper_base_en
   ```

2. Prepare a `.wav` audio file (any sample rate, mono or stereo, 16-bit int or 32-bit float).

## Run

```sh
cargo run -p motlie-models --example models_v0_5 \
  --no-default-features --features model-whisper-base-en \
  -- --wav path/to/audio.wav
```

### Options

| Flag | Description |
|------|-------------|
| `--wav <path>` | Path to the `.wav` file to transcribe (required) |
| `--artifact-root <path>` | Override the default HF cache root for model artifacts |
| `--language <code>` | Language hint (e.g., `en`); defaults to auto-detect |

## Expected Output

```
=== motlie v0.5 — ASR vertical slice ===
wav:   path/to/audio.wav
format: 16000 Hz, 1 ch, Int, 16 bits
artifacts: ../../artifacts/models/hf-cache

--- transcribing 640000 bytes of audio ---

  [final] [0.0s - 5.0s]  Hello world, this is a test.

--- done ---
```

## Architecture

The example demonstrates the source-adaptation boundary from the DESIGN:
- `.wav` decoding is caller-side (using the `hound` crate)
- PCM chunks are fed into the model-layer `TranscriptionStream` contract
- The backend normalizes to its native format (mono f32 16kHz) internally
