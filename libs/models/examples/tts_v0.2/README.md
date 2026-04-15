# tts_v0.2 — Qwen3-TTS Phase 2 Vertical Slice

Second TTS example: Qwen3-TTS 12Hz 0.6B with optional voice cloning via
`VoiceConditioning::ReferenceAudio`.

## ONNX Export Prerequisite

The upstream `Qwen/Qwen3-TTS-12Hz-0.6B-Base` model uses safetensors format.
This backend requires pre-exported ONNX components:

- `encoder.onnx` — text/phoneme encoder
- `decoder.onnx` — flow-matching mel decoder
- `vocoder.onnx` — mel-to-waveform vocoder (BigVGAN-derived)
- `config.json` — model configuration

Export these from the upstream model and place them under the artifact root
(or a custom `--artifact-root` path). See `libs/models/docs/DESIGN_TTS.md`
Phase 2 for the full export guide.

## Run

### Basic synthesis (no cloning)

```bash
cargo run -p motlie-models --example models_tts_v0_2 \
  --no-default-features --features model-qwen3-tts-0_6b \
  -- --text "Hello from Motlie." --wav /tmp/motlie-qwen3-tts.wav
```

### With voice cloning (3-second reference audio)

```bash
cargo run -p motlie-models --example models_tts_v0_2 \
  --no-default-features --features model-qwen3-tts-0_6b \
  -- --text "Hello from Motlie." --wav /tmp/motlie-qwen3-tts.wav \
     --reference-audio /path/to/reference.wav
```

## Options

| Flag | Description |
|------|-------------|
| `--text <value>` | Text to synthesize (required) |
| `--wav <path>` | Output `.wav` file path (required) |
| `--artifact-root <path>` | Override the default HF cache root for ONNX artifacts |
| `--reference-audio <path>` | `.wav` file for voice cloning conditioning (optional) |

## Environment

| Variable | Description |
|----------|-------------|
| `ORT_LIB_PATH` | Path to ONNX Runtime shared library (if not system-installed) |
| `MOTLIE_TEST_QWEN3_TTS_ARTIFACT_ROOT` | Env-gated test artifact root |

## Expected Behavior

- The example opens the curated `Qwen3-TTS 12Hz 0.6B` bundle.
- Text is synthesized through the multi-model ONNX pipeline
  (encoder → decoder → vocoder) via the shared `SpeechModel`/`SpeechStream` contract.
- If `--reference-audio` is provided, the decoder receives mel-spectrogram
  conditioning extracted from the reference audio for voice cloning.
- The resulting `.wav` file uses the backend-reported sample rate and encoding.
- Like Piper Phase 1, Qwen3-TTS performs whole-utterance synthesis in
  `open_stream()` and then emits buffered PCM chunks through `next_chunk()`.

## Architecture

This example demonstrates the Phase 2 TTS vertical slice:
- Same `SpeechModel`/`SpeechStream` contract as Piper Phase 1
- Reuses the shared `motlie-model-ort` ONNX Runtime helpers
- Adds `VoiceConditioning::ReferenceAudio` for voice cloning
- Multi-model ONNX pipeline (3 sessions: encoder, decoder, vocoder)
