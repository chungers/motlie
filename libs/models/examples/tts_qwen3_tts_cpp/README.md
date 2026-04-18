# `tts_qwen3_tts_cpp` — qwen3-tts.cpp Example

Third TTS example: `qwen3-tts.cpp` as the secondary curated TTS backend.

This slice proves the GGUF/C++ runtime path end-to-end:
- curated bundle resolution from Hugging Face cache or a direct model directory
- `SpeechModel` / `SpeechStream` output through the Motlie generic backend
- optional voice cloning from `VoiceConditioning::ReferenceAudio`
- `.wav` file output for manual verification and downstream ASR round-trips

## Artifacts

The curated bundle expects a model directory or HF snapshot for `koboldcpp/tts`
containing:

- `qwen3-tts-0.6b-q8_0.gguf` or `qwen3-tts-0.6b-f16.gguf`
- `qwen3-tts-tokenizer-f16.gguf`

The default artifact root is Motlie's standard HF cache root. Override it with
`--artifact-root` if needed.

## Run

Before building the example, initialize the native runtime checkout:

```bash
git submodule update --init --recursive \
  libs/model/backends/qwen3_tts_cpp/vendor/qwen3-tts.cpp
```

### Basic synthesis

```bash
cargo run -p motlie-models --example tts_qwen3_tts_cpp \
  --no-default-features --features model-qwen3-tts-cpp \
  -- --text "Hello from Motlie." --wav /tmp/motlie-qwen3-tts-cpp.wav
```

### With voice cloning

```bash
cargo run -p motlie-models --example tts_qwen3_tts_cpp \
  --no-default-features --features model-qwen3-tts-cpp \
  -- --text "Hello from Motlie." --wav /tmp/motlie-qwen3-tts-cpp.wav \
     --reference-audio /path/to/reference.wav
```

## Options

| Flag | Description |
|------|-------------|
| `--text <value>` | Text to synthesize (required) |
| `--wav <path>` | Output `.wav` file path (required) |
| `--artifact-root <path>` | Override the default HF cache root for GGUF artifacts |
| `--reference-audio <path>` | Optional `.wav` file used for voice cloning |

## Expected Behavior

- The example opens the curated `Qwen3-TTS CPP 0.6B` bundle.
- Text is synthesized through the pinned `qwen3-tts.cpp` submodule C API.
- If `--reference-audio` is provided, the backend downsamples it to 24 kHz mono
  float PCM before passing it through the voice-cloning entry point.
- The backend performs whole-utterance synthesis in `open_stream()` and then
  emits buffered PCM chunks through `next_chunk()`, matching the Piper and
  Qwen ONNX TTS stream contract.
- The resulting `.wav` file uses the runtime-reported sample rate and PCM
  encoding.
