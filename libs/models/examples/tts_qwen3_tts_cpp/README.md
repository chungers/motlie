# `tts_qwen3_tts_cpp` — qwen3-tts.cpp Example

Third TTS example: `qwen3-tts.cpp` as the secondary curated TTS backend.

Canonical build prerequisites live in
[`libs/models/docs/BUILD_MODELS.md`](../../docs/BUILD_MODELS.md).

This slice proves the GGUF/C++ runtime path end-to-end:
- curated bundle resolution from Hugging Face cache or a direct model directory
- typed `SpeechSynthesizer` / `SpeechStream` output through the Motlie backend
- optional voice cloning from a typed `CloneReference<16000, Mono>`
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

If `--wav` is omitted, the example writes WAV bytes to stdout:

```bash
echo "Hello from Motlie." | cargo run -p motlie-models --example tts_qwen3_tts_cpp \
  --no-default-features --features model-qwen3-tts-cpp \
  -- > /tmp/motlie-qwen3-tts-cpp.wav
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
| `--text <value>` | Text to synthesize (optional; stdin is used when omitted) |
| `--wav <path>` | Output `.wav` file path (optional; stdout WAV is used when omitted) |
| `--artifact-root <path>` | Override the default HF cache root for GGUF artifacts |
| `--reference-audio <path>` | Optional `.wav` file used for voice cloning |
| `--quiet` | Suppress example-layer and backend-native stderr diagnostics |

## Expected Behavior

- The example opens the curated `Qwen3-TTS CPP 0.6B` bundle.
- Text is synthesized through the pinned `qwen3-tts.cpp` submodule C API.
- If `--reference-audio` is provided, the example downsamples it to a typed
  `CloneReference<16000, Mono>` before calling the voice-cloning entry point.
- The backend performs whole-utterance synthesis in `synthesize()` and then
  emits buffered typed audio chunks through `next_chunk()`, matching the Piper
  and Qwen ONNX TTS stream contract.
- The resulting `.wav` output uses the backend's fixed 24 kHz mono float output,
  whether written to a file or stdout.
- Diagnostics are written to stderr so stdout stays clean when it is carrying
  WAV bytes.
- `--quiet` suppresses example-layer and backend-native stderr diagnostics.
