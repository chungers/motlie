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

Linux linker note:

- Motlie's `build.rs` for `qwen3-tts.cpp` adds `-Wl,-Bsymbolic` so the shared
  `libqwen3tts` keeps its bundled `ggml` symbols local when it is co-linked
  with other `ggml` users such as `whisper.cpp`. Without that, `tts_qwen3_tts_cpp`
  pipelines into ASR examples can fail due to symbol interposition.
- The dedicated scripted regression check for this lives in
  `./scripts/check_curated_model_examples.sh --mode smoke-qwen3-whisper`.

### Basic synthesis

```bash
cargo run -p motlie-models --example tts_qwen3_tts_cpp \
  --no-default-features --features model-qwen3-tts-cpp \
  -- --text "Hello from Motlie." --wav /tmp/motlie-qwen3-tts-cpp.wav
```

### Dedicated qwen3-tts.cpp -> Whisper smoke

This smoke is the issue `#211` regression check for Linux-side `ggml`
symbol interposition:

```bash
export QWEN3_TTS_CPP_ARTIFACT_ROOT="/tmp/qwen3-tts-models"
export WHISPER_ARTIFACT_ROOT="$HOME/.cache/huggingface/hub"

./scripts/check_curated_model_examples.sh --mode smoke-qwen3-whisper
```

Expected behavior:

- the pipeline completes without a native `GGML_ASSERT`
- Whisper receives a valid WAV stream on stdin
- the final transcript is non-empty and still contains `hello`

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
- When stdout is used, the example writes an aligned indefinite-length WAV
  header and then flushes chunks incrementally so downstream consumers can read
  until EOF without waiting for the full utterance buffer.
- Diagnostics are written to stderr so stdout stays clean when it is carrying
  WAV bytes.
- `--quiet` suppresses example-layer and backend-native stderr diagnostics.
  Because that is a whole-process stderr redirect, panic output may also be
  silent while `--quiet` is active.
- `--reference-audio` currently uses simple linear resampling to 16 kHz mono
  with no anti-alias filter. That is acceptable for an example-layer cloning
  path but is still a known quality limitation for high-rate reference audio.

## Stream To macOS `play` Over SSH

If Homebrew `sox` is installed on the remote Mac host, the stdout WAV stream can
be piped directly over SSH into `/opt/homebrew/bin/play -t wav -`.

Recommended artifact env var:

```bash
export QWEN3_TTS_CPP_ARTIFACT_ROOT="/tmp/qwen3-tts-models"
```

### Short input

```bash
printf '%s\n' "Hello from qwen3-tts.cpp over SSH." \
| ./target/release/examples/tts_qwen3_tts_cpp \
    --quiet \
    --artifact-root "$QWEN3_TTS_CPP_ARTIFACT_ROOT" \
| ssh motliehost '/opt/homebrew/bin/play -t wav -'
```

### Medium input

```bash
printf '%s\n' "This is a medium-length qwen3-tts.cpp synthesis sample streamed over SSH to a macOS host for immediate playback through Homebrew sox." \
| ./target/release/examples/tts_qwen3_tts_cpp \
    --quiet \
    --artifact-root "$QWEN3_TTS_CPP_ARTIFACT_ROOT" \
| ssh motliehost '/opt/homebrew/bin/play -t wav -'
```

### Long input

```bash
printf '%s\n' "qwen3-tts.cpp can also handle longer shell-composed utterances where text arrives on standard input, the example writes a WAV container to standard output, SSH forwards that byte stream to the remote macOS host, and Homebrew sox plays it without any intermediate file staging." \
| ./target/release/examples/tts_qwen3_tts_cpp \
    --quiet \
    --artifact-root "$QWEN3_TTS_CPP_ARTIFACT_ROOT" \
| ssh motliehost '/opt/homebrew/bin/play -t wav -'
```
