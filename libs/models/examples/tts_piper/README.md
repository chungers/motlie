# `tts_piper` — Piper TTS Example

This example demonstrates the Phase 1 TTS vertical slice using the curated
Piper `en_US-ljspeech-medium` ONNX voice.

Canonical build prerequisites live in
[`libs/models/docs/BUILD_MODELS.md`](../../docs/BUILD_MODELS.md).

## Run

```bash
cargo run -p motlie-models --example tts_piper \
  --no-default-features --features model-piper-en-us-ljspeech-medium \
  -- --text "Hello from Motlie." --wav /tmp/motlie-tts.wav
```

If `--wav` is omitted, the example writes WAV bytes to stdout instead:

```bash
echo "Hello from Motlie." | cargo run -p motlie-models --example tts_piper \
  --no-default-features --features model-piper-en-us-ljspeech-medium \
  -- > /tmp/motlie-tts.wav
```

## Preconditions

- The curated Piper artifacts must already be downloaded under the default
  HF cache root, or `--artifact-root` may point directly at the resolved
  artifact directory that contains the voice ONNX + JSON sidecar.
- The env-gated test path uses `MOTLIE_TEST_PIPER_ARTIFACT_ROOT` to point at
  the same Hugging Face cache root.
- A compatible ONNX Runtime installation must be provided explicitly for the
  `ort` crate. Set `ORT_LIB_PATH` unless ONNX Runtime is already discoverable
  via `pkg-config` or another explicit system installation path.
- A system `libespeak-ng` installation must be available at build and runtime.
  Motlie no longer vendors or bootstraps `espeak-ng` during `cargo build`.
  If the library is installed in a non-standard location, set
  `ESPEAK_NG_LIB_DIR` to the directory containing `libespeak-ng.so` or
  `libespeak-ng.so.1`.
- eSpeak-ng data must be discoverable for phonemization. If it is not installed
  system-wide, set `PIPER_ESPEAKNG_DATA_DIRECTORY` to the directory that
  contains `espeak-ng-data/`.

## Expected Behavior

- The example opens the curated `Piper en_US ljspeech medium` bundle.
- Text is synthesized through the typed `SpeechSynthesizer` / `SpeechStream`
  contract and collected into a `.wav` file or stdout WAV stream.
- In Phase 1, Piper performs whole-utterance synthesis in `synthesize()` and
  then emits buffered typed audio chunks through `next_chunk()`.
- The resulting file uses Piper's fixed `22050 Hz` mono `i16` output format.
- If `--text` is omitted, the example reads synthesis text from stdin.
- Diagnostics are written to stderr so stdout stays clean when it is carrying
  WAV bytes.
- `--quiet` suppresses example-layer and backend-native stderr diagnostics.

## Stream To macOS `play` Over SSH

If Homebrew `sox` is installed on the remote Mac host, the stdout WAV stream can
be piped directly over SSH into `/opt/homebrew/bin/play -t wav -`.

### Short input

```bash
printf '%s\n' "Hello from Piper over SSH." \
| ./target/release/examples/tts_piper \
    --quiet \
    --artifact-root /home/dchung/sessions/cdx-dgx-e2e/motlie/artifacts/models/hf-cache \
| ssh motliehost '/opt/homebrew/bin/play -t wav -'
```

### Medium input

```bash
printf '%s\n' "This is a medium-length Piper synthesis sample streamed over SSH to a macOS host for immediate playback through Homebrew sox." \
| ./target/release/examples/tts_piper \
    --quiet \
    --artifact-root /home/dchung/sessions/cdx-dgx-e2e/motlie/artifacts/models/hf-cache \
| ssh motliehost '/opt/homebrew/bin/play -t wav -'
```

### Long input

```bash
printf '%s\n' "Piper can also handle longer shell-composed utterances where text arrives on standard input, the example writes a WAV container to standard output, SSH forwards that byte stream to the remote macOS host, and Homebrew sox plays it without any intermediate file staging." \
| ./target/release/examples/tts_piper \
    --quiet \
    --artifact-root /home/dchung/sessions/cdx-dgx-e2e/motlie/artifacts/models/hf-cache \
| ssh motliehost '/opt/homebrew/bin/play -t wav -'
```
