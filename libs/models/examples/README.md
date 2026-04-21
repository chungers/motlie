# `libs/models/examples`

This directory contains the runnable example binaries for curated model bundles.

Current example groups:

- TTS
  - [`tts_piper`](./tts_piper/README.md)
  - [`tts_qwen3_tts_cpp`](./tts_qwen3_tts_cpp/README.md)
- ASR
  - [`asr_whisper`](./asr_whisper/README.md)
  - [`asr_sherpa_onnx`](./asr_sherpa_onnx/README.md)
  - [`asr_moonshine`](./asr_moonshine/README.md)
- Embeddings
  - [`embeddings`](./embeddings/README.md)
- Chat
  - [`chat_mistral_qwen3`](./chat_mistral_qwen3/README.md)
  - [`chat_multimodal_gemma4`](./chat_multimodal_gemma4/README.md)
  - [`chat_gguf_gwen3_gemma4`](./chat_gguf_gwen3_gemma4/README.md)

The detailed 2x3 TTS-to-ASR validation matrix lives in
[`../docs/VALIDATION_TTS_ASR_PIPELINES.md`](../docs/VALIDATION_TTS_ASR_PIPELINES.md).

## Release Builds

### CPU release build used for the published stats

The WER and latency numbers in the validation report and the charts below were
measured from this CPU-oriented release build on this host:

```bash
export ORT_LIB_PATH=/tmp/onnxruntime-cuda/build/Linux-sm121/Release
export LD_LIBRARY_PATH=/tmp/onnxruntime-cuda/build/Linux-sm121/Release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PIPER_ESPEAKNG_DATA_DIRECTORY=/usr/lib/aarch64-linux-gnu

cargo build -p motlie-models --release \
  --example tts_piper \
  --example tts_qwen3_tts_cpp \
  --example asr_whisper \
  --example asr_sherpa_onnx \
  --example asr_moonshine \
  --no-default-features \
  --features model-piper-en-us-ljspeech-medium,model-qwen3-tts-cpp,model-whisper-base-en,model-sherpa-onnx-streaming,model-moonshine-streaming
```

### Optional accelerated build

The workspace currently exposes CUDA feature flags for Piper, qwen3-tts.cpp,
Sherpa ONNX, and Whisper CPP. Moonshine does not currently expose a separate
CUDA feature in this workspace.

```bash
export ORT_LIB_PATH=/tmp/onnxruntime-cuda/build/Linux-sm121/Release
export LD_LIBRARY_PATH=/tmp/onnxruntime-cuda/build/Linux-sm121/Release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PIPER_ESPEAKNG_DATA_DIRECTORY=/usr/lib/aarch64-linux-gnu

cargo build -p motlie-models --release \
  --example tts_piper \
  --example tts_qwen3_tts_cpp \
  --example asr_whisper \
  --example asr_sherpa_onnx \
  --example asr_moonshine \
  --no-default-features \
  --features model-piper-en-us-ljspeech-medium,model-qwen3-tts-cpp,model-whisper-base-en,model-sherpa-onnx-streaming,model-moonshine-streaming,piper-cuda,qwen3-tts-cpp-cuda,sherpa-onnx-cuda,whisper-cpp-cuda
```

Important caveat:

- The published validation stats in this README and in
  [`../docs/VALIDATION_TTS_ASR_PIPELINES.md`](../docs/VALIDATION_TTS_ASR_PIPELINES.md)
  correspond to the CPU-oriented build above, not the optional accelerated one.
- `whisper-cpp-cuda` exists as a feature flag, but it was not part of the
  published validation matrix on this host.

## Speech Pipeline Snapshot

These are the average results from the 60-run validation matrix:

### Average WER

Lower is better.

![Average WER by speech pipeline](./assets/speech_pipeline_wer.svg)

### Average Elapsed Time

Lower is better.

![Average elapsed time by speech pipeline](./assets/speech_pipeline_latency.svg)

### Comparison Table

| Pipeline | Avg WER | Avg Elapsed (s) | Worst WER | Non-zero Exit Runs | Observation |
|----------|---------|-----------------|-----------|--------------------|-------------|
| `tts_piper -> asr_whisper` | `0.042` | `2.4` | `0.125` | `0` | Best overall CPU-first shell pipeline on this host. |
| `tts_qwen3_tts_cpp -> asr_moonshine` | `0.050` | `30.0` | `0.159` | `0` | Best qwen3-tts.cpp pairing by WER, but much slower. |
| `tts_qwen3_tts_cpp -> asr_whisper` | `0.055` | `24.1` | `0.188` | `0` | Stable qwen3-tts.cpp round-trip verification path. |
| `tts_piper -> asr_moonshine` | `0.260` | `12.1` | `1.000` | `2` | Mixed quality; two blank-output failures in the matrix. |
| `tts_piper -> asr_sherpa_onnx` | `0.272` | `2.5` | `0.500` | `0` | Fast, but noticeably noisier than Whisper. |
| `tts_qwen3_tts_cpp -> asr_sherpa_onnx` | `0.300` | `20.6` | `1.000` | `0` | Weakest qwen3-tts.cpp recognizer pairing; includes a catastrophic `QQQ...` case. |

## ASR From macOS Mic Over SSH

If Homebrew `sox` is installed on the Mac host and `motliehost` resolves to
that machine, the Mac microphone can be streamed over SSH as a WAV container and
fed directly into the shipped ASR examples.

Preconditions:

- `sox` is installed on the Mac host and `/opt/homebrew/bin/rec` exists.
- The recording context on the Mac host has microphone permission.
- These commands are run from the Motlie Linux host where the example binaries
  and artifacts live.

### Whisper CPU

```bash
ssh motliehost '/opt/homebrew/bin/rec -q -c 1 -r 16000 -b 16 -e signed-integer -t wav - trim 0 8' \
| ./target/release/examples/asr_whisper \
    --quiet \
    --artifact-root /home/dchung/sessions/cdx-tts/motlie/artifacts/models/hf-cache
```

### Whisper CUDA-capable build

Use this only if the `whisper-cpp-cuda` release build succeeded on the current
host:

```bash
ssh motliehost '/opt/homebrew/bin/rec -q -c 1 -r 16000 -b 16 -e signed-integer -t wav - trim 0 8' \
| ./target/release/examples/asr_whisper \
    --quiet \
    --artifact-root /home/dchung/sessions/cdx-tts/motlie/artifacts/models/hf-cache
```

### Sherpa ONNX

```bash
ssh motliehost '/opt/homebrew/bin/rec -q -c 1 -r 16000 -b 16 -e signed-integer -t wav - trim 0 8' \
| ./target/release/examples/asr_sherpa_onnx \
    --quiet \
    --artifact-root /home/dchung/.cache/huggingface/hub
```

### Moonshine

```bash
ssh motliehost '/opt/homebrew/bin/rec -q -c 1 -r 16000 -b 16 -e signed-integer -t wav - trim 0 8' \
| ./target/release/examples/asr_moonshine \
    --quiet \
    --artifact-root /home/dchung/cld-mistral/motlie/artifacts/models/hf-cache
```

### Longer capture window

To record for longer than 8 seconds, adjust the final `trim 0 <seconds>` value
on the Mac side:

```bash
ssh motliehost '/opt/homebrew/bin/rec -q -c 1 -r 16000 -b 16 -e signed-integer -t wav - trim 0 20' \
| ./target/release/examples/asr_whisper \
    --quiet \
    --artifact-root /home/dchung/sessions/cdx-tts/motlie/artifacts/models/hf-cache
```

## Practical Defaults

- For the best shell-pipeline experience on this host, use:
  - `tts_piper -> asr_whisper`
- For qwen3-tts.cpp validation with stable ASR, use:
  - `tts_qwen3_tts_cpp -> asr_whisper`
- For the best measured qwen3-tts.cpp WER, use:
  - `tts_qwen3_tts_cpp -> asr_moonshine`
