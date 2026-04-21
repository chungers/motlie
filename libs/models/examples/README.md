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

## Practical Defaults

- For the best shell-pipeline experience on this host, use:
  - `tts_piper -> asr_whisper`
- For qwen3-tts.cpp validation with stable ASR, use:
  - `tts_qwen3_tts_cpp -> asr_whisper`
- For the best measured qwen3-tts.cpp WER, use:
  - `tts_qwen3_tts_cpp -> asr_moonshine`
