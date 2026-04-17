# TTS↔ASR End-to-End Validation Suite

This suite benchmarks a 2 TTS x 3 ASR matrix by synthesizing text, feeding the
PCM directly into transcription, and comparing the transcript against the
reference text with word error rate (WER).

## Pipeline Matrix

| Pipeline | TTS | ASR | Status on DGX |
|----------|-----|-----|---------------|
| `piper_whisper` | Piper en_US ljspeech medium | whisper.cpp base.en | Completed on CPU and CUDA |
| `piper_sherpa` | Piper en_US ljspeech medium | sherpa-onnx streaming | Completed on CPU and CUDA |
| `piper_moonshine` | Piper en_US ljspeech medium | Moonshine streaming EN | CPU full pass failed `0/100`; CUDA smoke reproduced same failure |
| `qwen3cpp_whisper` | qwen3-tts.cpp 0.6B | whisper.cpp base.en | Blocked at TTS init on CPU and CUDA |
| `qwen3cpp_sherpa` | qwen3-tts.cpp 0.6B | sherpa-onnx streaming | Blocked at TTS init on CPU and CUDA |
| `qwen3cpp_moonshine` | qwen3-tts.cpp 0.6B | Moonshine streaming EN | Blocked at TTS init on CPU and CUDA |

The older Qwen3-TTS ONNX path is still dead for this benchmark suite and is not
built here.

## Test Dataset

`dataset/samples.json` contains `100` English text samples:
- 25 short (`5-15` words)
- 25 medium (`30-60` words)
- 25 long (`100-200` words)
- 25 paragraphs (`300-1000` words)

## Build

```bash
# CPU release matrix
cargo build --release -p motlie-models \
  --example tts_asr_piper_whisper \
  --example tts_asr_piper_sherpa \
  --example tts_asr_piper_moonshine \
  --example tts_asr_qwen3cpp_whisper \
  --example tts_asr_qwen3cpp_sherpa \
  --example tts_asr_qwen3cpp_moonshine \
  --example tts_asr_aggregate \
  --no-default-features \
  --features model-piper-en-us-ljspeech-medium,model-qwen3-tts-cpp,model-whisper-base-en,model-sherpa-onnx-streaming,model-moonshine-streaming

# CUDA release matrix
cargo build --release -p motlie-models \
  --example tts_asr_piper_whisper \
  --example tts_asr_piper_sherpa \
  --example tts_asr_piper_moonshine \
  --example tts_asr_qwen3cpp_whisper \
  --example tts_asr_qwen3cpp_sherpa \
  --example tts_asr_qwen3cpp_moonshine \
  --example tts_asr_aggregate \
  --no-default-features \
  --features model-piper-en-us-ljspeech-medium,model-qwen3-tts-cpp,model-whisper-base-en,model-sherpa-onnx-streaming,model-moonshine-streaming,piper-cuda,qwen3-tts-cpp-cuda,whisper-cpp-cuda,sherpa-onnx-cuda
```

Moonshine ASR itself is CPU-only. A `*_moonshine` CUDA build only changes the
TTS side.

## Output Format

Successful pipelines emit one JSON line per completed sample:

```json
{
  "pipeline": "piper_whisper",
  "sample_id": "short_001",
  "category": "short",
  "word_count": 5,
  "original_text": "Hello world, how are you?",
  "transcribed_text": "hello world how are you",
  "wer": 0.0,
  "tts_latency_ms": 120,
  "asr_latency_ms": 450,
  "total_latency_ms": 570,
  "pcm_bytes": 44100,
  "pcm_duration_ms": 1000,
  "resident_memory_bytes": 73400320,
  "peak_resident_memory_bytes": 81264640
}
```

Blocked lanes use a single status JSON object instead of per-sample rows so the
failure reason and wall-clock are still preserved in a JSONL artifact.

## DGX Spark Summary

Measured on `2026-04-16` by `@codex-dgx-e2e` on the DGX Spark GB10 host used
for PR `#182`.

### Successful aggregate results

| Pipeline | Mode | Samples | Mean WER | Mean Total Latency | Wall Clock |
|----------|------|---------|----------|--------------------|------------|
| Piper → sherpa-onnx | CPU | 100 | 0.285 | 10,988 ms | 1,101.66 s |
| Piper → sherpa-onnx | CUDA | 100 | 0.300 | 10,826 ms | 1,085.57 s |
| Piper → whisper.cpp | CPU | 100 | 0.463 | 61,960 ms | 6,198.48 s |
| Piper → whisper.cpp | CUDA | 100 | 0.459 | 13,261 ms | 1,328.87 s |

### Blocked lanes

| Pipeline | Mode | Outcome |
|----------|------|---------|
| Piper → Moonshine | CPU | Full `100`-sample pass completed with `0/100` transcripts; Moonshine ASR rejected every sample |
| Piper → Moonshine | CUDA | `60s` confirmation run reproduced the same deterministic Moonshine ASR failure |
| qwen3-tts.cpp → any ASR | CPU/CUDA | TTS init aborts before first sample with `GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx))` |

### Main takeaways

- Best WER: `Piper → sherpa-onnx` CPU (`0.285`)
- Best latency: `Piper → sherpa-onnx` CUDA (`10,826 ms`)
- Best GPU speedup: `Piper → whisper.cpp` (`4.67x` by both mean latency and
  full-run wall clock)
- Moonshine is not currently benchmarkable in this harness/backend combination
- qwen3-tts.cpp is blocked before transcript generation starts

Full tables and failure details are in [RESULTS.md](RESULTS.md).
