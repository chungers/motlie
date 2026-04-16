# TTS↔ASR End-to-End Validation Suite

Active validation covers the two Piper-backed TTS→ASR pipelines by
synthesizing text, feeding the PCM directly into transcription, and comparing
the output against the original text via word error rate (WER).

## Pipeline Matrix

| Pipeline | TTS | ASR | Features |
|----------|-----|-----|----------|
| `piper_whisper` | Piper en_US ljspeech medium | whisper.cpp base.en | `model-piper-en-us-ljspeech-medium`, `model-whisper-base-en` |
| `piper_sherpa` | Piper en_US ljspeech medium | sherpa-onnx streaming | `model-piper-en-us-ljspeech-medium`, `model-sherpa-onnx-streaming` |
| ~~`qwen3_whisper`~~ | ~~Qwen3-TTS 12Hz 0.6B ONNX~~ | ~~whisper.cpp base.en~~ | Deprecated / removed from PR `#182` |
| ~~`qwen3_sherpa`~~ | ~~Qwen3-TTS 12Hz 0.6B ONNX~~ | ~~sherpa-onnx streaming~~ | Deprecated / removed from PR `#182` |

The Qwen3-TTS ONNX path is now considered dead for this benchmark suite: the
exported decoder/vocoder path produced noisy output and ~100% WER, so the PR
no longer ships those binaries. Follow-on TTS work is moving to `qwen3_tts_rs`
and F5-TTS instead of extending the ONNX wrappers.

## Test Dataset

`dataset/samples.json` contains 100 English text samples:
- 25 short (5-15 words)
- 25 medium (30-60 words)
- 25 long (100-200 words)
- 25 paragraphs (300-1000 words)

## Prerequisites

1. Download model artifacts for each backend you want to test.

2. Environment requirements:
   - **Piper:** ONNX Runtime (`ORT_LIB_PATH`) + eSpeak-ng data (`PIPER_ESPEAKNG_DATA_DIRECTORY`)
   - **whisper.cpp:** Pre-downloaded `ggml-base.en.bin` under the artifact root
   - **sherpa-onnx:** ONNX Runtime + pre-downloaded sherpa zipformer model files
   - **Qwen3-TTS ONNX:** Deprecated for this suite; the PR no longer builds the
     dead example binaries. Replacement work is expected to use `qwen3_tts_rs`
     and F5-TTS.

## Build

Each pipeline binary requires its specific feature flags:

```bash
# Piper + whisper.cpp
cargo build -p motlie-models --example tts_asr_piper_whisper \
  --no-default-features --features model-piper-en-us-ljspeech-medium,model-whisper-base-en

# Piper + sherpa-onnx
cargo build -p motlie-models --example tts_asr_piper_sherpa \
  --no-default-features --features model-piper-en-us-ljspeech-medium,model-sherpa-onnx-streaming

# Aggregator (no model features needed)
cargo build -p motlie-models --example tts_asr_aggregate --no-default-features
```

CUDA-capable binaries are separate builds that add the relevant `*-cuda`
features. The examples do not use a runtime `--cuda` switch; CPU vs CUDA is
selected by the compiled feature set you run.

## Run the Full Matrix

```bash
DATA_DIR=libs/models/examples/tts_asr/dataset

# Run each active pipeline, capturing JSONL output
cargo run -p motlie-models --example tts_asr_piper_whisper \
  --no-default-features --features model-piper-en-us-ljspeech-medium,model-whisper-base-en \
  -- --data-dir $DATA_DIR > results_piper_whisper.jsonl 2>piper_whisper.log

cargo run -p motlie-models --example tts_asr_piper_sherpa \
  --no-default-features --features model-piper-en-us-ljspeech-medium,model-sherpa-onnx-streaming \
  -- --data-dir $DATA_DIR > results_piper_sherpa.jsonl 2>piper_sherpa.log
```

For CUDA-enabled runs, build and execute the corresponding binary with the
appropriate CUDA feature flags, and write the output to a `_cuda.jsonl` file so
the aggregator can distinguish it from the CPU run.

## Aggregate Results

```bash
cargo run -p motlie-models --example tts_asr_aggregate --no-default-features -- \
  --input results_piper_whisper.jsonl \
  --input results_piper_sherpa.jsonl
```

Or via stdin:
```bash
cat results_*.jsonl | cargo run -p motlie-models --example tts_asr_aggregate --no-default-features
```

## Output Format

### Pipeline Binaries

Each binary outputs one JSON line per sample to stdout:

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

`word_count` is recomputed from `text.split_whitespace()` when the dataset is
loaded, so the runtime metadata stays accurate even if the JSON fixture drifts.
`resident_memory_bytes` and `peak_resident_memory_bytes` are process RSS
snapshots captured during the sample run.

Status messages go to stderr so they don't interfere with JSONL output.

### Aggregator

The aggregator produces a human-readable comparison table:

```
=== TTS↔ASR Pipeline Comparison Report ===

Pipeline             Samples   Mean WER    Med WER    P95 WER    Max WER  Mean Lat(ms)  Med Lat(ms)
------------------------------------------------------------------------------------------------------
piper_sherpa             100      0.150      0.120      0.400      0.800          1200         1100
piper_whisper            100      0.080      0.050      0.250      0.600          2500         2200

=== Rankings ===

Best WER (accuracy):  piper_sherpa (mean WER = 0.150)
Best latency (speed): piper_sherpa (mean = 1200ms)
```

When the input filenames include `_cpu` / `_cuda`, the aggregator reports them as
separate variants and prints a CPU-vs-CUDA comparison section.

## Measured DGX Spark Results

Measured on `2026-04-15` and refreshed on `2026-04-16` by `@codex-dgx-e2e` on
the DGX Spark GB10 host used for PR `#182`. The `2026-04-16` refresh reran all
committed Piper JSONLs after adding per-sample RSS capture and correcting the
aggregator median computation. `Piper → sherpa-onnx` was rerun after the merged
PR `#183` fix on `feature/models`. Full run notes and tables are in
[RESULTS.md](RESULTS.md).

### Matrix status

| Pipeline | CPU | CUDA | Notes |
|----------|-----|------|-------|
| Piper → whisper.cpp | 100/100 samples completed | 100/100 samples completed | Full benchmark data captured |
| Piper → sherpa-onnx | 100/100 samples completed | 100/100 samples completed | Full benchmark data captured after PR `#183` |
| ~~Qwen3-TTS ONNX → whisper.cpp~~ | Removed | Removed | Dead/deprecated path; replaced by `qwen3_tts_rs` / F5-TTS follow-on work |
| ~~Qwen3-TTS ONNX → sherpa-onnx~~ | Removed | Removed | Dead/deprecated path; replaced by `qwen3_tts_rs` / F5-TTS follow-on work |

### Actual benchmark summary

| Pipeline | Mode | Samples | Mean WER | Mean Total Latency | Mean TTS | Mean ASR |
|----------|------|---------|----------|--------------------|----------|----------|
| Piper → sherpa-onnx | CPU | 100 | 0.296 | 11,195 ms | 6,544 ms | 4,651 ms |
| Piper → sherpa-onnx | CUDA | 100 | 0.303 | 10,870 ms | 6,341 ms | 4,529 ms |
| Piper → whisper.cpp | CPU | 100 | 0.464 | 60,223 ms | 6,104 ms | 54,119 ms |
| Piper → whisper.cpp | CUDA | 100 | 0.441 | 13,754 ms | 6,422 ms | 7,331 ms |

### CPU vs CUDA

| Pipeline | CPU Mean WER | CUDA Mean WER | CPU Mean Latency | CUDA Mean Latency | Speedup |
|----------|--------------|---------------|------------------|-------------------|---------|
| Piper → sherpa-onnx | 0.296 | 0.303 | 11,195 ms | 10,870 ms | 1.03x |
| Piper → whisper.cpp | 0.464 | 0.441 | 60,223 ms | 13,754 ms | 4.38x |

Overall, `Piper → sherpa-onnx` is now the best measured pipeline on this host for
aggregate WER and end-to-end latency. `Piper → whisper.cpp` still sees the only
material GPU acceleration in this matrix, though the refreshed RSS-enabled
harness reduced the observed speedup relative to the earlier latency-only runs.

## Predicted Pipeline Rankings

Based on model architecture characteristics (not yet validated with real data):

### CPU (no CUDA)

| Rank | Pipeline | Rationale |
|------|----------|-----------|
| 1 | Piper → whisper.cpp | Both CPU-optimized. Piper ~60 MiB ONNX + whisper-base.en ~142 MiB ggml. Fastest startup, lowest memory. |
| 2 | Piper → sherpa-onnx | True streaming ASR but heavier ORT runtime overhead |

### CUDA (DGX Spark GB10)

| Rank | Pipeline | Rationale |
|------|----------|-----------|
| 1 | Piper → sherpa-onnx | Best measured aggregate latency, but almost no GPU-specific benefit |
| 2 | Piper → whisper.cpp | Only active lane with material GPU speedup |

### By text length

| Length | Best CPU | Best CUDA |
|--------|----------|-----------|
| Short (5-15 words) | Piper → whisper.cpp (least overhead) | Piper → whisper.cpp (startup dominates) |
| Medium (30-60 words) | Piper → sherpa-onnx | Piper → whisper.cpp |
| Long (100-200 words) | Piper → sherpa-onnx (streaming) | Piper → sherpa-onnx |
| Paragraphs (300-1000) | Piper → sherpa-onnx (memory bounded) | Piper → sherpa-onnx |

These are **PREDICTIONS** for the active Piper matrix only. The deprecated
Qwen3-TTS ONNX path is excluded from forward-looking recommendations because the
example binaries have been removed from PR `#182`.

## Interpreting Results

- **WER** (Word Error Rate): lower is better. 0.0 = perfect transcription.
  WER = (substitutions + insertions + deletions) / reference_word_count
- **Latency**: TTS + ASR combined. Lower is better.
- **Current measured CPU recommendation**: `Piper → sherpa-onnx`
- **Current measured CUDA recommendation**: `Piper → sherpa-onnx` for best aggregate WER/latency, or `Piper → whisper.cpp` if the main goal is GPU speedup over CPU baseline
- **Qwen3 status**: the ONNX path is deprecated/dead for this suite and has been replaced on the roadmap by `qwen3_tts_rs` and F5-TTS
