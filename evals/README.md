# Motlie Curated Model Evals

`evals/` contains scenario manifests, datasets, and generated reports for
curated model acceptance.

Human-facing examples stay under `libs/models/examples`. The eval suite is for
repeatable curation evidence: bundle behavior, runtime configuration, platform
profile, performance, resources, and final acceptance status.

## Layout

```text
evals/
  scenarios/       # TOML scenario manifests
  datasets/        # prompts, images, audio, and expected outputs
  reports/         # generated outputs, ignored by default
```

## Scenario Conventions

- `id` is stable snake_case and matches the filename without `.toml`.
- Top-level `capability` names the eval capability and serializes as snake_case, for example `embeddings`, `chat`, `asr`, `tts`, or `perf`.
- `bundle_filter.capability` selects the primary model capability; `bundle_filter.required_capabilities` adds model-surface requirements such as `completion` or `tool_use`.
- `input` describes prompts, audio, images, or text inputs.
- `assertions` defines behavioral pass/fail checks.
- `metrics` requests performance and resource capture.
- `profiles.<name>.gates` may define optional acceptance thresholds.

## Current Scenarios

- `embeddings_similarity`: embedding dimensions plus similar-vs-dissimilar cosine ordering.
- `chat_smoke`: single-turn and follow-up chat response checks.
- `chat_completion_smoke`: chat plus completion-path smoke for bundles that advertise `completion`.
- `chat_tool_use_smoke`: tool-call smoke for bundles that advertise `tool_use`.
- `asr_short_transcription`: short WAV transcription smoke over the checked-in 16 kHz speech reference.
- `tts_synthesis_smoke`: short speech synthesis smoke over curated TTS bundles.
- `bench_chat_startup`: comparable chat startup and steady-state request latency benchmark.

## Run Examples

Embeddings:

```sh
cargo run -p evals --features "model-google-gemma-300m model-qwen3-embedding-06b" -- run --bundle embeddinggemma_300m --scenario embeddings_similarity
```

Chat:

```sh
cargo run -p evals --features "model-qwen3-4b" -- run --bundle qwen3_4b --scenario chat_smoke
```

ASR:

```sh
cargo run -p evals --features "model-whisper-base-en" -- run --bundle whisper_base_en --scenario asr_short_transcription
```

TTS:

```sh
cargo run -p evals --features "model-piper-en-us-ljspeech-medium" -- run --bundle piper_en_us_ljspeech_medium --scenario tts_synthesis_smoke
```

Perf:

```sh
cargo run -p evals --features "model-qwen3-4b" -- run --bundle qwen3_4b --scenario bench_chat_startup
```

The commands parse scenario TOML, start one compiled bundle from local
artifacts, emit a sectioned JSONL result record, and apply capability-specific
assertions. Use `--artifact-root ~/.cache/huggingface/hub` when the default
repo-local artifact cache does not contain the bundle artifacts.

On GB10/Linux AArch64, the repo `.cargo/config.toml` wires the required
`+fp16,+fhm` target features, so no manual `RUSTFLAGS` are needed for the
default Cargo command.

For CUDA-class hosts, pass the matching profile:

```sh
cargo run -p evals --features "model-google-gemma-300m model-qwen3-embedding-06b" -- run --bundle embeddinggemma_300m --scenario embeddings_similarity --profile dgx-spark
```

When `nvidia-smi` is available, platform records include `gpu_backend =
"nvidia"`, GPU identity, and driver/CUDA metadata. Resource acceptance uses
process swap delta gates on Linux/CUDA profiles; `apple-metal` intentionally
does not gate on machine-wide swap because macOS reports system swap rather
than per-process bundle swap through the current sampler. Performance output
keeps common latency fields plus a nested `capability_metrics` object tagged by
capability.

The `model-qwen3-tts-cpp` feature depends on the native submodule checkout:

```sh
git submodule update --init --recursive libs/model/backends/qwen3_tts_cpp/vendor/qwen3-tts.cpp
```

Generated JSONL results should be consumed by `evals report` once the reporting
command lands.
