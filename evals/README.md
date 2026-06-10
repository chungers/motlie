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
  snapshots/       # pinned matrix manifests
  results/         # per-host raw JSONL/logs, ignored except committed .md summaries
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

## Gate Scope

Issue #399 gates the eval suite with eval-scoped formatting plus the eval binary
build, tests, and clippy. The intended commands are:

```sh
cargo fmt -p evals --check
cargo build -p evals --all-targets
cargo test -p evals --all-targets
cargo clippy -p evals --all-targets -- -D warnings
```

Full-workspace `cargo fmt --check` may report pre-existing formatting drift
outside the eval suite; that broader cleanup is separate from this issue.

## Current Scenarios

- `embeddings_similarity`: embedding dimensions plus similar-vs-dissimilar cosine ordering.
- `chat_smoke`: single-turn and follow-up chat response checks.
- `chat_completion_smoke`: chat plus completion-path smoke for bundles that advertise `completion`.
- `chat_tool_use_smoke`: legacy chat-scoped tool-call smoke for bundles that advertise `tool_use`.
- `tool_use_weather_cel_smoke`: first-class tool-use smoke with deterministic weather/CEL handlers.
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

For GGUF snapshot cells on Linux, `evals matrix` wires `BINDGEN_EXTRA_CLANG_ARGS`
for child builds with the repo-local `tools/clang-compat/include` shim; host
compiler builtin include directories are appended when discovered. Direct,
hand-run Linux GGUF feature builds need the same include arguments until
`llama-cpp-sys` handles this C-header path itself:

```sh
BINDGEN_EXTRA_CLANG_ARGS="-I$PWD/tools/clang-compat/include" \
  cargo build -p evals --no-default-features --features model-qwen3-4b-gguf --all-targets
```

For CUDA-class hosts, pass the matching profile:

```sh
cargo run -p evals --features "model-google-gemma-300m model-qwen3-embedding-06b" -- run --bundle embeddinggemma_300m --scenario embeddings_similarity --profile dgx-spark
```

For Apple Metal GGUF verification, run the pinned matrix on the Metal host. The
curated snapshot includes GGUF cells for `apple-metal`; those cells verify the
Apple clang + llama.cpp Metal path by building and running with the profile's
`metal` marker. If a GGUF Metal cell is not configured for that verification,
the matrix emits a blocked `gguf_metal_unverified` record instead of leaving the
quantization x platform row empty.

```sh
cargo run -p evals -- matrix --snapshot evals/snapshots/curated-v2-smoke.toml --profile apple-metal
```

When `nvidia-smi` is available, platform records include `gpu_backend =
"nvidia"`, GPU identity, and driver/CUDA metadata. Resource acceptance uses
process swap delta gates on Linux/CUDA profiles; `apple-metal` intentionally
does not gate on machine-wide swap because macOS reports system swap rather
than per-process bundle swap through the current sampler. Performance output
keeps common latency fields plus a nested `capability_metrics` object tagged by
capability. Current `apple-metal` mistralrs rows may report
`accelerator_mismatch`; that reflects missing/blocked mistralrs Metal backend
wiring at this head and is called out in generated aggregate report platform
notes.

The `model-qwen3-tts-cpp` feature depends on the native submodule checkout:

```sh
git submodule update --init --recursive libs/model/backends/qwen3_tts_cpp/vendor/qwen3-tts.cpp
```

## Distributed Matrix Commands

The canonical v2 snapshot command is host self-selecting and feature-light:

```sh
cargo run -p evals -- matrix --snapshot evals/snapshots/curated-v2-smoke.toml
```

Useful variants:

```sh
cargo run -p evals -- matrix --snapshot evals/snapshots/curated-v2-smoke.toml --profile dgx-spark --artifact-root ~/.cache/huggingface/hub
cargo run -p evals -- matrix --snapshot evals/snapshots/curated-v2-smoke.toml --profile apple-metal --artifact-root ~/.cache/huggingface/hub
cargo run -p evals -- matrix --snapshot evals/snapshots/curated-v2-smoke.toml --dry-run --results-root /tmp/motlie-evals-dry-run
```

`HF_TOKEN` is read only from the environment for gated Hugging Face artifacts.
The runner records token presence as a boolean and never logs or serializes the
token value.

Per-host raw output lands under:

```text
evals/results/<snapshot-id>/<run-id>/
```

That directory contains `results.jsonl`, `summary.md`, `run-manifest.toml`, and
per-cell logs. Raw result artifacts are ignored by default; committed Markdown
coverage summaries can live in `evals/results/`.

Aggregate cross-host reports are generated with:

```sh
cargo run -p evals -- report --aggregate 'evals/results/**/results.jsonl' --snapshot evals/snapshots/curated-v2-smoke.toml --output evals/reports/curated-v2-smoke/coverage.md
```
