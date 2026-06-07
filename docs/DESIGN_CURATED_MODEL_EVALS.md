# Curated Model Evals And Example Organization

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-06-07 | @codex-399-impl | Addressed Metal review: process-swap resource gates, section-specific failure reasons, tagged per-capability scenarios, and nested capability metrics. |
| 2026-06-06 | @codex-399-impl | Wired GB10 Linux/AArch64 fp16/fhm Cargo flags, NVIDIA platform capture, and CUDA profile gates for final pattern review. |
| 2026-06-06 | @codex-399-impl | Applied early pattern review changes: sectioned result record, explicit `RunContext`, runnable embeddings eval path, support namespace, snake_case capability TOML, and GB10/AArch64 build blocker note. |
| 2026-06-06 | @codex-399-impl | Updated scope to keep eval engine modules in the single `bins/evals` binary crate; no separate eval library expansion. |
| 2026-06-06 | @codex-399-impl | Initial brownfield design for separating human examples, curation evals, performance/resource evidence, and generated reports for issue #399. |

## Problem

The curated model catalog now spans chat, tool use, multimodal chat,
embeddings, ASR, streaming ASR, and TTS bundles. The existing
`libs/models/examples` targets are useful, but they mix four roles:

- human-facing demonstrations of how to call a capability
- repeatable acceptance evidence for curated bundles
- performance and resource measurement
- report material for curation review

That mix does not scale across curated bundles, backend variants, and host
profiles. A bundle should not become accepted because an ad hoc example once
printed plausible output.

## Decision

Separate the surfaces by responsibility:

- `libs/models/examples` remains for small human-facing demos.
- `bins/evals` owns the CLI plus binary-local scenario, runner, metrics, result, and report modules.
- `bins/evals` will be the CLI wrapper for list, run, matrix, and report
  commands.
- `evals/` stores repo-level scenario manifests, datasets, and generated
  reports.

Examples answer: how do I call this capability?

Evals answer: does this bundle satisfy this capability on this profile, with
traceable artifacts, runtime config, performance, resources, and acceptance
status?

## Target Layout

```text
libs/models/examples/
  README.md
  support/
    embeddings.rs
  chat_basic/main.rs
  tool_use_basic/main.rs
  multimodal_basic/main.rs
  embeddings_basic/main.rs
  asr_basic/main.rs
  asr_streaming/main.rs
  tts_basic/main.rs

bins/evals/
  Cargo.toml
  src/main.rs
  src/cli.rs
  src/scenario.rs
  src/result.rs
  src/platform.rs
  src/metrics.rs
  src/report.rs
  src/runner/

evals/
  README.md
  scenarios/
  datasets/{prompts,images,audio,expected}/
  reports/
```

## Naming Rules

Examples are capability-first and human-facing:

- `<capability>_<mode>` for new example targets
- `--bundle <bundle_id>` as the primary selector where feasible
- `--selector <capability:provider/name>` accepted for existing selector flows
- legacy target names stay registered while the migration is underway

Evals are scenario-first:

- `<capability>_<scenario>.toml`
- stable snake_case `id`
- one declared `capability`
- explicit bundle filters, inputs, assertions, metrics, and optional profile
  gates
- capability values serialize as snake_case in TOML, for example `embeddings`,
  and serde maps them to Rust enums
- `capability` is the serde tag for `ScenarioKind`, so each scenario carries
  capability-specific `input` and `assertions` payloads; embeddings is the
  runnable exemplar, with chat, ASR, TTS, and perf runners using the same tagged scenario and nested capability-metrics shape

Performance is not a human example. It is captured by perf scenarios and the
standard JSONL result schema. `bench_chat` should become a compatibility wrapper
around the eval runner once `evals` has a `bench_chat_startup` scenario.

Reports are generated from result JSONL and should be ignored by default unless
a curated release intentionally checks in a snapshot.

## Result Record

Every eval result should include these sections:

- top-level `schema_version`: result contract version
- `identity`: run id, git SHA, branch, command line
- `selection`: bundle id, selector, backend, checkpoint, artifact snapshot
- `profile`: explicit profile such as `local-cpu-x86_64`, `apple-metal`,
  `dgx-spark`, or `cuda-workstation`
- `platform`: OS, kernel, libc, target triple, CPU, RAM, swap, limits, GPU and
  accelerator inventory, including NVIDIA identity and driver/CUDA metadata via
  `nvidia-smi` when available
- `runtime`: cargo features, build profile, quantization, context, generation
  params, relevant env vars
- `performance`: startup, warmup, common request latency, and a nested
  `capability_metrics` object tagged by capability for embedding vectors per
  second, chat token throughput, ASR/TTS real-time factor, or perf summaries
- `resources`: RSS, peak RSS, CPU time, page faults, process swap delta,
  GPU memory and utilization where available
- `acceptance`: behavior, performance, resource, and overall statuses

Unknown platform fields are represented as `null` or `unavailable`, not omitted.

Acceptance failure reasons must name the section and gate that failed. A resource
failure cannot reuse a passing behavior assertion message; for example a process
swap gate failure should report the `max_process_swap_delta_bytes` threshold and
the observed `process_swap_delta_peak_bytes` value.

## Runner Boundary

`bins/evals/src/runner/mod.rs` owns the binary-local runner trait. Each
runner receives a `RunContext` carrying the parsed scenario, selected bundle,
profile, artifact root, runtime flags, platform collector, metrics sampler, and
output sink. Capability runners are responsible for turning backend observations
into the shared sectioned `ResultRecord`; the CLI is responsible only for
parsing flags, loading scenarios, selecting a runner, and writing emitted
records.

The embeddings exemplar proves this boundary with `evals run --bundle
embeddinggemma_300m --scenario embeddings_similarity`: it parses TOML, validates
the snake_case capability filter, starts one compiled embedding bundle, samples
startup and request latencies plus RSS/process-swap resources, emits JSONL, and applies
the `similar_gt_dissimilar` assertion.

## Migration Strategy

This is brownfield. Migration must preserve existing example behavior while
introducing capability-first names.

| Current target | New human example | Eval or perf scenario |
|---|---|---|
| `embeddings` | `embeddings_basic` | `embeddings_similarity` |
| `chat_tool_binding` | `tool_use_basic` | `chat_tool_use_smoke` |
| `chat_mistral_qwen3` | `chat_basic` | `chat_smoke`, `chat_completion_smoke`, `chat_tool_use_smoke` |
| `chat_gguf_gwen3_gemma4` | `chat_basic` | `chat_smoke`, `chat_completion_smoke`, `chat_tool_use_smoke`, `bench_chat_startup` |
| `chat_multimodal_gemma4` | `multimodal_basic` | `chat_smoke`, `multimodal_image_caption`, `chat_tool_use_smoke` |
| `chat_multimodal_qwen3_6_27b` | `chat_basic` | `chat_smoke`, `chat_completion_smoke` |
| `bench_chat` | none | `bench_chat_startup` |
| `asr_whisper` | `asr_basic` | `asr_short_transcription` |
| `asr_sherpa_onnx` | `asr_streaming` | `asr_short_transcription` |
| `asr_moonshine` | `asr_streaming` | `asr_short_transcription` |
| `tts_piper` | `tts_basic` | `tts_synthesis_smoke` |
| `tts_qwen3_tts_cpp` | `tts_basic` | `tts_synthesis_smoke` |

## Alternatives Considered

### Put The Eval Runner In `libs/models`

Rejected. `libs/models` should own curated bundle definitions and catalog
metadata. Substantial runner, scenario, metrics, and report code belongs in
`bins/evals`.

### Keep Performance As `bench_chat`

Rejected as the target shape. Human examples and benchmarks have different
audiences and output contracts. The compatibility target can remain while the
authoritative evidence moves to eval JSONL.

### Add Only Scenario Manifests

Rejected. Manifests are useful, but acceptance requires a reusable runner,
result schema, platform/profile capture, and reports.

## Platform Notes

GB10/Linux AArch64 needs `+fp16,+fhm` target features for the `gemm-f16`
inline assembly used by the embedding backend. The repo wires this through
`.cargo/config.toml` for `cfg(all(target_arch = "aarch64", target_os =
"linux"))`, so the default eval command builds on GB10 without manual
`RUSTFLAGS`.

CUDA-class profiles such as `dgx-spark` and `cuda-workstation` should emit
NVIDIA identity when `nvidia-smi` is present: `gpu_backend = "nvidia"`, one
`gpus[]` entry per device, and `accelerator_metadata` containing the collector,
driver version, and CUDA version when available. Memory fields may be `null` on
GB10 when `nvidia-smi` reports `N/A`; GPU identity should still be present.

D decision: keep `PlatformCollector` as the single profile-aware platform
section source. NVIDIA population is wired through `nvidia-smi` now. Metal population should use a macOS command fallback such as `system_profiler` or a
small Metal probe and can land as a fast-follow after A-C, before applying the
chat/ASR/TTS/bench batch migration.
