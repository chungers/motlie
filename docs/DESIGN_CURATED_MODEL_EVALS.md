# Curated Model Evals And Example Organization

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-06-09 | @codex-399-impl | Expanded v2 design for decentralized host-self-selecting matrix runs, PR-based aggregation, artifact provisioning, first-class tool-use evals, and eval-depth/performance dimensions. |
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

V2 extends that answer from a single-host smoke harness to a decentralized,
self-describing, cross-host matrix. There is no central scheduler. Every
participating host runs the same command against the same pinned eval snapshot,
detects what it can run, records what it skipped or blocked with explicit
reasons, and writes a collision-free local results directory that can become a
small PR. Consolidated coverage is produced later by report aggregation over
merged result directories.

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
  src/driver.rs
  src/snapshot.rs

libs/eval-tools/
  Cargo.toml
  src/lib.rs
  src/registry.rs
  src/weather.rs
  src/cel.rs

evals/
  README.md
  scenarios/
  datasets/{smoke,enriched}/{prompts,images,audio,expected,tools}/
  snapshots/
  results/
    <snapshot-id>/<run-id>-<host>-<arch>-<accelerator>/
      results.jsonl
      summary.md
      run-manifest.toml
      logs/
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
- one declared `capability`; `tool_use` is a first-class capability, not a chat
  sub-scenario
- one declared eval `depth`, initially `smoke` or `enriched`
- explicit bundle filters, inputs, assertions, metrics, and optional profile
  gates
- capability values serialize as snake_case in TOML, for example `embeddings`,
  and serde maps them to Rust enums
- `capability` is the serde tag for `ScenarioKind`, so each scenario carries
  capability-specific `input` and `assertions` payloads; embeddings is the
  runnable exemplar, with chat, tool-use, ASR, TTS, and perf runners using the
  same tagged scenario and nested capability-metrics shape

Performance is not a human example. It is captured by perf scenarios and the
standard JSONL result schema. `bench_chat` should become a compatibility wrapper
around the eval runner once `evals` has a `bench_chat_startup` scenario.

Reports are generated from result JSONL. Local per-host result directories are
gitignored by default until a platform agent intentionally opens a result PR.
Consolidated reports are committed by a separate aggregation PR after all
platform result PRs land.


## V2 Distributed Driver

The v2 driver is decentralized and host-self-selecting. It is not a scheduler
and does not coordinate work across hosts at runtime. The canonical command is
run independently by each recruited agent:

```sh
cargo run -p evals -- matrix --snapshot evals/snapshots/<snapshot-id>.toml
```

Driver responsibilities:

- Detect host identity: OS, CPU architecture, libc/toolchain where available,
  accelerator class (`cuda`, `metal`, or `cpu-only`), and concrete profile
  mapping such as `dgx-spark`, `apple-metal`, `local-cpu-x86_64`, or
  `local-cpu-aarch64`.
- Load a pinned snapshot manifest from the repo. The snapshot enumerates the
  bundle x scenario x depth matrix, expected feature groups, artifact
  requirements, platform constraints, and skip policies at the commit under
  test. Identical snapshot plus identical driver logic yields comparable
  coverage across hosts.
- For every snapshot cell, decide deterministically whether to run or skip on
  this host. A skip is an emitted result with `overall_status = "skipped"` and
  a structured reason such as `profile_not_applicable`,
  `feature_group_not_supported`, `artifact_missing`, `hf_token_missing`,
  `native_toolchain_missing`, or `submodule_missing`.
- Run what is applicable, block what is applicable but not currently runnable,
  and record every outcome. There should be no silent missing cells.
- Write a collision-free local directory:

```text
evals/results/<snapshot-id>/<run-id>-<host>-<arch>-<accelerator>/
```

`run-id` includes UTC timestamp, short git SHA, sanitized hostname, arch, and
accelerator class. The directory contains `results.jsonl`, `summary.md`,
`run-manifest.toml`, and per-cell logs. Results never include secrets.

Each host result directory is intended to be committed through its own PR. Git
is the aggregation substrate; there is no central daemon or shared mutable
state.

## Pinned Eval Snapshot

The snapshot manifest is the stable matrix input for a distributed run. It
records:

- snapshot id and git SHA it was generated from
- bundles and selectors included in the curated run
- scenarios by capability and depth
- feature groups required to compile each cell
- platform constraints and accelerator preferences
- artifact requirements and provisioning strategy
- expected output location policy

The driver may discover bundles and scenarios from catalog metadata, but the
distributed run uses the pinned snapshot as the authority. If catalog metadata
changes, a new snapshot is generated and reviewed.

## PR-Based Aggregation And Reports

`evals report --aggregate` must be functional for v2. It consumes merged host
result directories, not a live scheduler:

```sh
cargo run -p evals -- report --aggregate 'evals/results/**/results.jsonl' \
  --output evals/reports/<snapshot-id>/coverage.md
```

The aggregate report includes:

- per-cell coverage: bundle, capability, depth, scenario, profile, host,
  status (`ran`, `passed`, `failed`, `blocked`, `skipped`), failure or skip
  reason, and key metrics
- 2-D slices: model x capability, capability x platform/profile, capability x
  depth, and backend x platform
- missing-coverage lists for cells not represented by any merged result PR
- blocker rollups grouped by artifact, host profile, native toolchain, feature
  matrix, and behavior/performance/resource gate
- links or paths to the contributing host result directories

Aggregation produces a separate PR after the per-host result PRs merge. The
final human merge gate for #399 is a committed consolidated report that shows
which curated bundles actually ran, passed, failed, blocked, or skipped across
the recruited platforms.

## Artifact Provisioning And Native Toolchains

Artifact provisioning is a harness precondition, not an implicit side effect of
model startup. The driver should offer a provisioning phase:

```sh
cargo run -p evals -- provision --snapshot evals/snapshots/<snapshot-id>.toml \
  --artifact-root "$HOME/.cache/huggingface/hub"
```

Provisioning rules:

- Use `HF_TOKEN` from the environment when Hugging Face authentication is
  required. The token must never be printed, written to JSONL, committed, or
  passed as a command-line argument.
- Logs may record `hf_token_present = true/false`, target repos, and artifact
  validation status, but never token values.
- Every model artifact required by the snapshot is validated before matrix
  execution. Missing or unauthorized artifacts become structured skipped or
  blocked records.
- Artifact roots stay local to the host unless a release process explicitly
  packages them.

The GGUF axis is blocked until the `llama-cpp-sys` bindgen/toolchain
`stdbool.h` failure is fixed on Linux x86 and GB10. The design target is a
durable repo or documented toolchain fix, not a one-off operator workaround, so
GGUF quant-comparison cells can run under the same driver command as other
cells.

## Tool-Use Capability And `libs/eval-tools`

`tool_use` is a first-class eval capability. Chat can still advertise tool-use
support in descriptors, but eval selection, scenarios, metrics, and reports
treat tool-use independently from plain chat.

`libs/eval-tools` owns reusable tool fixtures for evals:

- a registry of tool specs exposed to models
- executable handlers for each tool
- transcript capture for model tool calls, tool execution, result replay, and
  final model answer
- CEL assertion support over the captured round-trip

The seed registry contains the example weather tool and CEL-backed math tool.
Scenarios parameterize tool cases in TOML, for example expected tool name,
required arguments, tool result payload, final-answer expectations, and CEL
assertions over the transcript. The tool-use runner performs the full
round-trip: model emits call, harness validates and executes the tool, feeds the
tool result back to the model, and records the final answer.

Tool-use metrics include tool-selection precision/recall, argument
precision/recall, number of repair turns, round-trip latency, tool execution
latency, and final-answer assertion status.

## Eval Depth And Metrics

Eval depth is a matrix dimension parallel to platform and accelerator:

- `smoke`: small, fast scenarios used to prove runtime wiring and basic
  behavior
- `enriched`: larger datasets that measure capability quality and regressions
  with category-appropriate metrics

Every capability should have both depths once v2 is complete. Enriched datasets
live under `evals/datasets/enriched/<capability>/` and are referenced by
scenario TOML rather than embedded in code.

Accuracy and performance are captured together:

- chat/LLM: scored correctness, warm-up time, time-to-first-token,
  tokens/second, output tokens, context length, peak memory, and resource gates
- tool-use: tool-selection and argument precision/recall, round-trip latency,
  repair turns, final-answer correctness, and CEL assertion outcomes
- ASR: real-time factor, WER, transcript length, segment count, and peak memory
- embeddings: vectors/second, embedding dimensions, similarity gap, cosine
  scores, and peak memory
- TTS: real-time factor, synthesis latency, audio duration, sample count,
  chunk count, clipping/silence checks where available, and peak memory
- perf scenarios: warm-up, iteration latencies, mean/p95, throughput, and
  resource peaks

Reviewers should explicitly check for relevant metrics that the runner cannot
currently expose. Missing but relevant metrics should appear in result
`unavailable` fields and in the aggregate report's metric-gap section.

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
  second, chat token throughput and TTFT where available, tool-use precision and
  round-trip latency, ASR/TTS real-time factor, or perf summaries
- `resources`: RSS, peak RSS, CPU time, page faults, process swap delta,
  GPU memory and utilization where available
- `acceptance`: behavior, performance, resource, and overall statuses
- `coverage`: optional matrix metadata such as snapshot id, depth, cell id,
  host applicability, skip reason, and aggregation grouping keys

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

The matrix driver builds on the same runner boundary. It constructs a
`RunContext` per runnable snapshot cell, or emits a skipped/blocked record when
the host cannot or should not run that cell. The report aggregator consumes only
records and does not re-run model code.

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

In v2, `chat_tool_use_smoke` migrates to `tool_use_smoke`, and enriched
tool-use scenarios live under the first-class `tool_use` capability. Legacy
chat-tool scenario names may remain as aliases during migration, but reports
group them under `tool_use`.

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
