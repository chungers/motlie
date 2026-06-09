# Curated Model Evals And Example Organization

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-06-09 | @codex-399-impl | Implemented the v2 framework shape: pinned smoke snapshot, host-self-selecting matrix driver, aggregate report command, raw-result ignore policy, and first-class tool-use crate/runner. |
| 2026-06-09 | @codex-399-impl | Tightened v2 design after R1 review: feature-light matrix driver, concrete skip/block schema, accelerator detection/use proof, portable host identity, quant grouping, metric source semantics, and cross-platform gates. |
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
  src/accelerator.rs
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
    <snapshot-id>/<run-id>/
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
cargo run -p evals -- matrix --snapshot evals/snapshots/curated-v2-smoke.toml
```

The top-level `evals` binary must remain feature-light: it cannot require every
model feature to compile before the driver can start. The matrix driver reads
the snapshot, then launches per-cell or per-feature child `cargo` invocations
for the model bundle being evaluated. A child build failure is captured as a
structured `blocked` record with reason `feature_build_failed`,
`native_toolchain_missing`, or a more specific reason such as
`gguf_toolchain_failed`; it must not prevent the outer command from writing
records for the rest of the matrix. This keeps the identical command runnable
on CPU-only, CUDA, and Metal hosts even when some cells require broken or
unavailable native feature groups.

Driver responsibilities:

- Detect portable host identity using `sysinfo::System::host_name()` or a
  `gethostname(2)` wrapper, with platform command fallbacks such as `uname -n`
  or `scutil --get LocalHostName`; never rely only on `$HOSTNAME`. The
  sanitized host slug, arch, accelerator class, UTC timestamp, short git SHA,
  and a short nonce form a single `run-id`. The output directory is
  `evals/results/<snapshot-id>/<run-id>/`; host is not duplicated in another
  path segment.
- Detect OS, CPU architecture, libc/toolchain where available, and accelerator
  class by probe, not by OS/arch heuristic alone. CUDA detection uses NVML when
  available and `nvidia-smi` as fallback. Metal detection uses a small
  `MTLCreateSystemDefaultDevice` probe when possible, with
  `system_profiler SPDisplaysDataType` or IOKit as fallback for inventory. A
  macOS host maps to `apple-metal` only when the Metal probe confirms a usable
  device/backend; otherwise Metal-required cells are skipped or blocked with a
  structured reason.
- Load a pinned snapshot manifest from the repo. The snapshot enumerates the
  bundle x scenario x depth x checkpoint-format x quantization matrix, expected
  Cargo feature groups, artifact requirements, platform constraints, runtime
  and resource budgets, accelerator requirements, and skip policies at the
  commit under test. Identical snapshot plus identical driver logic yields
  comparable coverage across hosts.
- For every snapshot cell, decide deterministically whether to run, skip, or
  block on this host. A skip/block is an emitted result with a terminal outcome
  and structured reason; there should be no silent missing cells.
- Enforce per-cell and per-profile budgets from the snapshot, including max
  wall time, max artifact/model size, max RSS or memory footprint, allowed
  eval depth on CPU, and optional accelerator requirement. A large cell that is
  theoretically compatible but impractical on CPU must end as a structured
  `blocked` or `skipped` record such as `cpu_profile_not_practical` or
  `runtime_budget_exceeded`, not an endless run with no JSONL.
- Run what is applicable, block what is applicable but not currently runnable,
  and record every outcome.
- Write a collision-free local directory:

```text
evals/results/<snapshot-id>/<run-id>/
```

The directory contains `results.jsonl`, `summary.md`, `run-manifest.toml`, and
per-cell logs. Results never include secrets.

Each host result directory is intended to be committed through its own PR. Git
is the aggregation substrate; there is no central daemon or shared mutable
state.

## Accelerator Detection And Use Proof

Accelerator detection and per-cell accelerator-use proof are first-class v2
requirements because matrix cells must prove they ran on the platform they
claim. The platform inventory answers "what devices exist on this host"; the
per-cell accelerator block answers "what accelerator this cell actually used."

The platform collector records CUDA and Metal inventory where available:

- CUDA: device id, name, UUID where available, driver version, CUDA runtime or
  toolkit version where available, total memory when reported, and collector
  source (`nvml` or `nvidia-smi`).
- Metal: device name, registry/location id where available, unified-memory
  flag, `recommendedMaxWorkingSetSize` where available, OS/Metal framework
  version where available, and collector source (`metal_probe`,
  `system_profiler`, or `iokit`).

Every result record includes a non-optional `accelerator` section:

```text
accelerator.requested_class        # cpu | cuda | metal | any
accelerator.resolved_class         # cpu | cuda | metal | unavailable
accelerator.selected_devices[]     # stable ids/names used by the backend
accelerator.backend_mode           # cuda, metal, cpu, mixed, or unavailable
accelerator.offload                # gpu_layers, device_map, or backend-specific mode
accelerator.driver_versions        # CUDA/Metal/runtime versions when known
accelerator.fallback_reason        # null unless requested != resolved
accelerator.use_proof_source       # backend callback, env, logs, nvml sample, metal probe, etc.
```

If a CUDA or Metal cell resolves to CPU fallback, the record cannot count as
green CUDA/Metal coverage. Depending on snapshot policy, the terminal outcome
is `failed` with reason `accelerator_mismatch`, or `blocked` with reason
`accelerator_unavailable` or `backend_offload_unverified`. Aggregation groups
by both requested and resolved accelerator so fallback coverage is visible.

## Pinned Eval Snapshot

The snapshot manifest is the stable matrix input for a distributed run. It
records:

- snapshot id and git SHA it was generated from
- bundles and selectors included in the curated run
- model family, checkpoint format (`hf_safetensors`, `gguf`, `onnx`,
  `piper`, etc.), artifact quantization label (`default`, `fp16`, `q4_0`,
  `qat_q4_0`, etc.), backend, and runtime feature group for each cell
- scenarios by capability and depth
- Cargo package/features or build command needed for each cell
- platform constraints, accelerator requirements, and fallback policy
- runtime and resource budgets by cell and profile: wall-clock timeout, max
  artifact/model size, max RSS or footprint, CPU depth limits, and optional
  accelerator-required flags
- artifact requirements and provisioning strategy
- expected output location policy

The driver may discover bundles and scenarios from catalog metadata, but the
distributed run uses the pinned snapshot as the authority. If catalog metadata
changes, a new snapshot is generated and reviewed.

## Skip/Block And Coverage Schema

Every matrix cell emits exactly one terminal coverage record per host attempt.
Coverage metadata is non-optional, including records that never start a model:

```text
coverage.snapshot_id
coverage.cell_id
coverage.depth                         # smoke | enriched
coverage.capability
coverage.scenario_id
coverage.bundle_id
coverage.model_family
coverage.checkpoint_format             # hf_safetensors | gguf | onnx | ...
coverage.quantization                  # default | fp16 | q4_0 | qat_q4_0 | ...
coverage.backend
coverage.profile
coverage.host_id
coverage.host_slug
coverage.arch
coverage.requested_accelerator
coverage.resolved_accelerator
coverage.applicability                 # applicable | not_applicable | blocked_pre_run
coverage.terminal_outcome              # passed | failed | blocked | skipped
coverage.reason                        # enum, required unless passed
coverage.grouping_keys                 # normalized report keys
```

Reason enum values start with `profile_not_applicable`,
`feature_group_not_supported`, `feature_build_failed`, `artifact_missing`,
`artifact_unauthorized`, `hf_token_missing`, `native_toolchain_missing`,
`gguf_toolchain_failed`, `gguf_metal_unverified`, `submodule_missing`,
`cpu_profile_not_practical`, `runtime_budget_exceeded`,
`resource_budget_exceeded`, `accelerator_unavailable`,
`accelerator_mismatch`, `backend_offload_unverified`,
`metric_unavailable_required`, `behavior_assertion_failed`,
`performance_gate_failed`, and `resource_gate_failed`. Implementations may add
new enum variants only with a schema-version bump or explicit compatibility
entry so aggregation remains deterministic.

## PR-Based Aggregation And Reports

`evals report --aggregate` must be functional for v2. It consumes merged host
result directories, not a live scheduler:

```sh
cargo run -p evals -- report --aggregate 'evals/results/**/results.jsonl' \
  --output evals/reports/<snapshot-id>/coverage.md
```

The aggregate report includes:

- per-cell coverage: bundle, model family, checkpoint format, quantization,
  capability, depth, scenario, backend, profile, host, requested accelerator,
  resolved accelerator, terminal outcome (`passed`, `failed`, `blocked`,
  `skipped`), reason, gate statuses, and key metrics
- 2-D and 3-D slices: model x capability, model x quantization x
  backend/platform/depth, capability x platform/profile, capability x depth,
  backend x platform, and requested x resolved accelerator
- missing-coverage lists for snapshot cells not represented by any merged
  result PR
- blocker rollups grouped by artifact, host profile, native toolchain, feature
  matrix, accelerator mismatch, behavior/performance/resource gate, and metric
  unavailability
- metric-gap rollups that distinguish `metric_unsupported_by_backend`,
  `metric_not_instrumented`, `metric_unavailable_on_platform`, and
  `metric_collection_failed` from collected metrics that failed gates
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
- Artifact checkpoint format and artifact quantization are first-class coverage
  grouping keys, independent from runtime precision flags. For GGUF this is the
  actual file quantization label such as `Q4_0`; for HF checkpoints it is the
  reviewed checkpoint precision or default label.

The GGUF axis is blocked until native toolchain checks pass on every recruited
platform that claims GGUF coverage. Linux x86 and GB10 must get a durable repo
or documented fix for the `llama-cpp-sys` bindgen/toolchain `stdbool.h`
failure. macOS/Metal must separately verify the Apple clang, Metal backend
feature flags, shader compilation, and any codesign/runtime requirements. If
GGUF-on-Metal is not supported for a snapshot, the Metal cells are emitted as
structured `blocked` records with reason `gguf_metal_unverified` or
`native_toolchain_missing`; the quant x platform slice must not be silently
empty.

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
  tokens/second, output tokens, context length, peak memory, accelerator use,
  and resource gates
- tool-use: tool-selection and argument precision/recall, round-trip latency,
  repair turns, final-answer correctness, and CEL assertion outcomes
- ASR: real-time factor, WER, transcript length, segment count, and peak memory
- embeddings: vectors/second, embedding dimensions, similarity gap, cosine
  scores, and peak memory
- TTS: real-time factor, synthesis latency, audio duration, sample count,
  chunk count, clipping/silence checks where available, and peak memory
- perf scenarios: warm-up, iteration latencies, mean/p95, throughput, and
  resource peaks

LLM metric collection has an explicit observation boundary. Runners should
prefer streaming/token callback instrumentation for TTFT and output-token count.
When a backend cannot stream, it may expose backend-specific timing hooks or
usage counters. If neither is available, the metric is recorded as unavailable
with a reason such as `metric_unsupported_by_backend` or
`metric_not_instrumented`. Tokens/second is computed only from a collected
output-token count and decode interval; it is not inferred from character count
without an explicit tokenizer source. Warm-up duration is a named phase before
the measured request. CUDA peak VRAM and utilization are sampled for the
selected device through NVML or `nvidia-smi` during warm-up and request windows,
recording start/peak/final memory plus sample source and cadence.

Peak-memory metrics are source-tagged so CUDA discrete VRAM and Apple unified
memory are not conflated. The resource schema supports entries such as:

```text
resources.memory_peaks[] = {
  kind: "process_rss" | "cuda_vram" | "metal_current_allocated" |
        "apple_footprint" | "system_unified_memory",
  bytes: <u64 or null>,
  device_id: <string or null>,
  source: "procfs" | "nvml" | "nvidia_smi" | "metal_api" |
          "mach_task_info" | "sysinfo" | "unavailable",
  unavailable_reason: <enum or null>
}
```

On Apple Silicon, the primary comparable peak-memory value is process footprint
or RSS plus, where backend hooks exist, Metal current allocated bytes and
`recommendedMaxWorkingSetSize` headroom. Reports label these as unified-memory
metrics; they are not shown as CUDA VRAM.

Resource and performance gates have explicit degraded semantics. Each gate
records `gate_status = passed | failed | skipped | blocked`, observed value,
threshold, metric source, and reason. If a required gate's metric is unavailable
on a platform, the default outcome is `blocked` with
`metric_unavailable_required` unless the snapshot marks that gate advisory. An
advisory gate may be `skipped` with reason `metric_unavailable_on_platform`,
but it must appear in the record and aggregate report; it never silently passes.
The process-swap gate remains valid on Linux via `/proc/self/status`; macOS
either gets an explicit provider such as `mach_task_info`/`footprint`/`vm_stat`
or records a skipped/blocked gate according to snapshot policy.

Reviewers should explicitly check for relevant metrics that the runner cannot
currently expose. Missing but relevant metrics should appear in result
`unavailable` fields and in the aggregate report's metric-gap section.

## Result Record

Every eval result should include these sections:

- top-level `schema_version`: result contract version
- `identity`: run id, git SHA, branch, command line, portable host id and host
  slug
- `coverage`: non-optional matrix metadata from the pinned snapshot: snapshot
  id, cell id, capability, scenario id, depth, bundle/model, checkpoint format,
  quantization label, backend, profile, requested/resolved accelerator,
  applicability, terminal outcome, reason enum, and aggregation grouping keys
- `selection`: bundle id, selector, backend, checkpoint, artifact snapshot,
  checkpoint format, artifact quantization label, and artifact digests where
  available
- `profile`: explicit profile such as `local-cpu-x86_64`, `apple-metal`,
  `dgx-spark`, or `cuda-workstation`
- `platform`: OS, kernel, libc, target triple, CPU, RAM, swap, limits, GPU and
  accelerator inventory, including NVIDIA and Metal identity plus driver/runtime
  metadata when available
- `accelerator`: per-cell requested-vs-resolved accelerator, selected device
  ids, backend mode, offload settings, driver/runtime versions, use-proof
  source, and fallback reason
- `runtime`: cargo package/features, child build command/status, build profile,
  runtime precision, context, generation params, relevant redacted env vars, and
  runtime budgets
- `performance`: startup, warmup, common request latency, and a nested
  `capability_metrics` object tagged by capability for embedding vectors per
  second, chat token throughput and TTFT where available, tool-use precision and
  round-trip latency, ASR/TTS real-time factor, or perf summaries. Each metric
  has source and unavailable-reason metadata when needed.
- `resources`: RSS, peak RSS, CPU time, page faults, process swap delta where
  available, source-tagged memory peaks, GPU utilization where available, and
  unavailable reasons
- `acceptance`: behavior, performance, resource, accelerator, and overall gate
  statuses. Each gate names observed value, threshold, source, status, and
  reason.

Unknown platform fields are represented as `null` or `unavailable`, not omitted.
Unknown coverage fields are not allowed in v2 records; if the driver cannot
resolve a coverage key, it emits a `blocked` record that names the missing key.

Acceptance failure reasons must name the section and gate that failed. A resource
failure cannot reuse a passing behavior assertion message; for example a process
swap gate failure should report the `max_process_swap_delta_bytes` threshold and
the observed `process_swap_delta_peak_bytes` value.

## Runner Boundary

`bins/evals/src/runner/mod.rs` owns the binary-local runner trait. Each
runner receives a `RunContext` carrying the parsed scenario, selected bundle,
pinned snapshot cell metadata, build artifact location, profile, artifact root,
runtime flags, budget limits, requested accelerator, platform collector,
metrics sampler, and output sink. Capability runners are responsible for turning
backend observations into the shared sectioned `ResultRecord`; the CLI is
responsible only for parsing flags, loading scenarios, selecting a runner, and
writing emitted records.

Backends that can expose streaming tokens, token counts, accelerator use, device
maps, GPU-layer counts, or memory samples should report them through a small
observation adapter used by the runner. Backends that cannot expose a metric must
return a structured unavailable reason; the runner must not fabricate derived
metrics from unrelated data.

The matrix driver builds on the same runner boundary. It constructs a
`RunContext` per runnable snapshot cell, or emits a skipped/blocked record when
the host cannot or should not run that cell. The report aggregator consumes only
records and does not re-run model code.

The embeddings exemplar proves this boundary with `evals run --bundle
embeddinggemma_300m --scenario embeddings_similarity`: it parses TOML, validates
the snake_case capability filter, starts one compiled embedding bundle, samples
startup and request latencies plus RSS/process-swap resources, emits JSONL, and
applies the `similar_gt_dissimilar` assertion.

## Migration Strategy

This is brownfield. Migration must preserve existing example behavior while
introducing capability-first names.

| Current target | New human example | Eval or perf scenario |
|---|---|---|
| `embeddings` | `embeddings_basic` | `embeddings_similarity` |
| `chat_tool_binding` | `tool_use_basic` | `tool_use_smoke` |
| `chat_mistral_qwen3` | `chat_basic` | `chat_smoke`, `chat_completion_smoke`, `tool_use_smoke` |
| `chat_gguf_gwen3_gemma4` | `chat_basic` | `chat_smoke`, `chat_completion_smoke`, `tool_use_smoke`, `bench_chat_startup` |
| `chat_multimodal_gemma4` | `multimodal_basic` | `chat_smoke`, `multimodal_image_caption`, `tool_use_smoke` |
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

CUDA-class profiles such as `dgx-spark` and `cuda-workstation` must emit
NVIDIA identity when NVML or `nvidia-smi` is present: `gpu_backend =
"nvidia"`, one `gpus[]` entry per device, and accelerator metadata containing
the collector, driver version, CUDA runtime/toolkit version when available, and
memory/utilization sampling source. Memory fields may be `null` on GB10 when
`nvidia-smi` reports `N/A`; GPU identity should still be present. Per-cell
accelerator use proof determines CUDA coverage, not host inventory alone.

Metal-class profiles such as `apple-metal` must probe for a usable Metal device
rather than infer support from macOS/AArch64. Platform inventory should use a
small Metal API probe where possible, with `system_profiler` or IOKit fallback.
Per-cell accelerator use proof must report the resolved Metal backend/device and
offload mode; CPU fallback cannot count as Apple Metal coverage. Unified-memory
peak metrics are source-tagged separately from CUDA VRAM metrics.

GGUF verification is platform-specific. Linux x86 and GB10 must cover the
`stdbool.h`/bindgen toolchain path, while macOS/Metal must independently cover
Apple clang, Metal backend flags, shader build, and runtime loading. Unsupported
GGUF platform cells are represented by blocked records and visible quant x
platform report gaps.
