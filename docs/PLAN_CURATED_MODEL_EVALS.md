# Curated Model Evals And Example Organization Plan

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-06-09 | @codex-399-impl | Addressed v2 R1 design review with tasks for feature-light per-cell builds, accelerator probes/use proof, concrete coverage schema, quant grouping, portable metrics/gates, host identity, and Metal GGUF verification. |
| 2026-06-09 | @codex-399-impl | Added v2 review-first plan for decentralized full-matrix distributed evals, result PR aggregation, artifact provisioning, GGUF toolchain repair, first-class tool-use, and smoke/enriched depth. |
| 2026-06-07 | @codex-399-impl | Fixed Metal review items A-C and recorded the then-current Metal platform population decision. |
| 2026-06-06 | @codex-399-impl | Added durable GB10 Linux/AArch64 build flags, NVIDIA platform inventory, and dgx-spark/cuda-workstation profile gates. |
| 2026-06-06 | @codex-399-impl | Incorporated early pattern review: sectioned result schema, explicit runner context, runnable embeddings path, support namespace, canonical TOML capability values, and platform blocker tracking. |
| 2026-06-06 | @codex-399-impl | Updated plan for the single `bins/evals` binary crate and marked the exemplar module scaffolding now added. |
| 2026-06-06 | @codex-399-impl | Initial implementation plan for issue #399, aligned with `DESIGN_CURATED_MODEL_EVALS.md`. |

Derived from [DESIGN_CURATED_MODEL_EVALS.md](./DESIGN_CURATED_MODEL_EVALS.md).

## Phase 0: V2 Design Review Gate

No v2 implementation starts until @codex-399-amd-rv, @codex-399-cuda-rv,
and @opus-399-mac-rv approve the updated design and this plan. The design
review checks quality, comprehensiveness, correctness, relevance, and practical
usability across x86 CPU, GB10/CUDA, and Metal.

- [x] Read the #399 v2 requirements expansion.
  DESIGN reference: V2 Distributed Driver.
- [x] Update DESIGN and PLAN docs with the v2 target architecture.
  DESIGN reference: PR-Based Aggregation And Reports.
- [x] Post the v2 DESIGN/PLAN summary to #399 and Discussion #404.
  DESIGN reference: V2 Distributed Driver.
- [x] Incorporate R1 needs-work review from AMD, CUDA, and Metal reviewers.
  DESIGN reference: Accelerator Detection And Use Proof.
- [x] Wait for all three reviewer design approvals before coding.
  DESIGN reference: Artifact Provisioning And Native Toolchains.

## Phase 1: Design And Exemplar

- [x] Post initial DESIGN to GitHub issue #399.
  DESIGN reference: Decision, Target Layout.
- [x] Add repo design and plan docs for the binary/data organization.
  DESIGN reference: Decision.
- [x] Convert one example path to the capability-first pattern.
  DESIGN reference: Migration Strategy.
- [x] Add the first scenario manifest under `evals/scenarios`.
  DESIGN reference: Naming Rules.
- [x] Post the exemplar diff to discussion #404 for early review.
  DESIGN reference: Migration Strategy.
- [x] Incorporate early CHANGE-PATTERN feedback before batch refactoring.
  DESIGN reference: Runner Boundary.
- [x] Re-ping for quick pattern re-review before batch refactoring.
  DESIGN reference: Migration Strategy.

## Phase 2: Eval Binary Module Skeleton

- [x] Add `bins/evals/src/scenario.rs` for scenario listing and tagged per-capability TOML metadata.
  DESIGN reference: Naming Rules.
- [x] Add `bins/evals/src/result.rs` for the sectioned JSONL result contract with nested capability metrics.
  DESIGN reference: Result Record.
- [x] Add binary-local platform/profile skeletons.
  DESIGN reference: Result Record.
- [x] Add binary-local report model skeletons for future matrix, leaderboard, and per-bundle evidence.
  DESIGN reference: Decision.

## Phase 3: CLI Wrapper

- [x] Add `bins/evals` as the workspace binary crate.
  DESIGN reference: Target Layout.
- [x] Implement `evals list bundles`.
  DESIGN reference: Decision.
- [x] Implement `evals list scenarios`.
  DESIGN reference: Naming Rules.
- [x] Implement `run --bundle <id> --scenario ...` for embeddings, chat, ASR, TTS, and perf scenarios.
  DESIGN reference: Result Record.
- [x] Implement `report --input <jsonl> --format markdown`.
  DESIGN reference: Decision.
- [x] Implement `report --aggregate 'evals/results/**/results.jsonl' --output <path>` for consolidated cross-host coverage.
  DESIGN reference: PR-Based Aggregation And Reports.
- [x] Validate aggregate input records against the non-optional coverage schema before reporting; strict aggregate mode fails on invalid records, with `--allow-invalid-records` reserved for local forensics.
  DESIGN reference: Skip/Block And Coverage Schema.

## Phase 4: Batch Example Migration

- [ ] Migrate chat examples to `chat_basic`, `tool_use_basic`, and
  `multimodal_basic` while preserving legacy targets.
  DESIGN reference: Migration Strategy.
- [ ] Migrate ASR and streaming ASR examples to `asr_basic` and
  `asr_streaming` while preserving legacy targets.
  DESIGN reference: Migration Strategy.
- [ ] Migrate TTS examples to `tts_basic` while preserving legacy targets.
  DESIGN reference: Migration Strategy.
- [x] Convert `bench_chat` evidence into `bench_chat_startup`.
  DESIGN reference: Naming Rules.

## Phase 5: Verification

- [x] `cargo fmt`
- [x] `cargo build`
- [x] `cargo test`
- [x] `cargo clippy -- -D warnings`
- [ ] targeted example checks for each migrated compatibility target
- [x] targeted `evals` checks after the CLI exists

## Platform Requirements

- GB10/Linux AArch64 now uses repo-wired `+fp16,+fhm` target features through
  `.cargo/config.toml`, so default `cargo build -p evals --features ...` and
  `cargo run -p evals --features ...` commands no longer require manual
  `RUSTFLAGS`.
- CUDA and Metal accelerator detection, inventory, and per-cell use proof are
  first-class v2 deliverables, not follow-up work. The driver must not rely on
  arch/OS heuristics alone for accelerator profile selection.
  DESIGN reference: Accelerator Detection And Use Proof.
- Cross-platform resource gates must report `passed`, `failed`, `skipped`, or
  `blocked`; unavailable platform metrics cannot silently pass.
  DESIGN reference: Eval Depth And Metrics.

## Phase 6: V2 Snapshot And Host-Self-Selecting Driver

- [x] Add a pinned eval snapshot manifest under `evals/snapshots/` for the curated bundle x scenario x depth x checkpoint-format x quantization matrix.
  DESIGN reference: Pinned Eval Snapshot.
- [x] Add `bins/evals/src/snapshot.rs` to parse and validate snapshot id, git SHA, bundles, scenarios, depth, feature groups, checkpoint format, artifact quantization, artifact requirements, platform constraints, accelerator requirements, fallback policy, and runtime/resource budgets.
  DESIGN reference: Pinned Eval Snapshot.
- [x] Add `bins/evals/src/driver.rs` for `evals matrix --snapshot ...` as a feature-light outer driver that can start without compiling all model features.
  DESIGN reference: V2 Distributed Driver.
- [x] Spawn per-cell or per-feature child `cargo` invocations so compile/build failures become structured blocked records instead of preventing the top-level matrix command from running.
  DESIGN reference: V2 Distributed Driver.
- [x] Capture child build failures with reason enums such as `feature_build_failed`, `native_toolchain_missing`, or `gguf_toolchain_failed`.
  DESIGN reference: Skip/Block And Coverage Schema.
- [x] Detect portable host identity with `sysinfo::System::host_name()` or `gethostname(2)` plus platform fallbacks; do not depend on `$HOSTNAME`.
  DESIGN reference: V2 Distributed Driver.
- [x] Detect host architecture and accelerator class by probe, then map to `local-cpu-x86_64`, `local-cpu-aarch64`, `apple-metal`, `cuda-workstation`, or `dgx-spark` only when the required accelerator is usable.
  DESIGN reference: Accelerator Detection And Use Proof.
- [x] Implement CUDA inventory through NVML or `nvidia-smi`, including driver/runtime versions, device ids, names, and memory/utilization source where available.
  DESIGN reference: Accelerator Detection And Use Proof.
- [ ] Implement Metal inventory through a Metal API probe with `system_profiler` or IOKit fallback, including device identity, unified-memory attributes, and recommended working-set size where available.
  DESIGN reference: Accelerator Detection And Use Proof.
- [x] Add a non-optional per-cell accelerator section with requested vs resolved accelerator, selected device ids, backend mode, offload settings (`gpu_layers` or device map), driver/runtime versions, use-proof source, and fallback reason.
  DESIGN reference: Accelerator Detection And Use Proof.
- [ ] Enforce snapshot budgets: max wall time, max artifact/model size, max RSS or footprint, CPU depth limits, and accelerator-required flags.
  DESIGN reference: V2 Distributed Driver.
- [x] Emit blocked/skipped records for practical CPU cutoffs such as `cpu_profile_not_practical` or `runtime_budget_exceeded`; no cell may stall a result PR without JSONL.
  DESIGN reference: V2 Distributed Driver.
- [x] For every snapshot cell, emit exactly one record with non-optional coverage metadata, terminal outcome, reason enum, and grouping keys.
  DESIGN reference: Skip/Block And Coverage Schema.
- [x] Write collision-free local result directories under `evals/results/<snapshot-id>/<run-id>/` with `results.jsonl`, `summary.md`, `run-manifest.toml`, and logs.
  DESIGN reference: V2 Distributed Driver.
- [x] Keep local raw results gitignored until a platform agent intentionally opens a result PR.
  DESIGN reference: PR-Based Aggregation And Reports.

## Phase 7: Artifact Provisioning And GGUF Toolchain

- [x] Add a provisioning command or driver phase that validates required artifacts before eval execution.
  DESIGN reference: Artifact Provisioning And Native Toolchains.
- [x] Use `HF_TOKEN` from the environment for gated Hugging Face artifacts without logging, serializing, or committing the token.
  DESIGN reference: Artifact Provisioning And Native Toolchains.
- [x] Redact secrets in logs and result records; record only token presence and artifact validation status.
  DESIGN reference: Artifact Provisioning And Native Toolchains.
- [x] Produce structured blocked/skipped records for missing artifacts, unauthorized artifacts, incomplete submodules, and native runtime setup.
  DESIGN reference: Artifact Provisioning And Native Toolchains.
- [x] Promote checkpoint format and artifact quantization label to snapshot/result/report grouping keys, independent from runtime precision flags.
  DESIGN reference: Artifact Provisioning And Native Toolchains.
- [x] Fix or document a durable repo-wired GGUF toolchain path for the `llama-cpp-sys` `stdbool.h` bindgen failure on Linux x86 and GB10.
  Implemented in `evals matrix` child builds by injecting repo-local `stdbool.h` plus host compiler builtin include dirs via `BINDGEN_EXTRA_CLANG_ARGS`; direct manual GGUF Cargo builds use the documented equivalent env command. DESIGN reference: Artifact Provisioning And Native Toolchains.
- [x] Add an explicit macOS/Metal GGUF build+verify task covering Apple clang, Metal backend feature flags, shader compilation, and runtime loading.
  Implemented through `apple-metal` GGUF snapshot cells plus the matrix command documented in `evals/README.md`; those cells build/run the llama.cpp GGUF path on Apple Metal hosts. DESIGN reference: Artifact Provisioning And Native Toolchains.
- [x] If GGUF-on-Metal is unsupported for a snapshot, emit blocked Metal GGUF cells with `gguf_metal_unverified` or `native_toolchain_missing` instead of leaving the quant x platform row empty.
  Implemented by matrix preflight: Metal GGUF cells missing the `metal` profile feature marker emit blocked `gguf_metal_unverified` records, and Metal-specific GGUF child build failures are classified as `gguf_metal_unverified`. DESIGN reference: Artifact Provisioning And Native Toolchains.
- [ ] Re-run GGUF feature builds on amd1, dgx/GB10, amd2, and mac/Metal after the platform-specific toolchain checks.
  DESIGN reference: Artifact Provisioning And Native Toolchains.

## Phase 8: First-Class Tool-Use Capability

- [x] Add `libs/eval-tools` with a registry, executable handlers, transcript capture, and CEL assertion support.
  DESIGN reference: Tool-Use Capability And `libs/eval-tools`.
- [x] Seed `libs/eval-tools` with weather and CEL math handlers equivalent to the examples.
  DESIGN reference: Tool-Use Capability And `libs/eval-tools`.
- [x] Split `tool_use` scenarios, runner selection, metrics, and reports from `chat` while preserving legacy chat-tool aliases during migration.
  DESIGN reference: Tool-Use Capability And `libs/eval-tools`.
- [x] Implement the full tool-use round trip: model emits call, harness validates and executes the tool, feeds the result back, and evaluates the final answer.
  DESIGN reference: Tool-Use Capability And `libs/eval-tools`.
- [x] Add TOML-parameterized tool-use cases with CEL assertions over tool calls, arguments, tool results, and final answers.
  DESIGN reference: Tool-Use Capability And `libs/eval-tools`.

## Phase 9: Eval Depth And Metrics

- [x] Add `depth = "smoke" | "enriched"` to scenario parsing, result coverage metadata, and aggregate reports.
  DESIGN reference: Eval Depth And Metrics.
- [ ] Add enriched datasets under `evals/datasets/enriched/<capability>/` for chat, tool-use, ASR, embeddings, and TTS.
  DESIGN reference: Eval Depth And Metrics.
- [x] Extend chat/LLM metrics with warm-up time, time-to-first-token, tokens/sec, output tokens, context length, accelerator use, and peak memory.
  DESIGN reference: Eval Depth And Metrics.
- [ ] Add runner/backend observation adapters for streaming token callbacks or backend timing hooks, usage/token counts, decode intervals, and explicit unavailable reasons.
  DESIGN reference: Runner Boundary.
- [ ] Collect CUDA start/peak/final VRAM and utilization for the selected device through NVML or `nvidia-smi` sampling where available.
  DESIGN reference: Eval Depth And Metrics.
- [x] Define source-tagged peak-memory resources for process RSS, CUDA VRAM, Metal current allocation, Apple footprint, and system unified memory.
  DESIGN reference: Eval Depth And Metrics.
- [x] Implement cross-platform resource-gate statuses `passed`, `failed`, `skipped`, and `blocked`; unavailable metrics must produce gate records with reasons.
  DESIGN reference: Eval Depth And Metrics.
- [x] Extend ASR metrics with WER and RTF, embeddings with vectors/sec and similarity gap, TTS with RTF/audio metrics, and tool-use with precision/recall plus round-trip latency.
  DESIGN reference: Eval Depth And Metrics.
- [x] Record relevant-but-unavailable metrics explicitly so reports expose metric gaps rather than hiding them.
  DESIGN reference: Eval Depth And Metrics.

## Implementation Notes

- `evals/snapshots/curated-v2-smoke.toml` is the current pinned smoke snapshot. It contains per-profile feature overlays so CUDA/Metal model features are only added for matching host profiles.
- `evals matrix --snapshot ... --dry-run` has been validated to emit one JSONL record per snapshot cell without launching model children.
- `evals report --aggregate` has been validated against the dry-run JSONL and renders per-cell, model/capability, quant/backend/profile/depth, accelerator, blocker, and metric-gap slices.
- Raw JSONL/log directories under `evals/results/` are ignored by default; committed Markdown summaries remain visible.

## Phase 10: PR-Based Distributed Dry Run And Aggregation

- [ ] Recruit x86 CPU, GB10/CUDA, and Metal agents to run the identical `evals matrix --snapshot ...` command.
  DESIGN reference: V2 Distributed Driver.
- [ ] Each host opens one results PR containing only its local result directory for that snapshot/run id.
  DESIGN reference: PR-Based Aggregation And Reports.
- [ ] After result PRs merge, run `evals report --aggregate` and open a separate report PR with the consolidated coverage report.
  DESIGN reference: PR-Based Aggregation And Reports.
- [ ] Ensure the consolidated report contains per-cell status, model x capability, model x quantization x backend/platform/depth, capability x platform, requested x resolved accelerator, depth coverage, blocker rollups, and metric-gap rollups.
  DESIGN reference: PR-Based Aggregation And Reports.
- [ ] Treat the committed consolidated report as the final human merge gate for #399.
  DESIGN reference: PR-Based Aggregation And Reports.
