# Curated Model Evals And Example Organization Plan

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-06-09 | @codex-399-impl | Added v2 review-first plan for decentralized full-matrix distributed evals, result PR aggregation, artifact provisioning, GGUF toolchain repair, first-class tool-use, and smoke/enriched depth. |
| 2026-06-07 | @codex-399-impl | Fixed Metal review items A-C and recorded the Metal platform population fast-follow decision. |
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
- [ ] Post the v2 DESIGN/PLAN summary to #399 and Discussion #404.
  DESIGN reference: V2 Distributed Driver.
- [ ] Wait for all three reviewer design approvals before coding.
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
- [ ] Implement `report --input <jsonl> --format markdown`.
  DESIGN reference: Decision.
- [ ] Implement `report --aggregate 'evals/results/**/results.jsonl' --output <path>` for consolidated cross-host coverage.
  DESIGN reference: PR-Based Aggregation And Reports.

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

## Platform Follow-Up

- GB10/Linux AArch64 now uses repo-wired `+fp16,+fhm` target features through
  `.cargo/config.toml`, so default `cargo build -p evals --features ...` and
  `cargo run -p evals --features ...` commands no longer require manual
  `RUSTFLAGS`.
- CUDA-class profiles now have initial scenario gates and `PlatformCollector`
  records NVIDIA identity through `nvidia-smi` when available. Future profile
  hardening can add GPU memory/utilization acceptance gates once result policy
  requires them.
- D decision: Metal device population stays in `PlatformCollector` and can land
  as a focused fast-follow before batch migration; A-C are fixed in the
  exemplar first so the schema and resource acceptance semantics are stable.

## Phase 6: V2 Snapshot And Host-Self-Selecting Driver

- [ ] Add a pinned eval snapshot manifest under `evals/snapshots/` for the curated bundle x scenario x depth matrix.
  DESIGN reference: Pinned Eval Snapshot.
- [ ] Add `bins/evals/src/snapshot.rs` to parse and validate snapshot id, git SHA, bundles, scenarios, depth, feature groups, artifact requirements, and platform constraints.
  DESIGN reference: Pinned Eval Snapshot.
- [ ] Add `bins/evals/src/driver.rs` for `evals matrix --snapshot ...`.
  DESIGN reference: V2 Distributed Driver.
- [ ] Detect host architecture and accelerator class, then map to `local-cpu-x86_64`, `local-cpu-aarch64`, `apple-metal`, `cuda-workstation`, or `dgx-spark`.
  DESIGN reference: V2 Distributed Driver.
- [ ] For every snapshot cell, emit a record for run, pass, fail, block, or skip; no silent missing cells.
  DESIGN reference: V2 Distributed Driver.
- [ ] Write collision-free local result directories under `evals/results/<snapshot-id>/<run-id>-<host>-<arch>-<accelerator>/` with `results.jsonl`, `summary.md`, `run-manifest.toml`, and logs.
  DESIGN reference: V2 Distributed Driver.
- [ ] Keep local raw results gitignored until a platform agent intentionally opens a result PR.
  DESIGN reference: PR-Based Aggregation And Reports.

## Phase 7: Artifact Provisioning And GGUF Toolchain

- [ ] Add a provisioning command or driver phase that validates required artifacts before eval execution.
  DESIGN reference: Artifact Provisioning And Native Toolchains.
- [ ] Use `HF_TOKEN` from the environment for gated Hugging Face artifacts without logging, serializing, or committing the token.
  DESIGN reference: Artifact Provisioning And Native Toolchains.
- [ ] Redact secrets in logs and result records; record only token presence and artifact validation status.
  DESIGN reference: Artifact Provisioning And Native Toolchains.
- [ ] Produce structured blocked/skipped records for missing artifacts, unauthorized artifacts, incomplete submodules, and native runtime setup.
  DESIGN reference: Artifact Provisioning And Native Toolchains.
- [ ] Fix or document a durable repo-wired GGUF toolchain path for the `llama-cpp-sys` `stdbool.h` bindgen failure on Linux x86 and GB10.
  DESIGN reference: Artifact Provisioning And Native Toolchains.
- [ ] Re-run GGUF feature builds on amd1, dgx/GB10, amd2, and mac/Metal after the toolchain fix.
  DESIGN reference: Artifact Provisioning And Native Toolchains.

## Phase 8: First-Class Tool-Use Capability

- [ ] Add `libs/eval-tools` with a registry, executable handlers, transcript capture, and CEL assertion support.
  DESIGN reference: Tool-Use Capability And `libs/eval-tools`.
- [ ] Seed `libs/eval-tools` with weather and CEL math handlers equivalent to the examples.
  DESIGN reference: Tool-Use Capability And `libs/eval-tools`.
- [ ] Split `tool_use` scenarios, runner selection, metrics, and reports from `chat` while preserving legacy chat-tool aliases during migration.
  DESIGN reference: Tool-Use Capability And `libs/eval-tools`.
- [ ] Implement the full tool-use round trip: model emits call, harness validates and executes the tool, feeds the result back, and evaluates the final answer.
  DESIGN reference: Tool-Use Capability And `libs/eval-tools`.
- [ ] Add TOML-parameterized tool-use cases with CEL assertions over tool calls, arguments, tool results, and final answers.
  DESIGN reference: Tool-Use Capability And `libs/eval-tools`.

## Phase 9: Eval Depth And Metrics

- [ ] Add `depth = "smoke" | "enriched"` to scenario parsing, result coverage metadata, and aggregate reports.
  DESIGN reference: Eval Depth And Metrics.
- [ ] Add enriched datasets under `evals/datasets/enriched/<capability>/` for chat, tool-use, ASR, embeddings, and TTS.
  DESIGN reference: Eval Depth And Metrics.
- [ ] Extend chat/LLM metrics with warm-up time, time-to-first-token where available, tokens/sec, output tokens, context length, and peak memory.
  DESIGN reference: Eval Depth And Metrics.
- [ ] Extend ASR metrics with WER and RTF, embeddings with vectors/sec and similarity gap, TTS with RTF/audio metrics, and tool-use with precision/recall plus round-trip latency.
  DESIGN reference: Eval Depth And Metrics.
- [ ] Record relevant-but-unavailable metrics explicitly so reports expose metric gaps rather than hiding them.
  DESIGN reference: Eval Depth And Metrics.

## Phase 10: PR-Based Distributed Dry Run And Aggregation

- [ ] Recruit x86 CPU, GB10/CUDA, and Metal agents to run the identical `evals matrix --snapshot ...` command.
  DESIGN reference: V2 Distributed Driver.
- [ ] Each host opens one results PR containing only its local result directory for that snapshot/run id.
  DESIGN reference: PR-Based Aggregation And Reports.
- [ ] After result PRs merge, run `evals report --aggregate` and open a separate report PR with the consolidated coverage report.
  DESIGN reference: PR-Based Aggregation And Reports.
- [ ] Ensure the consolidated report contains per-cell status, model x capability slice, capability x platform slice, depth coverage, blocker rollups, and metric-gap rollups.
  DESIGN reference: PR-Based Aggregation And Reports.
- [ ] Treat the committed consolidated report as the final human merge gate for #399.
  DESIGN reference: PR-Based Aggregation And Reports.
