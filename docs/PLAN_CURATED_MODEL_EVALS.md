# Curated Model Evals And Example Organization Plan

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-06-06 | -399-impl | Updated plan for the single `bins/evals` binary crate and marked the exemplar module scaffolding now added. |
| 2026-06-06 | @codex-399-impl | Initial implementation plan for issue #399, aligned with `DESIGN_CURATED_MODEL_EVALS.md`. |

Derived from [DESIGN_CURATED_MODEL_EVALS.md](./DESIGN_CURATED_MODEL_EVALS.md).

## Phase 1: Design And Exemplar

- [x] Post initial DESIGN to GitHub issue #399.
  DESIGN reference: Decision, Target Layout.
- [x] Add repo design and plan docs for the binary/data organization.
  DESIGN reference: Decision.
- [x] Convert one example path to the capability-first pattern.
  DESIGN reference: Migration Strategy.
- [x] Add the first scenario manifest under `evals/scenarios`.
  DESIGN reference: Naming Rules.
- [ ] Post the exemplar diff to discussion #404 and wait for early review before
  batch refactoring.
  DESIGN reference: Migration Strategy.

## Phase 2: Eval Binary Module Skeleton

- [x] Add `bins/evals/src/scenario.rs` for scenario listing and future parsed TOML metadata.
  DESIGN reference: Naming Rules.
- [x] Add `bins/evals/src/result.rs` for JSONL result record skeletons.
  DESIGN reference: Result Record.
- [x] Add binary-local platform/profile skeletons.
  DESIGN reference: Result Record.
- [x] Add binary-local report model skeletons for future matrix, leaderboard, and per-bundle evidence.
  DESIGN reference: Decision.

## Phase 3: CLI Wrapper

- [x] Add `bins/evals` as the workspace binary crate.
  DESIGN reference: Target Layout.
- [ ] Implement `evals list bundles`.
  DESIGN reference: Decision.
- [x] Implement `evals list scenarios`.
  DESIGN reference: Naming Rules.
- [ ] Implement a minimal `run --bundle <id> --scenario embeddings_similarity`.
  DESIGN reference: Result Record.
- [ ] Implement `report --input <jsonl> --format markdown`.
  DESIGN reference: Decision.

## Phase 4: Batch Example Migration

- [ ] Migrate chat examples to `chat_basic`, `tool_use_basic`, and
  `multimodal_basic` while preserving legacy targets.
  DESIGN reference: Migration Strategy.
- [ ] Migrate ASR and streaming ASR examples to `asr_basic` and
  `asr_streaming` while preserving legacy targets.
  DESIGN reference: Migration Strategy.
- [ ] Migrate TTS examples to `tts_basic` while preserving legacy targets.
  DESIGN reference: Migration Strategy.
- [ ] Convert `bench_chat` evidence into `perf_chat_startup`.
  DESIGN reference: Naming Rules.

## Phase 5: Verification

- [ ] `cargo fmt`
- [ ] `cargo build`
- [ ] `cargo test`
- [ ] `cargo clippy -- -D warnings`
- [ ] targeted example checks for each migrated compatibility target
- [ ] targeted `evals` checks after the CLI exists
