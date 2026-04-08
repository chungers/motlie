# Model Evaluation Tooling — Implementation Plan

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-07 | @codex-researcher | Initial PLAN for `libs/model-eval` vertical slice work. Covers the minimum harness functionality needed to exercise the first embedding bundle and validate the `model::eval` abstractions. |
| 2026-04-08 | @codex-researcher | Added the first implemented cross-crate proof point after PR review: `libs/model-eval` now consumes the `CapabilityDescriptor` -> `EvalTrack` mapping from `motlie_model::eval` and verifies that the default curated embedding bundle is track-eligible without bundle-specific branching. The fuller suite/runner/report work remains intentionally pending. | Phases 3-4 |

Derived from [DESIGN.md](./DESIGN.md). This PLAN focuses on only the minimum tooling needed to support the first embedding vertical slice.

---

## Phase 1: Harness Foundations

Build the smallest substantial tooling that proves the `libs/model::eval` split is worthwhile.

### 1.1 — Suite and runner skeleton

- [ ] Add internal module structure for:
  `suite`, `runner`, `report`, and `tracks/embeddings`.
  DESIGN reference: `Core Responsibilities`
- [ ] Finalize a small `EvalSuite` shape layered on `model::eval::EvalCase`.
  DESIGN reference: `API Sketch`
- [ ] Finalize a small `EvalReport` / `EvalSummary` shape layered on `model::eval::EvalResult`.
  DESIGN reference: `API Sketch`

## Phase 2: Embedding Track Runner

Use the first curated bundle to prove the harness architecture.

### 2.1 — Embedding runner

- [ ] Implement an embeddings runner that accepts:
  a `BundleHandle`,
  verifies `CapabilityKind::Embeddings`,
  and executes `EmbeddingRequest` cases.
  DESIGN reference: `Core Responsibilities`
- [ ] Implement simple result scoring for the first slice, for example:
  vector count/shape checks and deterministic placeholder-output checks until real runtime inference lands.
  DESIGN reference: `Evaluation Serves Curation`
- [ ] Add tests for unsupported-capability failure when a non-embedding bundle is passed.

### 2.2 — First embedding suite

- [ ] Add a small in-crate suite for the first vertical slice.
- [ ] Include at least:
  single-input case,
  multi-input case,
  and one deterministic comparison case.
- [ ] Keep suite assets/code simple enough to move later if a repo-level `evals/` directory is adopted.
  DESIGN reference: `Open Concerns`

## Phase 3: Catalog-Driven Evaluation

Use the curated catalog, not ad hoc bundle construction, to drive the first slice.

### 3.1 — Catalog selection

- [x] Add a helper that determines whether a capability set is eligible for `EvalTrack::Embeddings` using only `motlie_model::eval` vocabulary.
  DESIGN reference: `Core Responsibilities`
- [x] Ensure the helper works with `libs/models::Catalog`.
  DESIGN reference: `Architecture`
- [x] Add tests for track-based selection using the first curated bundle.

## Phase 4: Vertical Slice Validation

### 4.1 — End-to-end embedding eval

- [ ] Validate:
  `Catalog::with_defaults()` -> instantiate `embeddinggemma_300m` -> start bundle -> run embedding suite -> produce report.
- [ ] Keep the first report format minimal but structured.
  DESIGN reference: `API Sketch`

### 4.2 — Required verification commands

- [x] `cargo check -p motlie-model-eval`
- [x] `cargo test -p motlie-model-eval`
- [x] `cargo check -p motlie-model -p motlie-model-mistral -p motlie-models -p motlie-model-eval`
