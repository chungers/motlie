# Model Contract Library — Implementation Plan

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-07 | @codex-researcher | Initial PLAN for `libs/model` vertical slice support. Covers contract finalization for the first embedding example, including capability introspection, request/response envelopes, lifecycle traits, and lightweight `model::eval` vocabulary. |
| 2026-04-07 | @codex-researcher | Marked the completed contract and test work for the first embedding slice after `cargo check`/`cargo test` verification. | Phases 1-5 |
| 2026-04-08 | @codex-researcher | Closed the missing capability-ordering and eval-track mapping gaps after PR review. Added explicit `CapabilityDescriptor` -> `EvalTrack` helpers/tests and a minimal `libs/model-eval` cross-crate consumption proof so the remaining unchecked items are true deferrals rather than silent omissions. | Phases 1, 4, 5 |

Derived from [DESIGN.md](./DESIGN.md). This PLAN covers the contract work needed to support the first end-to-end embedding vertical slice while preserving the longer-term curated-bundle architecture.

---

## Phase 1: Core Identity and Introspection

Establish stable contract vocabulary that all bundles, backends, and harnesses will share.

### 1.1 — Bundle identity and metadata

- [x] Finalize `BundleId` as the stable bundle selection type.
  DESIGN reference: `Core Abstractions` / `Bundle Identity`
- [x] Finalize `BundleMetadata` and `LoadedBundleDescriptor` so both carry:
  `id`, `display_name`, and `Capabilities`.
  DESIGN reference: `Core Abstractions` / `Bundle Contract`
- [x] Finalize `StartOptions` as the minimal deployment-oriented load surface for the first slice.
  DESIGN reference: `Lifecycle`
- [x] Add unit tests for `BundleId` construction/display/ordering and metadata cloning/equality.
  DESIGN reference: `Testing Scope for PLAN`

### 1.2 — Capability introspection model

- [x] Finalize `CapabilityKind` for the first supported set:
  `Chat`, `Completion`, `Embeddings`, `Vision`, `Ocr`.
  DESIGN reference: `Capability Model`
- [x] Finalize normalized `ContentKind` and `InteractionStyle`.
  DESIGN reference: `Capability Model`, `Capability Surfaces`
- [x] Finalize `CapabilityDescriptor` with:
  `kind`, `summary`, `inputs`, `outputs`, `interaction`.
  DESIGN reference: `Capability Model`
- [x] Finalize `Capabilities` as a descriptor-backed set with:
  `new`, `descriptors`, `supports`, and first built-ins such as `embeddings()`.
  DESIGN reference: `Capability Model`
- [x] Add unit tests for:
  `supports()`, duplicate/ordering behavior, and built-in descriptor helpers.
  DESIGN reference: `Testing Scope for PLAN`

## Phase 2: Lifecycle and Capability Traits

Define the actual contract backends and curated bundles must satisfy.

### 2.1 — Error and lifecycle traits

- [x] Finalize `ModelError` for the first slice:
  `Internal`, `InvalidConfiguration`, `UnsupportedCapability`.
  DESIGN reference: `Core Abstractions`, `Lifecycle`
- [x] Finalize `ModelBundle` and `BundleHandle` trait signatures.
  DESIGN reference: `Bundle Contract`, `Loaded Handle`
- [x] Ensure `BundleHandle` exposes both:
  `capabilities()` and `supports(CapabilityKind)`.
  DESIGN reference: `Capability Model`
- [x] Add unit tests using a small fake bundle/handle to verify unsupported-capability behavior.
  DESIGN reference: `Lifecycle Rules`

### 2.2 — Capability-specific traits

- [x] Finalize `ChatModel`, `CompletionModel`, and `EmbeddingModel`.
  DESIGN reference: `Capability Surfaces`
- [x] Ensure the first vertical slice only needs `EmbeddingModel`, but keep the other traits compilable and stable.
  DESIGN reference: `Capability Surfaces`
- [x] Add a small fake embedding-only handle test verifying:
  `chat()` and `completion()` return `UnsupportedCapability`,
  while `embeddings()` succeeds.
  DESIGN reference: `Lifecycle Rules`, `Testing Scope for PLAN`

## Phase 3: Request/Response Envelopes

Keep the first shared request/response surface small but explicit.

### 3.1 — Text-producing request/response types

- [x] Finalize `ChatRole`, `ChatMessage`, `GenerationParams`, `ChatRequest`, `ChatResponse`.
  DESIGN reference: `Shared Request/Response Types`
- [x] Finalize `CompletionRequest`, `CompletionResponse`.
  DESIGN reference: `Shared Request/Response Types`
- [x] Add tests for request construction and default parameter behavior.
  DESIGN reference: `Testing Scope for PLAN`

### 3.2 — Embedding request/response types

- [x] Finalize `EmbeddingRequest { inputs: Vec<String> }`.
  DESIGN reference: `Capability Surfaces` / `Embeddings`
- [x] Finalize `EmbeddingResponse { vectors: Vec<Vec<f32>> }`.
  DESIGN reference: `Capability Surfaces` / `Embeddings`
- [x] Add tests for empty-input and multi-input request/response shape expectations.
  DESIGN reference: `Testing Scope for PLAN`

## Phase 4: Lightweight `model::eval`

Keep only the small stable vocabulary needed by higher-level tooling.

### 4.1 — Eval-facing core types

- [x] Finalize `EvalTrack`.
  DESIGN reference: `Evaluation Harness Support`
- [x] Finalize `EvalCaseId`, `EvalCase`, and `EvalResult`.
  DESIGN reference: `Evaluation Harness Support`
- [x] Ensure `EvalTrack::Embeddings` is sufficient for the first vertical slice.
  DESIGN reference: `Evaluation Harness Support`
- [x] Add unit tests for eval-case identity and simple result construction.
  DESIGN reference: `Testing Scope for PLAN`

### 4.2 — Introspection compatibility for eval tooling

- [x] Document and test the mapping between `CapabilityDescriptor` and `EvalTrack` selection for embeddings.
  DESIGN reference: `Evaluation Harness Support`
- [x] Ensure the contract is sufficient for `libs/models` to tag bundles and for `libs/model-eval` to select runners without bundle-specific branching.
  DESIGN reference: `Framework Principles` / `Evaluability as a First-Class Concern`

## Phase 5: Vertical Slice Validation

Validate that the contract is sufficient for the first curated embedding example.

### 5.1 — Cross-crate validation tasks

- [x] Verify `libs/model/backends/mistral` can implement an embedding-only bundle without adding backend-specific types to `libs/model`.
  DESIGN reference: `Architecture`
- [x] Verify `libs/models` can describe and instantiate `embeddinggemma_300m` using only stable `libs/model` contracts.
  DESIGN reference: `Architecture`
- [x] Verify `libs/model-eval` can consume `CapabilityDescriptor` and `EvalTrack::Embeddings` without extra contract changes.
  DESIGN reference: `Evaluation Harness Support`

### 5.2 — Required verification commands

- [x] `cargo check -p motlie-model`
- [x] `cargo check -p motlie-model -p motlie-model-mistral -p motlie-models -p motlie-model-eval`
- [x] Add unit tests in `libs/model` and run `cargo test -p motlie-model`
- [x] Add a minimal cross-crate `libs/model-eval` consumption test and run `cargo test -p motlie-model-eval`
