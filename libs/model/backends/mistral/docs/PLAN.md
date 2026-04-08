# Mistral Backend — Implementation Plan

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-07 | @codex-researcher | Initial PLAN for the first `libs/model/backends/mistral` vertical slice. Focused on a generic embedding backend path for `google/embeddinggemma-300m` that satisfies `libs/model` contracts. |
| 2026-04-07 | @codex-researcher | Marked the completed backend scaffold, runtime integration, and unit-test work after the first `mistralrs`-backed embedding slice passed compile/test verification. | Phases 1-4 |
| 2026-04-07 | @codex-researcher | Updated the backend plan after investigating NaN embeddings. The slice now validates true `LocalOnly` startup from a local snapshot, requires the sentence-transformers module stack, and adds an env-gated finite-vector test. |
| 2026-04-08 | @codex-researcher | Clarified the remaining backend TODOs after PR review. Real multi-input inference is now validated by the env-gated test; the suggested `spec.rs` / `handle.rs` file split remains an explicit cleanup follow-up rather than an implied completed task. | Phases 1, 4 |
| 2026-04-08 | @codex-researcher | Moved the embedding loader choice into `MistralEmbeddingSpec` so the backend implementation is genuinely spec-driven rather than hardcoding `EmbeddingGemma` in the generic builder path. | Phase 2, Phase 3 |
| 2026-04-08 | @codex-researcher | Tightened the backend boundary after PR 139 review. Provider-specific local artifact validation moved back into the curated bundle layer, the backend crate dropped its direct `hf-hub` dependency, and `LocalOnly` startup now consumes a resolved local model path instead of reconstructing Hugging Face cache layout internally. | Phases 1, 3 |

Derived from [../../docs/DESIGN.md](../../docs/DESIGN.md). This PLAN covers the generic `mistral` backend implementation work needed for the first embedding-only curated bundle.

---

## Phase 1: Backend Scaffold

Make the backend crate structurally ready to host generic `mistral.rs` implementations.

### 1.1 — Crate shape and module structure

- [x] Keep the crate limited to generic backend logic, not curated bundle metadata.
  DESIGN reference: `libs/model/docs/DESIGN.md` / `Architecture`
- [ ] Add internal module structure for the first slice, for example:
  `embeddings.rs`, `spec.rs`, `handle.rs`.
  `@codex-researcher 2026-04-08 -- Deferred intentionally. The first slice still fits comfortably in one file, and splitting it before a second backend surface lands would be churn without a behavioral benefit.`
  DESIGN reference: `Architecture`
- [x] Keep the backend crate dependent only on `libs/model` plus `mistral.rs`-related runtime dependencies.
  DESIGN reference: `Architecture`

## Phase 2: Embedding Spec and Bundle Implementation

Implement the first generic embedding path.

### 2.1 — Static backend-facing spec

- [x] Finalize `MistralEmbeddingSpec` for:
  `id`, `display_name`, `model_id`, embedding architecture, and `Capabilities`.
  DESIGN reference: `libs/model/docs/DESIGN.md` / `Bundle Contract`
- [x] Provide a built-in constructor for `embeddinggemma_300m`.
  DESIGN reference: vertical slice target agreed during planning

### 2.2 — Generic embedding bundle implementation

- [x] Finalize `MistralEmbeddingBundle` as a generic `ModelBundle` implementation.
  DESIGN reference: `libs/model/docs/DESIGN.md` / `Bundle Contract`
- [x] Finalize `MistralEmbeddingHandle` as an embedding-only `BundleHandle`.
  DESIGN reference: `Loaded Handle`
- [x] Ensure unsupported capabilities return `UnsupportedCapability(CapabilityKind::...)`.
  DESIGN reference: `Lifecycle Rules`

## Phase 3: Replace Placeholder Embedder with Real `mistral.rs`

The current placeholder embedder is only acceptable for contract validation. The real slice must exercise `mistral.rs`.

### 3.1 — Runtime integration

- [x] Add the `mistral.rs` dependency set required for embeddings.
- [x] Identify the concrete embedding load path for `google/embeddinggemma-300m`.
- [x] Implement backend initialization from `StartOptions`.
- [x] Implement actual `EmbeddingModel::embed()` using the `mistral.rs` embedding API.
- [x] Normalize backend errors into `ModelError`.

### 3.2 — Operational behavior

- [x] Define how the backend consumes local model artifacts for the first slice.
- [x] Keep provider-specific cache-layout resolution and validation out of the generic backend crate.
- [x] Define what happens when required weights are unavailable.
- [x] Define whether the first slice assumes local sidecar assets only, or permits a preconfigured model path.

## Phase 4: Tests and Verification

### 4.1 — Unit tests

- [x] Add unit tests for spec construction and unsupported-capability behavior.
- [x] Add tests for deterministic handle metadata and capability exposure.

### 4.2 — Integration validation

- [x] Add at least one integration-style test or env-gated test for real `mistral.rs` embedding inference.
- [x] Validate multi-input embedding requests.
- [x] Validate that `embeddinggemma_300m` can be instantiated through `libs/models::Catalog`.

### 4.3 — Required verification commands

- [x] `cargo check -p motlie-model-mistral`
- [x] `cargo test -p motlie-model-mistral`
- [x] `cargo check -p motlie-model -p motlie-model-mistral -p motlie-models`
