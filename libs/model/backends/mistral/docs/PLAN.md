# Mistral Backend — Implementation Plan

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-07 | @codex-researcher | Initial PLAN for the first `libs/model/backends/mistral` vertical slice. Focused on a generic embedding backend path for `google/embeddinggemma-300m` that satisfies `libs/model` contracts. |

Derived from [../../docs/DESIGN.md](../../docs/DESIGN.md). This PLAN covers the generic `mistral` backend implementation work needed for the first embedding-only curated bundle.

---

## Phase 1: Backend Scaffold

Make the backend crate structurally ready to host generic `mistral.rs` implementations.

### 1.1 — Crate shape and module structure

- [ ] Keep the crate limited to generic backend logic, not curated bundle metadata.
  DESIGN reference: `libs/model/docs/DESIGN.md` / `Architecture`
- [ ] Add internal module structure for the first slice, for example:
  `embeddings.rs`, `spec.rs`, `handle.rs`.
  DESIGN reference: `Architecture`
- [ ] Keep the backend crate dependent only on `libs/model` plus `mistral.rs`-related runtime dependencies.
  DESIGN reference: `Architecture`

## Phase 2: Embedding Spec and Bundle Implementation

Implement the first generic embedding path.

### 2.1 — Static backend-facing spec

- [ ] Finalize `MistralEmbeddingSpec` for:
  `id`, `display_name`, `model_id`, and `Capabilities`.
  DESIGN reference: `libs/model/docs/DESIGN.md` / `Bundle Contract`
- [ ] Provide a built-in constructor for `embeddinggemma_300m`.
  DESIGN reference: vertical slice target agreed during planning

### 2.2 — Generic embedding bundle implementation

- [ ] Finalize `MistralEmbeddingBundle` as a generic `ModelBundle` implementation.
  DESIGN reference: `libs/model/docs/DESIGN.md` / `Bundle Contract`
- [ ] Finalize `MistralEmbeddingHandle` as an embedding-only `BundleHandle`.
  DESIGN reference: `Loaded Handle`
- [ ] Ensure unsupported capabilities return `UnsupportedCapability(CapabilityKind::...)`.
  DESIGN reference: `Lifecycle Rules`

## Phase 3: Replace Placeholder Embedder with Real `mistral.rs`

The current placeholder embedder is only acceptable for contract validation. The real slice must exercise `mistral.rs`.

### 3.1 — Runtime integration

- [ ] Add the `mistral.rs` dependency set required for embeddings.
- [ ] Identify the concrete embedding load path for `google/embeddinggemma-300m`.
- [ ] Implement backend initialization from `StartOptions`.
- [ ] Implement actual `EmbeddingModel::embed()` using the `mistral.rs` embedding API.
- [ ] Normalize backend errors into `ModelError`.

### 3.2 — Operational behavior

- [ ] Define how local model artifacts are located for the first slice.
- [ ] Define what happens when required weights are unavailable.
- [ ] Define whether the first slice assumes local sidecar assets only, or permits a preconfigured model path.

## Phase 4: Tests and Verification

### 4.1 — Unit tests

- [ ] Add unit tests for spec construction and unsupported-capability behavior.
- [ ] Add tests for deterministic handle metadata and capability exposure.

### 4.2 — Integration validation

- [ ] Add at least one integration-style test or env-gated test for real `mistral.rs` embedding inference.
- [ ] Validate multi-input embedding requests.
- [ ] Validate that `embeddinggemma_300m` can be instantiated through `libs/models::Catalog`.

### 4.3 — Required verification commands

- [ ] `cargo check -p motlie-model-mistral`
- [ ] `cargo test -p motlie-model-mistral`
- [ ] `cargo check -p motlie-model -p motlie-model-mistral -p motlie-models`

