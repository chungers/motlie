# Model Contract API

## Status: Draft

## Change Log

| Date | Change | Sections |
|------|--------|----------|
| 2026-04-07 | @codex-researcher: Initial API sketch for `libs/model` and `libs/model::eval`. Reflects the current scaffold and keeps the focus on stable contract shapes. | All |
| 2026-04-07 | @codex-researcher: Updated the API sketch to reflect capability introspection helpers and the explicit separation between curated artifact staging and backend startup. | Overview, Bundle API Sketch, Notes |
| 2026-04-08 | @codex-researcher: Clarified the contract-level error model and the library-versus-application error boundary. `ModelError` is now explicitly specified as a typed library error derived with `thiserror`. | Overview, Core Types, Notes |
| 2026-04-08 | @codex-researcher: Added the bundle-level embedding metadata contract (`Embedding`, `EmbeddingSpec`, distance, normalization) so curated embedding bundles can expose vector semantics before runtime startup. | Overview, Core Types, Bundle API Sketch |

This document sketches the concrete contract shapes currently introduced in `libs/model`. It covers both the core bundle lifecycle/capability contracts and the lightweight `model::eval` vocabulary that higher-level harness tooling should build on.

For implementers, this document should be read as the current contract specification for `libs/model`, not merely aspirational pseudocode. DESIGN explains why these shapes exist; this API document captures what downstream crates are expected to implement against.

## Overview

The first concrete `libs/model` API now includes:

- stable `BundleId`
- `CapabilityKind`, `ContentKind`, `InteractionStyle`, `CapabilityDescriptor`, and `Capabilities`
- `BundleMetadata`, `LoadedBundleDescriptor`, and `StartOptions`
- `ArtifactPolicy`
- `ModelError`
- `EmbeddingDistance`, `EmbeddingNormalization`, `EmbeddingSpec`, and `Embedding`
- request/response envelopes for chat, completion, and embeddings
- the `ModelBundle`, `BundleHandle`, `ChatModel`, `CompletionModel`, and `EmbeddingModel` traits
- lightweight eval types in `model::eval`

The goal is to give downstream crates a stable contract surface while keeping the implementation burden low in this first pass.

## Core Types

Primary bundle-contract types:

- `BundleId`
- `CapabilityKind`
- `ContentKind`
- `InteractionStyle`
- `CapabilityDescriptor`
- `Capabilities`
- `BundleMetadata`
- `StartOptions`
- `ArtifactPolicy`
- `LoadedBundleDescriptor`
- `ModelError`
- `EmbeddingDistance`
- `EmbeddingNormalization`
- `EmbeddingSpec`

Primary capability request/response types:

- `ChatRole`
- `ChatMessage`
- `GenerationParams`
- `ChatRequest`
- `ChatResponse`
- `CompletionRequest`
- `CompletionResponse`
- `EmbeddingRequest`
- `EmbeddingResponse`

Primary traits:

- `ModelBundle`
- `Embedding`
- `BundleHandle`
- `ChatModel`
- `CompletionModel`
- `EmbeddingModel`

`ModelError` is a typed library error. Binaries and examples may wrap it with `anyhow`, but libraries should return it directly from bundle lifecycle and capability APIs.

## Bundle API Sketch

### Bundle Metadata and Capability Introspection

```rust
use motlie_model::{
    BundleId, BundleMetadata, Capabilities, CapabilityDescriptor,
};

let metadata = BundleMetadata {
    id: BundleId::new("qwen3_5_instruct"),
    display_name: "Qwen 3.5 Instruct".into(),
    capabilities: Capabilities::new(vec![
        CapabilityDescriptor::chat(),
        CapabilityDescriptor::completion(),
    ]),
};
```

### Request Envelopes

```rust
use motlie_model::{
    ChatMessage, ChatRequest, ChatRole, CompletionRequest, EmbeddingRequest, GenerationParams,
};

let chat = ChatRequest {
    messages: vec![
        ChatMessage::new(ChatRole::System, "Be concise."),
        ChatMessage::new(ChatRole::User, "Summarize the bundle model."),
    ],
    params: GenerationParams {
        max_tokens: Some(256),
        ..Default::default()
    },
};

let completion = CompletionRequest {
    prompt: "Explain deterministic extraction in one paragraph.".into(),
    params: GenerationParams::default(),
};

let embeddings = EmbeddingRequest {
    inputs: vec!["motlie model bundle".into(), "deterministic package bundle".into()],
};
```

### Trait Shapes

```rust
use async_trait::async_trait;
use motlie_model::{
    ArtifactPolicy, BundleHandle, BundleId, BundleMetadata, Capabilities, ChatModel,
    CompletionModel, EmbeddingModel, LoadedBundleDescriptor, ModelBundle, ModelError,
    StartOptions,
};

#[async_trait]
impl ModelBundle for MyBundle {
    fn id(&self) -> &BundleId { todo!() }
    fn metadata(&self) -> &BundleMetadata { todo!() }
    fn capabilities(&self) -> &Capabilities { todo!() }

    async fn start(
        &self,
        options: StartOptions,
    ) -> Result<Box<dyn BundleHandle>, ModelError> {
        todo!()
    }
}
```

### Bundle-Level Embedding Metadata

```rust
use motlie_model::{
    ContentKind, Embedding, EmbeddingDistance, EmbeddingNormalization, EmbeddingSpec,
};

let spec = EmbeddingSpec {
    dimensions: Some(768),
    distance: EmbeddingDistance::Cosine,
    normalization: EmbeddingNormalization::L2,
    input: ContentKind::Text,
    output: ContentKind::EmbeddingVector,
    summary: "Normalized text embeddings for semantic similarity and retrieval.",
};
```

This bundle-level metadata trait is intentionally separate from the runtime `EmbeddingModel` trait:

- `Embedding` describes the curated bundle's vector semantics
- `EmbeddingModel` executes embedding generation on the loaded handle

Loaded handles also expose `supports(CapabilityKind)` as a convenience over `capabilities().supports(...)`, which keeps harness and catalog-driven code simple when it only needs to branch on capability presence.

### Startup Artifact Policy

```rust
use std::path::PathBuf;

use motlie_model::{ArtifactPolicy, StartOptions};

let regulated = StartOptions {
    artifact_policy: Some(ArtifactPolicy::LocalOnly {
        root: PathBuf::from("artifacts/models/hf-cache"),
    }),
    ..Default::default()
};

let permissive = StartOptions {
    artifact_policy: Some(ArtifactPolicy::AllowFetch {
        root: Some(PathBuf::from("artifacts/models/hf-cache")),
    }),
    ..Default::default()
};
```

## `model::eval` API Sketch

Lightweight eval types currently include:

- `EvalTrack`
- `EvalCaseId`
- `EvalCase`
- `EvalResult`

These are intentionally small. They exist to let `libs/models` annotate bundles for evaluation tracks and to give `libs/model-eval` stable vocabulary to build on.

```rust
use motlie_model::eval::{EvalCase, EvalCaseId, EvalResult, EvalTrack};

let case = EvalCase {
    id: EvalCaseId::new("reasoning-basic-001"),
    track: EvalTrack::Reasoning,
    prompt: "If all A are B and all B are C, are all A C?".into(),
};

let result = EvalResult {
    case_id: case.id.clone(),
    score: 1.0,
    passed: true,
    notes: vec!["matched expected reasoning chain".into()],
};
```

## Notes

- `BundleId` lives in `libs/model`, not `libs/models`, because it is part of the stable contract surface.
- capability introspection now includes task kind, input/output content kinds, and interaction style
- `model::eval` contains small declarative types only; runners, scoring, suite loading, and reports belong in `libs/model-eval`.
- Curated artifact download and provenance control live above this crate in `libs/models`. Backend startup consumes the resulting artifact root and policy through `StartOptions` rather than using `libs/model` to initiate curated downloads itself.
