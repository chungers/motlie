# Model Contract API

## Status: Draft

## Change Log

| Date | Change | Sections |
|------|--------|----------|
| 2026-04-07 | @codex-researcher: Initial API sketch for `libs/model` and `libs/model::eval`. Reflects the current scaffold and keeps the focus on stable contract shapes. | All |
| 2026-04-07 | @codex-researcher: Updated the API sketch to reflect capability introspection helpers and the explicit separation between curated artifact staging and backend startup. | Overview, Bundle API Sketch, Notes |
| 2026-04-08 | @codex-researcher: Clarified the contract-level error model and the library-versus-application error boundary. `ModelError` is now explicitly specified as a typed library error derived with `thiserror`. | Overview, Core Types, Notes |
| 2026-04-08 | @codex-researcher: Added the bundle-level embedding metadata contract (`Embedding`, `EmbeddingSpec`, distance, normalization) so curated embedding bundles can expose vector semantics before runtime startup. | Overview, Core Types, Bundle API Sketch |
| 2026-04-08 | @codex-researcher: Added the end-to-end embedding bundle contract flow and the explicit bridge to `motlie_db::vector::EmbeddingSpec` / `Distance` so implementers can wire curated embedding bundles into the vector subsystem without guessing. | Overview, Bundle API Sketch, Notes |
| 2026-04-08 | @codex-researcher: Added explicit notes on the planned additive chat/multimodal extensions so the current API doc is honest about what is implemented today versus what is already queued for the first chat-capable bundles. | Overview, Core Types, Notes |
| 2026-04-08 | @codex-researcher: Clarified the implemented loaded-descriptor and backend-error contract after PR 139 review. `LoadedBundleDescriptor` is an alias of `BundleMetadata`, `ModelError` now distinguishes backend initialization and execution failures, and `ArtifactPolicy::LocalOnly` is documented as consuming a curated bundle-resolved local model path. | Overview, Core Types, Bundle API Sketch, Notes |

This document sketches the concrete contract shapes currently introduced in `libs/model`. It covers both the core bundle lifecycle/capability contracts and the lightweight `model::eval` vocabulary that higher-level harness tooling should build on.

For implementers, this document should be read as the current contract specification for `libs/model`, not merely aspirational pseudocode. DESIGN explains why these shapes exist; this API document captures what downstream crates are expected to implement against.

## Overview

The first concrete `libs/model` API now includes:

- stable `BundleId`
- `CapabilityKind`, `ContentKind`, `InteractionStyle`, `CapabilityDescriptor`, and `Capabilities`
- `BundleMetadata` (with `LoadedBundleDescriptor` as the loaded-instance alias) and `StartOptions`
- `ArtifactPolicy`
- `ModelError`
- `EmbeddingDistance`, `EmbeddingNormalization`, `EmbeddingSpec`, and `Embedding`
- request/response envelopes for chat, completion, and embeddings
- the `ModelBundle`, `BundleHandle`, `ChatModel`, `CompletionModel`, and `EmbeddingModel` traits
- lightweight eval types in `model::eval`

The goal is to give downstream crates a stable contract surface while keeping the implementation burden low in this first pass.

Important scope note:

- this API is complete for the current embedding vertical slice
- the first chat-capable bundles will require additive extensions already tracked in `DESIGN.md` / `PLAN.md`, including multimodal chat content, `ChatRole::Tool`, richer `ChatResponse` metadata, and more detailed `StartOptions`

For the current vertical slice, this contract is intended to support an end-to-end flow of:

1. a curated embedding bundle defines bundle metadata and an `EmbeddingSpec`
2. `libs/models` exposes that bundle through a direct module and a curated selector enum
3. the caller starts the bundle with `ArtifactPolicy::LocalOnly`
   after `libs/models` has resolved any curated cache layout into a concrete local model path for the backend
4. the loaded handle exposes `EmbeddingModel`
5. downstream crates such as `motlie_db::vector` consume the bundle-level `EmbeddingSpec` to configure vector storage and search semantics consistently

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

Known near-term additive follow-ups for chat-capable bundles:

- `ChatRole::Tool`
- multimodal chat content parts in place of `ChatMessage.content: String`
- richer `ChatResponse` metadata

Primary traits:

- `ModelBundle`
- `Embedding`
- `BundleHandle`
- `ChatModel`
- `CompletionModel`
- `EmbeddingModel`

`ModelError` is a typed library error. Binaries and examples may wrap it with `anyhow`, but libraries should return it directly from bundle lifecycle and capability APIs.

The currently implemented error variants are:

- `Internal`
- `InvalidConfiguration`
- `BackendInitialization`
- `BackendExecution`
- `UnsupportedCapability`

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

`Capabilities::new(...)` canonicalizes descriptors by `CapabilityKind`: the first descriptor for a kind wins, later duplicates are dropped, and `supports(...)` always reflects exactly the descriptor set stored in the struct.

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

### End-to-End Embedding Bundle Contract

For a curated embedding bundle, the intended contract flow is:

```rust
use motlie_model::{
    ArtifactPolicy, ContentKind, Embedding, EmbeddingDistance, EmbeddingModel,
    EmbeddingNormalization, EmbeddingRequest, StartOptions,
};
use motlie_models::{default_artifact_root, embeddings::EmbeddingModels};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model = EmbeddingModels::GoogleGemma300m;

    // Bundle-level metadata used before startup.
    let spec = model.embedding_spec();
    assert_eq!(spec.dimensions, Some(768));
    assert_eq!(spec.distance, EmbeddingDistance::Cosine);
    assert_eq!(spec.normalization, EmbeddingNormalization::L2);
    assert_eq!(spec.input, ContentKind::Text);

    // Runtime bundle start.
    let bundle = model.bundle();
    let handle = bundle
        .start(StartOptions {
            artifact_policy: Some(ArtifactPolicy::LocalOnly {
                root: default_artifact_root(),
            }),
            ..Default::default()
        })
        .await?;

    // Runtime embedding generation.
    let embeddings: &dyn EmbeddingModel = handle.embeddings()?;
    let response = embeddings
        .embed(EmbeddingRequest {
            inputs: vec!["motlie curated model bundle".into()],
        })
        .await?;

    assert_eq!(response.vectors[0].len(), 768);
    handle.shutdown().await?;
    Ok(())
}
```

From a curator’s perspective, the implementation obligations are:

1. implement `ModelBundle` for the concrete curated bundle type
2. implement `Embedding` for the same bundle type
3. expose a stable `EmbeddingSpec`
4. ensure the runtime handle returns a working `EmbeddingModel`
5. ensure the bundle can be started under `ArtifactPolicy::LocalOnly` when local artifacts are required

### Integration with `motlie_db::vector`

`libs/model` must not depend on `motlie_db`, but the intended integration with the vector subsystem is explicit.

Relevant vector types live in:

- [distance.rs](/Users/dchung/projects/claude-mistral/motlie/libs/db/src/vector/distance.rs)
- [schema.rs](/Users/dchung/projects/claude-mistral/motlie/libs/db/src/vector/schema.rs)

The intended mapping is:

- `motlie_model::EmbeddingDistance::Cosine` -> `motlie_db::vector::Distance::Cosine`
- `motlie_model::EmbeddingDistance::SquaredL2` -> `motlie_db::vector::Distance::L2`
- `motlie_model::EmbeddingDistance::Dot` -> `motlie_db::vector::Distance::DotProduct`
- `motlie_model::EmbeddingSpec.dimensions` -> `motlie_db::vector::EmbeddingSpec.dim`
- curated bundle model identity -> `motlie_db::vector::EmbeddingSpec.model`

Illustrative bridge code:

```rust
use motlie_model::{EmbeddingDistance, EmbeddingSpec as ModelEmbeddingSpec};
use motlie_db::vector::{Distance, EmbeddingSpec as DbEmbeddingSpec, VectorElementType};

fn to_db_embedding_spec(
    model_name: &str,
    spec: &ModelEmbeddingSpec,
) -> anyhow::Result<DbEmbeddingSpec> {
    let dim = spec
        .dimensions
        .context("embedding bundle must declare dimensions for vector-space registration")?;

    let distance = match spec.distance {
        EmbeddingDistance::Cosine => Distance::Cosine,
        EmbeddingDistance::SquaredL2 => Distance::L2,
        EmbeddingDistance::Dot => Distance::DotProduct,
    };

    Ok(DbEmbeddingSpec {
        code: 0,
        model: model_name.to_owned(),
        dim: dim as u32,
        distance,
        storage_type: VectorElementType::F32,
        hnsw_m: 16,
        hnsw_ef_construction: 200,
        rabitq_bits: 1,
        rabitq_seed: 42,
    })
}
```

The important contract point is that bundle curation owns the semantic truth about the vectors, and `motlie_db::vector` should derive its storage/search configuration from that truth rather than requiring the application to restate it manually.

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
- `EvalTrack::primary_for_descriptor(...)`
- `tracks_for_capabilities(...)`

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

For the current vertical slice, `motlie_model::eval` also provides the stable bridge that higher-level tooling needs in order to select embedding-oriented runners without bundle-specific branching:

```rust
use motlie_model::{Capabilities, CapabilityDescriptor};
use motlie_model::eval::{capabilities_support_track, tracks_for_capabilities, EvalTrack};

let capabilities = Capabilities::new(vec![CapabilityDescriptor::embeddings()]);
let tracks = tracks_for_capabilities(&capabilities);

assert!(tracks.contains(&EvalTrack::Embeddings));
assert!(capabilities_support_track(&capabilities, EvalTrack::Embeddings));
```

The v0.1 contract is intentionally narrow: embedding capabilities map directly to `EvalTrack::Embeddings`, while text-generation capabilities still require higher-level harness configuration because they may participate in multiple tracks.

## Notes

- `BundleId` lives in `libs/model`, not `libs/models`, because it is part of the stable contract surface.
- capability introspection now includes task kind, input/output content kinds, and interaction style
- `CapabilityKind` does not imply that every variant immediately gets its own executable trait. For the planned multimodal chat path, `Vision` is expected to begin as a capability flag on the chat surface rather than as a separate `VisionModel` trait.
- `model::eval` contains small declarative types only; runners, scoring, suite loading, and reports belong in `libs/model-eval`.
- Curated artifact download and provenance control live above this crate in `libs/models`. Backend startup consumes the resulting artifact root and policy through `StartOptions` rather than using `libs/model` to initiate curated downloads itself.
