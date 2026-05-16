# Model Contract API

## Status: Draft

## Change Log

| Date | Change | Sections |
|------|--------|----------|
| 2026-04-24 | @codex-gpt55: Added `QuantizationBits::Five` and `QuantizationBits::FloatEight` so GGUF Q5 and FP8 can be represented without overloading Q8. | Core Types, Bundle API Sketch |
| 2026-04-07 | @codex-researcher: Initial API sketch for `libs/model` and `libs/model::eval`. Reflects the current scaffold and keeps the focus on stable contract shapes. | All |
| 2026-04-07 | @codex-researcher: Updated the API sketch to reflect capability introspection helpers and the explicit separation between curated artifact staging and backend startup. | Overview, Bundle API Sketch, Notes |
| 2026-04-08 | @codex-researcher: Clarified the contract-level error model and the library-versus-application error boundary. `ModelError` is now explicitly specified as a typed library error derived with `thiserror`. | Overview, Core Types, Notes |
| 2026-04-08 | @codex-researcher: Added the bundle-level embedding metadata contract (`Embedding`, `EmbeddingSpec`, distance, normalization) so curated embedding bundles can expose vector semantics before runtime startup. | Overview, Core Types, Bundle API Sketch |
| 2026-04-08 | @codex-researcher: Added the end-to-end embedding bundle contract flow and the explicit bridge to `motlie_db::vector::EmbeddingSpec` / `Distance` so implementers can wire curated embedding bundles into the vector subsystem without guessing. | Overview, Bundle API Sketch, Notes |
| 2026-04-08 | @codex-researcher: Added explicit notes on the planned additive chat/multimodal extensions so the current API doc is honest about what is implemented today versus what is already queued for the first chat-capable bundles. | Overview, Core Types, Notes |
| 2026-04-08 | @codex-researcher: Clarified the implemented loaded-descriptor and backend-error contract after PR 139 review. `LoadedBundleDescriptor` is an alias of `BundleMetadata`, `ModelError` now distinguishes backend initialization and execution failures, and `ArtifactPolicy::LocalOnly` is documented as consuming a curated bundle-resolved local model path. | Overview, Core Types, Bundle API Sketch, Notes |
| 2026-04-08 | @claude: Added `QuantizationBits` to `StartOptions` and documented the quantized startup pattern for the Qwen3-4B chat slice (#141). | Core Types, Bundle API Sketch |
| 2026-04-08 | @codex-researcher: Updated the chat contract for the Gemma 4 multimodal slice (#142). `ChatMessage` now carries `ContentPart`s, the first vision-capable bundle still uses `ChatModel`, and `examples/v0.3` is now the concrete end-to-end reference for text+image chat. | Overview, Core Types, Bundle API Sketch, Notes |
| 2026-04-09 | @codex-researcher: Added handle-level metric snapshots and unit-safe wrappers. Runtime/request aggregates now live on `BundleHandle::metric_snapshot()` instead of individual responses. | Overview, Core Types, Bundle API Sketch, Notes |
| 2026-04-09 | @codex-researcher: Documented the current cross-platform runtime-metrics implementation. `mistral` backends and examples use `sysinfo` for current RSS on macOS and Linux, with Motlie maintaining the observed peak in-handle rather than relying on an OS-native historical peak counter. | Handle-Level Metrics, Notes |
| 2026-04-09 | @codex-researcher: Added the second embedding slice to the quantization examples. `QuantizationSupport::without_recommended([Q8])` is now concretely exercised by the Qwen3-Embedding-0.6B bundle, while EmbeddingGemma remains unquantized. | Core Types |
| 2026-05-11 | @codex-tool-calling: Added the typed tool-calling chat vocabulary: `ToolSpec`, `ToolInputSchema`, `ToolArguments`, `ToolChoice`, `ToolCall`, `ChatRole::Tool`, tool-aware `ChatRequest`/`ChatResponse` fields, and descriptive `CapabilityKind::ToolUse`. Backend adapters still gate tool-bearing requests until concrete model paths are wired and tested. | Overview, Core Types, Request Envelopes |
| 2026-05-13 | @codex-tool-calling: Added typed Rust tool binding helpers, `Capabilities` helpers for chat/completion/tool-use combinations, and the safetensors `mistral.rs` adapter path for Qwen3/Gemma 4 tool calls. Runtime tool registries stay outside the core `motlie-model` contract; examples use `motlie_models::ToolRegistry`. GGUF tool-bearing requests remain gated by the llama.cpp adapter. | Overview, Core Types, Request Envelopes |
| 2026-05-13 | @codex-tool-calling: Added the llama.cpp GGUF adapter path for tool-bearing chat through OpenAI-compatible chat templates. GGUF descriptor advertising remains gated pending local artifact smoke validation. | Overview |

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
- request/response envelopes for chat, tool-aware chat, completion, and embeddings
- the `ModelBundle`, `BundleHandle`, `ChatModel`, `CompletionModel`, and `EmbeddingModel` traits
- lightweight eval types in `model::eval`

The goal is to give downstream crates a stable contract surface while keeping the implementation burden low in this first pass.

Important scope note:

- this API covers the embedding vertical slice, the first text-only chat slice (Qwen3-4B), and the first multimodal chat slice (Gemma 4 E2B-it)
- `QuantizationBits` has been added to `StartOptions` for ISQ quantization of local chat models
- the core tool-calling chat vocabulary is present; safetensors Qwen3/Gemma 4 advertise `CapabilityKind::ToolUse` after `mistral.rs` adapter tests, while GGUF variants have adapter support but remain descriptor-gated until local chat-template smoke validation lands

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
- `QuantizationBits`
- `LoadedBundleDescriptor`
- `ModelError`
- `ModelMetricSnapshot`
- `RuntimeMetrics`
- `TextGenerationMetrics`
- `EmbeddingMetrics`
- `Milliseconds`
- `Bytes`
- `Tokens`
- `TokensPerSecond`
- `EmbeddingDistance`
- `EmbeddingNormalization`
- `EmbeddingSpec`

Primary capability request/response types:

- `ChatRole`
- `ContentPart`
- `ChatMessage`
- `ToolName`
- `ToolCallId`
- `ToolInputSchema`
- `ToolSpec`
- `ToolArguments`
- `ToolChoice`
- `ToolCall`
- `GenerationParams`
- `ChatRequest`
- `ChatResponse`
- `ChatFinishReason`
- `GenerationUsage`
- `CompletionRequest`
- `CompletionResponse`
- `EmbeddingRequest`
- `EmbeddingResponse`

Known near-term additive follow-ups for chat-capable bundles:

- live smoke validation for GGUF chat-template preservation before adding `CapabilityKind::ToolUse` to GGUF descriptors
- curated descriptor gating for `CapabilityKind::ToolUse`
- optional examples that demonstrate caller-owned tool execution loops

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

### Quantization Support

`BundleMetadata.quantization` declares which precisions a bundle supports and the curated default. `LoadedBundleDescriptor.resolved_quantization` records which precision was actually applied at startup.

- `QuantizationSupport::none()` — no quantization supported (e.g., EmbeddingGemma 300M)
- `QuantizationSupport::with_recommended([Q4, Q8], Q4)` — Q4 and Q8 supported, Q4 auto-applied when caller omits
- `QuantizationSupport::without_recommended([Q8])` — Q8 available but F32 by default (e.g., Qwen3-Embedding-0.6B)

The `resolve()` method validates caller requests: unsupported precision → `InvalidConfiguration`, omitted precision → curated default. This invariant is enforced at construction: `recommended` must be in `supported` or `None`.

After startup, `handle.descriptor().resolved_quantization` reports what was actually loaded — not the curated recommendation, but the runtime-resolved value.

### Handle-Level Metrics

Loaded bundles may expose additive runtime metrics through `BundleHandle::metric_snapshot()`:

```rust
let snapshot = handle.metric_snapshot();
```

This is intentionally service-level, not response-level. It is the right ownership boundary for:

- current and peak RSS
- request counts
- last / max / average request latency
- aggregate token totals and aggregate token/sec for text-generation backends

The current unit wrappers live in `libs/model/src/units.rs`:

- `Milliseconds(pub u64)`
- `Bytes(pub u64)`
- `Tokens(pub u64)`
- `TokensPerSecond(pub u64)`

The current metric types live in `libs/model/src/metrics.rs`:

- `ModelMetricSnapshot`
- `RuntimeMetrics`
- `TextGenerationMetrics`
- `EmbeddingMetrics`

`ModelMetricSnapshot` uses `Option<T>` both at the section level and field level. This is intentional:

- some sections are capability-specific and absent on bundles that do not expose that capability
- some values are not meaningful until the handle has served at least one request
- some values are backend/platform dependent, such as runtime memory sampling

Current implementation note:

- runtime/process memory metrics are currently collected through `sysinfo`
- the current `mistral` implementation is intended to work on macOS and Linux without `cfg(target_os)` branches in Motlie code
- current RSS is sampled from the current process, and peak RSS is maintained by Motlie as the max observed sample over the handle lifetime
- the current `mistral` text-generation throughput fields are derived from cumulative token totals and cumulative prompt/generation time reported by upstream usage data
- this metrics path is currently always built into the `mistral` backend and example binaries; it is not separately feature-gated yet
- `ContentPart::ImageUrl` exists at the contract layer for future remote-capable bundles, but the current `mistral` multimodal path accepts inline image bytes only

## Bundle API Sketch

### Bundle Metadata and Capability Introspection

```rust
use motlie_model::{
    BundleId, BundleMetadata, Capabilities, CapabilityDescriptor,
    QuantizationBits, QuantizationSupport,
};

let metadata = BundleMetadata {
    id: BundleId::new("qwen3_5_instruct"),
    display_name: "Qwen 3.5 Instruct".into(),
    capabilities: Capabilities::new(vec![
        CapabilityDescriptor::chat(),
        CapabilityDescriptor::completion(),
    ]),
    quantization: QuantizationSupport::with_recommended(
        [QuantizationBits::Four, QuantizationBits::Eight],
        QuantizationBits::Four,
    ),
};
```

`QuantizationBits` is intentionally coarse but distinct enough for current
backends: `Four`, `Five`, `Eight`, and `FloatEight`. Backend adapters map these
to native labels such as ISQ Q4/Q8 or GGUF Q4_K_M/Q5_K_M/Q8_0/FP8. A bundle
must only advertise values that it can actually resolve to curated artifacts or
runtime behavior.

`Capabilities::new(...)` canonicalizes descriptors by `CapabilityKind`: the first descriptor for a kind wins, later duplicates are dropped, and `supports(...)` always reflects exactly the descriptor set stored in the struct.

Common helpers include `Capabilities::chat_and_completion()`,
`Capabilities::chat_completion_and_tool_use()`,
`Capabilities::multimodal_chat_and_vision()`, and
`Capabilities::multimodal_chat_vision_and_tool_use()`.

### Request Envelopes

```rust
use motlie_model::{
    ChatMessage, ChatRequest, ChatRole, CompletionRequest, EmbeddingRequest, GenerationParams,
};

let chat = ChatRequest {
    messages: vec![
        ChatMessage::text(ChatRole::System, "Be concise."),
        ChatMessage::with_parts(
            ChatRole::User,
            vec![
                ContentPart::image(image_bytes, "image/png"),
                ContentPart::text("Summarize the visible diagram."),
            ],
        ),
    ],
    params: GenerationParams {
        max_tokens: Some(256),
        ..Default::default()
    },
    ..Default::default()
};

let completion = CompletionRequest {
    prompt: "Explain deterministic extraction in one paragraph.".into(),
    params: GenerationParams::default(),
};

let embeddings = EmbeddingRequest {
    inputs: vec!["motlie model bundle".into(), "deterministic package bundle".into()],
};
```

Tool-aware chat uses typed Rust argument structs and generated JSON Schema.
Tool inputs are object-shaped; bind named argument structs rather than scalar
or tuple payloads. Tool names are validated once through `ToolName`, and model
tool-call correlation ids are carried as `ToolCallId` rather than plain strings.

```rust
use motlie_model::{ChatRequest, ToolChoice};
use motlie_models::{ToolError, ToolRegistry};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, JsonSchema)]
struct AddArgs {
    left: i64,
    right: i64,
}

#[derive(Serialize)]
struct AddOutput {
    value: i64,
}

async fn add(args: AddArgs) -> Result<AddOutput, ToolError> {
    Ok(AddOutput {
        value: args.left + args.right,
    })
}

let mut registry = ToolRegistry::new();
registry.insert_fn("add", "Add two signed integers.", add)?;

let chat = ChatRequest {
    tools: registry.specs(),
    tool_choice: Some(ToolChoice::Auto),
    ..Default::default()
};
```

An inline closure uses the same typed binding path:

```rust
registry.insert_fn(
    "multiply",
    "Multiply two signed integers.",
    |args: AddArgs| async move {
        Ok::<_, ToolError>(AddOutput {
            value: args.left * args.right,
        })
    },
)?;
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
    type Handle = MyHandle;

    fn id(&self) -> &BundleId { todo!() }
    fn metadata(&self) -> &BundleMetadata { todo!() }
    fn capabilities(&self) -> &Capabilities { todo!() }

    async fn start(
        &self,
        options: StartOptions,
    ) -> Result<Self::Handle, ModelError> {
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

### Quantized Startup

```rust
use motlie_model::{ArtifactPolicy, QuantizationBits, StartOptions};

let quantized_q4 = StartOptions {
    artifact_policy: Some(ArtifactPolicy::LocalOnly {
        root: PathBuf::from("artifacts/models/hf-cache"),
    }),
    quantization: Some(QuantizationBits::Four),
    ..Default::default()
};

let full_precision = StartOptions {
    artifact_policy: Some(ArtifactPolicy::LocalOnly {
        root: PathBuf::from("artifacts/models/hf-cache"),
    }),
    quantization: None,
    ..Default::default()
};
```

`QuantizationBits` is backend-agnostic. The mistral.rs backend maps `Four` → `IsqBits::Four` (ISQ Q4, ~2.5GB for Qwen3-4B) and `Eight` → `IsqBits::Eight`. Embedding bundles ignore the field since small models run comfortably in F32.

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
- `CapabilityKind` does not imply that every variant immediately gets its own executable trait. In the current Gemma 4 slice, `Vision` is implemented as a capability flag on the chat surface rather than as a separate `VisionModel` trait.
- `model::eval` contains small declarative types only; runners, scoring, suite loading, and reports belong in `libs/model-eval`.
- Curated artifact download and provenance control live above this crate in `libs/models`. Backend startup consumes the resulting artifact root and policy through `StartOptions` rather than using `libs/model` to initiate curated downloads itself.
