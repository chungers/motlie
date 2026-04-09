# Curated Model Bundle API

## Status: Draft

## Change Log

| Date | Change | Sections |
|------|--------|----------|
| 2026-04-07 | @codex-researcher: Initial API sketch for `libs/models` catalog and descriptor shapes. Reflects the current scaffold, not the final loaded-bundle runtime API. | All |
| 2026-04-07 | @codex-researcher: Added explicit curated artifact-control examples and updated the first embedding slice to use the real `mistralrs` builder path with separate pre-download support. | Overview, API Sketch, Notes |
| 2026-04-07 | @codex-researcher: Added the `examples/v0.1` runnable example for the current curated embedding bundle. | Example Program |
| 2026-04-07 | @codex-researcher: Added optional Hugging Face token support to the out-of-band downloader path only. | Explicit Artifact Control, Notes |
| 2026-04-08 | @codex-researcher: Clarified typed catalog/artifact errors and made the first vertical-slice artifact contract explicit so the API can be implemented without hidden bundle-specific assumptions. | Overview, Core Types, Explicit Artifact Control, Notes |
| 2026-04-08 | @codex-researcher: Updated the `v0.1` example contract so artifact download is opt-in. The default example path now exercises existing curated artifacts with `ArtifactPolicy::LocalOnly`, which is the intended regulated/offline behavior. | Example Program, Notes |
| 2026-04-08 | @codex-researcher: Added the direct curated embedding enum and parser-oriented `ModelSelector` path to the API sketch, and removed `SupportTier` / `PackagingMode` from the recommended public surface. | Core Types, API Sketch, Example Program, Notes |
| 2026-04-08 | @codex-researcher: Added an explicit end-to-end vertical-slice walkthrough and a curator implementation checklist so both callers and bundle implementers can follow the same documented path. | Overview, API Sketch, Example Program, Notes |
| 2026-04-08 | @codex-researcher: Documented the per-bundle feature-gating convention and the `ModelUnavailable` behavior for known selectors that are disabled in the current build. | Overview, Core Types, Notes |
| 2026-04-08 | @codex-researcher: Clarified the local-only startup boundary after PR 139 review. Curated bundle modules now resolve and validate provider-specific cache layout before startup, while generic backends consume only a resolved local model path. Also clarified that selector strings are composed from capability family plus model selector rather than hardcoded one-off branches. | Overview, API Sketch, Notes |
| 2026-04-08 | @claude: Added the Qwen3-4B chat bundle (#141). New `ChatModels` enum, `ModelSelector::Chat` variant, `chat:qwen/qwen3_4b` parsing, ISQ quantization via `StartOptions.quantization`, and `v0.2` example. | Overview, Core Types, API Sketch, Example Program |
| 2026-04-08 | @codex-researcher: Added the Gemma 4 E2B-it multimodal chat slice (#142). `ChatModels` now includes `Gemma4E2B`, `ModelSelector` supports `chat:google/gemma4_e2b`, and `examples/v0.3` documents the direct text-only and image+text caller paths. | Overview, Core Types, API Sketch, Example Program, Notes |
| 2026-04-09 | @codex-researcher: Tightened the example-build convention. Versioned examples now expect a single-bundle build and print `catalog-entry-count: 1`, and the Gemma example follows the same one-model-per-example rule. | Example Program, Notes |
| 2026-04-09 | @codex-researcher: Collapsed the duplicate Gemma 4 examples into a single `v0.3` flow. `v0.3` now carries the optional `--download-artifacts` behavior directly, so each versioned example once again maps to exactly one curated model. | Example Program, Notes |
| 2026-04-09 | @codex-researcher: Added handle-level metric snapshot usage to the examples. The current `mistral` backends now surface runtime latency/memory aggregates on the loaded bundle handle, with text-generation token metrics where the backend provides them. | Example Program, Notes |

This document sketches the concrete API shapes currently introduced in `libs/models`. The crate now owns both the descriptor catalog and the curated bundle constructors that bind those descriptors to a backend implementation.

For implementers, this document is the current API specification for the crate surface that exists today. It should be sufficient to build compatible bundle/catalog logic without reverse-engineering unstated assumptions from the example or backend implementation.

## Overview

The first concrete `libs/models` API is an in-memory `Catalog` of curated bundle descriptors. This lets Motlie start expressing:

- stable bundle IDs
- family metadata
- backend selection
- introspective capability metadata
- build/platform constraints
- evaluation-track membership
- curated bundle instantiation
- explicit artifact download control separate from backend cache population

The goal is to make the product-facing bundle layer tangible before the runtime-facing bundle handle APIs are finalized.

For the current vertical slice, this crate must document two concrete experiences clearly:

1. caller experience:
   choose a curated embedding bundle, inspect its descriptor and embedding metadata, start it locally, and call `embed(...)`
2. curator experience:
   define a bundle module, expose `descriptor()` and `bundle()`, implement the bundle-level `Embedding` contract, and register the bundle in `Catalog`

Curated bundle availability is build-dependent. The selector enums and default catalog only expose bundles compiled into the current build through per-bundle Cargo features.

## Core Types

Current public shapes:

- `BundleId`
- `BundleFamily`
- `BackendKind`
- `PlatformConstraint`
- `BuildConstraint`
- `BundleRequirements`
- `BundleDescriptor`
- `Catalog`
- `ChatModels`
- `EmbeddingModels`
- `ModelSelector`
- `ModelsError`

`BundleId`, `Capabilities`, and capability introspection types come from `motlie_model`. Evaluation-track membership is expressed with `motlie_model::eval::EvalTrack`.

`ModelsError` is the typed library error for catalog lookup and artifact staging. Binaries and examples may convert it to `anyhow::Error`, but the crate does not expose `anyhow::Result` as its library API.

For build-gated curated bundles, `ModelsError::ModelUnavailable` is the intended error when a selector is known to the codebase but disabled in the current build.

## API Sketch

### Defining a Bundle Descriptor

```rust
use motlie_model::eval::EvalTrack;
use motlie_models::{
    BackendKind, BuildConstraint, BundleDescriptor, BundleFamily, BundleId,
    BundleRequirements, PlatformConstraint,
};
use motlie_model::{Capabilities, CapabilityDescriptor};

let qwen = BundleDescriptor {
    id: BundleId::new("qwen3_5_instruct"),
    display_name: "Qwen 3.5 Instruct".into(),
    family: BundleFamily::Qwen,
    capabilities: Capabilities::new(vec![
        CapabilityDescriptor::chat(),
        CapabilityDescriptor::completion(),
    ]),
    backend: BackendKind::MistralRs,
    requirements: BundleRequirements {
        platform: vec![PlatformConstraint::Linux],
        build: vec![BuildConstraint::Feature("mistral".into())],
    },
    eval_tracks: vec![
        EvalTrack::Chat,
        EvalTrack::Reasoning,
        EvalTrack::Summarization,
        EvalTrack::Classification,
    ],
    artifacts: None,
};
```

### Registering Bundles in a Catalog

```rust
use motlie_model::eval::EvalTrack;
use motlie_models::{BundleId, Catalog};

let mut catalog = Catalog::new();
catalog.register(qwen.clone(), || {
    build_qwen_bundle()
});

let bundle = catalog
    .bundle(&BundleId::new("qwen3_5_instruct"))
    .expect("bundle should exist");

assert!(bundle.capabilities.supports(motlie_model::CapabilityKind::Chat));
assert!(bundle.supports_track(EvalTrack::Reasoning));
assert_eq!(
    bundle.capability_descriptors()[0].summary,
    "Multi-turn text interaction with text output."
);
```

### Direct Curated Embedding Enum

```rust
use motlie_models::embeddings::EmbeddingModels;

let model = EmbeddingModels::GoogleGemma300m;

assert_eq!(model.as_str(), "google/embeddinggemma_300m");

let descriptor = model.descriptor();
let bundle = model.bundle();
let spec = model.embedding_spec();
```

### Direct Curated Chat Enum

```rust
use motlie_models::chat::ChatModels;

let model = ChatModels::Qwen3_4B;

assert_eq!(model.as_str(), "qwen/qwen3_4b");

let descriptor = model.descriptor();
let bundle = model.bundle();
```

### Parser-Oriented Model Selector

The parser supports both `chat:` and `embedding:` prefixes:

```rust
use std::str::FromStr;

use motlie_models::ModelSelector;

let selector = ModelSelector::from_str("embedding:google/embeddinggemma_300m")?;
let descriptor = selector.descriptor();
let bundle = selector.bundle();
```

### End-to-End Vertical Slice

The current curated embedding slice is intended to be readable from the caller’s point of view as one continuous flow:

```rust
use motlie_model::{ArtifactPolicy, EmbeddingRequest, StartOptions};
use motlie_models::{default_artifact_root, embeddings::EmbeddingModels, ModelSelector};

// Direct curated enum path.
let direct = EmbeddingModels::GoogleGemma300m;
let direct_spec = direct.embedding_spec();
let direct_bundle = direct.bundle();

// Parser-oriented path.
let selected: ModelSelector = "embedding:google/embeddinggemma_300m".parse()?;
let selected_bundle = selected.bundle();

// Both paths resolve to the same curated bundle behavior.
// In LocalOnly mode, the curated bundle resolves the provider-specific cache
// layout to a concrete local snapshot path before delegating to the backend.
let handle = direct_bundle
    .start(StartOptions {
        artifact_policy: Some(ArtifactPolicy::LocalOnly {
            root: default_artifact_root(),
        }),
        ..Default::default()
    })
    .await?;

let response = handle
    .embeddings()?
    .embed(EmbeddingRequest {
        inputs: vec!["motlie curated model bundle".into()],
    })
    .await?;

assert_eq!(direct_spec.dimensions, Some(768));
assert_eq!(response.vectors[0].len(), 768);
```

The runnable realization of this flow is:

- [main.rs](/Users/dchung/projects/claude-mistral/motlie/libs/models/examples/v0.1/main.rs)
- [README.md](/Users/dchung/projects/claude-mistral/motlie/libs/models/examples/v0.1/README.md)

### Selecting Bundles for an Evaluation Track

```rust
use motlie_model::eval::EvalTrack;

let reasoning_bundles: Vec<_> = catalog
    .bundles_for_track(EvalTrack::Reasoning)
    .map(|bundle| bundle.id.as_str().to_owned())
    .collect();
```

### Instantiating the First Vertical Slice

```rust
 use motlie_model::{ArtifactPolicy, BundleId, EmbeddingRequest, StartOptions};
 use motlie_models::{default_artifact_root, Catalog};

let catalog = Catalog::with_defaults();
let bundle = catalog
    .instantiate(&BundleId::new("embeddinggemma_300m"))
    .expect("bundle should exist");

let handle = bundle
    .start(StartOptions {
        artifact_policy: Some(ArtifactPolicy::LocalOnly {
            root: default_artifact_root(),
        }),
        ..Default::default()
    })
    .await?;
let embeddings = handle.embeddings()?;
let response = embeddings
    .embed(EmbeddingRequest {
        inputs: vec![
            "curated bundle catalog".into(),
            "mistral embedding vertical slice".into(),
        ],
    })
    .await?;
```

### Explicit Artifact Control

```rust
use motlie_model::BundleId;
use motlie_models::{
    default_artifact_root, download_bundle_artifacts_with_options, ArtifactDownloadOptions,
    Catalog,
};

let catalog = Catalog::with_defaults();
let bundle_id = BundleId::new("embeddinggemma_300m");

let artifacts = catalog
    .artifacts(&bundle_id)
    .expect("bundle should expose artifact download control");

assert_eq!(artifacts.control_name, "embeddinggemma_300m");

let summary = download_bundle_artifacts_with_options(
    &catalog,
    &bundle_id,
    &default_artifact_root(),
    &ArtifactDownloadOptions {
        hf_token: std::env::var("HF_TOKEN").ok(),
    },
)?;

assert!(!summary.downloaded.is_empty());
```

For `embeddinggemma_300m`, the curated artifact set currently includes:

- `config.json`
- `modules.json`
- `tokenizer.json`
- `tokenizer.model`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `1_Pooling/config.json`
- `2_Dense/config.json`
- `2_Dense/model.safetensors`
- `3_Dense/config.json`
- `3_Dense/model.safetensors`
- root `.safetensors` files for the base model

The same flow is available from the binary target:

```sh
cargo run -p motlie-models --bin motlie-models-download -- embeddinggemma_300m
```

Authenticated out-of-band download:

```sh
export HF_TOKEN=...
cargo run -p motlie-models --bin motlie-models-download -- --hf-token-env HF_TOKEN embeddinggemma_300m
```

## Example Program

The runnable examples for this crate are:

- `v0.1` embedding slice
  - [README.md](/Users/dchung/projects/claude-mistral/motlie/libs/models/examples/v0.1/README.md)
  - [main.rs](/Users/dchung/projects/claude-mistral/motlie/libs/models/examples/v0.1/main.rs)
- `v0.2` text-only chat slice
  - [README.md](/Users/dchung/projects/claude-mistral/motlie/libs/models/examples/v0.2/README.md)
  - [main.rs](/Users/dchung/projects/claude-mistral/motlie/libs/models/examples/v0.2/main.rs)
- `v0.3` Gemma multimodal slice
  - [README.md](/tmp/motlie-issue142/libs/models/examples/v0.3/README.md)
  - [main.rs](/tmp/motlie-issue142/libs/models/examples/v0.3/main.rs)

All versioned examples now assume a single-bundle build and print `catalog-entry-count: 1`. Run them with `--no-default-features` and only the feature for the bundle under test.

Embedding example (`v0.1`):

```sh
cargo run -p motlie-models --no-default-features --features model-google-gemma-300m --example models_v0_1 -- "motlie curated model bundle"
```

Text-only chat example (`v0.2`):

```sh
cargo run -p motlie-models --no-default-features --features model-qwen3-4b --example models_v0_2 -- "What is Rust's ownership model?"
```

Gemma multimodal example (`v0.3`) with optional curated download:

```sh
cargo run -p motlie-models --no-default-features --features model-gemma4-e2b --bin motlie-models-download -- --hf-token-env HF_TOKEN gemma4_e2b
cargo run -p motlie-models --no-default-features --features model-gemma4-e2b --example models_v0_3 -- "Describe ownership in one paragraph"
```

Or:

```sh
cargo run -p motlie-models --no-default-features --features model-gemma4-e2b --example models_v0_3 -- --download-artifacts "Describe ownership in one paragraph"
```

## Curator Implementation

For a new curated embedding bundle, the intended implementation checklist is:

1. create one bundle file under the capability-family namespace
   - for example: `libs/models/src/embeddings/my_model.rs`
2. expose:
   - `descriptor() -> BundleDescriptor`
   - `bundle() -> Box<dyn ModelBundle>`
   - `embedding_spec() -> &'static motlie_model::EmbeddingSpec`
3. define a concrete bundle type that implements:
   - `motlie_model::ModelBundle`
   - `motlie_model::Embedding`
4. add a curated enum variant to `EmbeddingModels`
5. wire `EmbeddingModels::{as_str, descriptor, bundle, embedding_spec}`
6. optionally support parser-driven selection through `ModelSelector`
7. register the bundle in `Catalog::with_defaults()` if it should be in the default curated set
8. define explicit artifact rules if the bundle uses local artifacts
9. add a focused bundle test that proves:
   - descriptor is reviewable as data
   - embedding spec matches expected semantics
   - local-only startup works when curated artifacts are present

## Notes

- `BundleId` is currently a string-backed newtype rather than an enum. This keeps the catalog extensible while still giving the crate a stable selection key.
- capability metadata comes from `motlie_model`, so the catalog can describe input/output shape and interaction style without inventing its own parallel schema
- `BackendKind` is metadata for catalog reasoning and observability. It does not make runtime choice part of the application control path.
- `Catalog` now also owns curated bundle instantiation through registered constructors.
- The preferred direct curated path is the bundle-family enum, such as `EmbeddingModels::GoogleGemma300m`; `ModelSelector` is the parser-friendly wrapper above that.
- Known selectors for bundles disabled by Cargo features should return `ModelsError::ModelUnavailable`, not a generic unknown-selector error.
- The embedding caller path should be understandable by reading the `v0.1` example; the text-only chat caller path by reading `v0.2`; and the multimodal chat caller path by reading `v0.3`. The curator path should be understandable by reading the `google_gemma_300m`, `qwen3_4b`, or `gemma4_e2b` bundle modules and the checklist above.
- Curated artifact download is explicit and independent of the backend library's own cache-miss behavior. Backends consume the curated artifact policy through `StartOptions`. For regulated local bundles, `ArtifactPolicy::LocalOnly` is the intended fail-closed mode.
- `embeddinggemma_300m` local-only startup depends on the full sentence-transformers module stack being present in the curated artifact root. That requirement is part of the bundle contract, not an ambient `mistralrs` cache behavior.
- Authentication for protected upstream artifacts belongs only to the out-of-band download/build path. The runtime/bundle startup path does not accept tokens and remains artifact-consumption only.
- The `v0.1` example now defaults to existing local artifacts and only touches Hugging Face when `--download-artifacts` is passed explicitly.

## Next Step

The text-only and first multimodal chat paths are now functional (Qwen3-4B with ISQ and Gemma 4 E2B-it through the multimodal builder path). The next contract changes should focus on richer response metadata and tool-calling:

- richer `ChatResponse` metadata (finish reason, usage, tool calls)
- tool-calling message roles and correlation fields (`ChatRole::Tool`)
- additive `StartOptions` controls for device selection and context-length policy
