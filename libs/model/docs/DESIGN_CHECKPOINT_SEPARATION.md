# Design: ModelIdentity / ModelCheckpoint / BackendAdapter Separation

> @claude-llm-researcher 2026-04-11 — proposed
>
> Status: **Implemented core architecture**. The `motlie-model` foundation
> types, backend adapter shims, and `motlie-models` catalog resolution now
> follow this separation. Legacy alias bundle descriptors still exist as
> compatibility shims while startup flows through the shared adapter path.

## Changelog

| Date | Who | Summary |
|---|---|---|
| 2026-04-11 | @claude-llm-researcher | Initial design proposal |
| 2026-04-12 | @codex-models | Implemented the core separation in `motlie-model`, backend crates, and `motlie-models` catalog resolution |

---

## 1. Problem Statement

The current `libs/model` + `libs/models` architecture couples three independent
concerns into a single `ModelBundle` implementation:

1. **Model identity** — what the model is (Qwen3-4B, Gemma4 E2B, EmbeddingGemma)
2. **Weight checkpoint** — where and how the weights are stored (safetensors on HF, GGUF on HF)
3. **Execution backend** — what runtime loads and runs the model (mistral.rs, llama.cpp)

Each bundle file (e.g., `qwen3_4b.rs`, `qwen3_4b_gguf.rs`) hard-wires all three.

### Current state (6 bundle files)

```
libs/models/src/chat/
  qwen3_4b.rs         → MistralTextBundle    → safetensors (Qwen/Qwen3-4B)
  qwen3_4b_gguf.rs    → LlamaCppTextBundle   → GGUF (Qwen/Qwen3-4B-GGUF)
  gemma4_e2b.rs        → MistralMultimodal    → safetensors (google/gemma-4-E2B-it)
  gemma4_e2b_gguf.rs   → LlamaCppTextBundle   → GGUF (unsloth/gemma-4-E2B-it-GGUF)

libs/models/src/embeddings/
  google_gemma_300m.rs      → MistralEmbeddingBundle → safetensors
  qwen3_embedding_06b.rs    → MistralEmbeddingBundle → safetensors
```

### Why this is a problem

**GGUF is a convergent format.** Both mistral.rs (`GgufModelBuilder`) and
llama.cpp (`LlamaModel::load_from_file`) can load the same GGUF weights.
Vendors increasingly publish GGUF directly (Qwen, Meta, Mistral). Community
quantizers (unsloth, bartowski) produce GGUF for everything else within hours
of release.

Under the current design, to run Qwen3-4B GGUF on *both* backends, you need
two separate bundle files even though they share the same artifact. Adding a
third backend (ORT with ONNX) or a fifth model adds a full cross-product of
files. The count grows as `models × formats × backends`.

**Concrete missed opportunity today:** mistral.rs with flash-attn can outperform
llama.cpp for Qwen3 on DGX Spark (CUDA + unified memory). With shared GGUF
checkpoints, we could A/B test backends on the same weights at zero marginal
storage cost. Currently this requires maintaining parallel bundle definitions.

### Non-problem: ONNX/ORT

ONNX Runtime does **not** support GGUF. ORT uses `.onnx` (protobuf graph +
initializer tensors) — a fundamentally different format that encodes the
computation graph. GGUF stores only tensors and metadata; the runtime provides
the graph. There is no conversion bridge. `BackendKind::Ort` will require its
own `ModelCheckpoint` format variant, which this design accommodates cleanly.

---

## 2. Proposed Architecture

Separate into three layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                      ModelIdentity                              │
│  id: "qwen3_4b"                                                 │
│  display_name: "Qwen3 4B"                                       │
│  family: Qwen                                                   │
│  capabilities: Chat + Completion                                │
│  eval_tracks: [Chat, Reasoning, Summarization, Classification]  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ one-to-many
          ┌────────────────┼────────────────┐
          ▼                                  ▼
┌───────────────────┐             ┌───────────────────────┐
│  ModelCheckpoint   │             │  ModelCheckpoint       │
│  format: GGUF      │             │  format: Safetensors   │
│  repo: Qwen/...    │             │  repo: Qwen/Qwen3-4B  │
│    -GGUF           │             │  include: [config.json │
│  include:          │             │    tokenizer.json,     │
│    [*Q4_K_M.gguf]  │             │    *.safetensors]      │
│  quant: Q4_K_M     │             │  quant: none (ISQ at   │
│    (pre-quantized)  │             │    load time)          │
└────────┬───────────┘             └────────┬──────────────┘
         │ many-to-many                      │
    ┌────┼──────────┐                   ┌────┘
    ▼               ▼                   ▼
┌─────────┐  ┌───────────┐      ┌───────────┐
│llama.cpp│  │mistral.rs │      │mistral.rs │
│  GGUF   │  │  GGUF     │      │  HF/ST    │
│ Adapter │  │ Adapter   │      │ Adapter   │
└─────────┘  └───────────┘      └───────────┘
```

### 2.1 ModelIdentity

Extracted from what is currently `BundleDescriptor` minus artifacts and backend:

```rust
pub struct ModelIdentity {
    pub id: BundleId,
    pub display_name: String,
    pub family: BundleFamily,
    pub capabilities: Capabilities,
    pub eval_tracks: Vec<EvalTrack>,
    pub requirements: BundleRequirements,
}
```

One definition per logical model. Immutable. Registered once in the catalog.

### 2.2 ModelCheckpoint

Replaces the current `BundleArtifacts` + format-specific resolution logic:

```rust
pub enum CheckpointFormat {
    /// HuggingFace safetensors layout (config.json + tokenizer + *.safetensors)
    Safetensors,
    /// GGUF single-file quantized weights (*.gguf)
    Gguf,
    /// ONNX graph + initializers (*.onnx + optional external data)
    Onnx,
}

pub struct ModelCheckpoint {
    pub format: CheckpointFormat,
    pub source: ArtifactSource,
    pub include: Vec<ArtifactRule>,
    /// Pre-baked quantization in the checkpoint (e.g., Q4_K_M for GGUF).
    /// `None` means full-precision; backend may apply runtime quantization.
    pub quantization: Option<CheckpointQuantization>,
}

pub enum CheckpointQuantization {
    /// GGUF pre-quantized level
    Gguf { label: String },  // "Q4_K_M", "Q8_0", "f16"
    /// ONNX quantized
    Onnx { bits: u8 },
}
```

Multiple checkpoints per model identity. The catalog discovers available
checkpoints. The download binary fetches whichever checkpoint the user selects.

### 2.3 BackendAdapter

Replaces the current tight coupling between `MistralTextBundle` / `LlamaCppTextBundle`
and their format-specific loading:

```rust
pub trait BackendAdapter: Send + Sync {
    /// Which checkpoint formats this adapter can load.
    fn supported_formats(&self) -> &[CheckpointFormat];

    /// Backend identifier for display and selection.
    fn backend_kind(&self) -> BackendKind;

    /// Load a checkpoint and produce a live BundleHandle.
    ///
    /// The adapter receives the model identity (for capability validation),
    /// the resolved checkpoint path, and backend-specific start options.
    fn start(
        &self,
        identity: &ModelIdentity,
        checkpoint: &ResolvedCheckpoint,
        options: StartOptions,
    ) -> Result<Box<dyn BundleHandle>, ModelError>;
}
```

Each backend crate registers one or more adapters:

```rust
// libs/model/backends/llama_cpp/
pub struct LlamaCppGgufAdapter {
    pub arch: LlamaCppTextArch,  // Qwen3, Gemma4
    pub default_context_length: u32,
}

impl BackendAdapter for LlamaCppGgufAdapter {
    fn supported_formats(&self) -> &[CheckpointFormat] { &[CheckpointFormat::Gguf] }
    fn backend_kind(&self) -> BackendKind { BackendKind::LlamaCpp }
    // ...
}

// libs/model/backends/mistral/
pub struct MistralGgufAdapter { pub arch: MistralTextArch, ... }
pub struct MistralSafetensorsAdapter { pub arch: MistralTextArch, ... }
pub struct MistralMultimodalAdapter { pub arch: MistralMultimodalArch, ... }
pub struct MistralEmbeddingAdapter { pub arch: MistralEmbeddingArch, ... }
```

### 2.4 Catalog Composition

The catalog registers triples of (identity, checkpoint, adapter):

```rust
impl Catalog {
    pub fn register(
        &mut self,
        identity: ModelIdentity,
        checkpoints: Vec<ModelCheckpoint>,
        adapters: Vec<Arc<dyn BackendAdapter>>,
    );

    /// Select a runnable combination for a model.
    /// Matches checkpoint format to adapter support.
    pub fn resolve(
        &self,
        model_id: &BundleId,
        backend_preference: Option<BackendKind>,
        format_preference: Option<CheckpointFormat>,
    ) -> Option<ResolvedBundle>;
}
```

Example registration for Qwen3-4B:

```rust
catalog.register(
    ModelIdentity::qwen3_4b(),
    vec![
        ModelCheckpoint::gguf("Qwen/Qwen3-4B-GGUF", &["*Q4_K_M.gguf"]),
        ModelCheckpoint::safetensors("Qwen/Qwen3-4B", &["config.json", "*.safetensors", ...]),
    ],
    vec![
        Arc::new(LlamaCppGgufAdapter::qwen3()),
        Arc::new(MistralGgufAdapter::qwen3()),
        Arc::new(MistralSafetensorsAdapter::qwen3()),
    ],
);
```

One model. Two checkpoints. Three backend adapters. The catalog resolves the
best (checkpoint, adapter) pair at startup based on:
- What's downloaded locally
- User backend preference (`--backend=llama-cpp`)
- Adapter format compatibility

### 2.5 User-facing selector surface

Current:
```sh
--chat=qwen/qwen3_4b           # mistral + safetensors (implicit)
--chat=qwen/qwen3_4b_gguf      # llama_cpp + GGUF (implicit)
```

Proposed:
```sh
--chat=qwen/qwen3_4b                              # auto-select best available
--chat=qwen/qwen3_4b --backend=llama-cpp           # GGUF + llama.cpp
--chat=qwen/qwen3_4b --backend=mistral             # auto-select checkpoint
--chat=qwen/qwen3_4b --backend=mistral --format=gguf  # explicit
```

The `_gguf` suffix selectors can be preserved as aliases for backwards
compatibility during migration.

---

## 3. Scope

### In scope

| Area | Files affected | Change |
|---|---|---|
| `libs/model/src/lib.rs` | Core contracts | Add `ModelIdentity`, `ModelCheckpoint`, `BackendAdapter` trait |
| `libs/model/backends/mistral/` | All src files | Refactor into adapter impls; add `MistralGgufAdapter` |
| `libs/model/backends/llama_cpp/` | All src files | Refactor into adapter impl |
| `libs/models/src/chat/` | 4 files → 2 | Collapse format-specific bundles into identity + checkpoint declarations |
| `libs/models/src/embeddings/` | 2 files | Same pattern |
| `libs/models/src/lib.rs` | Catalog, ModelSelector | New resolution logic, simplified feature flags |
| `libs/models/examples/v0.1–v0.4` | Example entrypoints | Updated selector surface |
| `libs/models/Cargo.toml` | Features | Simplify: per-model + per-backend instead of per-combo |

### Out of scope

- **ORT/ONNX backend implementation** — future work; this refactor makes it a clean addition
- **Embedding GGUF support** — no known GGUF embedding models today; the architecture supports it when they arrive
- **Multimodal GGUF** — llama.cpp `mtmd` feature exists but not wired; this refactor doesn't block it
- **Runtime backend selection API** — the adapter trait enables it; the UX for `--backend=` is deferred to implementation

### Migration strategy

1. Introduce new types (`ModelIdentity`, `ModelCheckpoint`, `BackendAdapter`) alongside existing `ModelBundle`
2. Implement adapters in each backend crate, wrapping existing logic
3. Rewrite bundle files as identity + checkpoint registrations
4. Update catalog resolution
5. Preserve `_gguf` selectors as aliases
6. Remove old `ModelBundle` impls once all consumers migrate

No breaking API change at the `BundleHandle` level — the handle interface
(`ChatModel`, `CompletionModel`, `EmbeddingModel`, metrics) is unchanged.

---

## 4. Alternatives Considered

### A. Keep current coupling, add `mistral + GGUF` as yet another bundle file

- Pro: zero refactor cost
- Con: 8 bundle files for 4 models × 2 formats. 12 files when ORT arrives. Does not scale.
- Con: cannot A/B backends on same weights without separate artifact downloads

### B. Format-polymorphic bundles (bundle picks format at startup)

- Pro: fewer files (one per model × backend)
- Con: still couples identity to backend. Adding a third backend still adds N files.
- Con: artifact resolution becomes runtime conditional rather than declarative

### C. Proposed three-layer separation (recommended)

- Pro: O(models + formats + backends) instead of O(models × formats × backends)
- Pro: shared GGUF artifacts across backends
- Pro: clean extension point for ORT, future backends
- Con: upfront refactor cost across ~15 files
- Con: catalog resolution logic is more complex (match format → adapter)

---

## 5. Open Questions

1. **Capability variance by backend** — Gemma4 on mistral.rs supports vision
   (multimodal content parts). On llama.cpp text-only, it's chat-only. Should
   the adapter narrow the identity's capability set, or should the identity
   declare the superset and adapters reject unsupported content types at runtime?

   *Proposed:* Adapters declare their *effective* capabilities as a subset of
   the identity's capabilities. The catalog exposes the intersection.

2. **Quantization semantics** — mistral.rs ISQ quantizes at load time from
   full-precision safetensors. GGUF is pre-quantized. `QuantizationBits::Four`
   means "apply ISQ Q4" for mistral/safetensors but "select the Q4_K_M .gguf
   file" for GGUF. Should `StartOptions::quantization` be backend-agnostic or
   checkpoint-aware?

   *Proposed:* `StartOptions::quantization` remains backend-agnostic (Q4/Q8/none).
   The adapter maps it to the appropriate mechanism (ISQ precision vs GGUF file
   selection) internally. The checkpoint declares what's available.

3. **Feature flag granularity** — Currently `model-qwen3-4b-gguf` activates the
   llama_cpp dep. Under the new design, should it be `model-qwen3-4b` +
   `backend-llama-cpp` (orthogonal), or keep compound flags?

   *Proposed:* Orthogonal flags: `model-qwen3-4b`, `backend-llama-cpp`,
   `backend-mistral`. The catalog only registers combinations where both the
   model and backend features are enabled.
