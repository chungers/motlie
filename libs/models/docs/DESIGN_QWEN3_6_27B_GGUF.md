# Qwen3.6 27B GGUF Bundle Design

## Status: Implementation in progress

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-24 | @codex-gpt55 | Started implementation: added the text-only llama.cpp/GGUF bundle shape, Q4/Q5/Q8 curated quant policy, and the dedicated `chat_multimodal_qwen3_6_27b` example. FP8 remains blocked because no curated FP8 GGUF artifact is available. |
| 2026-04-24 | @codex-gpt55 | Clarified the interface decision: `libs/model::ChatModel` already supports image content through `ContentPart`, so Qwen3.6 does not need a new core chat interface. Any additional work is limited to llama.cpp backend support for mmproj/image execution and capability advertisement. |
| 2026-04-24 | @codex-gpt55 | Initial design for issue `#224`: Qwen3.6 27B GGUF through the existing llama.cpp backend, with platform-aware quant defaults, feature-gated catalog wiring, CUDA reuse, and a multimodal example path. |

## Scope

Issue `#224` adds a curated Qwen3.6 27B GGUF bundle to `motlie-models`.
The bundle must reuse the existing llama.cpp backend rather than adding a new
runtime.

Planned public identity:

| Field | Value |
|-------|-------|
| Selector | `qwen/qwen3_6_27b_gguf` |
| Bundle id | `qwen3_6_27b_gguf` |
| Display name | `Qwen3.6 27B (GGUF)` |
| Backend | `BackendKind::LlamaCpp` |
| Checkpoint format | `CheckpointFormat::Gguf` |
| Cargo feature | `model-qwen3-6-27b-gguf` |
| Optional CUDA feature | `llama-cpp-cuda` |

The current curated source is `unsloth/Qwen3.6-27B-GGUF`, which provides
Q4_K_M, Q5_K_M, Q8_0, and mmproj artifacts. It does not provide an FP8 GGUF.
The official `Qwen/Qwen3.6-27B-FP8` repository is Transformers/safetensors
format, so it is not compatible with this llama.cpp/GGUF bundle without a
separate GGUF conversion artifact.

## Goals

- Add Qwen3.6 27B as a curated chat bundle under `libs/models`.
- Reuse `motlie-model-llama-cpp` and its GGUF loading path.
- Preserve the existing curated-bundle layering:
  `libs/models` owns catalog/artifacts; `libs/model/backends/llama_cpp` owns
  runtime execution.
- Gate the model with `model-qwen3-6-27b-gguf`.
- Reuse `llama-cpp-cuda` for CUDA builds and GPU offload.
- Support the currently available curated GGUF quants: Q5-class default, with
  Q4 and Q8 explicit overrides.
- Keep the CUDA FP8 default as a design target, but do not advertise FP8 until a
  real curated FP8 GGUF artifact exists.
- Expose only the interfaces that the backend actually implements.

## Non-Goals

- Do not add another backend crate.
- Do not add a new core chat interface. The existing `ChatModel` /
  `ChatRequest` / `ContentPart` contract already models text and image content.
- Do not route this model through `mistral.rs`; the `qwen35` GGUF path is a
  llama.cpp-backed slice.
- Do not make arbitrary GGUF loading public. This remains a curated bundle with
  known artifacts.
- Do not advertise image input unless a real llama.cpp multimodal path is
  implemented and validated.

## Existing Patterns to Follow

Existing llama.cpp GGUF curated bundles:

- `libs/models/src/chat/qwen3_4b_gguf.rs`
- `libs/models/src/chat/gemma4_e2b_gguf.rs`

Existing llama.cpp backend:

- `libs/model/backends/llama_cpp/src/text.rs`
- `LlamaCppTextSpec` owns curated runtime spec data.
- `LlamaCppTextBundle` implements `ModelBundle`.
- `LlamaCppTextHandle` implements `ChatModel` and `CompletionModel`.
- `LLAMA_CPP_TEXT_FORMATS` is `[CheckpointFormat::Gguf]`.
- Current GGUF filename selection maps Q4, Q5, Q8, reserved FP8, and
  F16-style default through one centralized `Option<QuantizationBits>` suffix
  helper after callers resolve support through the bundle spec.

Existing multimodal example pattern:

- `libs/models/examples/chat_multimodal_gemma4/main.rs`
- `libs/models/examples/chat_multimodal_gemma4/README.md`
- Uses `--download-artifacts`, `--precision`, optional `--image`, local-only
  startup after optional download, descriptor/capability printing, shared
  `support.rs`, and handle metrics.

## Interface Design

### Core Chat Contract Decision

No `libs/model` chat API change is required for multimodal Qwen3.6 support.
The core contract already accepts image-bearing chat messages:

- `ChatRequest` contains `ChatMessage` values.
- `ChatMessage.content` is a list of `ContentPart`.
- `ContentPart` already includes `Text`, `Image`, and `ImageUrl`.
- Capability descriptors already distinguish text chat from multimodal
  chat/vision through `Capabilities::chat_and_completion()` and
  `Capabilities::multimodal_chat_and_vision()`.

The open work is therefore not a public interface problem. It is a
backend-execution problem: the llama.cpp backend must either process image
content through a real mmproj/image path or continue to reject image parts and
advertise text-only capabilities.

### Required Text Interfaces

The Qwen3.6 27B GGUF bundle must expose:

- `ChatModel`
- `CompletionModel`, if it uses the existing `LlamaCppTextHandle` path
- no embeddings

For a text-only first slice, descriptor capabilities must be
`Capabilities::chat_and_completion()`. The backend must continue to reject
image content parts with `UnsupportedCapability(CapabilityKind::Vision)`.

### Multimodal Interface

Qwen3.6 27B is a multimodal model, so full support should include image+text
chat when the selected GGUF distribution and Rust llama.cpp binding support
the required projector path.

Multimodal support must not be simulated in the text backend. If feasible, add
or refactor llama.cpp backend runtime code so it consumes the existing
`ChatRequest` image parts with:

- backend support for `ContentPart::Image` and/or `ContentPart::ImageUrl`
- mmproj artifact resolution in the curated bundle module
- `Capabilities::multimodal_chat_and_vision()`
- tests proving image input is accepted by the loaded handle

This backend work may be implemented as a new `llama_cpp` multimodal module or
as a refactor of the existing handle, but it must preserve the same public
`ChatModel` interface. The important decision is that the catalog advertises
vision only when llama.cpp can actually execute image+text requests.

If the binding does not expose the required mmproj/image APIs, the text path
may land as an intermediate slice, but the bundle must remain text-only in its
descriptor until multimodal execution is real.

## GGUF Quantization

The implementation extends `QuantizationBits` with `Five` and `FloatEight` so
callers do not overload `Eight` for both Q8 and FP8.

Current curated runtime policy:

| Host profile | Default when unspecified | Required explicit overrides |
|--------------|--------------------------|-----------------------------|
| macOS / Apple Silicon | Q5_K_M GGUF | Q4_K_M and Q8_0 |
| CUDA-enabled Linux/DGX | Q5_K_M GGUF until FP8 GGUF exists | Q8_0 and Q4_K_M |
| CPU-only Linux | Q5_K_M GGUF | Q4_K_M and Q8_0 |

The requested CUDA policy remains: CUDA on by default through `llama-cpp-cuda`,
FP8 by default, with Q8 and Q4 switchable. The implementation intentionally
does not advertise FP8 yet because the validated GGUF repo does not contain an
FP8 GGUF artifact.

If an FP8 GGUF artifact becomes available, the next step is to add it to the
Qwen3.6 `QuantizationSupport`, artifact include rules, example parser, and
CUDA-default resolver without changing the core chat API.

```rust
QuantizationBits::Four       // Q4_K_M
QuantizationBits::Five       // Q5_K_M
QuantizationBits::Eight      // Q8_0
QuantizationBits::FloatEight // reserved for FP8 GGUF
```

Default selection can later be resolved from:

1. explicit caller request from the example or future public option
2. compiled CUDA feature plus runtime GPU policy
3. target OS
4. model-specific fallback

CUDA builds should continue to use the existing llama.cpp GPU layer policy:

- default to full GPU offload when CUDA is enabled
- honor `MOTLIE_MODEL_FORCE_CPU=1`
- honor `MOTLIE_MODEL_GPU_LAYERS=<n>`

## Artifact Contract

The curated bundle module must own provider-specific artifact rules. The
backend should receive resolved local GGUF paths and optional projector paths.

Artifact validation must cover:

- selected GGUF repo: `unsloth/Qwen3.6-27B-GGUF`
- exact artifact filenames for Q4_K_M, Q5_K_M, and Q8_0
- absence of FP8 GGUF in the curated repo
- expected tokenizer/config sidecars used by the repo
- optional mmproj file if multimodal support is implemented
- local-only startup failure messages for missing selected quant artifacts

The implementation hardcodes only the validated Q4/Q5/Q8 GGUF filenames.

## Planned Files and Directories

Documentation introduced by this planning slice:

```text
libs/models/docs/DESIGN_QWEN3_6_27B_GGUF.md
```

Planned backend changes:

```text
libs/model/backends/llama_cpp/src/lib.rs
libs/model/backends/llama_cpp/src/text.rs
libs/model/backends/llama_cpp/src/common.rs
libs/model/backends/llama_cpp/src/quantization.rs
libs/model/backends/llama_cpp/src/multimodal.rs
```

Notes:

- `quantization.rs` is expected to hold native GGUF quant label mapping and
  platform-aware default resolution.
- `multimodal.rs` is only introduced if the Rust llama.cpp binding can support
  mmproj/image input. Otherwise image support remains blocked and the bundle
  must stay text-only.

Planned curated catalog changes:

```text
libs/models/src/chat/qwen3_6_27b_gguf.rs
libs/models/src/chat/mod.rs
libs/models/src/lib.rs
libs/models/Cargo.toml
```

Planned example changes:

```text
libs/models/examples/chat_multimodal_qwen3_6_27b/main.rs
libs/models/examples/chat_multimodal_qwen3_6_27b/README.md
libs/models/examples/README.md
```

The example should exist only when it demonstrates the same public interfaces
advertised by the bundle. If Qwen3.6 lands text-only first, either delay the
multimodal example or make the image path explicitly blocked until the backend
can support it.

Planned validation and release-doc updates:

```text
libs/models/docs/BUILD_MODELS.md
scripts/check_curated_model_examples.sh
.github/workflows/models-build.yml
```

Only update the script/workflow if the new example or feature set should become
part of the regular curated-model check matrix.

## Example Contract

The dedicated example should be:

```text
libs/models/examples/chat_multimodal_qwen3_6_27b/
```

Expected command style:

```bash
cargo run -p motlie-models \
  --no-default-features \
  --features model-qwen3-6-27b-gguf \
  --example chat_multimodal_qwen3_6_27b -- \
  [--download-artifacts] [--precision=q4|q5|q8|fp8] [--image=/path/to/image] <prompt>
```

CUDA command style:

```bash
cargo run -p motlie-models \
  --no-default-features \
  --features model-qwen3-6-27b-gguf,llama-cpp-cuda \
  --example chat_multimodal_qwen3_6_27b -- \
  [--download-artifacts] [--precision=q5|q8|q4|fp8] [--image=/path/to/image] <prompt>
```

The example must:

- use shared `../support.rs`
- use `default_artifact_root()`
- optionally download curated artifacts with `--download-artifacts`
- start with `ArtifactPolicy::LocalOnly`
- print descriptor capabilities
- print selected quant label and GPU offload state
- run text chat
- run completion when the handle exposes completion
- run image+text chat only when the bundle advertises vision capability
- document that `llama-cpp-cuda` is the CUDA gate

## Testing Strategy

Backend tests:

- Qwen3.6 spec identity and capabilities
- native GGUF quant label mapping
- Q5 default and Q4/Q8 override support
- FP8 rejection until a curated FP8 GGUF artifact exists
- CUDA default offload policy remains overrideable by env vars
- text prompt formatting for Qwen3.6/Qwen35
- image content rejection for text-only path
- image content acceptance for multimodal path if implemented

Catalog tests:

- descriptor reports `BackendKind::LlamaCpp`
- descriptor reports `CheckpointFormat::Gguf`
- selector parses `chat:qwen/qwen3_6_27b_gguf`
- disabled feature returns `ModelUnavailable`
- catalog registers the bundle only when `model-qwen3-6-27b-gguf` is enabled
- local GGUF snapshot resolver accepts required artifacts and rejects missing
  platform-default artifacts

Example/build tests:

- `cargo check -p motlie-models --no-default-features --features model-qwen3-6-27b-gguf --example chat_multimodal_qwen3_6_27b`
- `cargo check -p motlie-models --no-default-features --features model-qwen3-6-27b-gguf,llama-cpp-cuda --example chat_multimodal_qwen3_6_27b`
- env-gated smoke with pre-downloaded Qwen3.6 GGUF artifacts

## Open Decisions

- Whether to switch from `unsloth/Qwen3.6-27B-GGUF` if an official repo later
  provides the complete Q4/Q5/Q8/FP8 set.
- Exact FP8 GGUF filename and source, once one exists.
- Whether the current `llama-cpp-2` binding exposes enough multimodal APIs for
  mmproj image input.
- Whether the 27B bundle should stay out of the default feature set. The
  current design assumes it is opt-in only.
