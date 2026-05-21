# Gemma 4 E4B Bundle Design

## Status: Implemented (PR #309)

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-05-20 | @claude-reviewer | Design-as-built record for issue `#308`: curated Gemma 4 E4B bundles on both the llama.cpp/GGUF and mistral.rs/safetensors backends, the per-model `recommended_*` spec fields, the core `ThinkingMode` move, and the `#310` safetensors tool-transcript fix. |

## Scope

Issue `#308` adds curated Gemma 4 **E4B-it** bundles to `motlie-models`, the
larger sibling of the already-shipped E2B bundles. The work reuses both
existing backends rather than adding a runtime, and adds a small core surface
so per-model sampling/thinking/system-prompt guidance is curated data rather
than demo-side constants.

Public identity:

| Field | GGUF bundle | safetensors bundle |
|-------|-------------|--------------------|
| Selector | `google/gemma4_e4b_gguf` | `google/gemma4_e4b` |
| Bundle id | `gemma4_e4b_gguf` | `gemma4_e4b` |
| Display name | `Gemma 4 E4B-it (GGUF)` | `Gemma 4 E4B-it` |
| Backend | `BackendKind::LlamaCpp` | `BackendKind::MistralRs` |
| Checkpoint format | `CheckpointFormat::Gguf` | `CheckpointFormat::Safetensors` |
| Cargo feature | `model-gemma4-e4b-gguf` | `model-gemma4-e4b` |
| Curated source | `unsloth/gemma-4-E4B-it-GGUF` | `google/gemma-4-E4B-it` |
| Quantization | Q8_0 default, Q4_K_M override | ISQ Q8 default, ISQ Q4 override |
| Capabilities | `Chat` + `Completion` + `ToolUse` | `Chat` + `Vision` + `ToolUse` |

Both bundles are **opt-in only** — neither is in the `motlie-models` default
feature set, consistent with the size-based policy (E4B Q8_0 GGUF is ~8 GB;
safetensors is larger). E2B remains the default-feature Gemma 4 bundle.

## Goals

- Add Gemma 4 E4B as a curated chat bundle on both backends.
- Reuse `motlie-model-llama-cpp` (GGUF text) and `motlie-model-mistral`
  (safetensors multimodal); no new backend crate, no new core chat trait.
- Carry per-model curated guidance — sampling defaults, recommended system
  prompt — on the bundle spec, not hard-coded in examples.
- Default the GGUF bundle to Q8_0 because E4B's reasoning quality is sensitive
  to Q4-level quantization loss.
- Allow callers to flip thinking mode per request.
- Advertise `ToolUse` only against a backend path with multi-round tool-loop
  smoke coverage (`#272` rule).

## Non-Goals

- No new backend crate; no new core `ChatModel` interface.
- No arbitrary GGUF/safetensors loading — curated bundles with known artifacts.
- No audio input. Gemma 4 E2B/E4B accept audio upstream, but `motlie-model`
  has no audio-input chat capability; tracked separately if pursued.
- No 128K context by default. E4B supports it; the curated default is 32768 to
  bound KV-cache memory. A future `StartOptions` context override can lift it.

## Core Surface Added

E4B needs model-specific sampling and prompt guidance to be first-class. Three
additive changes in `libs/model`, all non-breaking (`Option`/`Default`):

- `GenerationParams::with_defaults(self, &Self) -> Self` — per-field merge;
  caller-set values win, `None` falls through to the supplied defaults, and an
  empty `stop_sequences` list falls through.
- `ThinkingMode { Disabled, Auto }` — moved into `motlie_model::generation`
  (was llama.cpp-local) and re-exported from the llama.cpp backend for
  source-compatibility. `ChatRequest` gains `thinking: Option<ThinkingMode>`;
  the backend resolves `request.thinking.unwrap_or(spec.thinking)`.
- `LlamaCppTextSpec`, `MistralTextSpec`, and `MistralMultimodalSpec` each gain
  `recommended_generation_params: GenerationParams` and
  `recommended_system_prompt: Option<&'static str>`. Existing constructors set
  `GenerationParams::default()` / `None`, preserving current behavior.

E4B sets `recommended_generation_params` to `temperature = 1.0`, `top_p = 0.95`
(Google's published guidance for E4B reasoning quality), and
`recommended_system_prompt` to `"You are Gemma, a helpful assistant."`. The
GGUF spec defaults `thinking` to `ThinkingMode::Auto`.

These are *recommendations*, not enforcement: backends do not read the spec's
`recommended_*` fields. Callers (the demos) merge them into the `ChatRequest`
via `with_defaults`. The three-way precedence is caller value → spec
recommendation → backend constant.

## Quantization

| Backend | Default | Override | Rationale |
|---------|---------|----------|-----------|
| llama.cpp GGUF | Q8_0 | Q4_K_M | E4B reasoning degrades measurably at Q4; Q8_0 (~8 GB) is effectively lossless |
| mistral.rs safetensors | ISQ Q8 | ISQ Q4 | Same reasoning-quality rationale; ISQ applies at load |

The GGUF artifact rule lists `-Q8_0.gguf` first and `-Q4_K_M.gguf` second, so
`motlie-models-download` stages both and the curated default resolves to Q8_0.
FP8 is not advertised — no curated FP8 artifact exists.

## Artifact Contract

- GGUF: `unsloth/gemma-4-E4B-it-GGUF`, `-Q8_0.gguf` + `-Q4_K_M.gguf` suffix
  rules; local-only startup fails closed when the selected-quant artifact is
  absent.
- safetensors: `google/gemma-4-E4B-it`, the standard Gemma 4 multimodal sidecar
  set (config, tokenizer, processor/preprocessor config, chat template,
  `.safetensors` shards + index). Local-only startup additionally requires the
  multimodal processor config and `chat_template.jinja`.

## Tool Calling

Both E4B bundles advertise `ToolUse`. The GGUF path routes tool-bearing
requests through llama.cpp's OpenAI-compatible chat-template helpers; the
safetensors path goes through the `mistral.rs` adapter.

During implementation, live validation surfaced a `mistral.rs` safetensors
tool-transcript regression (`#310`): the upstream `mistralrs::RequestBuilder`
replayed assistant calls under `function` instead of `tool_calls`, omitted
`name` on tool-result replay, and let `enable_thinking` default to `true`,
consuming the tool-demo generation budget. The fix replaces `RequestBuilder`
with a custom `MotlieMistralRequest` implementing `mistralrs::RequestLike`,
which controls the exact transcript shape. With the fix, all three `mistral.rs`
safetensors bundles (`qwen3_4b`, `gemma4_e2b`, `gemma4_e4b`) honestly advertise
`ToolUse`.

`gemma4_e4b` safetensors has no dedicated example; it shares the
`MistralMultimodalAdapter` path with `gemma4_e2b`, whose tool loop is smoke-
validated. See `VALIDATION_CHAT_TOOL_CALLING.md` for the matrix.

## Files

Curated catalog:

```text
libs/models/src/chat/gemma4_e4b.rs        # safetensors bundle
libs/models/src/chat/gemma4_e4b_gguf.rs   # GGUF bundle
libs/models/src/chat/mod.rs               # ChatModels::Gemma4E4B{,_Gguf}
libs/models/Cargo.toml                    # model-gemma4-e4b{,-gguf} features
```

Backend specs:

```text
libs/model/backends/llama_cpp/src/text.rs        # LlamaCppTextSpec::gemma4_e4b
libs/model/backends/mistral/src/multimodal.rs    # MistralMultimodalSpec::gemma4_e4b
libs/model/src/generation.rs                     # ThinkingMode, with_defaults, ChatRequest::thinking
```

Examples and docs:

```text
libs/models/examples/chat_gguf_gwen3_gemma4/     # --chat=google/gemma4_e4b_gguf + --thinking/--system/--assistant
libs/models/examples/chat_tool_binding/          # exercises recommended_* fields without an LLM
libs/models/examples/README.md                   # capability + demo-output ledgers
libs/models/docs/VALIDATION_CHAT_TOOL_CALLING.md # end-to-end validation matrix
```

## Testing

- Backend: `gemma4_e4b()` spec identity, Q8 default, `temperature=1.0`/
  `top_p=0.95` recommendations, system prompt, `ThinkingMode::Auto`, advertised
  capabilities.
- Core: `GenerationParams::with_defaults` field-precedence and
  empty-`stop_sequences` fallthrough.
- Catalog: descriptor identity, GGUF local-snapshot resolution, artifact rules.
- Example: `cargo check` for the GGUF example with the E4B feature; live
  Apple Silicon Metal smoke recorded in `VALIDATION_CHAT_TOOL_CALLING.md`.

## Open Decisions

- Whether to add a dedicated `gemma4_e4b` safetensors example so its `ToolUse`
  advertisement is directly smoked rather than relying on the shared adapter.
- Whether to raise the default context length above 32768 once a
  `StartOptions` context override exists.
- Whether to capture a richer, model-card-derived `recommended_system_prompt`
  than the current generic string.
