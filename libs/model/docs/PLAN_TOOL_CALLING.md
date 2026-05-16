# Model Tool-Calling Contract Plan

## Status: In Progress

## Owner

@codex-tool-calling

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-05-11 | @codex-tool-calling | Initial staged plan for the unified tool-calling API contract, backend adapters, curated capability gating, and validation work. |
| 2026-05-13 | @codex-tool-calling | Implemented the core typed Rust tool binding contract, added examples, wired the `mistral.rs` text/multimodal request and response adapter, and enabled `ToolUse` on the safetensors Qwen3/Gemma 4 curated descriptors. |
| 2026-05-13 | @codex-tool-calling | Wired the llama.cpp GGUF tool-aware path through OpenAI-compatible message/tool JSON, chat-template rendering, grammar sampling, response parsing, and the GGUF live example switch. GGUF `ToolUse` descriptors remain gated pending artifact smoke validation. |
| 2026-05-13 | @codex-tool-calling | Attempted the Qwen3 GGUF tool-demo smoke against the default local artifact cache. It failed before model load because `Qwen/Qwen3-4B-GGUF` is not cached locally; GGUF descriptors remain intentionally unadvertised for `ToolUse`. |

Derived from [DESIGN_TOOL_CALLING.md](./DESIGN_TOOL_CALLING.md). This plan covers the work needed to make Gemma 4, Qwen3, and Qwen3.6 tool calling available through the existing `ChatModel` surface.

## Scope Summary

Expected implementation shape:

- 3 to 5 focused PRs
- no vendored dependency changes
- one core contract change in `libs/model`
- one `mistral.rs` adapter change
- one `llama.cpp` adapter change
- curated descriptor/test updates in `libs/models`
- optional examples after the core behavior is validated

## Phase 0: Design and Tracking

- [x] Document the common API shape in `libs/model/docs/DESIGN_TOOL_CALLING.md`.
- [x] Document the staged implementation plan in this file.
- [x] Link the design/plan from GitHub issues #272 and #273.
- [x] Keep issue updates signed as `@codex-tool-calling`.

## Phase 1: Core `libs/model` Contract

Add the public vocabulary needed to represent tool-calling turns without backend-specific types.

### 1.1 - Dependencies

- [x] Add `serde_json` and `schemars` to `libs/model/Cargo.toml`.
- [x] Keep the dependency in the contract crate only; backend crates can reuse the public types.

### 1.2 - Chat roles and messages

Files:

- `libs/model/src/chat.rs`

Tasks:

- [x] Add `ChatRole::Tool`.
- [x] Add `ToolName`, `ToolInputSchema`, `ToolSpec`, `ToolChoice`, `ToolArguments`, and `ToolCall`.
- [x] Extend `ChatMessage` with:
  - `name: Option<ToolName>`
  - `tool_call_id: Option<ToolCallId>`
  - `tool_calls: Vec<ToolCall>`
  - `reasoning: Option<String>`
- [x] Preserve existing constructors:
  - `ChatMessage::new`
  - `ChatMessage::text`
  - `ChatMessage::with_parts`
  - `ChatMessage::text_and_image`
- [x] Add ergonomic constructors for assistant tool calls and tool result messages.
- [x] Add validation helpers only where they remove duplicated backend checks.

### 1.3 - Requests and responses

Files:

- `libs/model/src/generation.rs`

Tasks:

- [x] Extend `ChatRequest` with `tools: Vec<ToolSpec>`.
- [x] Extend `ChatRequest` with `tool_choice: Option<ToolChoice>`.
- [x] Extend `ChatResponse` with:
  - `tool_calls: Vec<ToolCall>`
  - `finish_reason: Option<ChatFinishReason>`
  - `reasoning: Option<String>`
  - `usage: Option<GenerationUsage>`
- [x] Add `ChatFinishReason`.
- [x] Add `GenerationUsage`.
- [x] Preserve text response readability for existing text-only callers through constructors/defaults.

### 1.3a - Typed Rust tool binding

Files:

- `libs/model/src/tool.rs`
- `libs/models/src/tool_registry.rs`

Tasks:

- [x] Add `Tool` for typed Rust argument/output bindings.
- [x] Keep the core crate free of runtime tool execution/type-erased registries.
- [x] Add `ToolRegistry` in `motlie-models` for caller-owned example/application tool execution.
- [x] Support binding an existing async function with `insert_fn` in the `motlie-models` helper.
- [x] Support binding an inline async closure with `insert_fn` in the `motlie-models` helper.
- [x] Return executed tool results as `ChatRole::Tool` messages through the helper's `call_to_message`.
- [x] Preserve raw model argument JSON through `ToolArguments`.

### 1.4 - Capability introspection

Files:

- `libs/model/src/lib.rs`
- `libs/model/src/eval.rs` if eval-track mapping needs an explicit no-op decision

Tasks:

- [x] Add `CapabilityKind::ToolUse`.
- [x] Add `CapabilityDescriptor::tool_use()`.
- [x] Add `Capabilities` built-ins/helpers if needed by curated descriptors.
- [x] Decide whether `EvalTrack::primary_for_descriptor` maps `ToolUse` to an existing chat track or returns `None`.

### 1.5 - Core tests and docs

Files:

- `libs/model/src/chat.rs`
- `libs/model/src/generation.rs`
- `libs/model/src/lib.rs`
- `libs/model/docs/API.md`
- `libs/model/docs/DESIGN.md`
- `libs/model/docs/PLAN.md`

Tasks:

- [x] Add unit tests for constructors, defaults, and equality.
- [x] Add capability tests for `ToolUse`.
- [x] Update API docs with the new concrete type shapes.
- [x] Update the main DESIGN/PLAN docs to point at the dedicated tool-calling docs.
- [x] Run `cargo test -p motlie-model`.

## Phase 2: `mistral.rs` Backend Adapter

Wire the new contract into the safetensors-backed Qwen3 and Gemma 4 paths.

Files:

- `libs/model/backends/mistral/src/common.rs`
- `libs/model/backends/mistral/src/text.rs`
- `libs/model/backends/mistral/src/multimodal.rs`
- `libs/model/backends/mistral/Cargo.toml` only if an explicit dependency is needed

Tasks:

- [x] Extend role mapping to support `ChatRole::Tool`.
- [x] Map `ToolSpec` to `mistralrs::Tool`.
- [x] Map `ToolChoice` to `mistralrs::ToolChoice`.
- [x] Map assistant replay messages with `tool_calls`.
- [x] Map tool result messages with `tool_call_id`; Motlie retains the tool `name`, while the current `mistral.rs` builder API accepts only content and call ID.
- [x] Include `tools` and `tool_choice` in generated requests.
- [x] Extract response `tool_calls` into `ChatResponse.tool_calls`.
- [x] Map upstream finish reason into `ChatFinishReason`.
- [x] Map upstream usage into `GenerationUsage`.
- [x] Preserve existing text-only Qwen3 behavior.
- [x] Preserve existing multimodal Gemma 4 image behavior.
- [x] Advertise `ToolUse` on safetensors Qwen3/Gemma 4 descriptors after adapter unit tests pass.

Known limitation:

- `ToolChoice::Required` returns `ModelError::InvalidConfiguration` on the `mistral.rs` backend because the upstream builder does not expose required-tool enforcement. `Auto`, `None`, and `Named` are mapped.

Validation:

- [x] Unit-test shared tool schema/call mapping for Qwen3/Gemma 4 `mistral.rs` requests.
- [x] Unit-test response extraction with tool calls.
- [x] Run `RUSTFLAGS="-C target-feature=+fullfp16" cargo test -p motlie-model-mistral --lib`.

## Phase 3: `llama.cpp` Backend Adapter

Replace or bypass hand-written prompt formatting for tool-aware chats.

Files:

- `libs/model/backends/llama_cpp/src/text.rs`
- `libs/model/backends/llama_cpp/Cargo.toml` only if an explicit dependency is needed

Tasks:

- [x] Convert Motlie `ChatMessage` values to OpenAI-compatible JSON messages.
- [x] Convert Motlie `ToolSpec` values to OpenAI-compatible JSON tools.
- [x] Load the model chat template through `LlamaModel::chat_template(None)`.
- [x] Use `OpenAIChatTemplateParams` for tool-aware requests.
- [x] Feed returned grammar metadata into generation when present.
- [x] Parse generated output with `ChatTemplateResult::parse_response_oaicompat`.
- [x] Map parsed tool calls into `ChatResponse.tool_calls`.
- [x] Preserve the current image rejection behavior for text-only GGUF handles.
- [x] Keep current hand formatter for no-tool chat in this PR; route only tool-bearing chat through llama.cpp chat templates.

Known limitation:

- `ToolChoice::Named` returns `ModelError::InvalidConfiguration` on llama.cpp because upstream's OpenAI-compatible tool-choice parser accepts `auto`, `none`, and `required`.

Validation:

- [x] Unit-test OpenAI-compatible message JSON for system/user/assistant/tool turns.
- [x] Unit-test OpenAI-compatible tool JSON.
- [x] Unit-test OpenAI-compatible parsed response tool-call mapping.
- [ ] Add model-family generated-output fixtures if live smoke reveals parser differences.
- [ ] Validate that GGUF artifact templates exist before advertising `ToolUse`; blocked locally until `Qwen/Qwen3-4B-GGUF`, `bartowski/gemma-4-E2B-it-GGUF`, and `Qwen/Qwen3.6-27B-GGUF` artifacts are present in the curated HF cache.
- [x] Run `cargo test -p motlie-model-llama-cpp`.

## Phase 4: Curated Bundle Capability Gating

Advertise `ToolUse` only for backend/artifact pairs that pass adapter validation.

Files:

- `libs/models/src/chat/gemma4_e2b.rs`
- `libs/models/src/chat/gemma4_e2b_gguf.rs`
- `libs/models/src/chat/qwen3_4b.rs`
- `libs/models/src/chat/qwen3_4b_gguf.rs`
- `libs/models/src/chat/qwen3_6_27b_gguf.rs`
- `libs/models/src/chat/mod.rs`

Tasks:

- [x] Add `ToolUse` to `google/gemma4_e2b` after `mistral.rs` multimodal tests pass.
- [x] Add `ToolUse` to `qwen/qwen3_4b` after `mistral.rs` text tests pass.
- [ ] Add `ToolUse` to GGUF variants only after chat-template validation passes for the selected artifacts.
- [ ] Keep `Vision` unchanged: only the safetensors Gemma 4 path currently advertises vision.
- [x] Add descriptor tests for safetensors capability sets.
- [ ] Add descriptor tests for GGUF tool capability gating when the llama.cpp adapter lands.
- [ ] Run `cargo test -p motlie-models --features model-gemma4-e2b,model-gemma4-e2b-gguf,model-qwen3-4b,model-qwen3-4b-gguf,model-qwen3-6-27b-gguf`.

## Phase 5: Examples and Smoke Tests

Prove the caller-owned loop works without adding automatic tool execution to `libs/model`.

Candidate files:

- `libs/models/examples/chat_tool_binding/main.rs`
- `scripts/check_curated_model_examples.sh`
- model backend docs under `libs/model/backends/docs/`

Tasks:

- [x] Add a small example with deterministic local tools, including `get_weather`.
- [x] Show both existing function binding and closure binding.
- [x] Show the local loop scaffold: send tool specs, handle simulated tool calls, execute local functions, and produce tool-result messages.
- [x] Add a Qwen3 safetensors live backend example path that requests a final answer after tool-result messages.
- [x] Add a GGUF llama.cpp live backend example path that requests a final answer after tool-result messages.
- [ ] Add the same live backend tool loop to Gemma 4 safetensors once a suitable multimodal/text prompt path is selected.
- [x] Keep example execution optional because curated LLM weights may not be present in CI.
- [x] Add fixture-level tests that do not require downloading model weights.
- [x] Add manual smoke instructions for local model owners.

Manual smoke commands for the remaining GGUF descriptor gate:

```sh
cargo run -p motlie-models --no-default-features --features model-qwen3-4b-gguf \
  --bin motlie-models-download -- qwen3_4b_gguf

cargo run -p motlie-models --no-default-features --features model-qwen3-4b-gguf \
  --example chat_gguf_gwen3_gemma4 -- --tool-demo "What is Rust?"
```

Local status on 2026-05-13: the smoke command was attempted without
`--download-artifacts` and failed before model load with
`artifact policy LocalOnly requires cached GGUF artifacts for Qwen/Qwen3-4B-GGUF`;
this is an artifact availability blocker, not a code-path failure.

## Phase 6: Final Documentation and Issue Closure

Files:

- `libs/model/docs/API.md`
- `libs/model/docs/DESIGN_TOOL_CALLING.md`
- `libs/model/docs/PLAN_TOOL_CALLING.md`
- GitHub issues #272 and #273

Tasks:

- [ ] Update API docs with the final implemented structs/enums.
- [ ] Mark completed plan items.
- [ ] Document any backend-specific limitations that remain.
- [ ] Confirm no files under vendored dependency paths changed.
- [ ] Comment on #272 and #273 with final implementation status.

## Acceptance Criteria

- A caller can send a `ChatRequest` with tools to a tool-capable curated chat bundle.
- The model can return structured `ChatResponse.tool_calls`.
- The caller can append assistant tool-call and tool-result messages.
- The model can produce a final assistant answer after tool results are supplied.
- Plain chat callers continue to compile with constructor/default-based usage.
- Curated descriptors advertise `ToolUse` only after backend-specific tests pass.
- No vendored dependency files are modified.

## Risk Register

| Risk | Mitigation |
| --- | --- |
| GGUF artifact lacks a usable tool-aware template. | Validate template availability before adding `ToolUse` to that variant. |
| Qwen3.6 thinking content is confused with tool-call JSON. | Keep `reasoning` separate from `content` and parsed `tool_calls`. |
| Backend cannot enforce `ToolChoice::Required` or `ToolChoice::Named`. | Return `InvalidConfiguration` rather than silently degrading. |
| Existing exhaustive `ChatRole` matches break. | Update all role matches in the same PR as `ChatRole::Tool`. |
| Tool schema generation accepts Rust types that do not model an object-shaped argument payload. | Prefer named argument structs in examples and consider stricter schema validation before backend adapter PRs. |
| Tool parsing bugs surface only with live models. | Add parser/rendering fixture tests plus optional model smoke examples. |
