# Model Tool-Calling Contract Design

## Status: In Progress

## Owner

@codex-tool-calling

## Change Log

| Date | Change | Sections |
|------|--------|----------|
| 2026-05-11 | @codex-tool-calling: Initial design for a unified tool-calling chat contract that supports the curated Gemma 4 and Qwen3/Qwen3.6 LLMs with backend-local adaptation. | All |
| 2026-05-13 | @codex-tool-calling: Refined the API around typed Rust tool binding, documented existing function and closure binding, implemented the `mistral.rs` adapter path, and enabled `ToolUse` for safetensors Qwen3/Gemma 4 descriptors. | Proposed Core API, Backend Adaptation, Validation Strategy |
| 2026-05-13 | @codex-tool-calling: Implemented the llama.cpp GGUF tool-aware adapter path using OpenAI-compatible chat templates and documented the remaining GGUF descriptor gate. | Backend Adaptation, Validation Strategy |
| 2026-05-16 | @codex-tool-calling: Replaced runtime-erased tool registries with static `ToolList` dispatch and documented the future MCP data-routing path tracked by issue #284. | Design Principle, Typed Rust Tool Binding, Dispatch Architecture |

This document extends the `libs/model` chat contract with a common tool-calling API. It is intentionally focused on the Motlie contract layer, but it includes backend and curated-bundle implications because the design only works if the common shape can map cleanly to the current `mistral.rs` and `llama.cpp` paths.

Related tracking:

- GitHub issue #272: unified tool-calling chat contract for Gemma 4 and Qwen3
- GitHub issue #273: gap map, implementation scope, and current model convention notes
- GitHub issue #284: MCP integration architecture and future transport work
- [PLAN_TOOL_CALLING.md](./PLAN_TOOL_CALLING.md)

## Problem

The curated chat LLMs in `libs/models` are capable of tool calling at the model/template level, but the current Motlie chat contract cannot represent a tool-calling turn:

- `ChatRole` has only `System`, `User`, and `Assistant`.
- `ChatMessage` has only `role` and `content`.
- `ChatRequest` has no `tools` or `tool_choice`.
- `ChatResponse` returns only assistant text.
- `CapabilityKind` has no descriptive `ToolUse` capability.
- The `mistral.rs` wrappers use request/response paths that drop upstream tool metadata.
- The `llama.cpp` wrapper hand-formats Qwen/Gemma prompts, bypassing the OpenAI-compatible chat-template/tool helpers exposed by `llama-cpp-2`.

This blocks a standard LLM tool loop:

1. caller sends messages plus tool definitions
2. model returns structured tool call(s)
3. caller executes those tools outside the model runtime
4. caller appends tool result message(s)
5. model produces the final assistant response

## Goals

- Define one Motlie-level tool-calling shape that works for Gemma 4, Qwen3, and Qwen3.6.
- Keep `ChatModel::generate(ChatRequest) -> ChatResponse` as the executable surface.
- Make tool execution caller-owned for the first implementation.
- Preserve text-only and multimodal chat behavior.
- Preserve current constructors such as `ChatMessage::text(...)` for normal chat users.
- Keep model-family formatting in backend adapters, not in callers.
- Allow curated bundles to advertise `ToolUse` only after their backend path passes round-trip tests.
- Avoid changes to vendored dependencies.

## Non-Goals

- Automatic tool execution inside `libs/model`.
- A callback registry, MCP client, sandbox, or policy engine.
- Streaming tool-call deltas in the first slice.
- Provider-specific tool APIs in the public contract.
- A separate `ToolModel` trait.
- Tool calling for non-chat models such as embeddings, ASR, TTS, or completion-only APIs.

## Curated Model Capability

The current curated chat LLMs are model-capable for tool calling, but Motlie should not advertise that capability until the API and adapters below are implemented.

| Curated selector | Backend | Common capability stance | Convention behind the adapter |
| --- | --- | --- | --- |
| `google/gemma4_e2b` | `mistral.rs` multimodal | ToolUse advertised after adapter unit tests. | Gemma 4 chat template with `tools`, assistant `tool_calls`, tool-result turns, and Gemma tool-call/tool-response tokens. |
| `google/gemma4_e4b` | `mistral.rs` multimodal | ToolUse advertised on the same generic adapter path as E2B. | Same Gemma 4 convention as E2B; the `mistral.rs` adapter is bundle-agnostic once the template and request fields are available. |
| `google/gemma4_e2b_gguf` | `llama.cpp` text | Tool-capable if the GGUF template is preserved and routed through llama.cpp chat-template helpers. | Same Gemma 4 convention, exposed to Motlie through an OpenAI-compatible JSON bridge. |
| `google/gemma4_e4b_gguf` | `llama.cpp` text | ToolUse advertised after model-specific GGUF template smoke. | Same Gemma 4 convention; GGUF bundles are validated per model because the embedded chat template owns the concrete tool markers. |
| `qwen/qwen3_4b` | `mistral.rs` text | ToolUse advertised after adapter unit tests. | Qwen3/Hermes-style tools, `<tool_call>`, and `<tool_response>` through the model chat template. |
| `qwen/qwen3_4b_gguf` | `llama.cpp` text | Tool-capable if the GGUF template is preserved and routed through llama.cpp chat-template helpers. | Same Qwen3/Hermes convention, exposed to Motlie through an OpenAI-compatible JSON bridge. |
| `qwen/qwen3_6_27b_gguf` | `llama.cpp` text | Tool-capable after GGUF template validation and adapter support. | Qwen-family tool convention plus Qwen3.6 thinking-mode handling. |

Capability gating policy: safetensors bundles on `mistral.rs` may advertise `ToolUse`
when the shared adapter path is covered because the backend owns the generic tool
request/response mapping. GGUF bundles on `llama.cpp` require model-specific
chat-template smoke before advertising `ToolUse` because the artifact's embedded
template defines the concrete tool and thinking markers.

## Design Principle

The public Motlie contract should use an OpenAI-compatible logical shape, not OpenAI-specific transport types. That shape is the smallest common denominator already understood by `mistral.rs`, `llama.cpp`, Gemma 4 templates, and Qwen templates:

- tool definitions are named functions with JSON Schema parameters
- assistant turns may carry one or more tool calls
- tool-result turns correlate back to an assistant tool call
- the final assistant turn may carry text and no tool calls

Backend adapters translate the common shape into the concrete family convention:

- Gemma 4: template `tools`, assistant `tool_calls`, and tool responses as the template expects.
- Qwen3/Qwen3.6: Qwen/Hermes `<tools>`, `<tool_call>`, and `<tool_response>`.
- `mistral.rs`: native `Tool`, `ToolChoice`, request fields, response `tool_calls`, and template preprocessing.
- `llama.cpp`: `OpenAIChatTemplateParams`, generated tool grammar, and OpenAI-compatible response parsing.

## Dispatch Architecture

The implementation deliberately separates three mechanisms that are often
collapsed under the word "dispatch":

| Term | Mechanism | Uses `Box<dyn>`? | Motlie use |
| --- | --- | --- | --- |
| Static dispatch | Compile-time monomorphization | No | Local typed Rust tools via `ToolList` tuple recursion |
| Dynamic dispatch | Trait-object vtable | Yes | Avoided for tool dispatch in `libs/` |
| Data routing | Runtime match or lookup on values | No | Future MCP server/tool selection by name |

Typed Rust tools are a type-level problem: every tool has its own `Args`,
`Output`, and `Error` associated types. Motlie models that with a recursive
`ToolList` over tuples, so the compiler monomorphizes each tool call path.
`ToolListError` may still box a source error to preserve arbitrary tool error
chains; that is error reporting, not tool dispatch.

MCP tools are a data-level routing problem: every MCP tool crosses the same
JSON-RPC `tools/call` boundary with JSON arguments and content results. PR #279
only adds the `Mcp` scaffolding. Concrete stdio/SSE/HTTP transports,
initialize/shutdown lifecycle, JSON-RPC framing, capability negotiation, and
permission policy are explicitly future work for issue #284.

## Proposed Core API

The public tool API is typed at the Rust boundary and raw JSON only at the model
adapter boundary. Callers bind Rust argument/output types, `libs/model`
generates JSON Schema from those argument types, and model-emitted arguments are
kept lossless until the caller executes the selected tool.

### Tool Spec

```rust
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolName(String);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolInputSchema {
    raw_json_schema: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolSpec {
    pub name: ToolName,
    pub description: String,
    pub input_schema: ToolInputSchema,
}

impl ToolSpec {
    pub fn from_args<T: schemars::JsonSchema>(
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Result<Self, ToolSchemaError>;

    pub fn from_json_schema(
        name: impl Into<String>,
        description: impl Into<String>,
        raw_json_schema: impl AsRef<str>,
    ) -> Result<Self, ToolSchemaError>;
}
```

Rules:

- callers binding Rust functions should normally use `ToolSpec::from_args::<T>()`
- imported schemas should use the explicit `from_json_schema(...)` escape hatch
- `ToolName::new(...)` validates the OpenAI-compatible name shape once in the shared contract: non-empty, at most 64 characters, and ASCII letters, digits, `_`, or `-`
- `ToolInputSchema` validates that the schema document is JSON and describes an object-shaped argument payload before it reaches a backend
- adapters serialize `ToolSpec` to the model/backend-specific JSON shape
- the contract does not execute functions through `ToolSpec`; execution is caller-owned, with static `motlie_models::ToolList` helpers available for local typed Rust tools

### Tool Choice

```rust
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Named(ToolName),
}
```

Rules:

- `Auto` lets the model decide.
- `None` suppresses tool calls even if tools are present.
- `Required` requires at least one tool call if the backend/template can enforce it.
- `Named` requires a specific tool if the backend/template can enforce it; callers can use `ToolChoice::named(...)` to validate the name while constructing the choice.
- Unsupported enforcement should return `ModelError::InvalidConfiguration`, not silently downgrade.

### Tool Call

```rust
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub id: ToolCallId,
    pub name: ToolName,
    pub arguments: ToolArguments,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolArguments {
    raw_json: Box<serde_json::value::RawValue>,
}

impl ToolArguments {
    pub fn from_json_str(raw_json: impl Into<String>) -> Result<Self, ToolArgumentError>;
    pub fn from_serializable<T: serde::Serialize>(value: &T) -> Result<Self, ToolArgumentError>;
    pub fn parse<T: serde::de::DeserializeOwned>(&self) -> Result<T, ToolArgumentError>;
    pub fn raw_json(&self) -> &serde_json::value::RawValue;
    pub fn raw_json_str(&self) -> &str;
}
```

Rules:

- `id` is a stable per-response correlation value represented as `ToolCallId`, not a bare string. Empty ids are rejected at construction.
- `name` should match a registered `ToolSpec.name`; unknown tools still surface so the caller can reject them explicitly.
- `ToolArguments` preserves the exact model/backend argument JSON while making typed parsing the normal Rust path.
- `ToolArguments` rejects non-object JSON payloads; Rust tool arguments should be named structs.
- If argument JSON cannot be parsed or deserialized into the requested Rust type, the caller receives `ToolArgumentError`; the `motlie_models::ToolList` helper surfaces that through `ToolListError::InvalidArguments`.

### Typed Rust Tool Binding

`libs/model` keeps model invocation separate from tool execution. The core crate defines the typed `Tool` trait and portable model-facing vocabulary. The curated examples use `motlie_models::ToolList`, which collects model-facing `ToolSpec`s and executes structured `ToolCall`s through statically dispatched tuple recursion after the model asks for a tool.

Core binding trait:

```rust
pub trait Tool: Send + Sync + 'static {
    type Args: serde::de::DeserializeOwned + schemars::JsonSchema + Send + 'static;
    type Output: serde::Serialize + Send + 'static;
    type Error: std::error::Error + Send + Sync + 'static;

    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;

    fn call(
        &self,
        args: Self::Args,
    ) -> impl std::future::Future<Output = Result<Self::Output, Self::Error>> + Send;

    fn spec(&self) -> Result<ToolSpec, ToolSchemaError> {
        ToolSpec::from_args::<Self::Args>(self.name(), self.description())
    }
}
```

Static tool-list helper, provided by `motlie-models`:

```rust
pub trait ToolList: Send + Sync {
    fn collect_specs(&self, out: &mut Vec<ToolSpec>) -> Result<(), ToolListError>;
    fn dispatch(
        &self,
        call: ToolCall,
    ) -> impl Future<Output = Result<ToolDispatch, ToolListError>> + Send + '_;
}

pub enum ToolDispatch {
    Handled(ChatMessage),
    NotMine(ToolCall),
}

let tools = motlie_models::tool_list!(WeatherTool, MathTool);
```

Binding an existing function by delegating from a concrete tool:

```rust
use motlie_model::Tool;
use std::future::Future;

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct WeatherArgs {
    city: String,
    units: Units,
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
enum Units {
    Fahrenheit,
    Celsius,
}

#[derive(serde::Serialize)]
struct WeatherOutput {
    temperature: f32,
    summary: String,
}

#[derive(Debug, thiserror::Error)]
#[error("weather lookup failed")]
struct WeatherError;

async fn get_weather(args: WeatherArgs) -> Result<WeatherOutput, WeatherError> {
    Ok(WeatherOutput {
        temperature: 72.0,
        summary: format!("clear in {}", args.city),
    })
}

struct WeatherTool;

impl Tool for WeatherTool {
    type Args = WeatherArgs;
    type Output = WeatherOutput;
    type Error = WeatherError;

    fn name(&self) -> &'static str { "get_weather" }
    fn description(&self) -> &'static str { "Get current weather for a city." }
    fn call(&self, args: Self::Args) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send {
        get_weather(args)
    }
}
```

Binding a closure uses the same pattern: the caller owns a concrete tool struct
that stores the closure type, so the dispatch remains static:

```rust
use motlie_model::Tool;
use std::future::Future;

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct AddArgs {
    left: i64,
    right: i64,
}

#[derive(serde::Serialize)]
struct AddOutput {
    value: i64,
}

#[derive(Debug, thiserror::Error)]
#[error("add failed")]
struct AddError;

struct AddTool<F> {
    f: F,
}

impl<F, Fut> Tool for AddTool<F>
where
    F: Fn(AddArgs) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<AddOutput, AddError>> + Send,
{
    type Args = AddArgs;
    type Output = AddOutput;
    type Error = AddError;

    fn name(&self) -> &'static str { "add" }
    fn description(&self) -> &'static str { "Add two integers." }
    fn call(&self, args: Self::Args) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send {
        (self.f)(args)
    }
}

let tools = motlie_models::tool_list!(AddTool {
    f: |args: AddArgs| async move {
        Ok(AddOutput { value: args.left + args.right })
    },
});
```

Using the tool list with chat:

```rust
let tool_specs = tools.specs()?;

let response = chat.generate(ChatRequest {
    messages,
    tools: tool_specs,
    tool_choice: Some(ToolChoice::Auto),
    ..Default::default()
}).await?;

for call in response.tool_calls {
    messages.push(ChatMessage::assistant_tool_calls(vec![call.clone()]));
    match tools.dispatch(call).await? {
        ToolDispatch::Handled(message) => messages.push(message),
        ToolDispatch::NotMine(call) => {
            // Future per #284: iterate mcp_servers here.
            return Err(AppError::UnknownTool(call.name));
        }
    }
}
```

### Chat Role and Message

```rust
pub enum ChatRole {
    Assistant,
    System,
    Tool,
    User,
}

pub struct ChatMessage {
    pub role: ChatRole,
    pub content: Vec<ContentPart>,
    pub name: Option<ToolName>,
    pub tool_call_id: Option<ToolCallId>,
    pub tool_calls: Vec<ToolCall>,
    pub reasoning: Option<String>,
}
```

Rules:

- Existing constructors keep working and default the new fields to empty/`None`.
- `Assistant` messages may include `tool_calls` when callers replay prior assistant tool-call turns.
- `Tool` messages must provide `tool_call_id`; `name` should be set when known. `ChatMessage::tool_result(...)` validates both fields, and `tool_result_parts(...)` accepts already validated newtypes.
- `System` and `User` messages must not carry `tool_calls`.
- Field-public hand-built messages should pass `validate_tool_metadata()` before backend serialization.
- `reasoning` is optional transcript metadata for model families that expose thinking content. It must not be fed to backends that cannot represent it unless the adapter explicitly supports it.

### Chat Request

```rust
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    pub params: GenerationParams,
    pub tools: Vec<ToolSpec>,
    pub tool_choice: Option<ToolChoice>,
    pub thinking: Option<ThinkingMode>,
}
```

Rules:

- Empty `tools` means ordinary chat.
- `tool_choice` without tools is invalid unless it is `None`.
- Backends that do not support tools must reject requests with non-empty `tools` using `UnsupportedCapability(CapabilityKind::ToolUse)`.
- `thinking` is an optional per-request override for model families with thinking/reasoning modes. Backends without a thinking control may accept and ignore it, but must document that behavior.

### Chat Response

```rust
pub struct ChatResponse {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: Option<ChatFinishReason>,
    pub reasoning: Option<String>,
    pub usage: Option<GenerationUsage>,
}

pub enum ChatFinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    Other(String),
}

pub struct GenerationUsage {
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub total_tokens: Option<u32>,
}
```

Rules:

- Existing text callers can continue reading `content`.
- `finish_reason == Some(ToolCalls)` should be set when the assistant is asking the caller to execute tools.
- `tool_calls` is structured output. Callers should not parse `content` for tool calls.
- `reasoning` carries thinking/reasoning text when the backend can split it from the final assistant content.
- `usage` mirrors request-local token accounting. Existing handle-level aggregate metrics stay in `BundleHandle::metric_snapshot()`.
- Backend-specific raw message blobs are intentionally not part of `ChatResponse`; adapters should use logs/tests for diagnostics and keep the public response portable.

## Capability Model

Add:

```rust
pub enum CapabilityKind {
    // existing variants...
    ToolUse,
}

impl CapabilityDescriptor {
    pub fn tool_use() -> Self { ... }
}
```

`ToolUse` is descriptive and attached to the chat surface. It does not imply a separate executable trait. A bundle may support `Chat` without `ToolUse`; tool-bearing requests to such a bundle return `UnsupportedCapability(CapabilityKind::ToolUse)`.

Suggested descriptor:

- inputs: `Text`, `StructuredJson`
- outputs: `Text`, `StructuredJson`
- interaction: `MultiTurn`

Curated bundle descriptors should include `ToolUse` only when that concrete backend/artifact path supports round-trip tool calls.

## Tool Loop

The caller-owned loop is:

```rust
let request = ChatRequest {
    messages,
    tools,
    tool_choice: Some(ToolChoice::Auto),
    params,
};

let response = chat.generate(request).await?;

if !response.tool_calls.is_empty() {
    let assistant_turn = ChatMessage::assistant_tool_calls(response.tool_calls.clone());
    let tool_turns = execute_tools(response.tool_calls).await?;
    let followup = ChatRequest {
        messages: [prior_messages, vec![assistant_turn], tool_turns].concat(),
        tools,
        tool_choice: Some(ToolChoice::Auto),
        params,
    };
    let final_response = chat.generate(followup).await?;
}
```

The helper constructors should make this ergonomic, but the contract should keep execution explicit.

## Backend Adaptation

### `mistral.rs`

Current implementation status: the Motlie `mistral.rs` wrappers now map the common tool contract into the native builder and response types for the safetensors Qwen3 text and Gemma 4 multimodal paths.

Implemented adaptation:

- Route text and multimodal wrappers through the shared `MistralProfile` runtime layer for backend startup, handle plumbing, metrics, request construction, and response mapping.
- Map `ToolSpec` to `mistralrs::Tool`.
- Map `ToolChoice` to `mistralrs::ToolChoice`.
- Map `ChatRole::Tool` and assistant `tool_calls` into the native request message shape.
- Preserve multimodal image handling in `MistralMultimodalHandle`.
- Extract `choice.message.tool_calls` into `ChatResponse.tool_calls`.
- Map upstream finish reason and usage into the new response fields.
- Let `mistral.rs` handle Gemma 4 tool-template preprocessing instead of reimplementing it in Motlie.

Backend-specific limitations:

- `ToolChoice::Required` returns `ModelError::InvalidConfiguration` because the current `mistral.rs` builder API exposes `Auto`, `None`, and specific-tool choice, but not required-tool enforcement.
- Tool-result `ChatMessage::name` remains part of the Motlie transcript, but the current `mistral.rs` tool-message builder accepts only content and `tool_call_id`.

### `llama.cpp`

The GGUF adapter now keeps current hand-formatted prompts for ordinary no-tool chat and routes tool-bearing chat through llama.cpp's OpenAI-compatible chat-template path.

- `format_qwen3_prompt`
- `format_gemma4_prompt`

The hand-formatting path cannot support tool definitions, tool results, or model-template-specific tool rendering, so it is bypassed when `ChatRequest::requires_tool_use()` is true.

Implemented adaptation:

- Convert Motlie messages to OpenAI-compatible JSON messages.
- Convert Motlie `ToolSpec` values to OpenAI-compatible JSON tools.
- Use `LlamaModel::chat_template(None)` to load the GGUF template.
- Use `apply_chat_template_oaicompat(...)` with `OpenAIChatTemplateParams`.
- Use the returned grammar, when present, during generation.
- Use `ChatTemplateResult::parse_response_oaicompat(...)` to parse tool calls from generated text.
- Keep rejecting image parts on current text-only GGUF handles until mmproj support lands.

Backend-specific limitations:

- `ToolChoice::Named` returns `ModelError::InvalidConfiguration` because upstream's OpenAI-compatible parser accepts only `auto`, `none`, and `required`.
- GGUF descriptors should not advertise `ToolUse` until local artifact smoke tests confirm that each selected GGUF preserves a usable tool-aware chat template.
- Local status on 2026-05-13: the Qwen3 GGUF `--tool-demo` smoke was attempted against the default curated cache and failed before model load because `Qwen/Qwen3-4B-GGUF` artifacts were not cached. This keeps the remaining GGUF gate at artifact validation rather than API/adapter design.

## Model-Family Notes

### Gemma 4

Gemma 4 should be treated as a native tool-calling template family, not a plain text prompt family. Adapters should use structured messages and template rendering so:

- `tools` are rendered by the model template
- assistant `tool_calls` are represented structurally on replay
- tool results are represented in the form expected by the Gemma template
- Gemma-specific tool-call tokens remain an adapter concern

### Qwen3

Qwen3 should use the Qwen/Hermes convention:

- tools listed as function schemas
- assistant calls represented as structured tool calls
- template rendering may produce `<tool_call>...</tool_call>`
- tool results render through `<tool_response>...</tool_response>`

The common Motlie contract should not expose those XML-like tokens.

### Qwen3.6

Qwen3.6 shares the Qwen-family tool surface, but it also has thinking-mode behavior. The contract separates `reasoning` from `content` so callers can preserve or drop thinking content by policy without confusing it with tool-call JSON.

## Validation Strategy

Contract tests:

- request constructors preserve backwards-compatible text-only behavior
- `ChatRole::Tool` and assistant `tool_calls` round-trip through owned values
- invalid tool-choice combinations are rejected
- `Capabilities::tool_use()` canonicalizes and reports support correctly

Backend unit tests:

- Qwen3/Gemma 4 shared `mistral.rs` tool schema and tool-call conversion maps to native request/response types
- malformed model tool JSON returns a typed backend error
- no-tools chat still behaves as before
- image content remains accepted only by the Gemma 4 multimodal path

Integration/smoke tests:

- safetensors Qwen3 4B returns a structured tool call for a simple deterministic tool prompt
- safetensors Gemma 4 E2B returns a structured tool call for the same prompt
- GGUF Qwen3 4B and Gemma 4 E2B validate that the selected artifact has a usable chat template
- Qwen3.6 GGUF validates template availability and thinking/tool-call separation before advertising `ToolUse`

## Compatibility

The API changes are additive at the field/enum level but require source updates where exhaustive matching exists:

- `ChatRole` gains `Tool`, so backend role matchers must be updated.
- `ChatRequest` gains new fields; constructors/defaults must be used to minimize call-site churn.
- `ChatResponse` gains new fields; existing direct struct literals must be updated.
- `ToolArguments` is lossless raw JSON, so response/message structs that carry tool calls use `PartialEq` rather than `Eq`.

No vendored dependency changes are required or expected.

## Open Questions
- Should Motlie expose a `parallel_tool_calls` request option in v1, or leave it backend-defaulted until a caller needs policy control?
- Should `reasoning` be one field on `ChatMessage`/`ChatResponse`, or should it become a `ContentPart::Reasoning` variant later?
- Should `ToolUse` also imply `ContentKind::StructuredJson` output on `Chat`, or remain a separate descriptive capability only?
