# `chat_tool_binding`

This example demonstrates the common typed tool-binding API without loading a
model backend. It covers the static caller-owned shape:

- `WeatherTool` delegates to the existing async Rust function `get_weather`
- `EvaluateMathExpressionTool` delegates to the typed CEL-backed math function
- `tool_list!(...)` builds the statically dispatched local tool set
- the E4B recommended generation params and system prompt are merged into an
  effective request shape without starting an LLM

Run it with:

```bash
cargo run -p motlie-models --example chat_tool_binding --no-default-features
```

Select the recommendation source explicitly:

```bash
cargo run -p motlie-models --no-default-features --example chat_tool_binding -- --model=gemma4-e4b
```

To exercise the real Gemma 4 E4B GGUF spec recommendations without loading an
LLM, enable the GGUF feature:

```bash
cargo run -p motlie-models --no-default-features --features model-gemma4-e4b-gguf \
  --example chat_tool_binding
```

The example builds a `ChatRequest` from `tools.specs()`, simulates assistant
`ToolCall`s for weather in three cities plus a math-expression average, and
converts each `ToolDispatch::Handled` result back into a `ChatRole::Tool`
message. The `ToolDispatch::NotMine` arm is where caller code can route to
future MCP servers from issue `#284`.

The backend-specific examples that should get live model tool-call smoke tests
after adapter wiring are:

- `chat_mistral_qwen3`: Qwen3 safetensors through `mistral.rs`
- `chat_multimodal_gemma4`: Gemma 4 safetensors through `mistral.rs`
- `chat_gguf_gwen3_gemma4`: Qwen3 and Gemma 4 GGUF through `llama.cpp`
- `chat_multimodal_qwen3_6_27b`: Qwen3.6 GGUF through `llama.cpp`
