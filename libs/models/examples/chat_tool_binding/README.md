# `chat_tool_binding`

This example demonstrates the common typed tool-binding API without loading a
model backend. It covers both supported caller styles:

- binding an existing async Rust function with typed arguments and output
- binding an inline async closure with typed arguments and output

Run it with:

```bash
cargo run -p motlie-models --example chat_tool_binding --no-default-features
```

The example registers two tools in `ToolRegistry`, builds a `ChatRequest` from
`registry.specs()`, simulates assistant `ToolCall`s, and converts each executed
tool result back into a `ChatRole::Tool` message.

The backend-specific examples that should get live model tool-call smoke tests
after adapter wiring are:

- `chat_mistral_qwen3`: Qwen3 safetensors through `mistral.rs`
- `chat_multimodal_gemma4`: Gemma 4 safetensors through `mistral.rs`
- `chat_gguf_gwen3_gemma4`: Qwen3 and Gemma 4 GGUF through `llama.cpp`
- `chat_multimodal_qwen3_6_27b`: Qwen3.6 GGUF through `llama.cpp`
