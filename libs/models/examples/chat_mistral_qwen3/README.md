# `motlie-models` `chat_mistral_qwen3` Example

This example demonstrates the curated Qwen3-4B chat bundle with ISQ quantization.

## What It Demonstrates

1. Direct curated enum selection through `ChatModels::Qwen3_4B`
2. Optional parser-driven selection through `--chat=qwen/qwen3_4b`
3. ISQ quantization control (Q4 default, Q8, or F32)
4. Descriptor/capability introspection
5. Local-only startup through `ArtifactPolicy::LocalOnly`
6. Single-turn chat with system prompt + user message
7. Multi-turn follow-up demonstrating message history
8. Completion path (delegates to single-turn chat)
9. Latency reporting for startup and each request
10. Process/memory snapshots before startup, after startup, and after each request path
11. Handle-level model metrics after startup and each request path
12. Optional `--tool-demo` path for end-to-end local tool calling

## Run

Default path (direct enum, ISQ Q4):

```sh
cargo run -p motlie-models --no-default-features --features model-qwen3-4b --example chat_mistral_qwen3 -- "What is Rust's ownership model?"
```

Parser-driven selector:

```sh
cargo run -p motlie-models --no-default-features --features model-qwen3-4b --example chat_mistral_qwen3 -- --chat=qwen/qwen3_4b "Explain borrow checking"
```

Full precision (no quantization):

```sh
cargo run -p motlie-models --no-default-features --features model-qwen3-4b --example chat_mistral_qwen3 -- --precision=f32 "What is Rust's ownership model?"
```

Pre-download artifacts:

```sh
cargo run -p motlie-models --no-default-features --features model-qwen3-4b --example chat_mistral_qwen3 -- --download-artifacts "What is Rust?"
```

Tool-calling loop:

```sh
cargo run -p motlie-models --no-default-features --features model-qwen3-4b --example chat_mistral_qwen3 -- --tool-demo "What is Rust?"
```

Tool-calling smoke only:

```sh
cargo run -p motlie-models --no-default-features --features model-qwen3-4b --example chat_mistral_qwen3 -- --tool-demo-only "What is Rust?"
```

The tool demo registers `get_weather` and `evaluate_math_expression`, sends
their generated schemas to the model, executes model-requested tool calls
through `ToolRegistry`, appends each tool-result message, and lets the model
combine a plain Rust explanation with a weather-derived average temperature.

## Preconditions

- Pre-downloaded Qwen3-4B artifacts in the curated artifact root, OR use `--download-artifacts`
- The example expects a single-bundle build and prints `catalog-entry-count: 1`; use `--no-default-features --features model-qwen3-4b` as shown above
- Sufficient memory: ~2.5GB for Q4, ~4.5GB for Q8, ~8GB for F32
- For authenticated download: pre-download with the artifact utility:
  ```sh
  export HF_TOKEN=...
  cargo run -p motlie-models --no-default-features --features model-qwen3-4b --bin motlie-models-download -- --hf-token-env HF_TOKEN qwen3_4b
  ```

## Source

- Example entrypoint: [main.rs](main.rs)
- Bundle definition: [qwen3_4b.rs](/libs/models/src/chat/qwen3_4b.rs)
