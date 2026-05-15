# `motlie-models` `chat_multimodal_gemma4` Example

This example demonstrates the single Gemma 4 E2B-it multimodal chat flow for this bundle. It supports both:

1. pure local-only startup using pre-downloaded curated artifacts
2. an optional in-example download step via `--download-artifacts`

## What It Demonstrates

1. Direct curated enum selection through `ChatModels::Gemma4E2B`
2. ISQ quantization control (Q4 default, Q8, or F32)
3. Descriptor/capability introspection showing `Chat` + `Vision` + `ToolUse`
4. Catalog self-check showing exactly one compiled curated bundle
5. Optional curated artifact download via `--download-artifacts`
6. Local-only startup through `ArtifactPolicy::LocalOnly`
7. Text-only chat through the multimodal runtime path
8. Image + text chat when `--image=/path/to/image` is provided
9. Optional `--tool-demo` path for end-to-end local tool calling
10. Latency reporting for startup, text-only chat, image+text chat, and tool calls
11. Process/memory snapshots before startup, after startup, and after each request path
12. Handle-level model metrics after startup and each request path

## Step 1: Download Artifacts

You can pre-download artifacts out of band with the standalone downloader:

```sh
export HF_TOKEN=...
cargo run -p motlie-models --no-default-features --features model-gemma4-e2b --bin motlie-models-download -- --hf-token-env HF_TOKEN gemma4_e2b
```

Or let the example perform the curated download first:

```sh
export HF_TOKEN=...
cargo run -p motlie-models --no-default-features --features model-gemma4-e2b --example chat_multimodal_gemma4 -- --download-artifacts "What is Rust's ownership model?"
```

## Step 2: Run the Example

Default path (direct enum, text-only, ISQ Q4, using existing local artifacts only):

```sh
cargo run -p motlie-models --no-default-features --features model-gemma4-e2b --example chat_multimodal_gemma4 -- "What is Rust's ownership model?"
```

Image + text:

```sh
cargo run -p motlie-models --no-default-features --features model-gemma4-e2b --example chat_multimodal_gemma4 -- --image=photo.jpg "Describe this image"
```

Full precision (no quantization):

```sh
cargo run -p motlie-models --no-default-features --features model-gemma4-e2b --example chat_multimodal_gemma4 -- --precision=f32 "What is Rust's ownership model?"
```

Tool-calling smoke only:

```sh
cargo run -p motlie-models --no-default-features --features model-gemma4-e2b --example chat_multimodal_gemma4 -- --tool-demo-only "What is Rust?"
```

The tool demo registers a local `get_weather` Rust function, sends its generated
schema to the model, executes the model-requested tool call through
`ToolRegistry`, appends the tool-result message, and asks the model for a final
answer.

## Preconditions

- Either pre-downloaded Gemma 4 E2B-it artifacts in the curated artifact root, or `--download-artifacts` plus the required Hugging Face access in the current environment
- Sufficient memory for the chosen precision
- The example must be built with only the Gemma 4 bundle enabled, which is why the commands above use `--no-default-features --features model-gemma4-e2b`

## Source

- Example entrypoint: [main.rs](main.rs)
- Bundle definition: `libs/models/src/chat/gemma4_e2b.rs`
