# `motlie-models` `chat_multimodal_gemma4` Example

This example demonstrates the Gemma 4 multimodal chat flow for the compiled
curated Gemma 4 safetensors variants: E2B and E4B. It supports both:

1. pure local-only startup using pre-downloaded curated artifacts
2. an optional in-example download step via `--download-artifacts`

> 2026-06-04 21:38 PDT @gemma4-cdx: added the `--model` selector so the same
> live example can exercise `gemma4-e2b` or `gemma4-e4b`.

## What It Demonstrates

1. Direct curated enum selection through the Gemma 4 `ChatModels` variants
2. ISQ quantization control using each bundle's recommendation, with Q4, Q8, or F32 overrides
3. Descriptor/capability introspection showing `Chat` + `Vision` + `ToolUse`
4. Catalog self-check for the selected compiled curated bundle
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

Default path for E2B (direct enum, text-only, ISQ Q4, using existing local artifacts only):

```sh
cargo run -p motlie-models --no-default-features --features model-gemma4-e2b --example chat_multimodal_gemma4 -- "What is Rust's ownership model?"
```

Select a specific Gemma 4 variant with `--model`. When multiple Gemma 4
features are compiled in, the default is E2B, then E4B.

```sh
cargo run -p motlie-models --no-default-features --features model-gemma4-e2b,model-gemma4-e4b --example chat_multimodal_gemma4 -- --model=gemma4-e4b "What is Rust's ownership model?"
```

Alternate curated artifact root:

```sh
cargo run -p motlie-models --no-default-features --features model-gemma4-e2b --example chat_multimodal_gemma4 -- --artifact-root /path/to/hf-cache "What is Rust's ownership model?"
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

The tool demo registers `get_weather` and `evaluate_math_expression`, sends
their generated schemas to the model, executes model-requested tool calls
through static `ToolList` dispatch, appends each tool-result message, and lets the model
combine a plain Rust explanation with a weather-derived average temperature.

## Preconditions

- Either pre-downloaded Gemma 4 artifacts for the selected variant in the curated artifact root, `--artifact-root /path/to/hf-cache`, or `--download-artifacts` plus the required Hugging Face access in the current environment
- Sufficient memory for the chosen precision
- The example must be built with at least one Gemma 4 feature enabled: `model-gemma4-e2b` or `model-gemma4-e4b`

## Source

- Example entrypoint: [main.rs](main.rs)
- Bundle definitions: `libs/models/src/chat/gemma4_e2b.rs`, `libs/models/src/chat/gemma4_e4b.rs`
