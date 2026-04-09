# `motlie-models` v0.3 Example

This example demonstrates the curated Gemma 4 E2B-it multimodal chat bundle.

## What It Demonstrates

1. Direct curated enum selection through `ChatModels::Gemma4E2B`
2. Optional parser-driven selection through `--chat=google/gemma4_e2b`
3. ISQ quantization control (Q4 default, Q8, or F32)
4. Descriptor/capability introspection showing `Chat` + `Vision`
5. Local-only startup through `ArtifactPolicy::LocalOnly`
6. Text-only chat through the multimodal runtime path
7. Image + text chat when `--image=/path/to/image` is provided
8. Latency reporting for startup, text-only chat, and image+text chat

## Run

Default path (direct enum, text-only, ISQ Q4):

```sh
cargo run -p motlie-models --example models_v0_3 -- "What is Rust's ownership model?"
```

Parser-driven selector:

```sh
cargo run -p motlie-models --example models_v0_3 -- --chat=google/gemma4_e2b "Summarize ownership in one paragraph"
```

Image + text:

```sh
cargo run -p motlie-models --example models_v0_3 -- --image=photo.jpg "Describe this image"
```

Full precision (no quantization):

```sh
cargo run -p motlie-models --example models_v0_3 -- --precision=f32 "What is Rust's ownership model?"
```

Pre-download artifacts:

```sh
cargo run -p motlie-models --example models_v0_3 -- --download-artifacts "What is Rust?"
```

## Preconditions

- Pre-downloaded Gemma 4 E2B-it artifacts in the curated artifact root, OR use `--download-artifacts`
- Sufficient memory for the chosen precision
- For authenticated download: pre-download with the artifact utility:
  ```sh
  export HF_TOKEN=...
  cargo run -p motlie-models --bin motlie-models-download -- --hf-token-env HF_TOKEN gemma4_e2b
  ```

## Source

- Example entrypoint: [main.rs](main.rs)
- Bundle definition: [/tmp/motlie-issue142/libs/models/src/chat/gemma4_e2b.rs](/tmp/motlie-issue142/libs/models/src/chat/gemma4_e2b.rs)
