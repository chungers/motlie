# `motlie-models` `chat_gguf_gwen3_gemma4` Example — llama.cpp Backend (GGUF)

This example demonstrates chat generation via the **llama.cpp** backend using
GGUF-quantized weights. It supports switching between two models at runtime:

- **Qwen3 4B** (default) — `qwen/qwen3_4b_gguf`
- **Gemma 4 E2B-it** — `google/gemma4_e2b_gguf`

## Weight Format Compatibility

| Backend | Weight format | Qwen3-4B repo | Gemma4 E2B repo |
|---------|--------------|---------------|-----------------|
| **mistral.rs** (`chat_mistral_qwen3` / `chat_multimodal_gemma4`) | safetensors | `Qwen/Qwen3-4B` | `google/gemma-4-E2B-it` |
| **llama.cpp** (`chat_gguf_gwen3_gemma4`) | GGUF | `Qwen/Qwen3-4B-GGUF` | `bartowski/gemma-4-E2B-it-GGUF` |

The two weight formats are **not interchangeable**. Each backend requires its
own artifact set. However, both target the identical upstream model
architectures and produce equivalent inference results at comparable
quantization levels.

### Quantization mapping

| `--precision` | mistral.rs (ISQ) | llama.cpp (GGUF) |
|---------------|------------------|------------------|
| `q4` (default) | ISQ Q4 | Q4_K_M |
| `q8` | ISQ Q8 | Q8_0 |
| `f16` / `f32` | F32 | F16 |

## What It Demonstrates

1. Direct curated enum selection through `ChatModels::Qwen3_4B_Gguf`
2. Runtime model switching through `--chat=google/gemma4_e2b_gguf`
3. GGUF quantization control (Q4_K_M default, Q8_0, or F16)
4. Descriptor/capability introspection showing `Chat` + `Completion`
5. Optional curated artifact download via `--download-artifacts`
6. Local-only startup through `ArtifactPolicy::LocalOnly`
7. Single-turn and multi-turn chat
8. Text completion
9. Latency reporting for startup and each request path
10. Process/memory snapshots before startup, after startup, and after each request
11. Handle-level model metrics after startup and each request

## Step 1: Download GGUF Artifacts

Pre-download the Qwen3-4B GGUF artifacts:

```sh
cargo run -p motlie-models --no-default-features --features model-qwen3-4b-gguf \
  --bin motlie-models-download -- qwen3_4b_gguf
```

Or let the example perform the curated download first:

```sh
cargo run -p motlie-models --no-default-features --features model-qwen3-4b-gguf \
  --example chat_gguf_gwen3_gemma4 -- --download-artifacts "What is Rust's ownership model?"
```

For Gemma 4, enable both features:

```sh
cargo run -p motlie-models --no-default-features \
  --features model-qwen3-4b-gguf,model-gemma4-e2b-gguf \
  --example chat_gguf_gwen3_gemma4 -- --download-artifacts --chat=google/gemma4_e2b_gguf \
  "What is Rust's ownership model?"
```

## Step 2: Run the Example

Default path (Qwen3 4B, GGUF Q4_K_M, using existing local artifacts):

```sh
cargo run -p motlie-models --no-default-features --features model-qwen3-4b-gguf \
  --example chat_gguf_gwen3_gemma4 -- "What is Rust's ownership model?"
```

Switch to Gemma 4 (requires both features enabled):

```sh
cargo run -p motlie-models --no-default-features \
  --features model-qwen3-4b-gguf,model-gemma4-e2b-gguf \
  --example chat_gguf_gwen3_gemma4 -- --chat=google/gemma4_e2b_gguf \
  "Summarize ownership in one paragraph"
```

Full precision (F16, no quantization):

```sh
cargo run -p motlie-models --no-default-features --features model-qwen3-4b-gguf \
  --example chat_gguf_gwen3_gemma4 -- --precision=f16 "What is Rust's ownership model?"
```

## Preconditions

- Pre-downloaded GGUF artifacts in the curated artifact root, or `--download-artifacts`
- Sufficient memory for the chosen precision and model size
- The `model-qwen3-4b-gguf` feature (minimum); add `model-gemma4-e2b-gguf` for model switching

## Source

- Example entrypoint: [main.rs](main.rs)
- Qwen3 GGUF bundle: `libs/models/src/chat/qwen3_4b_gguf.rs`
- Gemma4 GGUF bundle: `libs/models/src/chat/gemma4_e2b_gguf.rs`
- llama.cpp backend: `libs/model/backends/llama_cpp/`
