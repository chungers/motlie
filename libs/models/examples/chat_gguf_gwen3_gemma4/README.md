# `motlie-models` `chat_gguf_gwen3_gemma4` Example — llama.cpp Backend (GGUF)

This example demonstrates chat generation via the **llama.cpp** backend using
GGUF-quantized weights. It supports switching between these models at runtime:

- **Qwen3 4B** (default) — `qwen/qwen3_4b_gguf`
- **Gemma 4 E2B-it** — `google/gemma4_e2b_gguf`
- **Gemma 4 E4B-it** — `google/gemma4_e4b_gguf`

## Weight Format Compatibility

| Backend | Weight format | Qwen3-4B repo | Gemma4 E2B repo | Gemma4 E4B repo |
|---------|---------------|---------------|------------------|------------------|
| **mistral.rs** (`chat_mistral_qwen3` / `chat_multimodal_gemma4`) | safetensors | `Qwen/Qwen3-4B` | `google/gemma-4-E2B-it` | `google/gemma-4-E4B-it` |
| **llama.cpp** (`chat_gguf_gwen3_gemma4`) | GGUF | `Qwen/Qwen3-4B-GGUF` | `unsloth/gemma-4-E2B-it-GGUF` | `unsloth/gemma-4-E4B-it-GGUF` |

The two weight formats are **not interchangeable**. Each backend requires its
own artifact set. However, both target the identical upstream model
architectures and produce equivalent inference results at comparable
quantization levels.

### Quantization mapping

| `--precision` | mistral.rs (ISQ) | llama.cpp (GGUF) |
|---------------|------------------|------------------|
| `q4` | ISQ Q4 | Q4_K_M |
| `q5` | n/a | Q5_K_M |
| `q8` | ISQ Q8 | Q8_0 |
| `f16` / `f32` | F32 | F16 |

When `--precision` is omitted, the example uses each GGUF spec's recommended
quantization. Qwen3 4B and Gemma 4 E2B default to Q4_K_M. Gemma 4 E4B defaults
to Q8_0.

## What It Demonstrates

1. Direct curated enum selection through `ChatModels::Qwen3_4B_Gguf`
2. Runtime model switching through `--chat=google/gemma4_e2b_gguf` or `--chat=google/gemma4_e4b_gguf`
3. GGUF quantization control from curated spec defaults, plus `--precision=q4|q5|q8|f16`
4. Descriptor/capability introspection showing each selected bundle's advertised capabilities
5. Optional curated artifact download via `--download-artifacts`
6. Local-only startup through `ArtifactPolicy::LocalOnly`
7. Single-turn and multi-turn chat
8. Text completion
9. Latency reporting for startup and each request path
10. Process/memory snapshots before startup, after startup, and after each request
11. Handle-level model metrics after startup and each request
12. Optional `--tool-demo` path for caller-owned tool calling through llama.cpp chat templates
13. Per-request prompt controls through `--system=...`, `--no-system`, `--assistant=...`, and `--thinking=off|auto`

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

For Gemma 4 E2B, enable its GGUF feature:

```sh
cargo run -p motlie-models --no-default-features \
  --features model-gemma4-e2b-gguf \
  --example chat_gguf_gwen3_gemma4 -- --download-artifacts --chat=google/gemma4_e2b_gguf \
  "What is Rust's ownership model?"
```

For Gemma 4 E4B, the curated GGUF download includes Q8_0 first and Q4_K_M
second:

```sh
cargo run -p motlie-models --no-default-features --features model-gemma4-e4b-gguf \
  --bin motlie-models-download -- gemma4_e4b_gguf
```

## Step 2: Run the Example

Default path (Qwen3 4B, GGUF Q4_K_M, using existing local artifacts):

```sh
cargo run -p motlie-models --no-default-features --features model-qwen3-4b-gguf \
  --example chat_gguf_gwen3_gemma4 -- "What is Rust's ownership model?"
```

Switch to Gemma 4 E2B:

```sh
cargo run -p motlie-models --no-default-features \
  --features model-gemma4-e2b-gguf \
  --example chat_gguf_gwen3_gemma4 -- --chat=google/gemma4_e2b_gguf \
  "Summarize ownership in one paragraph"
```

Switch to Gemma 4 E4B. With no `--precision` flag this uses the E4B spec's
recommended Q8_0 GGUF artifact, `temperature=1.0`, `top_p=0.95`, and
`thinking=Auto`:

```sh
cargo run -p motlie-models --no-default-features --features model-gemma4-e4b-gguf \
  --example chat_gguf_gwen3_gemma4 -- --chat=google/gemma4_e4b_gguf \
  --system="You are Gemma, a helpful assistant." \
  --thinking=auto \
  "Summarize ownership in one paragraph"
```

Full precision (F16, no quantization):

```sh
cargo run -p motlie-models --no-default-features --features model-qwen3-4b-gguf \
  --example chat_gguf_gwen3_gemma4 -- --precision=f16 "What is Rust's ownership model?"
```

Tool-calling loop (Qwen3 GGUF):

```sh
cargo run -p motlie-models --no-default-features --features model-qwen3-4b-gguf \
  --bin motlie-models-download -- qwen3_4b_gguf

cargo run -p motlie-models --no-default-features --features model-qwen3-4b-gguf \
  --example chat_gguf_gwen3_gemma4 -- --tool-demo "What is Rust's ownership model?"
```

Tool-calling smoke only (skips ordinary chat and completion):

```sh
cargo run -p motlie-models --no-default-features --features model-qwen3-4b-gguf \
  --example chat_gguf_gwen3_gemma4 -- --tool-demo-only "What is Rust's ownership model?"
```

Tool-calling loop (Gemma4 GGUF):

```sh
cargo run -p motlie-models --no-default-features --features model-gemma4-e2b-gguf \
  --bin motlie-models-download -- gemma4_e2b_gguf

cargo run -p motlie-models --no-default-features \
  --features model-gemma4-e2b-gguf \
  --example chat_gguf_gwen3_gemma4 -- --chat=google/gemma4_e2b_gguf --tool-demo \
  "What is Rust's ownership model?"
```

The tool demo registers `get_weather` and `evaluate_math_expression`, sends
their generated schemas through the llama.cpp OpenAI-compatible chat-template
path, executes model-requested tool calls through static `ToolList` dispatch, appends each
tool-result message, and lets the model combine a plain Rust explanation with a
weather-derived average temperature.

## Preconditions

- Pre-downloaded GGUF artifacts in the curated artifact root, or `--download-artifacts`
- Sufficient memory for the chosen precision and model size
- At least one GGUF chat feature: `model-qwen3-4b-gguf`, `model-gemma4-e2b-gguf`, `model-gemma4-e4b-gguf`, or `model-qwen3-6-27b-gguf`

Validated tool-use smoke:

- `qwen3_4b_gguf` Q4_K_M passed locally on 2026-05-13.
- `gemma4_e2b_gguf` Q4_K_M passed locally on 2026-05-13.
- `gemma4_e4b_gguf` is wired and builds with E4B-only features. It intentionally advertises `Chat` + `Completion` only until a local tool-loop smoke passes.

## Source

- Example entrypoint: [main.rs](main.rs)
- Qwen3 GGUF bundle: `libs/models/src/chat/qwen3_4b_gguf.rs`
- Gemma4 GGUF bundle: `libs/models/src/chat/gemma4_e2b_gguf.rs`
- Gemma4 E4B GGUF bundle: `libs/models/src/chat/gemma4_e4b_gguf.rs`
- llama.cpp backend: `libs/model/backends/llama_cpp/`
