# `motlie-models` `chat_gguf_gwen3_gemma4` Example — llama.cpp Backend (GGUF)

This example demonstrates chat generation via the **llama.cpp** backend using
GGUF-quantized weights. It supports switching between these models at runtime:

- **Qwen3 4B** (default) — `qwen/qwen3_4b_gguf`
- **Gemma 4 E2B-it** — `google/gemma4_e2b_gguf`
- **Gemma 4 E4B-it** — `google/gemma4_e4b_gguf`
- **Gemma 4 12B-it** — `google/gemma4_12b_gguf`

## Weight Format Compatibility

| Backend | Weight format | Qwen3-4B repo | Gemma4 E2B repo | Gemma4 E4B repo | Gemma4 12B repo |
|---------|---------------|---------------|------------------|------------------|------------------|
| **mistral.rs** (`chat_mistral_qwen3` / `chat_multimodal_gemma4`) | safetensors | `Qwen/Qwen3-4B` | `google/gemma-4-E2B-it` | `google/gemma-4-E4B-it` | `google/gemma-4-12B-it` |
| **llama.cpp** (`chat_gguf_gwen3_gemma4`) | GGUF | `Qwen/Qwen3-4B-GGUF` | `unsloth/gemma-4-E2B-it-GGUF` | `unsloth/gemma-4-E4B-it-GGUF` | `unsloth/gemma-4-12b-it-GGUF` |

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
quantization. Qwen3 4B, Gemma 4 E2B, and Gemma 4 12B default to Q4_K_M.
Gemma 4 E4B defaults to Q8_0.

## What It Demonstrates

1. Direct curated enum selection through `ChatModels::Qwen3_4B_Gguf`
2. Runtime model switching through `--chat=google/gemma4_e2b_gguf`, `--chat=google/gemma4_e4b_gguf`, or `--chat=google/gemma4_12b_gguf`
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

For Gemma 4 12B, the curated GGUF download includes the exact root Q4_K_M and
Q8_0 files from `unsloth/gemma-4-12b-it-GGUF`. It intentionally excludes MTP
drafter files, BF16, and multimodal projector files:

```sh
cargo run -p motlie-models --no-default-features --features model-gemma4-12b-gguf \
  --bin motlie-models-download -- gemma4_12b_gguf
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
`thinking=Auto`. The example prints the recommended settings, the effective
system prompt, the assistant priming turn, and any returned thinking trace:

```sh
cargo run -p motlie-models --no-default-features --features model-gemma4-e4b-gguf \
  --example chat_gguf_gwen3_gemma4 -- --chat=google/gemma4_e4b_gguf \
  --system="You are Gemma, a helpful assistant." \
  --assistant="I will keep the answer compact." \
  --thinking=auto \
  "Summarize ownership in one paragraph"
```

Switch to Gemma 4 12B GGUF. With no `--precision` flag this uses the 12B
spec's recommended Q4_K_M artifact; pass `--precision=q8` to select Q8_0:

```sh
cargo run -p motlie-models --no-default-features --features model-gemma4-12b-gguf \
  --example chat_gguf_gwen3_gemma4 -- --chat=google/gemma4_12b_gguf \
  --system="You are Gemma, a helpful assistant." \
  --thinking=auto \
  "Summarize ownership in one paragraph"
```

Representative prompt-control output:

```text
recommended-generation-params: GenerationParams { max_tokens: None, temperature: Some(1.0), top_p: Some(0.95), stop_sequences: [] }
recommended-system-prompt: Some("You are Gemma, a helpful assistant.")
recommended-quantization: GGUF Q8_0
recommended-thinking: Auto
effective-chat-params: GenerationParams { max_tokens: None, temperature: Some(1.0), top_p: Some(0.95), stop_sequences: [] }
thinking: Auto
system-prompt: enabled
system-prompt-content: You are Gemma, a helpful assistant.
assistant-priming: enabled
assistant-priming-content: I will keep the answer compact.
single-turn-thinking-trace: <reasoning trace when returned by the backend, otherwise none>
```

`--assistant` is intended as a priming turn alongside a system prompt. The
example prints an `assistant-priming-warning` when `--assistant` is combined
with `--no-system`, because some GGUF chat templates can return empty visible
content for conversations that start with an assistant turn and no system
anchor.

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

Tool-calling loop (Gemma 4 E4B GGUF):

```sh
cargo run -p motlie-models --no-default-features --features model-gemma4-e4b-gguf \
  --bin motlie-models-download -- gemma4_e4b_gguf

cargo run -p motlie-models --no-default-features \
  --features model-gemma4-e4b-gguf \
  --example chat_gguf_gwen3_gemma4 -- --chat=google/gemma4_e4b_gguf --tool-demo-only \
  "What is Rust? Then calculate the average temperature for Seattle, Portland, and San Francisco."
```

Tool-calling loop (Gemma 4 12B GGUF):

```sh
cargo run -p motlie-models --no-default-features --features model-gemma4-12b-gguf \
  --bin motlie-models-download -- gemma4_12b_gguf

cargo run -p motlie-models --no-default-features \
  --features model-gemma4-12b-gguf \
  --example chat_gguf_gwen3_gemma4 -- --chat=google/gemma4_12b_gguf --tool-demo-only \
  "What is Rust? Then calculate the average temperature for Seattle, Portland, and San Francisco."
```

For Gemma 4 12B GGUF, the example keeps normal chat on the model's recommended
`ThinkingMode::Auto`, but defaults the tool demo to `ThinkingMode::Disabled`.
@gemma4-cdx validated this on 2026-06-05 16:59 PDT after `Auto` consumed the
short tool-demo budget with reasoning text and produced no tool call.

The tool demo registers `get_weather` and `evaluate_math_expression`, sends
their generated schemas through the llama.cpp OpenAI-compatible chat-template
path, executes model-requested tool calls through static `ToolList` dispatch, appends each
tool-result message, and lets the model combine a plain Rust explanation with a
weather-derived average temperature.

## Preconditions

- Pre-downloaded GGUF artifacts in the curated artifact root, or `--download-artifacts`
- Sufficient memory for the chosen precision and model size
- At least one GGUF chat feature: `model-qwen3-4b-gguf`, `model-gemma4-e2b-gguf`, `model-gemma4-e4b-gguf`, `model-gemma4-12b-gguf`, `model-gemma4-12b-qat-q4-0-gguf`, or `model-qwen3-6-27b-gguf`
- llama.cpp build prerequisites: CMake plus clang/libclang headers visible to bindgen

Validated tool-use smoke:

- `qwen3_4b_gguf` Q4_K_M passed locally on 2026-05-13.
- `gemma4_e2b_gguf` Q4_K_M passed locally on 2026-05-13.
- `gemma4_e4b_gguf` Q8_0 passed from a cold curated download and release build on 2026-05-18, then passed again with `llama-cpp-2` 0.1.146 and cached artifacts. It advertises `Chat` + `Completion` + `ToolUse` and produced this tool sequence:

```text
tool-call-name: get_weather
tool-call-args: {"city":"Seattle","units":"fahrenheit"}
tool-result: {"city":"Seattle","temperature":72.0,"units":"fahrenheit","summary":"clear"}
tool-call-name: get_weather
tool-call-args: {"city":"Portland","units":"fahrenheit"}
tool-result: {"city":"Portland","temperature":68.0,"units":"fahrenheit","summary":"clear"}
tool-call-name: get_weather
tool-call-args: {"city":"San Francisco","units":"fahrenheit"}
tool-result: {"city":"San Francisco","temperature":64.0,"units":"fahrenheit","summary":"clear"}
tool-call-name: evaluate_math_expression
tool-call-args: {"expression":"(72.0 + 68.0 + 64.0) / 3.0"}
tool-result: {"expression":"(72.0 + 68.0 + 64.0) / 3.0","value":68.0,"formatted":"68","engine":"cel-cxx"}
tool-final-response: The average current temperature for Seattle, Portland, and San Francisco is 68.0 degrees Fahrenheit.
```


- `gemma4_12b_qat_q4_0_gguf` was added by @gemma4-cdx on 2026-06-05 17:45 PDT for issue #397 using `google/gemma-4-12B-it-qat-q4_0-gguf` and exact file `gemma-4-12b-it-qat-q4_0.gguf`. Local validation passed `--lib`, this example, and `bench_chat` cargo checks with feature `model-gemma4-12b-qat-q4-0-gguf`; the curated download fetched one file from snapshot `f6e7774e6148da3b7f201e42ba37cf084c1db35f`. Local CPU smoke loaded GGUF Q4_0, file size 6.48 GiB, startup 4.946s in `bench_chat`, one-word warmup 38.8s, one measured one-word request 43.6s, final RSS 12.5 GiB, peak RSS 18.7 GiB. The `--tool-demo-only` path passed with `tool-demo-thinking: Disabled`, four expected tool calls, final response, startup 5.2s, final RSS 12.4 GiB, and peak resident bytes 24.45 GB.

- `gemma4_12b_gguf` Q4_K_M/Q8_0 wiring was added by @gemma4-cdx on 2026-06-04 22:58 PDT. @gemma4-cdx fixed GGUF-only compile wiring on 2026-06-05 16:11 PDT; `--lib`, this example, and `bench_chat` pass `cargo check` with local CMake plus bindgen include paths. @gemma4-cdx tightened artifact rules and completed the corrected Q4/Q8 download on 2026-06-05 16:36 PDT; Q4 startup and one-word warmup passed locally with startup 14.5s, warmup 15.4s, final RSS 11.3 GiB, peak RSS 22.1 GiB, and no swaps. @gemma4-cdx fixed the 12B GGUF tool-demo default on 2026-06-05 16:59 PDT and validated the default Q4 `--tool-demo-only` path: startup 7.0s, `tool-demo-thinking: Disabled`, four expected tool calls, clean final response, final RSS 12.5 GiB, and peak resident bytes 24.55 GB.

## Source

- Example entrypoint: [main.rs](main.rs)
- Qwen3 GGUF bundle: `libs/models/src/chat/qwen3_4b_gguf.rs`
- Gemma4 GGUF bundle: `libs/models/src/chat/gemma4_e2b_gguf.rs`
- Gemma4 E4B GGUF bundle: `libs/models/src/chat/gemma4_e4b_gguf.rs`
- Gemma4 12B GGUF bundle: `libs/models/src/chat/gemma4_12b_gguf.rs`
- Gemma4 12B QAT Q4_0 GGUF bundle: `libs/models/src/chat/gemma4_12b_qat_q4_0_gguf.rs`
- llama.cpp backend: `libs/model/backends/llama_cpp/`
