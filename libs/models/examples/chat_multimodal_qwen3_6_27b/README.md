# `motlie-models` `chat_multimodal_qwen3_6_27b` Example

This example runs Qwen3.6 27B through the curated llama.cpp GGUF bundle:
`qwen/qwen3_6_27b_gguf`.

The upstream model is multimodal, and the core Motlie `ChatRequest` surface
already supports image content parts. This curated bundle is text-only until
the llama.cpp backend wires real mmproj/image execution. Passing `--image=...`
therefore reports the blocked path instead of sending image stubs to the model.

## What It Demonstrates

1. Direct curated enum selection through `ChatModels::Qwen3_6_27B_Gguf`
2. GGUF quantization control with Q5_K_M as the current default
3. Optional curated artifact download via `--download-artifacts`
4. Local-only startup through `ArtifactPolicy::LocalOnly`
5. Descriptor/capability introspection
6. Text chat and text completion
7. Startup/request latency, process snapshots, and handle metrics
8. Explicit image-path gating based on advertised `Vision` capability

## Run

Download curated artifacts and run the default Q5_K_M path:

```sh
cargo run -p motlie-models --no-default-features \
  --features model-qwen3-6-27b-gguf \
  --example chat_multimodal_qwen3_6_27b -- \
  --download-artifacts "Summarize Rust ownership in one paragraph"
```

Use existing local artifacts:

```sh
cargo run -p motlie-models --no-default-features \
  --features model-qwen3-6-27b-gguf \
  --example chat_multimodal_qwen3_6_27b -- \
  "Summarize Rust ownership in one paragraph"
```

Select an alternate curated GGUF quant:

```sh
cargo run -p motlie-models --no-default-features \
  --features model-qwen3-6-27b-gguf \
  --example chat_multimodal_qwen3_6_27b -- \
  --precision=q4 "Summarize Rust ownership in one paragraph"

cargo run -p motlie-models --no-default-features \
  --features model-qwen3-6-27b-gguf \
  --example chat_multimodal_qwen3_6_27b -- \
  --precision=q8 "Summarize Rust ownership in one paragraph"
```

CUDA build:

```sh
cargo run -p motlie-models --no-default-features \
  --features model-qwen3-6-27b-gguf,llama-cpp-cuda \
  --example chat_multimodal_qwen3_6_27b -- \
  --precision=q8 "Summarize Rust ownership in one paragraph"
```

`llama-cpp-cuda` enables the llama.cpp CUDA build path. By default the backend
uses full GPU offload unless `MOTLIE_MODEL_FORCE_CPU=1` or
`MOTLIE_MODEL_GPU_LAYERS=<n>` is set.

## Precision Policy

Current curated GGUF artifacts:

| `--precision` | GGUF artifact | Status |
|---------------|---------------|--------|
| omitted / `q5` | `Qwen3.6-27B-Q5_K_M.gguf` | default |
| `q4` | `Qwen3.6-27B-Q4_K_M.gguf` | supported |
| `q8` | `Qwen3.6-27B-Q8_0.gguf` | supported |
| `fp8` | `Qwen3.6-27B-FP8.gguf` | reserved, not curated yet |

The official Qwen FP8 release is currently a Transformers/safetensors artifact,
not a GGUF artifact. The example accepts the documented `fp8` spelling but
fails closed until a real curated FP8 GGUF is available.

## Preconditions

- Pre-downloaded Qwen3.6 GGUF artifacts in the curated artifact root, or
  `--download-artifacts`
- Sufficient RAM/VRAM for the selected precision
- `model-qwen3-6-27b-gguf`; add `llama-cpp-cuda` for CUDA builds

## Source

- Example entrypoint: [main.rs](main.rs)
- Curated bundle: `libs/models/src/chat/qwen3_6_27b_gguf.rs`
- llama.cpp backend: `libs/model/backends/llama_cpp/`
