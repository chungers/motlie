# Chat and Tool-Calling Validation

## Change Log

| Date | Change | Sections |
|------|--------|----------|
| 2026-05-20 | @claude-reviewer: First end-to-end validation matrix for the curated chat and tool-calling examples — GGUF (llama.cpp) and safetensors (mistral.rs) backends, the API-only typed-binding path, per-request thinking/system/assistant controls, and the `#310` safetensors tool-transcript fix. | All |

## Scope

This report validates the shipped chat and tool-calling examples after the
Gemma 4 E4B bundle addition (`#308`, PR #309) and the `mistral.rs` safetensors
tool-transcript fix (`#310`, commit `5ab8fc3c`):

- `chat_tool_binding` — API-only typed `ToolList` binding, no LLM
- `chat_gguf_gwen3_gemma4` — llama.cpp GGUF: Qwen3 4B, Gemma 4 E2B-it, Gemma 4 E4B-it
- `chat_mistral_qwen3` — mistral.rs safetensors: Qwen3 4B
- `chat_multimodal_gemma4` — mistral.rs safetensors: Gemma 4 E2B-it

`chat_multimodal_qwen3_6_27b` is excluded — its bundle does not advertise
`ToolUse` and the 27B GGUF artifact exceeds a practical local-validation
footprint.

## Method

- Host: Apple M4 Pro, macOS 15.5, Metal acceleration, no CUDA.
- Build: `--release` examples from `libs/models`; GGUF examples with the
  `model-*-gguf` features; safetensors examples with
  `RUSTFLAGS='-C target-feature=+fp16'` and `model-qwen3-4b` /
  `model-gemma4-e2b`.
- GGUF quantization: per-bundle curated default (Qwen3 4B and Gemma E2B →
  Q4_K_M, Gemma E4B → Q8_0). Safetensors: ISQ Q4.
- Tool loop: the shared `tool_demo_support::run_tool_demo_with_options` path —
  `get_weather` for Seattle / Portland / San Francisco, then
  `evaluate_math_expression` for the average, executed by the static
  `tool_list!()` registry.
- Each run's exit code, round-by-round tool calls, and final response captured
  from stdout; backend-native loader logs omitted.

## Result Matrix

| Example | Backend | Model | Path | Result |
|---------|---------|-------|------|--------|
| `chat_tool_binding` | none (API-only) | n/a | typed `ToolList` + spec recommendation round-trip | PASS — 2 tools registered, CEL average `68.0`, `recommended_*` fields surfaced into the request |
| `chat_gguf_gwen3_gemma4` | llama.cpp | Qwen3 4B GGUF Q4_K_M | plain chat + completion | PASS |
| `chat_gguf_gwen3_gemma4` | llama.cpp | Qwen3 4B GGUF Q4_K_M | `--tool-demo-only` | PASS — 4-round loop |
| `chat_gguf_gwen3_gemma4` | llama.cpp | Gemma 4 E2B-it GGUF Q4_K_M | plain chat | PASS |
| `chat_gguf_gwen3_gemma4` | llama.cpp | Gemma 4 E2B-it GGUF Q4_K_M | `--tool-demo-only` | PASS — 4-round loop |
| `chat_gguf_gwen3_gemma4` | llama.cpp | Gemma 4 E4B-it GGUF Q8_0 | plain chat | PASS |
| `chat_gguf_gwen3_gemma4` | llama.cpp | Gemma 4 E4B-it GGUF Q8_0 | `--tool-demo-only` | PASS — 4-round loop |
| `chat_gguf_gwen3_gemma4` | llama.cpp | Gemma 4 E4B-it GGUF Q8_0 | `--thinking=auto` / `--thinking=off` / `--system` / `--assistant` | PASS — flag overrides honored |
| `chat_mistral_qwen3` | mistral.rs | Qwen3 4B safetensors ISQ Q4 | plain chat + multi-turn + completion | PASS — ~20 tps generation, ~8–20s startup |
| `chat_mistral_qwen3` | mistral.rs | Qwen3 4B safetensors ISQ Q4 | `--tool-demo-only` | PASS — 4-round loop (fixed by `5ab8fc3c`; previously failed round 2) |
| `chat_multimodal_gemma4` | mistral.rs | Gemma 4 E2B-it safetensors ISQ Q4 | plain text chat + multi-turn | PASS — ~18 tps generation, ~22s startup |
| `chat_multimodal_gemma4` | mistral.rs | Gemma 4 E2B-it safetensors ISQ Q4 | `--tool-demo-only` | PASS — 4-round loop (fixed by `5ab8fc3c`; previously failed round 1) |

## Tool-Loop Transcript (Representative)

All four tool-loop runs produce the same arithmetic answer; final-sentence
phrasing varies at the recommended `temperature=1.0` for Gemma 4 E4B.

```text
tool-round: 1  get_weather {"city":"Seattle","units":"fahrenheit"}        -> 72.0
tool-round: 2  get_weather {"city":"Portland","units":"fahrenheit"}       -> 68.0
tool-round: 3  get_weather {"city":"San Francisco","units":"fahrenheit"}  -> 64.0
tool-round: 4  evaluate_math_expression {"expression":"(72.0 + 68.0 + 64.0) / 3.0"}
               -> {"value":68.0,"formatted":"68","engine":"cel-cxx"}
tool-final-response: The average current temperature for Seattle, Portland,
                     and San Francisco is 68.0[ degrees Fahrenheit].
```

## `#310` — mistral.rs Safetensors Tool-Transcript Regression

Before `5ab8fc3c`, the two mistral.rs safetensors examples advertised `ToolUse`
but failed live tool loops on Apple Silicon Metal:

- `chat_mistral_qwen3 --tool-demo-only` — failed at round 2
- `chat_multimodal_gemma4 --tool-demo-only` — failed at round 1

Both with `mistralrs backend failed during send_chat_request: response
contained neither text content nor tool calls`.

Root cause: upstream `mistralrs::RequestBuilder` replayed the assistant
tool-call under `function` rather than the `tool_calls` shape the Qwen3 and
Gemma 4 chat templates expect, omitted `name` on tool-result replay (Gemma
preprocessing needs it), and let `enable_thinking` default to `true`, which
consumed the tool-demo generation budget before a tool call could be emitted.

Fix: the Motlie `mistral.rs` adapter replaces `RequestBuilder` with a custom
`MotlieMistralRequest` implementing `mistralrs::RequestLike`, which controls
the exact transcript shape. After the fix, both examples complete the full
4-round loop (verified on Apple M4 Pro / Metal, ISQ Q4).

## Capability-Advertisement Status

| Bundle | Backend | Advertises `ToolUse` | Smoke evidence |
|--------|---------|----------------------|----------------|
| `qwen3_4b` | mistral.rs safetensors | yes | direct — `chat_mistral_qwen3 --tool-demo-only` |
| `gemma4_e2b` | mistral.rs safetensors | yes | direct — `chat_multimodal_gemma4 --tool-demo-only` |
| `gemma4_e4b` | mistral.rs safetensors | yes | indirect — identical `MistralMultimodalAdapter` path as `gemma4_e2b`; no example loads E4B safetensors |
| `qwen3_4b_gguf` | llama.cpp | yes | direct — `chat_gguf_gwen3_gemma4 --tool-demo-only` |
| `gemma4_e2b_gguf` | llama.cpp | yes | direct — `chat_gguf_gwen3_gemma4 --chat=google/gemma4_e2b_gguf --tool-demo-only` |
| `gemma4_e4b_gguf` | llama.cpp | yes | direct — `chat_gguf_gwen3_gemma4 --chat=google/gemma4_e4b_gguf --tool-demo-only` |
| `qwen3_6_27b_gguf` | llama.cpp | no (`Chat` + `Completion`) | n/a — not advertised |

## Known Follow-Ups

- `gemma4_e4b` safetensors advertises `ToolUse` without a dedicated example
  smoke; it shares the proven `MistralMultimodalAdapter` path with
  `gemma4_e2b`. A direct E4B-safetensors smoke would be belt-and-suspenders
  coverage.
- The `unique_temp_dir()` catalog tests in `motlie-models` showed one
  intermittent failure across roughly five runs — a filesystem-timing flake
  worth hardening.
