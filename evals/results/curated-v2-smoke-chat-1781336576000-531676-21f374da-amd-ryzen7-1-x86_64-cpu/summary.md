# Eval CHAT-capability Coverage Run — amd1 x86 CPU (uncapped, #492 metrics)

- snapshot: `curated-v2-smoke`
- source pin / git SHA: `21f374da2325e98328aaae1ac5f7391bd309b758` (the #510 merge — uncapped chat_smoke + `thinking_tokens_to_answer`)
- profile: `local-cpu-x86_64`
- host: `amd-ryzen7-1` (`x86_64`)
- build profile: `release` (features: model-qwen3-4b-gguf, model-gemma4-e2b-gguf, model-gemma4-e4b-gguf)
- run id: `chat-coverage-21f374da-amd-x86`
- HF token: `HF_TOKEN_PRESENT` (env-only; token value not committed)
- generation: **UNCAPPED** (no a-priori `max_tokens`; runs to natural EOS) with `--max-wall-time-secs 1200` opt-in backstop (did not fire)
- records: `3` total, `3` passed, `0` failed, `0` blocked; behavior pass `3/3`

## Results (chat capability)

| cell | outcome | behavior | thinking_tokens_to_answer | completion_tokens (natural EOS) | response_chars | ttft_first_answer_ms |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `qwen3_4b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `pass` | `pass` | 344 | 372 | 1867 | 28217 |
| `gemma4_e2b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `pass` | `pass` | 0 | 31 | 176 | 602 |
| `gemma4_e4b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `pass` | `pass` | 0 | 31 | 175 | 3058 |

## Notes

- **qwen3_4b_gguf (the #492 thinking model): PASS and measured** — `thinking_tokens_to_answer=344` reasoning tokens precede the answer, completion `372` tokens at natural EOS. Under the dropped pre-#510 caps (96, then the backend's hidden 512) this trajectory truncated mid-`<think>` into `response_chars=0`; uncapped it completes and the reasoning overhead is now a recorded, comparable metric.
- **Thinking vs non-thinking contrast:** the Qwen reasoner records `thinking_tokens_to_answer=344`; the Gemma chat models record `0` (answer from the first token) — the empirical cross-model reasoning-overhead signal the metric was added for.
- **Coverage scope:** CPU-feasible GGUF chat bundles. `qwen3_6_27b_gguf` (~16 GB) and `gemma4_12b_gguf` (~7 GB) were not run on amd1 due to the host's ~12 GB free disk; they remain for a higher-disk host. mistral.rs safetensors chat bundles report `thinking_tokens_to_answer` as a metric gap (no per-token boundary data on that backend).
