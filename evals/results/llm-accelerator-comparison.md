# LLM Accelerator Comparison — CUDA vs CPU vs Metal × backend-family (#514/#534)

Generated from `evals report --aggregate 'evals/results/**/results.jsonl' --snapshot evals/snapshots/curated-v2-smoke.toml --allow-invalid-records` on the evals branch. The table below is the report-generated LLM section, now keyed by `bundle × quant` so ISQ rows do not disappear into bundle averages.

## Notes

- `mistralrs/HF` CUDA is validated on DGX for `qwen3_4b`: existing `bf16` plus #534 `isq_q4` and `isq_q8` rows all resolve `cuda` with `backend_observation`.
- #534 ISQ CPU rows: `gemma4_e2b isq_q4` passed on x86_64 and aarch64; `gemma4_e2b isq_q8` passed on aarch64. The x86_64 q8 direct-run probe ended with `scenario exceeded --max-wall-time-secs=1200s wall-time backstop` before writing a JSONL row, so no x86 q8 metric is committed.
- Metal mistralrs rows remain blocked/not-applicable per the current upstream Metal support state; no new Metal run was feasible in this DGX-hosted fix.

## LLM Accelerator Comparison

Decode throughput (tok/s) and TTFT (ms) per LLM bundle and quantization scheme, by target accelerator. Values are from passing `perf` cells; `—` = no passing perf metric for that pairing.

| bundle | quant | family | cpu tok/s | cpu ttft ms | cuda tok/s | cuda ttft ms | metal tok/s | metal ttft ms |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `gemma4_12b_gguf` | `gguf_q4_k_m` | llama.cpp/GGUF | 5.5 | 51967 | 24.6 | 1023 | 27.9 | 2788 |
| `gemma4_12b_qat_gguf` | `gguf_q4_0` | llama.cpp/GGUF | 5.5 | 53668 | 27.0 | 944 | 28.6 | 2622 |
| `gemma4_e2b` | `bf16` | mistralrs/HF | 6.9 | 48262 | — | — | — | — |
| `gemma4_e2b` | `isq_q4` | mistralrs/HF | 7.0 | 48587 | — | — | — | — |
| `gemma4_e2b` | `isq_q8` | mistralrs/HF | 9.5 | 20418 | — | — | — | — |
| `gemma4_e2b_gguf` | `gguf_q4_k_m` | llama.cpp/GGUF | 21.5 | 9048 | 96.0 | 136 | 82.6 | 418 |
| `gemma4_e4b` | `bf16` | mistralrs/HF | 6.3 | 40039 | — | — | — | — |
| `gemma4_e4b_gguf` | `gguf_q4_k_m` | llama.cpp/GGUF | 12.0 | 19196 | 55.4 | 346 | 52.3 | 881 |
| `qwen3_4b` | `bf16` | mistralrs/HF | 6.8 | 89163 | 47.3 | 230 | — | — |
| `qwen3_4b` | `isq_q4` | mistralrs/HF | — | — | 49.4 | 217 | — | — |
| `qwen3_4b` | `isq_q8` | mistralrs/HF | — | — | 36.2 | 227 | — | — |
| `qwen3_4b_gguf` | `gguf_q4_k_m` | llama.cpp/GGUF | 13.3 | 16143 | 71.4 | 168 | 64.5 | 774 |
| `qwen3_6_27b_gguf` | `gguf_q4_k_m` | llama.cpp/GGUF | 3.2 | 115863 | 11.3 | 928 | 11.7 | 5125 |

### Backend-family viability

`on_target` = passed cells that actually resolved to the requested accelerator (a passed cell that silently fell back to CPU is counted in `passed` but not `on_target`). `mean decode tok/s` is averaged over on-target passing `perf` cells.

| family | accelerator | cells | passed | on_target | blocked | failed | mean decode tok/s |
|---|---|---:|---:|---:|---:|---:|---:|
| llama.cpp/GGUF | `cpu` | 162 | 35 | 35 | 125 | 2 | 11.0 |
| llama.cpp/GGUF | `cuda` | 23 | 22 | 22 | 0 | 1 | 47.6 |
| llama.cpp/GGUF | `metal` | 51 | 48 | 48 | 0 | 3 | 44.6 |
| mistralrs/HF | `cpu` | 87 | 21 | 21 | 66 | 0 | 7.1 |
| mistralrs/HF | `cuda` | 14 | 3 | 3 | 11 | 0 | 44.3 |
| mistralrs/HF | `metal` | 27 | 0 | 0 | 27 | 0 | — |

### Build provenance

Distinct build SHAs (`identity.git_sha`) of the on-target passing records backing the numbers above, per accelerator. An accelerator whose SHA set differs from the others is **prior-pin** data (pin mismatch — confirmatory only, not a fresh re-run).

| accelerator | build SHAs |
|---|---|
| `cpu` | `21f374da`, `65a9416f`, `99ac891d`, `e8f27b6e` |
| `cuda` | `65a9416f`, `99ac891d`, `e8f27b6e` |
| `metal` | `99ac891d`, `c91ba281`, `ec3edeba` |

