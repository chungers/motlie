# Eval Regression Run - amd x86 CPU

- snapshot: `curated-v2-smoke`
- source pin / git SHA: `167f5d898bab823ae39ca3acf0c7f798328c3ee9`
- profile: `local-cpu-x86_64`
- host: `amd-ryzen7-1` (`x86_64`)
- build profile: `release`
- release binary: `/tmp/motlie-regression-target-167f5d89-release/release/evals`
- run id: `regression-amd-x86-release-167f5d89`
- HF token: `HF_TOKEN_PRESENT`
- records: `27` total, `20` passed, `7` failed, `0` blocked
- behavior pass: `26/27`; resource gate failures: `7`

## Results

| cell | outcome | behavior | resources | quant | runtime | key metric | failure |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `embeddinggemma_300m__embeddings_similarity__smoke__hf_default` | `passed` | `pass` | `pass` | `default` | `` | vec/s=13.85; gap=0.784 |  |
| `qwen3_embedding_06b__embeddings_similarity__smoke__hf_default` | `passed` | `pass` | `pass` | `default` | `` | vec/s=4.41; gap=0.609 |  |
| `qwen3_4b__chat_smoke__smoke__hf_safetensors_default` | `failed` | `pass` | `fail` | `default` | `` | tok/s=3.0; chars=243 | resources section not accepted: resource gate max_process_swap_delta_bytes=0 exceeded: process_swap_delta_peak=1.82MiB (1908736 bytes) |
| `qwen3_4b__bench_chat_startup__smoke__hf_safetensors_default` | `passed` | `pass` | `pass` | `default` | `` | startup_ms=39739; warmup_ms=157211; mean_ms=144824; iters=3 |  |
| `qwen3_4b__tool_use_weather_cel_smoke__smoke__hf_safetensors_default` | `passed` | `pass` | `pass` | `default` | `` | tool_calls=1 |  |
| `gemma4_e2b__chat_smoke__smoke__hf_safetensors_default` | `failed` | `pass` | `fail` | `default` | `` | tok/s=4.0; chars=169 | resources section not accepted: resource gate max_process_swap_delta_bytes=0 exceeded: process_swap_delta_peak=1.46GiB (1565851648 bytes) |
| `gemma4_e2b__bench_chat_startup__smoke__hf_safetensors_default` | `passed` | `pass` | `pass` | `default` | `` | startup_ms=28410; warmup_ms=129537; mean_ms=126084; iters=3 |  |
| `gemma4_e2b__tool_use_weather_cel_smoke__smoke__hf_safetensors_default` | `passed` | `pass` | `pass` | `default` | `` | tool_calls=1 |  |
| `gemma4_e4b__chat_smoke__smoke__hf_safetensors_default` | `passed` | `pass` | `pass` | `default` | `` | tok/s=1.0; chars=218 |  |
| `gemma4_e4b__tool_use_weather_cel_smoke__smoke__hf_safetensors_default` | `failed` | `pass` | `fail` | `default` | `` | tool_calls=1 | resources section not accepted: resource gate max_process_swap_delta_bytes=0 exceeded: process_swap_delta_peak=3.45GiB (3701534720 bytes) |
| `qwen3_4b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `passed` | `pass` | `pass` | `q4_k_m` | `q4` | tok/s=9.0; chars=457 |  |
| `qwen3_4b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `passed` | `pass` | `pass` | `q4_k_m` | `q4` | startup_ms=2251; warmup_ms=44423; mean_ms=45697; iters=3 |  |
| `qwen3_4b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `passed` | `pass` | `pass` | `q4_k_m` | `q4` | tool_calls=1 |  |
| `gemma4_e2b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `failed` | `pass` | `fail` | `q4_k_m` | `q4` | tok/s=10.0; chars=180 | resources section not accepted: resource gate max_process_swap_delta_bytes=0 exceeded: process_swap_delta_peak=93.50MiB (98037760 bytes) |
| `gemma4_e2b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `passed` | `pass` | `pass` | `q4_k_m` | `q4` | startup_ms=1943; warmup_ms=21620; mean_ms=20876; iters=3 |  |
| `gemma4_e2b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `passed` | `pass` | `pass` | `q4_k_m` | `q4` | tool_calls=1 |  |
| `gemma4_e4b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `passed` | `pass` | `pass` | `q4_k_m` | `q4` | tok/s=5.0; chars=266 |  |
| `gemma4_e4b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `passed` | `pass` | `pass` | `q4_k_m` | `q4` | startup_ms=3113; warmup_ms=36437; mean_ms=36363; iters=3 |  |
| `gemma4_e4b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `passed` | `pass` | `pass` | `q4_k_m` | `q4` | tool_calls=1 |  |
| `gemma4_12b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `failed` | `pass` | `fail` | `q4_k_m` | `q4` | tok/s=1.0; chars=204 | resources section not accepted: resource gate max_process_swap_delta_bytes=0 exceeded: process_swap_delta_peak=67.32MiB (70586368 bytes) |
| `gemma4_12b_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_k_m` | `failed` | `pass` | `fail` | `q4_k_m` | `q4` | tool_calls=1 | resources section not accepted: resource gate max_process_swap_delta_bytes=0 exceeded: process_swap_delta_peak=12.85MiB (13475840 bytes) |
| `gemma4_12b_qat_q4_0_gguf__chat_smoke__smoke__gguf_q4_0` | `passed` | `pass` | `pass` | `q4_0` | `q4` | tok/s=1.0; chars=220 |  |
| `gemma4_12b_qat_q4_0_gguf__tool_use_weather_cel_smoke__smoke__gguf_q4_0` | `passed` | `pass` | `pass` | `q4_0` | `q4` | tool_calls=1 |  |
| `qwen3_6_27b_gguf__chat_smoke__smoke__gguf_q4_k_m` | `failed` | `fail` | `fail` | `q4_k_m` | `q4` | tok/s=0.0; chars=0 | behavior assertion `min_response_chars` failed: response_chars=0 min=16 |
| `gemma4_e4b__bench_chat_startup__smoke__hf_safetensors_default` | `passed` | `pass` | `pass` | `default` | `` | startup_ms=41979; warmup_ms=583686; mean_ms=561847; iters=3 |  |
| `gemma4_12b_gguf__bench_chat_startup__smoke__gguf_q4_k_m` | `passed` | `pass` | `pass` | `q4_k_m` | `q4` | startup_ms=11809; warmup_ms=144354; mean_ms=143808; iters=3 |  |
| `gemma4_12b_qat_q4_0_gguf__bench_chat_startup__smoke__gguf_q4_0` | `passed` | `pass` | `pass` | `q4_0` | `q4` | startup_ms=9780; warmup_ms=120355; mean_ms=119203; iters=3 |  |

## Notes

- Direct `evals run` cells used the release binary built from the pinned tree because the matrix child path may still use debug before #456.
- 27B q4_k_m artifact resolution selected one GGUF artifact and launched the model; the cell failed behavior with an empty response, not artifact resolution.
- Several model behavior checks passed but terminal outcome failed on `max_process_swap_delta_bytes=0` resource gates on this x86 host; see `failures/` excerpts for failed cells.
