# Non-passed cell: qwen3_embedding_06b__embeddings_similarity__smoke__hf_safetensors_default

- **bundle**: qwen3_embedding_06b | **capability**: embeddings | **quant**: default | **backend**: mistralrs
- **profile**: apple-metal | **host**: mac-mini-m4pro-local / aarch64 / Apple M4 Pro / macOS 15.5
- **outcome**: blocked | **reason**: accelerator_mismatch
- **failure_reason**: accelerator section not accepted: requested=metal resolved=cpu reason=accelerator_mismatch
- **accelerator**: requested=metal resolved=cpu backend_mode=mistralrs:cpu use_proof=backend_observation
- **build**: RELEASE @ 167f5d89 (direct `evals run`, no driver budget)
- **repro**: `/tmp/regression-167f-target/release/evals run --bundle qwen3_embedding_06b --scenario embeddings_similarity --profile apple-metal --root /Users/dchung/sessions/issue-399-eval-suite/fable5-399-rv/regression-rerun/evals --artifact-root /Users/dchung/sessions/issue-399-eval-suite/fable5-399-rv/eval-run/artifacts/models/hf-cache --requested-accelerator metal --snapshot-id curated-v2-smoke --cell-id qwen3_embedding_06b__embeddings_similarity__smoke__hf_safetensors_default --depth smoke --checkpoint-format hf_safetensors --artifact-quantization default --model-family qwen3 --backend mistralrs --run-id curated-v2-smoke-1781144844000-95309-167f5d89-mac-mini-m4pro-local-aarch64-metal --quiet-backend-logs --jsonl /tmp/reg167-results.jsonl`

@claude-fable5-399-rv 2026-06-10 PDT -- regression-rerun packaging per #435. Note: accelerator_mismatch rows are HONEST completed CPU runs (documented platform gap), not failures.
