# Non-passed cell: embeddinggemma_300m__embeddings_similarity__smoke__hf_default

- **bundle**: embeddinggemma_300m | **capability**: embeddings | **quant**: default | **backend**: mistralrs
- **profile**: apple-metal | **host**: mac-mini-m4pro-local / aarch64 / Apple M4 Pro / macOS 15.5
- **outcome**: blocked | **reason**: accelerator_mismatch
- **failure_reason**: accelerator section not accepted: requested=metal resolved=cpu reason=accelerator_mismatch
- **accelerator**: requested=metal resolved=cpu backend_mode=mistralrs:cpu use_proof=backend_observation
- **build**: RELEASE child @ 99ac891d (native matrix driver)
- **repro**: `/tmp/final-run-target/release/evals run --bundle embeddinggemma_300m --scenario embeddings_similarity --profile apple-metal --root /Users/dchung/sessions/issue-399-eval-suite/fable5-399-rv/final-run/evals --artifact-root /Users/dchung/sessions/issue-399-eval-suite/fable5-399-rv/eval-run/artifacts/models/hf-cache --jsonl /Users/dchung/sessions/issue-399-eval-suite/fable5-399-rv/final-run/evals/results/curated-v2-smoke/curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal/results.jsonl --run-id curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal --snapshot-id curated-v2-smoke --cell-id embeddinggemma_300m__embeddings_similarity__smoke__hf_default --depth smoke --checkpoint-format hf_safetensors --artifact-quantization default --model-family gemma --backend mistralrs --requested-accelerator metal --child-build-log /Users/dchung/sessions/issue-399-eval-suite/fable5-399-rv/final-run/evals/results/curated-v2-smoke/curated-v2-smoke-1781160252203-78445-99ac891d-mac-mini-m4pro-local-aarch64-metal/logs/embeddinggemma_300m__embeddings_similarity__smoke__hf_default.log --child-build-status 0 --child-build-duration-ms 111700 --quiet-backend-logs`

@claude-fable5-399-rv 2026-06-11 PDT -- final-run packaging per #435. accelerator_mismatch rows are honest completed CPU runs (documented platform note).
