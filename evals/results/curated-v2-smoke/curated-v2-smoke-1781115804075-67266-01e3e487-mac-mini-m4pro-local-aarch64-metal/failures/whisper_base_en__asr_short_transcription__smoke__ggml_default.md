# Failure: whisper_base_en__asr_short_transcription__smoke__ggml_default

- **bundle**: whisper_base_en | **capability**: asr | **depth**: smoke
- **model_family**: whisper | **checkpoint**: ggml | **quant**: default | **backend**: whisper_cpp
- **profile**: apple-metal | **host**: mac-mini-m4pro-local | **arch**: aarch64 | **platform**: macOS 15.5 / Apple M4 Pro (Metal)
- **outcome**: blocked | **reason**: accelerator_mismatch
- **failure_reason**: accelerator section not accepted: requested=metal resolved=cpu reason=accelerator_mismatch
- **accelerator**: requested=metal resolved=cpu backend_mode=whisper_cpp:cpu use_proof=backend_observation
- **budgets**: {}
- **child build profile**: debug (driver child `cargo build` default)
- **repro**: `/Users/dchung/sessions/issue-399-eval-suite/fable5-399-rv/motlie/target/debug/evals run --bundle whisper_base_en --scenario asr_short_transcription --profile apple-metal --root /Users/dchung/sessions/issue-399-eval-suite/fable5-399-rv/eval-run/evals --artifact-root /Users/dchung/sessions/issue-399-eval-suite/fable5-399-rv/eval-run/libs/models/../../artifacts/models/hf-cache --jsonl /tmp/eval-run-results/curated-v2-smoke/curated-v2-smoke-1781115804075-67266-01e3e487-mac-mini-m4pro-local-aarch64-metal/results.jsonl --run-id curated-v2-smoke-1781115804075-67266-01e3e487-mac-mini-m4pro-local-aarch64-metal --snapshot-id curated-v2-smoke --cell-id whisper_base_en__asr_short_transcription__smoke__ggml_default --depth smoke --checkpoint-format ggml --artifact-quantization default --model-family whisper --backend whisper_cpp --requested-accelerator metal --child-build-log /tmp/eval-run-results/curated-v2-smoke/curated-v2-smoke-1781115804075-67266-01e3e487-mac-mini-m4pro-local-aarch64-metal/logs/whisper_base_en__asr_short_transcription__smoke__ggml_default.log --child-build-status 0 --child-build-duration-ms 19675 --quiet-backend-logs`
- **log tail**: [whisper_base_en__asr_short_transcription__smoke__ggml_default.log](./whisper_base_en__asr_short_transcription__smoke__ggml_default.log) (last 200 lines)

@claude-fable5-399-rv 2026-06-10 PDT -- packaged per #435 failure-tracking requirement.
