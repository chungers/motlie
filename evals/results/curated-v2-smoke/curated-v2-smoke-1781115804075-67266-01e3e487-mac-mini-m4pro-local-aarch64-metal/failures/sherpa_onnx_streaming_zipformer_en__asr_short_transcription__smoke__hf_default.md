# Failure: sherpa_onnx_streaming_zipformer_en__asr_short_transcription__smoke__hf_default

- **bundle**: sherpa_onnx_streaming_zipformer_en | **capability**: asr | **depth**: smoke
- **model_family**: sherpa_onnx | **checkpoint**: onnx | **quant**: default | **backend**: sherpa_onnx
- **profile**: apple-metal | **host**: mac-mini-m4pro-local | **arch**: aarch64 | **platform**: macOS 15.5 / Apple M4 Pro (Metal)
- **outcome**: blocked | **reason**: artifact_unauthorized
- **failure_reason**: child eval invocation failed; see /tmp/eval-run-results/curated-v2-smoke/curated-v2-smoke-1781115804075-67266-01e3e487-mac-mini-m4pro-local-aarch64-metal/logs/sherpa_onnx_streaming_zipformer_en__asr_short_transcription__smoke__hf_default.log
- **accelerator**: requested=metal resolved=metal backend_mode=backend_offload_unverified use_proof=backend:unreported
- **budgets**: {'max_wall_time_secs': '900'}
- **child build profile**: debug (driver child `cargo build` default)
- **repro**: `/Users/dchung/sessions/issue-399-eval-suite/fable5-399-rv/motlie/target/debug/evals matrix --snapshot evals/snapshots/curated-v2-smoke.toml --profile apple-metal --results-root /tmp/eval-run-results`
- **log tail**: [sherpa_onnx_streaming_zipformer_en__asr_short_transcription__smoke__hf_default.log](./sherpa_onnx_streaming_zipformer_en__asr_short_transcription__smoke__hf_default.log) (last 200 lines)

@claude-fable5-399-rv 2026-06-10 PDT -- packaged per #435 failure-tracking requirement.
