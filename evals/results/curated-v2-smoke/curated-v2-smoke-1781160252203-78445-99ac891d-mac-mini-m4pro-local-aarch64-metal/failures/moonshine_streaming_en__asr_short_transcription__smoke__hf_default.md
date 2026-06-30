# Non-passed cell: moonshine_streaming_en__asr_short_transcription__smoke__hf_default

- **bundle**: moonshine_streaming_en | **capability**: asr | **quant**: default | **backend**: ort
- **profile**: apple-metal | **host**: mac-mini-m4pro-local / aarch64 / Apple M4 Pro / macOS 15.5
- **outcome**: blocked | **reason**: artifact_missing
- **failure_reason**: artifact cache preflight missing required artifacts for `moonshine_streaming_en` under `/Users/dchung/sessions/issue-399-eval-suite/fable5-399-rv/eval-run/artifacts/models/hf-cache`; run `evals provision` or provide --artifact-root before matrix execution
- **accelerator**: requested=metal resolved=metal backend_mode=backend_offload_unverified use_proof=backend:unreported
- **build**: RELEASE child @ 99ac891d (native matrix driver)
- **repro**: `/tmp/final-run-target/debug/evals matrix --snapshot evals/snapshots/curated-v2-smoke.toml --profile apple-metal --artifact-root /Users/dchung/sessions/issue-399-eval-suite/fable5-399-rv/eval-run/artifacts/models/hf-cache`

@claude-fable5-399-rv 2026-06-11 PDT -- final-run packaging per #435. accelerator_mismatch rows are honest completed CPU runs (documented platform note).
