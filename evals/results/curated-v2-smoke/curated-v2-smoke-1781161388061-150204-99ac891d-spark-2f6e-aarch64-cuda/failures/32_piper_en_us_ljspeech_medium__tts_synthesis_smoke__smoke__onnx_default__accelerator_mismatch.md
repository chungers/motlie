# piper_en_us_ljspeech_medium__tts_synthesis_smoke__smoke__onnx_default

## Record

- schema_version: 3
- git_sha: 99ac891d8a2adabe823cce61b2a9fec0aa5dbde3
- run_id: curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda
- profile: dgx-spark
- host_id: spark-2f6e
- arch: aarch64
- capability: tts
- bundle_id: piper_en_us_ljspeech_medium
- backend: ort
- checkpoint_format: onnx
- quantization: default
- requested_accelerator: cuda
- resolved_accelerator: cpu
- accelerator_backend_mode: piper:cpu
- accelerator_use_proof_source: backend_observation
- outcome: blocked
- reason: accelerator_mismatch
- overall_status: blocked
- failure_reason: accelerator section not accepted: requested=cuda resolved=cpu reason=accelerator_mismatch
- child_build_profile: release
- child_build_status: 0

## Repro

```sh
/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/target/release/evals run --bundle piper_en_us_ljspeech_medium --scenario tts_synthesis_smoke --profile dgx-spark --root /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/evals --artifact-root /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/libs/models/../../artifacts/models/hf-cache --jsonl /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/eval-runs-final-99ac891d/curated-v2-smoke/curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda/results.jsonl --run-id curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda --snapshot-id curated-v2-smoke --cell-id piper_en_us_ljspeech_medium__tts_synthesis_smoke__smoke__onnx_default --depth smoke --checkpoint-format onnx --artifact-quantization default --model-family piper --backend ort --requested-accelerator cuda --child-build-log /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/eval-runs-final-99ac891d/curated-v2-smoke/curated-v2-smoke-1781161388061-150204-99ac891d-spark-2f6e-aarch64-cuda/logs/piper_en_us_ljspeech_medium__tts_synthesis_smoke__smoke__onnx_default.log --child-build-status 0 --child-build-duration-ms 7604 --quiet-backend-logs
```

Child build command:

```sh
n/a
```

## Child Log Tail

```text
   Compiling rustls-pki-types v1.14.1
   Compiling motlie-model-espeak-ng v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/libs/model/backends/espeak_ng)
   Compiling motlie-models v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/libs/models)
   Compiling webpki-root-certs v1.0.7
   Compiling ureq v3.3.0
   Compiling ort-sys v2.0.0-rc.12 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/third_party/ort-sys)
   Compiling ort v2.0.0-rc.12
   Compiling motlie-model-ort v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/libs/model/backends/ort)
   Compiling motlie-model-piper v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/libs/model/backends/piper)
   Compiling evals v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/bins/evals)
    Finished `release` profile [optimized] target(s) in 7.52s

```
