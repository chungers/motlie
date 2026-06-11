# qwen3_tts_cpp_0_6b__tts_synthesis_smoke__smoke__gguf_q8_0

## Record

- schema_version: 3
- git_sha: e14eaa7e00ba51c4e35636f9312da5e5fb963238
- run_id: curated-v2-smoke-1781169935139-380628-e14eaa7e-spark-2f6e-aarch64-cpu
- profile: local-cpu-aarch64
- host_id: spark-2f6e
- arch: aarch64
- capability: tts
- bundle_id: qwen3_tts_cpp_0_6b
- backend: qwen3_tts_cpp
- checkpoint_format: gguf
- quantization: q8_0
- requested_accelerator: cpu
- resolved_accelerator: cpu
- accelerator_backend_mode: cpu
- accelerator_use_proof_source: profile:cpu
- outcome: blocked
- reason: artifact_missing
- overall_status: blocked
- failure_reason: child eval invocation failed; see /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/eval-runs-targeted-e14eaa7e/curated-v2-smoke/curated-v2-smoke-1781169935139-380628-e14eaa7e-spark-2f6e-aarch64-cpu/logs/qwen3_tts_cpp_0_6b__tts_synthesis_smoke__smoke__gguf_q8_0.log
- child_build_profile: release
- child_build_status: 0

## Repro

```sh
target/debug/evals matrix --snapshot /tmp/curated-v2-smoke-artifact-pattern-targeted.toml --profile local-cpu-aarch64 --artifact-root /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/artifacts/models/hf-cache --results-root /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/eval-runs-targeted-e14eaa7e
```

Child build command:

```sh
cargo build --release -p evals --no-default-features --features model-qwen3-tts-cpp
```

## Child Log Tail

```text
   Compiling motlie-model-qwen3-tts-cpp v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie-targeted-e14eaa7e/libs/model/backends/qwen3_tts_cpp)
   Compiling motlie-models v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie-targeted-e14eaa7e/libs/models)
warning: motlie-model-qwen3-tts-cpp@0.1.0: qwen3-tts.cpp built for target aarch64-unknown-linux-gnu
   Compiling evals v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie-targeted-e14eaa7e/bins/evals)
    Finished `release` profile [optimized] target(s) in 27.70s
/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie-targeted-e14eaa7e/target/release/evals: error while loading shared libraries: libqwen3tts.so.0: cannot open shared object file: No such file or directory

```
