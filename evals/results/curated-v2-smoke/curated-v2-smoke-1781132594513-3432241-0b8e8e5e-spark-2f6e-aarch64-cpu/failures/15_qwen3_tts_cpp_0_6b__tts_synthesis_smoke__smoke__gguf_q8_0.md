# qwen3_tts_cpp_0_6b__tts_synthesis_smoke__smoke__gguf_q8_0

- outcome: `blocked`
- reason: `artifact_missing`
- acceptance: `blocked`
- failure_reason: `child eval invocation failed; see /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/eval-runs-rerun-0b8e8e5e/curated-v2-smoke/curated-v2-smoke-1781132594513-3432241-0b8e8e5e-spark-2f6e-aarch64-cpu/logs/qwen3_tts_cpp_0_6b__tts_synthesis_smoke__smoke__gguf_q8_0.log`
- bundle: `qwen3_tts_cpp_0_6b`
- capability: `tts`
- scenario: `tts_synthesis_smoke`
- profile: `local-cpu-aarch64`
- host_id: `spark-2f6e`
- platform: `linux/aarch64`
- requested_accelerator: `cpu`
- resolved_accelerator: `cpu`
- backend: `qwen3_tts_cpp`
- checkpoint_format: `gguf`
- quantization: `q8_0`
- git_sha: `0b8e8e5ecb53c5256037b8970041446a1637515b`
- build_profile: `None`
- cargo_features: `model-qwen3-tts-cpp`
- accelerator_backend_mode: `cpu`
- accelerator_offload: `None`
- child_build_status: `0`
- child_build_duration_ms: `4916`

## Runtime Environment

- HF_TOKEN_PRESENT: `[REDACTED_PRESENT]`

## Repro Command

```sh
target/debug/evals matrix --snapshot evals/snapshots/curated-v2-smoke.toml --profile local-cpu-aarch64 --results-root /home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/eval-runs-rerun-0b8e8e5e
```

## Child Log Tail

```text
   Compiling motlie-model-qwen3-tts-cpp v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/libs/model/backends/qwen3_tts_cpp)
warning: motlie-model-qwen3-tts-cpp@0.1.0: qwen3-tts.cpp built for target aarch64-unknown-linux-gnu
   Compiling motlie-models v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/libs/models)
   Compiling evals v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/bins/evals)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 4.84s
/home/dchung/sessions/issue-399-eval-suite/codex-399-cuda-rv/motlie/target/debug/evals: error while loading shared libraries: libqwen3tts.so.0: cannot open shared object file: No such file or directory
```
