# Failure: `qwen3_tts_cpp_0_6b__tts_synthesis_smoke__smoke__gguf_q8_0`

- bundle: `qwen3_tts_cpp_0_6b`
- capability: `tts`
- profile: `local-cpu-x86_64`
- arch: `x86_64`
- backend: `qwen3_tts_cpp`
- checkpoint_format: `gguf`
- quantization: `q8_0`
- outcome: `blocked`
- reason: `submodule_missing`
- child_build_status: `101`
- child_build_profile: `debug`
- HF_TOKEN_PRESENT: `true`

## Child Build Command

```sh
cargo build -p evals --no-default-features --features model-qwen3-tts-cpp
```

## Repro Command

```sh
target/debug/evals matrix --snapshot evals/snapshots/curated-v2-smoke.toml --profile local-cpu-x86_64 --results-root /tmp/motlie-codex-399-amd-rv-real-eval-01e3e487
```

## Child Log Tail

```text
   Compiling motlie-model-qwen3-tts-cpp v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie/libs/model/backends/qwen3_tts_cpp)
error: failed to run custom build command for `motlie-model-qwen3-tts-cpp v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie/libs/model/backends/qwen3_tts_cpp)`

Caused by:
  process didn't exit successfully: `/home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie/target/debug/build/motlie-model-qwen3-tts-cpp-10ff89ad9fa0a370/build-script-build` (exit status: 101)
  --- stdout
  cargo:rerun-if-changed=build.rs
  cargo:rerun-if-changed=.gitmodules
  cargo:rerun-if-changed=vendor/qwen3-tts.cpp/.git
  cargo:rerun-if-changed=vendor/qwen3-tts.cpp/CMakeLists.txt
  cargo:rerun-if-changed=vendor/qwen3-tts.cpp/src/qwen3tts_c_api.cpp
  cargo:rerun-if-changed=vendor/qwen3-tts.cpp/src/qwen3tts_c_api.h
  cargo:rerun-if-changed=vendor/qwen3-tts.cpp/ggml/CMakeLists.txt

  --- stderr

  thread 'main' (3418822) panicked at libs/model/backends/qwen3_tts_cpp/build.rs:127:13:
  qwen3-tts.cpp submodule checkout at `vendor/qwen3-tts.cpp` is incomplete; missing `vendor/qwen3-tts.cpp/CMakeLists.txt`; run `git submodule update --init --recursive libs/model/backends/qwen3_tts_cpp/vendor/qwen3-tts.cpp`
  note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace

```
