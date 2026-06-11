# qwen3_tts_cpp_0_6b__tts_synthesis_smoke__smoke__gguf_q8_0

- runner: @codex-399-amd-rv
- host: amd-ryzen7-1
- profile: local-cpu-x86_64
- platform: linux x86_64 CPU
- build profile: release
- git_sha: e14eaa7e00ba51c4e35636f9312da5e5fb963238
- capability: tts
- backend: qwen3_tts_cpp
- checkpoint_format: gguf
- quantization: q8_0
- outcome: blocked
- coverage.reason: artifact_missing
- observed failure: child eval invocation failed after the native qwen3-tts.cpp child build completed; dynamic loader could not find libqwen3tts.so.0.

Repro command used for this targeted post-#470 rerun:

```sh
CARGO_TARGET_DIR=/tmp/motlie-rerun-e14eaa7e-0204-target \
  /tmp/motlie-rerun-e14eaa7e-0204-target/release/evals matrix \
  --snapshot /tmp/curated-v2-smoke-e14eaa7e-targeted-artifact-pattern.toml \
  --profile local-cpu-x86_64 \
  --artifact-root /home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie-rerun-e14eaa7e-20260611-0203/artifacts/models/hf-cache \
  --results-root evals/results
```

Submodule preflight log tail:

```text
qwen3-tts.cpp submodule preflight missing before init:
- /home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie-rerun-e14eaa7e-20260611-0203/libs/model/backends/qwen3_tts_cpp/vendor/qwen3-tts.cpp/CMakeLists.txt
- /home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie-rerun-e14eaa7e-20260611-0203/libs/model/backends/qwen3_tts_cpp/vendor/qwen3-tts.cpp/src/qwen3tts_c_api.cpp
- /home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie-rerun-e14eaa7e-20260611-0203/libs/model/backends/qwen3_tts_cpp/vendor/qwen3-tts.cpp/src/qwen3tts_c_api.h
- /home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie-rerun-e14eaa7e-20260611-0203/libs/model/backends/qwen3_tts_cpp/vendor/qwen3-tts.cpp/ggml/CMakeLists.txt
running: git submodule update --init --recursive libs/model/backends/qwen3_tts_cpp/vendor/qwen3-tts.cpp
Submodule path 'libs/model/backends/qwen3_tts_cpp/vendor/qwen3-tts.cpp': checked out '7a762e2ad4bacc6fdda81d81bf10a09ffb546f29'
Submodule 'ggml' (https://github.com/ggml-org/ggml.git) registered for path 'libs/model/backends/qwen3_tts_cpp/vendor/qwen3-tts.cpp/ggml'
Submodule path 'libs/model/backends/qwen3_tts_cpp/vendor/qwen3-tts.cpp/ggml': checked out '5cecdad692d868e28dbd2f7c468504770108f30c'
qwen3-tts.cpp submodule preflight satisfied after init
```

Child log tail:

```text
   Compiling motlie-model-qwen3-tts-cpp v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie-rerun-e14eaa7e-20260611-0203/libs/model/backends/qwen3_tts_cpp)
   Compiling motlie-models v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie-rerun-e14eaa7e-20260611-0203/libs/models)
warning: motlie-model-qwen3-tts-cpp@0.1.0: qwen3-tts.cpp built for target x86_64-unknown-linux-gnu
   Compiling evals v0.1.0 (/home/dchung/sessions/issue-399-eval-suite/codex-399-amd-rv/motlie-rerun-e14eaa7e-20260611-0203/bins/evals)
    Finished `release` profile [optimized] target(s) in 46.69s
/tmp/motlie-rerun-e14eaa7e-0204-target/release/evals: error while loading shared libraries: libqwen3tts.so.0: cannot open shared object file: No such file or directory
```
