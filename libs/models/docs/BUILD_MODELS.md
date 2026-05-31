# Curated Model Build Guide

## Change Log

| Date | Who | Summary |
|------|-----|---------|
| 2026-05-31 | @codex-364-impl | Clarified that ONNX Runtime source builds own their third-party C/C++ dependencies through FetchContent; Motlie runbooks should verify host build tools, not install ORT internal dependency packages. |
| 2026-05-31 | @codex-364-impl | Changed ONNX Runtime provisioning guidance from shared-library discovery to static source-built linkage through `ORT_LIB_PATH`; dynamic `ORT_PREFER_DYNAMIC_LINK` runbooks are no longer accepted. |
| 2026-04-24 | @codex-gpt55 | Added Qwen3.6 27B GGUF build guidance for the new llama.cpp curated example, including the feature gate, CUDA gate, and current FP8 GGUF limitation. |
| 2026-04-22 | @codex-tts | Added a dedicated `smoke-qwen3-whisper` mode to `scripts/check_curated_model_examples.sh` for issue `#211`. This exercises `tts_qwen3_tts_cpp | asr_whisper` under the same feature set to catch Linux `ggml` symbol-interposition regressions around `-Wl,-Bsymbolic`. |
| 2026-04-21 | @codex-tts | Removed the non-functional Qwen3-TTS ONNX curated backend per issue `#210`. The ONNX Runtime prerequisite list now covers only the surviving ORT-backed bundles: Piper, sherpa-onnx, and Moonshine. |
| 2026-04-18 | @codex-asr | Added the canonical build-prerequisite guide for curated model backends, including the Piper `libespeak-ng` system dependency, ONNX Runtime provisioning, qwen3-tts.cpp submodule setup, and the script/CI entry points that enforce these requirements. |

## Why This Exists

The curated model stack has a few host-level prerequisites that are easy to
forget if they only live in PR comments or reviewer memory. This guide is the
single source of truth for those requirements.

The main principles are:

- fail fast on missing system prerequisites
- never hide missing prerequisites behind build-time download/bootstrap logic
- keep the build checks scripted so CI and local validation call the same paths

## Backend Prerequisites

| Backend | Requirement | How it is enforced |
|--------|-------------|--------------------|
| Piper | System `libespeak-ng` shared library | `motlie-model-espeak-ng/build.rs` fails explicitly if the library cannot be found. |
| Piper | `espeak-ng-data` runtime assets | Runtime prerequisite. Set `PIPER_ESPEAKNG_DATA_DIRECTORY` if the data is not installed in a standard system location. |
| sherpa-onnx / Moonshine / Piper | Static ONNX Runtime build | `scripts/check_models_build_prereqs.sh --require-ort` requires `ORT_LIB_PATH` to point at a directory containing `libonnxruntime.a` or `libonnxruntime_common.a`, and rejects `ORT_PREFER_DYNAMIC_LINK`. |
| qwen3-tts.cpp | Vendored submodule checkout | `git submodule update --init --recursive libs/model/backends/qwen3_tts_cpp/vendor/qwen3-tts.cpp` |
| llama.cpp GGUF | Optional CUDA build | Add `llama-cpp-cuda` to the `motlie-models` feature set. Runtime GPU offload can be disabled with `MOTLIE_MODEL_FORCE_CPU=1` or controlled with `MOTLIE_MODEL_GPU_LAYERS=<n>`. |

## Canonical Environment Variables

| Variable | Purpose |
|----------|---------|
| `ESPEAK_NG_LIB_DIR` | Directory containing `libespeak-ng.so`, `libespeak-ng.so.1`, or `libespeak-ng.dylib` when the library is not in a standard linker path. |
| `PIPER_ESPEAKNG_DATA_DIRECTORY` | Directory containing `espeak-ng-data/` for Piper phonemization runtime assets. |
| `ORT_LIB_PATH` | Directory containing the static ONNX Runtime build used by ORT-backed backends and examples, such as `$HOME/src/onnxruntime-1.24.2/build/Linux/Release`. |
| `MOTLIE_MODEL_FORCE_CPU` | Set to `1` to force llama.cpp-backed bundles to use zero GPU-offloaded layers. |
| `MOTLIE_MODEL_GPU_LAYERS` | Explicit llama.cpp GPU layer count. When unset, llama.cpp-backed bundles request full offload and the compiled backend decides what is available. |

## Canonical Checks

### 1. Prerequisite Check

```bash
./scripts/check_models_build_prereqs.sh --require-espeak --require-qwen-submodule
```

For a full build that links ORT-backed examples:

```bash
./scripts/check_models_build_prereqs.sh --require-espeak --require-qwen-submodule --require-ort
```

### 2. Scripted Curated-Example Check

Fast `cargo check` path used by CI:

```bash
./scripts/check_curated_model_examples.sh --mode check
```

The scripted curated-example check intentionally targets the lightweight
speech-model matrix. Large chat bundles such as Qwen3.6 27B GGUF should use the
manual feature-gated checks below so CI does not pull heavyweight llama.cpp
model builds into the default path.

Full local build path when ONNX Runtime is available:

```bash
./scripts/check_curated_model_examples.sh --mode build
```

## Static ONNX Runtime Policy

ORT-backed Motlie examples and gateway flows should link ONNX Runtime
statically. Build ONNX Runtime from the release tag that matches the checked-in
`ort`/`ort-sys` bindings, set `ORT_LIB_PATH` to the static build output, and
leave `ORT_PREFER_DYNAMIC_LINK` unset.

Ubuntu static build for the current `ort-sys 2.0.0-rc.12` binding generation:

```bash
command -v git
python3 -c 'import sys; assert sys.version_info >= (3, 10), sys.version'
cmake --version
${CXX:-c++} --version

export ORT_VERSION=v1.24.2
export ORT_SRC="$HOME/src/onnxruntime-${ORT_VERSION#v}"
git clone --branch "$ORT_VERSION" --depth 1 --recursive --shallow-submodules \
  https://github.com/microsoft/onnxruntime.git "$ORT_SRC"
cd "$ORT_SRC"
./build.sh --config Release --parallel --compile_no_warning_as_error \
  --skip_submodule_sync --skip_tests \
  --cmake_extra_defines FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER

export ORT_LIB_PATH="$ORT_SRC/build/Linux/Release"
test -f "$ORT_LIB_PATH/libonnxruntime.a" || test -f "$ORT_LIB_PATH/libonnxruntime_common.a"
unset ORT_PREFER_DYNAMIC_LINK
```

Do not use `ORT_PREFER_DYNAMIC_LINK=1`, `LD_LIBRARY_PATH`, or an extracted
`onnxruntime-linux-*.tgz` shared-library package as the documented path for
local validation, CI, live tests, or deployment.

Do not install ORT internal dependencies such as protobuf, FlatBuffers, Abseil,
re2, nsync, or cpuinfo as host packages for this path. The
`FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER` CMake define keeps the source build
from preferring system packages for FetchContent-managed dependencies.

Dedicated qwen3-tts.cpp / whisper co-link smoke for Linux symbol-interposition regressions:

```bash
export QWEN3_TTS_CPP_ARTIFACT_ROOT=/path/to/qwen3-tts-models
export WHISPER_ARTIFACT_ROOT=/path/to/whisper-cache-or-snapshot

./scripts/check_curated_model_examples.sh --mode smoke-qwen3-whisper
```

This smoke exists specifically to catch the failure mode from issue `#211`,
where `libqwen3tts.so` and `whisper.cpp` can bind the wrong `ggml` symbols if
the qwen3-tts.cpp shared library is linked without Linux-side symbol isolation.

### 3. Qwen3.6 27B GGUF Example

CPU/macOS-oriented build:

```bash
cargo check -p motlie-models \
  --no-default-features \
  --features model-qwen3-6-27b-gguf \
  --example chat_multimodal_qwen3_6_27b
```

CUDA-enabled build:

```bash
cargo check -p motlie-models \
  --no-default-features \
  --features model-qwen3-6-27b-gguf,llama-cpp-cuda \
  --example chat_multimodal_qwen3_6_27b
```

Runtime example:

```bash
cargo run -p motlie-models \
  --no-default-features \
  --features model-qwen3-6-27b-gguf,llama-cpp-cuda \
  --example chat_multimodal_qwen3_6_27b -- \
  --precision=q8 "Summarize Rust ownership in one paragraph"
```

The curated Qwen3.6 GGUF slice currently advertises Q4_K_M, Q5_K_M, and Q8_0.
The requested CUDA FP8 default remains blocked until a real FP8 GGUF artifact is
available; the official FP8 release is safetensors/Transformers format rather
than GGUF.

Current validation status:

- The Qwen3.6 example, bundle metadata, selector/catalog wiring, and llama.cpp
  backend unit tests have build/unit coverage.
- Live generation from real Qwen3.6 27B GGUF artifacts is still a hardware- and
  artifact-gated manual smoke.
- Image input is intentionally fail-closed until llama.cpp `mtmd`/mmproj support
  is wired; the bundle does not advertise `Vision`.

## Design Rule: No Hidden Bootstrap

For Piper specifically, Motlie no longer vendors or bootstraps `espeak-ng`
during Cargo builds. The old `espeak-rs-sys` CMake / `FetchContent` path was
removed because it could fail mid-bootstrap and poison later runs.

The supported behavior is now:

1. require a system `libespeak-ng`
2. fail explicitly if it is missing
3. keep the runtime data requirement explicit with `PIPER_ESPEAKNG_DATA_DIRECTORY`

If a future backend introduces another host-level prerequisite, document it
here and add it to `scripts/check_models_build_prereqs.sh`.

## CI Hook

The GitHub Actions hook for these checks lives in:

- `.github/workflows/models-build.yml`

The workflow is intentionally thin. The real logic belongs in:

- `scripts/check_models_build_prereqs.sh`
- `scripts/check_curated_model_examples.sh`
