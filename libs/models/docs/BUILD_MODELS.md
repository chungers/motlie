# Curated Model Build Guide

## Change Log

| Date | Who | Summary |
|------|-----|---------|
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
| sherpa-onnx / Moonshine / Qwen3-TTS ONNX / Piper | ONNX Runtime shared library | Provide `ORT_LIB_PATH`, `pkg-config` metadata, or another explicit system installation path. |
| qwen3-tts.cpp | Vendored submodule checkout | `git submodule update --init --recursive libs/model/backends/qwen3_tts_cpp/vendor/qwen3-tts.cpp` |

## Canonical Environment Variables

| Variable | Purpose |
|----------|---------|
| `ESPEAK_NG_LIB_DIR` | Directory containing `libespeak-ng.so`, `libespeak-ng.so.1`, or `libespeak-ng.dylib` when the library is not in a standard linker path. |
| `PIPER_ESPEAKNG_DATA_DIRECTORY` | Directory containing `espeak-ng-data/` for Piper phonemization runtime assets. |
| `ORT_LIB_PATH` | Directory containing the ONNX Runtime shared library used by ORT-backed backends and examples. |

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

Full local build path when ONNX Runtime is available:

```bash
./scripts/check_curated_model_examples.sh --mode build
```

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
