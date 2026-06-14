# ORT / ONNX Backend Policy

## Status: Active

## Change Log

| Date | Change | Sections |
|------|--------|----------|
| 2026-06-13 | @513-impl: Updated policy for issue #513: DGX CUDA audio requires Motlie source-built static ONNX Runtime with the CUDA EP statically linked; Pyke is rejected for eval/deploy parity. | Policy, CUDA Static ORT, Sherpa ONNX Runtime Source |
| 2026-06-01 | @codex-364-impl: Split policy for workspace ORT backends and the upstream sherpa-onnx runtime boundary. | Superseded |
| 2026-05-31 | @codex-364-impl: Documented the older Pyke prebuilt static archive path. | Superseded |

This policy applies to every Motlie model backend that uses ONNX Runtime or loads `.onnx` artifacts. The workspace patches `ort-sys` under `libs/model/backends/ort/ort-sys` so eval and deployment builds use the selected Motlie static runtime source instead of host dynamic ORT libraries.

## Policy

- Motlie ORT/ONNX backends must link ONNX Runtime statically for local validation, CI, live tests, and deployment.
- Eval and deployment builds must keep `MOTLIE_ORT_SOURCE=sherpa-onnx` unless a task explicitly changes this policy. `MOTLIE_ORT_SOURCE=pyke` is not accepted for DGX CUDA audio because it does not provide the required aarch64 CUDA EP and would create an eval-only path.
- `ORT_LIB_PATH`, `ORT_LIB_LOCATION`, `ORT_PREFER_DYNAMIC_LINK`, shared ORT release archives, and `LD_LIBRARY_PATH`-based ORT selection are not accepted runbook paths.
- `ORT_SKIP_DOWNLOAD`, `ORT_OFFLINE`, and Cargo offline mode must not be used for first-time ORT-backed eval builds because they disable the selected static runtime fetch/build path.
- ORT backends that request CUDA must register the CUDA execution provider with `error_on_failure`; a compiled-but-unusable EP must fail loudly instead of silently running CPU.

## CUDA Static ORT

DGX CUDA audio builds must use a Motlie source-built static ONNX Runtime with `onnxruntime_providers_cuda` statically linked. The only runtime dynamic libraries allowed for this path are the host CUDA driver (`libcuda.so`) and cuDNN 9.

The patched `libs/model/backends/ort/ort-sys` resolver rejects the CPU-only `sherpa-onnx` static archive whenever `ort/cuda` is enabled. CUDA builds compile ONNX Runtime v1.24.2 from the submodule at `libs/model/backends/ort/vendor/onnxruntime`, apply the Motlie static-CUDA patch in the build output tree, and link split static ONNX Runtime archives plus `libonnxruntime_providers_cuda.a`. `MOTLIE_ORT_CUDA_STATIC_LIB_DIR` remains a developer override for an equivalent prebuilt static CUDA ORT directory, but it is not the committed eval/deploy path. The linker adds static CUDA toolkit archives when the CUDA provider archive is present.

cuDNN 9 headers and library are a hard prerequisite for this path. On DGX/SBSA CUDA 13 hosts, install `libcudnn9-cuda-13` and `libcudnn9-dev-cuda-13`, or set `CUDNN_HOME`/`CUDNN_PATH` to a cuDNN 9 installation.

## Sherpa ONNX Runtime Source

The `MOTLIE_ORT_SOURCE=sherpa-onnx` name is the stable Motlie vendoring shape for both CPU and CUDA ORT-backed audio evals:

- CPU builds use the upstream `sherpa-onnx` static native archive selected by the patched `ort-sys` build script.
- CUDA builds use the Motlie source-built static CUDA ORT from `libs/model/backends/ort/vendor/onnxruntime` instead of the upstream CPU archive.

The `libs/model/backends/sherpa_onnx` crate still uses the upstream `sherpa-onnx` Rust runtime boundary for recognition. That does not permit a separate dynamic ORT override for evals.

## Host Requirements

The host must provide a normal Rust/C++ build environment. CPU ORT builds need network access for the first static archive fetch unless the cache or `SHERPA_ONNX_ARCHIVE_DIR` is already populated. CUDA ORT builds additionally need CUDA 13 tooling and cuDNN 9.

For Linux targets, the final binary still links the C++ standard library used by the static archives.

## Crate Expectations

The shared `libs/model/backends/ort` crate and every concrete ORT-backed model crate should depend on `ort.workspace = true`. A concrete backend may add feature flags such as CUDA execution-provider support, but it must not silently change the linkage policy.

Validation scripts should fail when an environment override would bypass the selected static archive or source-built static CUDA ORT path.
