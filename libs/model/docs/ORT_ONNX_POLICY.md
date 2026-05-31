# ORT / ONNX Backend Policy

## Status: Active

## Change Log

| Date | Change | Sections |
|------|--------|----------|
| 2026-05-31 | @codex-364-impl: Established the general ONNX Runtime policy for all `libs/model` ORT/ONNX backends: static linkage, source-built ONNX Runtime, and ORT-owned third-party dependency builds. | All |

This policy applies to every Motlie model backend that uses ONNX Runtime or
loads `.onnx` artifacts, including Piper, Sherpa ONNX, Moonshine, and future
ORT-backed bundles.

## Policy

- Motlie ORT/ONNX backends must link ONNX Runtime statically for local
  validation, CI, live tests, and deployment.
- `ORT_LIB_PATH` must point at a source-built static ONNX Runtime release
  directory containing `libonnxruntime.a` or `libonnxruntime_common.a`.
- `ORT_PREFER_DYNAMIC_LINK` must remain unset.
- `LD_LIBRARY_PATH` and extracted `onnxruntime-linux-*.tgz` shared-library
  releases are not accepted runbook paths.
- Motlie crates must not enable `ort` build-time binary downloads for library
  backends.
- ONNX Runtime's source build owns its third-party C/C++ dependency builds.
  Operators should provide host build tools only, not ORT internal libraries
  such as protobuf, FlatBuffers, Abseil, re2, nsync, or cpuinfo.

## Host Requirements

The host must provide build tools that ONNX Runtime cannot compile without:

- `git`
- Python `3.10+`
- CMake `3.28+`
- a Linux C++ compiler such as GCC `8+`

The runbook should verify these tools. It should not install ONNX Runtime
internal dependencies as host packages.

## Canonical Static Build

Use the ONNX Runtime release that matches the checked-in `ort` / `ort-sys`
binding generation. For `ort-sys 2.0.0-rc.12`, use ONNX Runtime `v1.24.2`:

```sh
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

`FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER` keeps the ONNX Runtime source build
from preferring system packages for FetchContent-managed dependencies.

## Crate Expectations

The shared `libs/model/backends/ort` crate and every concrete ORT-backed model
crate should depend on `ort` with `default-features = false`. A concrete backend
may add feature flags such as CUDA execution-provider support, but it must not
silently change the linkage policy.

Validation scripts should fail when `ORT_PREFER_DYNAMIC_LINK` is set or when
`ORT_LIB_PATH` points only at shared ONNX Runtime libraries.
