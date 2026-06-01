# ORT / ONNX Backend Policy

## Status: Active

## Change Log

| Date | Change | Sections |
|------|--------|----------|
| 2026-06-01 | @codex-364-impl: Split the policy between Motlie-owned Pyke `ort` backends and the upstream `sherpa-onnx` backend, which must use the crate's downloaded static native archives instead of Motlie's workspace `ort` dependency. | Policy, Sherpa ONNX Exception |
| 2026-05-31 | @codex-364-impl: Changed the ORT/ONNX policy from source-built ONNX Runtime to the `ort/download-binaries` path, which downloads and statically links Pyke's `libonnxruntime.a` archive. | All |
| 2026-05-31 | @codex-364-impl: Superseded the earlier manual ONNX Runtime provisioning guidance with a general static-linkage policy for all `libs/model` ORT/ONNX backends. | All |

This policy applies to every Motlie model backend that uses ONNX Runtime or
loads `.onnx` artifacts. Motlie-owned Pyke `ort` backends such as Piper,
Moonshine, and future ORT-backed bundles follow the workspace `ort` path below.
The Sherpa ASR backend is an explicit exception because it now delegates runtime
ownership to the upstream `sherpa-onnx` Rust crate.

## Policy

- Motlie ORT/ONNX backends must link ONNX Runtime statically for local
  validation, CI, live tests, and deployment.
- Motlie ORT/ONNX backends must use the workspace `ort` dependency with
  `download-binaries`, `tls-native`, and `api-24` enabled.
- The accepted default path is the `ort-sys` prebuilt download for the target,
  statically linked as `libonnxruntime.a`.
- `ORT_LIB_PATH` and `ORT_LIB_LOCATION` must remain unset so builds do not
  bypass the downloaded static archive with a user-provided or source-built ORT.
- `ORT_PREFER_DYNAMIC_LINK` must remain unset.
- `ORT_SKIP_DOWNLOAD`, `ORT_OFFLINE`, and Cargo offline mode must not be used
  for ORT-backed build targets because they disable the prebuilt archive path.
- `LD_LIBRARY_PATH` and extracted `onnxruntime-linux-*.tgz` shared-library
  releases are not accepted runbook paths.
- Motlie docs and scripts must not ask operators to build ONNX Runtime from
  source for normal ORT-backed model use.
- Motlie crates must not add local ONNX Runtime build scripts, vendored ORT
  source trees, or manual `ORT_LIB_PATH` setup for this path.

## Sherpa ONNX Exception

`libs/model/backends/sherpa_onnx` must use the upstream `sherpa-onnx` Rust
crate as its runtime boundary. That crate statically links by default and, when
`SHERPA_ONNX_LIB_DIR` is not set, downloads a matching prebuilt
`sherpa-onnx` static native archive from upstream releases. The archive includes
the ONNX Runtime library used internally by Sherpa.

Sherpa runbooks must not set `ORT_LIB_PATH`, `ORT_PREFER_DYNAMIC_LINK`,
`LD_LIBRARY_PATH`, or require a local ONNX Runtime source build. They also
should not add a parallel Motlie decoder over Pyke `ort`; Motlie's backend
crate should adapt upstream `OnlineRecognizer` / `OnlineStream` to the typed
`StreamingTranscriber` contract.

## Host Requirements

The host must allow Cargo to fetch the `ort-sys` prebuilt archive for the
target. No ONNX Runtime source checkout or CMake build is part of the Motlie
runbook.

For Linux targets, the final binary still links the C++ standard library used by
the prebuilt static archive. A normal Rust/C++ toolchain environment is expected.

## Canonical Cargo Path

Use the checked-in workspace dependency:

```toml
ort = { version = "2.0.0-rc.12", default-features = false, features = ["std", "ndarray", "download-binaries", "tls-native", "api-24"] }
```

For `x86_64-unknown-linux-gnu`, `ort-sys 2.0.0-rc.12` resolves the
`ms@1.24.2` Pyke archive and links the downloaded file as:

```sh
cargo:rustc-link-lib=static=onnxruntime
```

Operator runbooks for ORT-backed builds should not set ORT-specific
environment variables.

Do not use `cargo --offline` for the first ORT-backed build. The downloaded
archive is cached under Cargo's cache directory after `ort-sys` fetches and
verifies it.

## Crate Expectations

The shared `libs/model/backends/ort` crate and every concrete ORT-backed model
crate should depend on `ort.workspace = true`. A concrete backend may add
feature flags such as CUDA execution-provider support, but it must not silently
change the linkage policy.

Validation scripts should fail when an environment override would bypass the
downloaded static archive path.
