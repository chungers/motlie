# ORT / ONNX Backend Policy

## Status: Active

## Change Log

| Date | Change | Sections |
|------|--------|----------|
| 2026-06-05 | @codex-369-rv: Corrected PR #393 after live Telnyx answer reproduced `free(): invalid pointer` with the rejected Pyke-ORT unification. The accepted single ORT provider is now workspace `ort-sys` patched to download/link the k2-fsa `sherpa-onnx` ORT archive that the upstream Sherpa C++ static bundle is built against. | Policy, Process-Level Rule, Sherpa ONNX Runtime Boundary, Canonical Cargo Path, Maintenance Procedures |
| 2026-06-05 | @codex-369-rv: Replaced the old Sherpa bundled-ORT exception with the unified one-process/one-ORT rule from PR #393 / #396. Added detailed integration scenarios for direct workspace `ort` backends versus upstream crates that bundle ONNX Runtime, plus model-family impact and maintenance procedures. | Policy, Sherpa ONNX Runtime Boundary, Future Model Integration Scenarios, Maintenance Procedures |
| 2026-06-01 | @codex-364-impl: Split the policy between Motlie-owned Pyke `ort` backends and the upstream `sherpa-onnx` backend, which must use the crate's downloaded static native archives instead of Motlie's workspace `ort` dependency. | Policy, Sherpa ONNX Exception |
| 2026-05-31 | @codex-364-impl: Changed the ORT/ONNX policy from source-built ONNX Runtime to the `ort/download-binaries` path, which downloads and statically links Pyke's `libonnxruntime.a` archive. | All |
| 2026-05-31 | @codex-364-impl: Superseded the earlier manual ONNX Runtime provisioning guidance with a general static-linkage policy for all `libs/model` ORT/ONNX backends. | All |

This policy applies to every Motlie model backend that uses ONNX Runtime or
loads `.onnx` artifacts. Motlie-owned Pyke `ort` backends such as Piper and
future ORT-backed bundles follow the workspace `ort` path below.
The Sherpa ASR backend still delegates recognition semantics to the upstream
`sherpa-onnx` Rust crate, but it no longer owns a separate ONNX Runtime instance
inside the Motlie gateway. The process-level rule is now simple: one binary gets
one static ONNX Runtime provider.

## Policy

- Motlie ORT/ONNX backends must link ONNX Runtime statically for local
  validation, CI, live tests, and deployment.
- Motlie ORT/ONNX backends must use the workspace `ort` dependency with
  `download-binaries`, `tls-native`, and `api-24` enabled.
- The accepted default path is the patched workspace `ort-sys` prebuilt
  download, which fetches the k2-fsa `sherpa-onnx` static package and links its
  `libonnxruntime.a` as the single process-wide ORT provider.
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

## Process-Level Rule

Every Motlie executable that links ONNX-backed model runtimes must have exactly
one ONNX Runtime provider. Multiple logical models may use that provider, but
the final native link must not include multiple independent ORT archives or
shared libraries.

| Case | Accepted? | Reason |
|------|-----------|--------|
| One patched workspace `ort-sys` static `libonnxruntime.a` linked by all ORT-backed backends | Yes | One process-wide ORT environment and allocator surface. In the Telnyx all-in-one binary this archive must come from the k2-fsa `sherpa-onnx` static package, not Pyke's `ms@1.24.2` archive. |
| Workspace `ort-sys` plus an upstream crate's bundled static `libonnxruntime.a` | No | Duplicates ORT global state and can crash at init, teardown, or allocator use. PR #393 reproduced this as `free(): invalid pointer`. |
| Workspace `ort-sys` plus a system or manually extracted `libonnxruntime.so` | No | Violates static-link policy and can still duplicate ORT state. |
| Multiple non-ORT native runtimes, such as `whisper.cpp` / GGML and Qwen3-TTS CPP, alongside workspace ORT | Yes, after validation | These are separate runtime stacks. They still need symbol-collision checks, but they are not ONNX Runtime providers. |

The current workspace provider is:

```toml
ort = { version = "2.0.0-rc.12", default-features = false, features = ["std", "ndarray", "download-binaries", "tls-native", "api-24"] }
```

The workspace also patches `ort-sys`:

```toml
[patch.crates-io]
ort-sys = { path = "third_party/ort-sys" }
```

On `x86_64-unknown-linux-gnu`, the patched `ort-sys` downloads
`sherpa-onnx-v1.13.2-linux-x64-static-lib.tar.bz2` and links that package's
`libonnxruntime.a`. The archive reports ONNX Runtime 1.24.4 and is ABI-safe for
the upstream `sherpa-onnx` C++ static bundle. Pyke's `ms@1.24.2` archive, and
the later Pyke `ms@1.24.4` archive tested during PR #393, both reproduced
`free(): invalid pointer` when Sherpa opened its upstream `OnlineRecognizer`.
This policy is not an ORT 1.22 policy and should not be documented as one.

## Sherpa ONNX Runtime Boundary

`libs/model/backends/sherpa_onnx` must use the upstream `sherpa-onnx` Rust crate
as the recognizer boundary. Motlie must not reintroduce a parallel in-house
greedy decoder over Pyke `ort` for Sherpa. The backend crate adapts upstream
`OnlineRecognizer` / `OnlineStream` to the typed `StreamingTranscriber`
contract.

The native-link boundary is different from the recognizer boundary:

| Layer | Crate / artifact | Owner | Policy |
|------|------------------|-------|--------|
| Motlie ASR contract adapter | `libs/model/backends/sherpa_onnx` | Motlie | Implements Motlie typed contracts and depends on upstream `sherpa-onnx`. |
| Upstream Sherpa Rust wrapper | `sherpa-onnx = "1.13.2"` | crates.io / k2-fsa | Provides `OnlineRecognizer` and `OnlineStream`; leave recognition semantics upstream. |
| Upstream Sherpa FFI/sys crate | `sherpa-onnx-sys = "1.13.2"` | Patched locally through `[patch.crates-io]` | Downloads/links Sherpa native archives, but Motlie filters out Sherpa's own emitted `onnxruntime` link so it cannot link a second ORT. |
| ONNX Runtime provider | patched workspace `ort-sys v2.0.0-rc.12` | Motlie workspace dependency | Downloads the k2-fsa Sherpa static package and supplies its `libonnxruntime.a` as the single static ORT archive for Sherpa, Piper, and future direct-ORT backends. |

The root workspace patch is:

```toml
[patch.crates-io]
ort-sys = { path = "third_party/ort-sys" }
sherpa-onnx-sys = { path = "third_party/sherpa-onnx-sys" }
```

The patched `third_party/ort-sys/build/main.rs` must continue to:

- default to `MOTLIE_ORT_SOURCE=sherpa-onnx`;
- fetch the k2-fsa `sherpa-onnx` static package for the target;
- link only that package's `libonnxruntime.a` through workspace `ort-sys`;
- keep `MOTLIE_ORT_SOURCE=pyke` as a debug escape hatch, not as an accepted
  Telnyx runbook path.

The patched `third_party/sherpa-onnx-sys/build.rs` must continue to:

- keep upstream Sherpa's static native archives and C API;
- remove `onnxruntime` from Sherpa's emitted static library list;
- expose a filtered static library directory that does not contain
  `libonnxruntime.a`;
- allow workspace `ort-sys` to satisfy ORT symbols, including `OrtGetApiBase`.

Sherpa runbooks must not set `ORT_LIB_PATH`, `ORT_PREFER_DYNAMIC_LINK`,
`LD_LIBRARY_PATH`, or require a local ONNX Runtime source build.

### Sherpa Model Family Impact

The Sherpa model artifacts are different checkpoints over the same runtime
family. A runtime/linkage change in `motlie-model-sherpa-onnx`,
`sherpa-onnx`, or `sherpa-onnx-sys` affects all Sherpa-backed curated models.

| Curated model | Artifact / selector | Runtime crate path | Upstream wrapper | Patched sys crate impact |
|---------------|---------------------|--------------------|------------------|--------------------------|
| `sherpa-2023` | `sherpa-zipformer-en-2023-06-26` | `libs/model/backends/sherpa_onnx` | `sherpa-onnx 1.13.2` | Yes. Uses patched `sherpa-onnx-sys 1.13.2` and workspace ORT. |
| `kroko-2025` | `sherpa-zipformer-en-kroko-2025-08-06` | `libs/model/backends/sherpa_onnx` | `sherpa-onnx 1.13.2` | Yes. Same runtime family and same patched sys crate. |

This maintenance burden is per runtime family, not per checkpoint. Adding a new
Sherpa Zipformer checkpoint should not need a new native-link patch. Upgrading
`sherpa-onnx` or `sherpa-onnx-sys` does.

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
patched Motlie `sherpa-onnx` ORT source and links the downloaded file as:

```sh
cargo:rustc-link-lib=static=onnxruntime
```

Operator runbooks for ORT-backed builds should not set ORT-specific
environment variables.

Do not use `cargo --offline` for the first ORT-backed build. The downloaded
archive is cached under Cargo's cache directory after patched `ort-sys` fetches
and extracts it.

## Crate Expectations

The shared `libs/model/backends/ort` crate and every concrete ORT-backed model
crate should depend on `ort.workspace = true`. A concrete backend may add
feature flags such as CUDA execution-provider support, but it must not silently
change the linkage policy.

Validation scripts should fail when an environment override would bypass the
downloaded static archive path.

## Future Model Integration Scenarios

Future models must be classified by how they reach ONNX Runtime before they are
added to `libs/model`, `libs/models`, or gateway feature sets. The classification
drives both the implementation shape and the long-term maintenance burden.

### Scenario A: Upstream Wrapper Or Sys Crate Bundles ONNX Runtime

This is the risky path. It was the Sherpa failure mode before PR #393: the
gateway linked workspace `ort-sys` for direct-ORT backends and also linked Sherpa's
bundled static `libonnxruntime.a`. The all-in-one ASR binary then had duplicate
ORT global state and aborted with `free(): invalid pointer`.

| Question | Required answer before merge |
|----------|------------------------------|
| Does the upstream crate emit `cargo:rustc-link-lib=static=onnxruntime` or `dylib=onnxruntime`? | If yes, block until it can be disabled or redirected to workspace `ort-sys`. |
| Does the upstream crate put a bundled `libonnxruntime.a` or `.so` on the link search path? | If yes, block or filter the directory so workspace `ort-sys` cannot accidentally pick the bundled archive. |
| Does the upstream crate expose `system-ort`, `external-ort`, `no-bundled-ort`, or equivalent? | Use that feature if it results in exactly one workspace ORT provider. |
| Does the upstream crate require a different ORT API/ABI than workspace `ort-sys`? | Align the workspace ORT source to the upstream-compatible archive if possible; otherwise isolate the model in another process. |
| Does the upstream crate own non-ORT native libraries too? | Validate those separately for symbol collisions, especially C++ or allocator-heavy libraries. |

Accepted implementation shapes for this scenario:

| Shape | Accepted? | Maintenance impact |
|-------|-----------|--------------------|
| Upstream crate has a supported feature to avoid bundled ORT and use external symbols from workspace `ort-sys` | Yes | Normal Cargo bump plus validation of the feature contract. |
| Local `[patch.crates-io]` copy of the upstream sys crate filters out bundled ORT | Yes, if necessary | Deliberate vendor refresh on each upstream sys-crate bump. |
| Upstream crate still links its bundled ORT while Motlie also links workspace ORT | No | Reintroduces duplicate ORT risk. |
| Upstream model runs in a separate process or service with its own ORT | Possible fallback | Avoids same-process duplicate ORT, but adds IPC/service complexity and is not the gateway default. |

The PR #393 Sherpa patch is the reference implementation for this scenario. It
keeps upstream recognition behavior, filters native ORT linkage from
`sherpa-onnx-sys`, and patches workspace `ort-sys` to link the k2-fsa Sherpa ORT
archive as the single provider. This is the combination that allows the Telnyx
gateway to run upstream Sherpa/Kroko plus Piper in one statically linked process
without reintroducing duplicate ORT state. A temporary Moonshine all-in-one
evaluation also validated the duplicate-ORT fix, but Moonshine is not an active
Telnyx live backend after #191/#394.

### Scenario B: Model Runs Directly On Workspace `ort`

This is the preferred path. The model backend owns the typed Motlie contract and
loads `.onnx` or `.ort` weights through the workspace `ort` crate.

| Requirement | Expected implementation |
|-------------|-------------------------|
| Runtime dependency | `ort.workspace = true` in the concrete backend crate. |
| Session construction | Use `libs/model/backends/ort` helpers where they fit; otherwise use `ort` APIs directly while preserving workspace linkage. |
| Artifact format | Curated `.onnx` or `.ort` artifacts under `libs/models` catalog entries. |
| Native linking | No model-specific ONNX Runtime build script, no bundled ORT archive, no `ORT_LIB_PATH`. |
| Validation | Build the concrete feature and at least one all-in-one target that also includes Sherpa. |

Maintenance remains normal Cargo maintenance:

| Change | Procedure |
|--------|-----------|
| Add a new direct-ORT model | Add backend crate code, curated artifacts, typed tests, and all-in-one build validation. No sys-crate patch expected. |
| Bump workspace `ort` | Bump once in root `Cargo.toml`, then run all ORT-backed model tests and gateway all-in-one validation. |
| Change model weights only | Re-run model-specific quality/latency tests; no native-link audit unless artifact format/runtime requirements change. |
| Add CUDA execution-provider support | Keep CPU static path as default; CUDA paths require separate DGX/appserver validation and must not silently change gateway linkage. |

### Scenario C: Non-ORT Native Model Runtime

Some model backends do not use ONNX Runtime at all, even if they are linked into
the same gateway or harness binary.

| Backend family | Runtime | ORT policy impact |
|----------------|---------|-------------------|
| `whisper.cpp` / `motlie-model-whisper-cpp` | GGML / whisper.cpp native runtime | Not an ORT provider. Validate separately for GGML/C++ symbol collisions and batch/final-pass behavior. |
| `qwen3-tts.cpp` / `motlie-model-qwen3-tts-cpp` | qwen3-tts.cpp plus GGML | Not an ORT provider. Keep its shared library and GGML symbols isolated from other GGML users. |
| Llama/Mistral style native backends | GGML, llama.cpp, mistral runtime, or other native stacks | Not an ORT provider unless they explicitly link ONNX Runtime. Validate native runtime coexistence independently. |

These backends do not need the `sherpa-onnx-sys` patch and do not affect the
workspace ORT version. They still belong in all-in-one validation if they ship in
the same binary.

## Current Backend Impact Matrix

| Backend / model family | Uses ONNX Runtime? | Current ORT source | Affected by `sherpa-onnx-sys` patch? | Notes |
|------------------------|-------------------|--------------------|--------------------------------------|-------|
| `sherpa-2023` | Yes | Upstream Sherpa recognizer resolves ORT symbols from patched workspace `ort-sys`, which links k2-fsa's Sherpa ORT archive | Yes | Same Sherpa runtime family as kroko; validates call-center profile. |
| `kroko-2025` | Yes | Upstream Sherpa recognizer resolves ORT symbols from patched workspace `ort-sys`, which links k2-fsa's Sherpa ORT archive | Yes | Same Sherpa runtime family; Telnyx balanced live default. |
| Moonshine | Yes | Direct workspace `ort` / patched `ort-sys`, using k2-fsa's Sherpa ORT archive if built with the gateway | No | Historical research candidate only for Telnyx after #191/#394; benefits from duplicate-ORT removal in combined binaries but is not an active live backend. |
| Piper TTS | Yes | Direct workspace `ort` / `motlie-model-ort`, using patched `ort-sys` | No | Telnyx outbound TTS path shares the single workspace ORT; no Sherpa maintenance burden. |
| Whisper | No | `whisper.cpp` native runtime | No | Batch/final-pass candidate; not an ORT provider. |
| Qwen3-TTS CPP | No | qwen3-tts.cpp / GGML | No | Not affected by ORT version or Sherpa sys patch; validate GGML coexistence separately. |

## Maintenance Procedures

### Adding A New ORT-Backed Model

1. Classify the model as Scenario A or Scenario B.
2. If Scenario B, depend on `ort.workspace = true` and keep model artifacts in
   the curated catalog.
3. If Scenario A, do not merge until the upstream wrapper can be made to avoid
   linking its bundled ORT in the Motlie process.
4. Build the concrete feature and an all-in-one binary that includes Sherpa and
   at least one other workspace-ORT backend.
5. Confirm no dynamic ORT appears in `ldd`.
6. Confirm `cargo tree -i ort-sys` shows one workspace `ort-sys` version.
7. Run model-specific functional tests and at least one representative
   all-in-one replay or smoke test.

### Upgrading Workspace `ort`

| Step | Check |
|------|-------|
| Update root `Cargo.toml` and the patched `third_party/ort-sys` source together | Keep `download-binaries`, `tls-native`, and the intended `api-*` feature explicit, and preserve the k2-fsa/Sherpa ORT source unless a full all-in-one validation proves a replacement. |
| Rebuild direct ORT backends | Piper and any future direct-ORT models must compile and run model-specific tests. If Moonshine is revived, it must pass this gate before gateway exposure. |
| Rebuild Sherpa | `cargo build -p motlie-telnyx-gateway --features sherpa` must still resolve ORT symbols from patched workspace `ort-sys`. |
| Re-run combined validation | Use an all-in-one binary containing Sherpa plus direct-ORT backends. |
| Re-check link evidence | `ldd` has no ORT, `cargo tree -i ort-sys` has one version, and no upstream bundled `libonnxruntime.a` is on a live link path. |

### Upgrading `sherpa-onnx` / `sherpa-onnx-sys`

This is no longer a simple Cargo version bump because Motlie carries a local
patch for the sys crate.

| Step | Check |
|------|-------|
| Update `sherpa-onnx` version | Start with the wrapper crate version required by the upstream release. |
| Refresh `third_party/sherpa-onnx-sys` to the matching version | Do not keep stale FFI bindings or build-script assumptions across versions. |
| Reapply Motlie's ORT filter | The sys build must not emit `static=onnxruntime`, and the filtered lib dir must not contain `libonnxruntime.a`. |
| Validate ORT ABI/API compatibility | If upstream Sherpa requires a newer ORT API/ABI than the current patched workspace `ort-sys` source, align the workspace ORT source rather than linking a second ORT. |
| Build Sherpa-only | `cargo build -p motlie-telnyx-gateway --features sherpa` must pass. |
| Build and run all-in-one | The all-in-one ASR/TTS harness must run without duplicate-ORT crashes. |
| Re-run WER/latency gate if ASR behavior can change | Both `sherpa-2023` and `kroko-2025` are affected by Sherpa runtime upgrades. |

### Verification Commands

These commands are examples. Adjust feature sets to include the model under
review.

| Purpose | Command |
|---------|---------|
| Build Sherpa-only live path | `cargo build -p motlie-telnyx-gateway --features sherpa` |
| Build combined Telnyx model path | `cargo build -p motlie-telnyx-gateway --features golden-ab` |
| Verify one workspace ORT crate | `cargo tree -p motlie-telnyx-gateway --features golden-ab -i ort-sys` |
| Verify no dynamic ORT in the binary | `ldd target/debug/telnyx-gateway \| rg -i 'onnx\|ort' \|\| true` |
| Verify patched Sherpa filtered dir excludes bundled ORT | `find target/debug/build/sherpa-onnx-sys-*/out/sherpa-onnx-static-without-ort -name libonnxruntime.a -print` |
| Verify patched workspace ORT source | `find ~/.cache -path '*motlie-sherpa-ort*' -name libonnxruntime.a -print` |
| Verify ORT symbol exists in final static binary | `nm target/debug/telnyx-gateway \| rg ' OrtGetApiBase$'` |

The `find` command above should print nothing. The `ldd` command should print
no ONNX Runtime shared libraries. `cargo tree -i ort-sys` should show one
`ort-sys` version and all ORT-backed backends routing through it.
