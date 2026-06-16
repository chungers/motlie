# evals Binary

`bins/evals` is the feature-light CLI for Motlie curated model evaluation. The
repo-level scenario, snapshot, result, and report data live under `evals/`.

## Build Policy

`evals matrix` compiles each snapshot cell in a child Cargo process with only
that cell's model features. The child environment is part of the eval contract:
results should describe repo policy, not whatever model runtime variables were
left in a host shell.

- GGUF cells on Linux receive repo-wired `BINDGEN_EXTRA_CLANG_ARGS` for
  `tools/clang-compat/include`, with compiler builtin include dirs appended when
  discovered.
- ORT-backed cells prefer static ONNX Runtime. The driver removes dynamic/offline
  ORT overrides (`ORT_LIB_PATH`, `ORT_LIB_LOCATION`, `ORT_PREFER_DYNAMIC_LINK`,
  `ORT_SKIP_DOWNLOAD`, `ORT_OFFLINE`, `CARGO_NET_OFFLINE`) and sets
  `MOTLIE_ORT_SOURCE=sherpa-onnx` so patched workspace `ort-sys` links the
  k2-fsa static ORT archive.
- Link-stage failures are classified as `native_link_failed`; they must not be
  reported as Hugging Face artifact authorization failures.

See `evals/README.md` for the index, `evals/RUNBOOK.md` for operator commands, and `evals/docs/PROCESS.md` for standing policies.
