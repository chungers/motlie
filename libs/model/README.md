# motlie-model

Core contracts for curated model bundles in the Motlie ecosystem.

This crate is intended to stay small and stable. It owns model lifecycle and
capability contracts plus lightweight `model::eval` abstractions. Substantial
evaluation tooling belongs in `motlie-model-eval`.

ORT/ONNX-backed model crates must follow the shared static-linking policy in
[docs/ORT_ONNX_POLICY.md](docs/ORT_ONNX_POLICY.md).
