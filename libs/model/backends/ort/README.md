# motlie-model-ort

Shared ONNX Runtime helpers for Motlie model backends.

All backends that use this crate must follow the general Motlie ORT/ONNX policy:
[../../docs/ORT_ONNX_POLICY.md](../../docs/ORT_ONNX_POLICY.md).

In short: use the workspace `ort/download-binaries` dependency, let Cargo
download Pyke's prebuilt `libonnxruntime.a`, and statically link it without
`ORT_LIB_PATH`, `LD_LIBRARY_PATH`, or a local ONNX Runtime source build.
